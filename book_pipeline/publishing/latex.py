"""Deterministic rendering, original-number anchors and exact source accounting."""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import unicodedata

from book_pipeline.config import ROOT, digest, save
from book_pipeline.structure.semantics import HEADINGS, LABELS
from book_pipeline.structure.numbered import CAPTION_ROLES
from book_pipeline.ingestion.source import corrected_content
from book_pipeline.ingestion.math_syntax import math_compatibility, formula_tags, compatible_equation_tag
from book_pipeline.structure.paragraphs import supported_joins
from book_pipeline.structure.heading_layout import HeadingLayout
from book_pipeline.publishing.colors import COLOR_KINDS, color_definitions, selected_color, validate_colors
from book_pipeline.publishing.figure_recovery import recover_fragmented_figures
from book_pipeline.publishing.image_rows import recover_horizontal_images
from book_pipeline.publishing.typography import validate_typography, subheading_size, trim_title_punctuation

ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


GREEK_MATH = {'α':r'\alpha','β':r'\beta','γ':r'\gamma','δ':r'\delta',
              'ε':r'\varepsilon','ζ':r'\zeta','η':r'\eta','θ':r'\theta',
              'ι':r'\iota','κ':r'\kappa','λ':r'\lambda','μ':r'\mu',
              'ν':r'\nu','ξ':r'\xi','π':r'\pi','ρ':r'\rho',
              'σ':r'\sigma','τ':r'\tau','υ':r'\upsilon','φ':r'\varphi',
              'χ':r'\chi','ψ':r'\psi','ω':r'\omega','Γ':r'\Gamma',
              'Δ':r'\Delta','Θ':r'\Theta','Λ':r'\Lambda','Ξ':r'\Xi',
              'Π':r'\Pi','Σ':r'\Sigma','Υ':r'\Upsilon','Φ':r'\Phi',
              'Ψ':r'\Psi','Ω':r'\Omega','∂':r'\partial','∇':r'\nabla',
              'Α':'A','Β':'B','Ε':'E','Ζ':'Z','Η':'H','Ι':'I','Κ':'K',
              'Μ':'M','Ν':'N','Ο':'O','Ρ':'P','Τ':'T','Χ':'X','ο':'o',
              'ς':r'\varsigma','ϵ':r'\epsilon','ϑ':r'\vartheta',
              'ϰ':r'\varkappa','ϕ':r'\phi','ϱ':r'\varrho','ϖ':r'\varpi',
              'ϴ':r'\Theta','ı':r'\imath','ȷ':r'\jmath'}


def latin_quote_context(text):
    """Require Latin prose without CJK before changing ambiguous quote widths."""
    return bool(re.search(r'[A-Za-z]', text) and not re.search(
        r'[\u2e80-\u9fff\uf900-\ufaff\U00020000-\U0003134f]', text))


def literal_text(text, latin_quotes=None):
    # PDF extraction may detach the combining negation slash from its equals.
    text = re.sub('\u0338'+r'\s*=', '≠', text)
    if latin_quotes is None:
        latin_quotes = latin_quote_context(text)
    result = []
    for n, char in enumerate(text):
        # A word-internal apostrophe is unambiguous even in Chinese prose.
        apostrophe = (char == '’' and 0 < n < len(text)-1
                      and text[n-1].isascii() and text[n-1].isalpha()
                      and text[n+1].isascii() and text[n+1].isalpha())
        if char in '‘’“”' and (latin_quotes or apostrophe):
            result.append(r'\BookLatinQuote{' + char + '}')
        elif ord(char) >= 32 or char == '\n':
            result.append(unicode_symbol(char) or ESCAPES.get(char, char))
    return ''.join(result)


URL_TEXT = re.compile(r"(?:https?://|www\.)[^\s<>(){}，。；：、]+")


def escape(text, latin_quotes=None):
    # URLs remain literal source text, with invisible break opportunities so a
    # bibliography address cannot extend beyond the physical page.
    if latin_quotes is None:
        latin_quotes = latin_quote_context(text)
    chunks, cursor = [], 0
    for match in URL_TEXT.finditer(text):
        chunks.append(literal_text(text[cursor:match.start()], latin_quotes))
        chunks.append(r"\allowbreak{}".join(literal_text(c, latin_quotes) for c in match[0]))
        cursor = match.end()
    chunks.append(literal_text(text[cursor:], latin_quotes))
    return "".join(chunks)


def unicode_symbol(char, math=False):
    # Circled footnote markers occur inside superscripts as well as ordinary text.
    # NewTX's OT1 math text font cannot represent these Unicode characters.
    if math and char in GREEK_MATH:
        return GREEK_MATH[char]
    code = ord(char)
    if 0x1D400 <= code <= 0x1D7FF:
        # OCR sometimes leaves mathematical alphanumerics in prose. Text
        # fonts do not contain them; retain the variable and its math style.
        name = unicodedata.name(char, '')
        base = unicodedata.normalize('NFKC', char)
        base = GREEK_MATH.get(base, base)
        if base != char:
            command = ('mathbb' if 'DOUBLE-STRUCK' in name else 'mathfrak' if 'FRAKTUR' in name
                       else 'mathcal' if 'SCRIPT' in name else 'mathtt' if 'MONOSPACE' in name
                       else 'mathsf' if 'SANS-SERIF' in name else None)
            if command:base = '\\'+command+'{'+base+'}'
            if 'BOLD' in name:base = r'\boldsymbol{'+base+'}'
            return base if math else r'\ensuremath{'+base+'}'
    symbols={'∈':r'\in','∉':r'\notin','≠':r'\neq','ℝ':r'\mathbb{R}',
             '♡':r'\heartsuit','♢':r'\diamondsuit','♣':r'\clubsuit','♠':r'\spadesuit',
             '⋯':r'\cdots','✓':r'\checkmark','∑':r'\sum','∏':r'\prod','∫':r'\int',
             '√':r'\surd','±':r'\pm','×':r'\times','÷':r'\div','≤':r'\leq',
             '≥':r'\geq','≈':r'\approx','→':r'\to','←':r'\leftarrow','↔':r'\leftrightarrow',
             '∀':r'\forall','∃':r'\exists','∅':r'\varnothing','⊂':r'\subset',
             '∪':r'\cup','∩':r'\cap',
             '⊆':r'\subseteq','⊥':r'\perp','⊤':r'\top','∥':r'\parallel',
             '∧':r'\land','∨':r'\lor','¬':r'\neg','∞':r'\infty',
             '⇒':r'\Rightarrow','⇐':r'\Leftarrow','⇔':r'\Leftrightarrow','∼':r'\sim',
             '∴':r'\therefore','∵':r'\because'}
    if char in symbols:
        return symbols[char] if math else r'\ensuremath{'+symbols[char]+'}'
    if char in '◀▶':
        symbol = r'\blacktriangleleft' if char == '◀' else r'\blacktriangleright'
        return symbol if math else r'\ensuremath{' + symbol + '}'
    if char in '₀₁₂₃₄₅₆₇₈₉':
        subscript='_{'+str('₀₁₂₃₄₅₆₇₈₉'.index(char))+'}'
        return subscript if math else r'\ensuremath{{}'+subscript+'}'
    cjk_math = math and (0x2E80 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF
                        or 0xFF01 <= code <= 0xFF60 or 0x20000 <= code <= 0x3134F)
    if (0x2460 <= code <= 0x24FF or 0x400 <= code <= 0x52F or 0x2160 <= code <= 0x2188
            or char in "∞☉↗↘∼≡′—" or cjk_math):
        glyph = r'{\BookUnicodeFont\char"' + f"{ord(char):X}" + "}"
        return r"\mbox{" + glyph + "}" if math else glyph
    return None


def math_tex(text):
    # TeX from OCR is mathematical input, never executable document instructions.
    forbidden = r"\\(?:input|include|write\d*|openout|openin|read|catcode|usepackage|documentclass|def|edef|gdef|xdef|newcommand|renewcommand|csname|endcsname|special|directlua|pdfobj|immediate|loop|repeat)\b"
    if re.search(forbidden, text):
        raise ValueError("Unsafe command in source formula")
    for env in re.findall(r"\\(?:begin|end)\{([^}]+)\}", text):
        if env not in ("aligned", "alignedat", "array", "cases", "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "gathered", "split"):
            raise ValueError("Unsupported formula environment: " + env)
    text,_=math_compatibility(text)
    chunks = []
    for n,char in enumerate(text):
        replacement = unicode_symbol(char, math=True)
        if (replacement and re.search(r'\\[A-Za-z]+$',replacement)
                and n+1 < len(text) and text[n+1].isascii() and text[n+1].isalpha()):
            replacement += ' '  # End the control word before the next variable.
        chunks.append(replacement or char)
    return ''.join(chunks)


def mixed_title(text):
    # Model title fields may copy explicit TeX math delimiters from the source.
    parts = re.split(r"(\$[^$]*\$)", text)
    return "".join("$" + math_tex(p[1:-1]) + "$" if p.startswith("$") and p.endswith("$") else escape(p) for p in parts)


def bookmark_title(index,lo,hi):
    """Plain Unicode fallback for PDF strings; printed headings keep native math."""
    symbols={'cdots':'⋯','ldots':'…','dots':'…','in':'∈','infty':'∞','times':'×','to':'→',
             'alpha':'α','beta':'β','gamma':'γ','delta':'δ','epsilon':'ε','lambda':'λ',
             'pi':'π','sigma':'σ','omega':'ω','mathbb':'','mathrm':'','text':'','operatorname':'',
             'left':'','right':'','quad':' ','qquad':' '}
    values=[]
    for atom,a,z in index.fragments(lo,hi):
        text=atom['content'][a:z]
        if atom['kind']=='math':
            text=re.sub(r'\\([A-Za-z]+)',lambda m:symbols.get(m[1],m[1]),text)
            text=re.sub(r'\\[!,;: ]',' ',text).replace('{','').replace('}','')
        values.append(text)
    text=re.sub(r'\s+',' ',''.join(values)).strip()
    return ''.join(ESCAPES.get(c,c) for c in text)


def reference_number_mismatch(reference, targets):
    labels = {"式": "equation", "公式": "equation", "算法": "algorithm", "例": "example", "例题": "example",
              "定理": "theorem", "引理": "lemma", "推论": "corollary", "定义": "definition", "图": "figure", "表": "table", "习题": "exercise"}
    match = re.fullmatch(r"(式|公式|算法|例题|例|定理|引理|推论|定义|图|表|习题)\s*[(（]?\s*([A-Z0-9]+(?:[.．]\d+)+)\s*[)）]?", reference["text"])
    target = targets.get(reference["target"])
    return bool(match and target and (target["kind"] != labels[match[1]] or
                target["number"].replace("．", ".") != match[2].replace("．", ".")))


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell, self.in_cell, self.in_math = [], None, "", False, False
        self.spans, self.row_spans = [], None
        self.styles = []
        self.complex = False

    def handle_starttag(self, tag, attrs):
        if tag not in ("table", "tbody", "thead", "tr", "td", "th", "eq", "br", "i", "b"):
            self.complex = True
        if tag in ('i', 'b'):
            if not self.in_cell or self.in_math or attrs:
                self.complex = True
            self.styles.append(tag)
            self.cell += r'{\itshape ' if tag == 'i' else r'{\bfseries '
        if tag == "tr":
            self.row = []
            self.row_spans = []
        if tag in ("td", "th"):
            if any(k == 'rowspan' and v != '1' for k,v in attrs):
                self.complex = True
            span = next((v for k,v in attrs if k == 'colspan'), '1')
            if not isinstance(span,str) or not re.fullmatch(r'[1-9]|10',span):
                self.complex = True
                span = '1'
            if self.row_spans is not None:
                self.row_spans.append(int(span))
            self.in_cell, self.cell = True, ""
        if tag == "eq":
            self.in_math = True
            self.cell += "$"
        if tag == "br":
            self.cell += " "

    def handle_endtag(self, tag):
        if tag in ('i', 'b'):
            if not self.styles or self.styles.pop() != tag:
                self.complex = True
            self.cell += '}'
        if tag == "eq":
            self.in_math = False
            self.cell += "$"
        if tag in ("td", "th"):
            if self.styles:
                self.complex = True
            if self.row is None:
                self.complex = True
            else:
                self.row.append(self.cell)
            self.in_cell = False
        if tag == "tr":
            if self.row:
                self.rows.append(self.row)
                self.spans.append(self.row_spans)
            self.row = None

    def handle_data(self, data):
        if self.in_cell:
            self.cell += math_tex(data) if self.in_math else escape(data)


def table_columns(parser):
    """Only a full-width first heading may span an otherwise regular grid."""
    if parser.complex or not parser.rows:
        return None
    columns = max(len(r) for r in parser.rows)
    if not 1 <= columns <= 10:
        return None
    for n,(row,spans) in enumerate(zip(parser.rows,parser.spans)):
        if n == 0 and len(row) == 1 and spans == [columns]:
            continue
        if len(row) != columns or spans != [1]*columns:
            return None
    return columns


def table_content(parser, rows=None):
    rows = parser.rows if rows is None else rows
    columns = table_columns(parser)
    result = []
    for n,row in enumerate(rows):
        cells = [cell.replace(',', r',\allowbreak{}') for cell in row]
        if n == 0 and columns > 1 and parser.spans[n] == [columns]:
            result.append('\\multicolumn{'+str(columns)+'}{c}{'+cells[0]+r'} \\'+'\n'+r'\midrule')
        else:
            result.append(' & '.join(cells)+r' \\')
    return '\n'.join(result)


def table_tex(html, in_box=False):
    parser = TableParser()
    parser.feed(html)
    rows = parser.rows
    columns = table_columns(parser)
    if columns is None:
        return None
    # Malformed OCR math in one cell must not abort the entire book. Preserve
    # the original table image via the existing complex-table fallback instead.
    for row in rows:
        for cell in row:
            depth, dollars, i = 0, 0, 0
            while i < len(cell):
                if cell[i] == '\\' and i+1 < len(cell) and cell[i+1] in '{}$\\':
                    i += 2
                    continue
                if cell[i] == '{': depth += 1
                if cell[i] == '}': depth -= 1
                if cell[i] == '$': dollars += 1
                if depth < 0: return None
                i += 1
            if depth or dollars % 2 or cell.count(r'\(') != cell.count(r'\)'):
                return None
    # Layer names such as Conv2.1,Conv2.2 must be able to wrap in a
    # longtable's fixed-width columns, without changing the visible text.
    content = table_content(parser)
    if len(rows) > 12 and not in_box:
        column = "p{\\dimexpr\\linewidth/" + str(columns) + "-2\\tabcolsep\\relax}"
        # Narrow plain-text tables need natural columns: equal fixed columns
        # needlessly split short Chinese headers. Long/math cells keep wrapping.
        plain_rows = []
        for row in rows:
            plain = []
            for cell in row:
                previous = None
                while previous != cell:
                    previous = cell
                    cell = re.sub(r'\{\\(?:itshape|bfseries) ([^{}]*)\}', r'\1', cell)
                plain.append(cell)
            plain_rows.append(plain)
        widths=[max(sum(10.5 if ord(c)>127 else 5.3 for c in row[n]) for row in plain_rows if len(row)==columns) for n in range(columns)]
        short=all('\\' not in cell and '$' not in cell for row in plain_rows for cell in row) and sum(widths)+12*columns<375
        specification='c'*columns if short else column*columns
        return "\n{\\setlength{\\LTleft}{\\fill}\\setlength{\\LTright}{\\fill}\n\\begin{longtable}{" + specification + "}\n\\toprule\n" + content + "\n\\bottomrule\n\\end{longtable}}\n"
    if len(rows) > 12:
        return None
    return "\n\\par\\addvspace{6pt}\\noindent\\makebox[\\linewidth][c]{%\n\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{" + "c" * columns + "}\n\\toprule\n" + content + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}}\\par\\addvspace{6pt}\n"


def footnote_marker_span(content, start, end):
    """Include an OCR marker's superscript wrappers without touching its base."""
    while True:
        prefix = re.search(r'(?:\^|\\(?:text|mbox|mathrm|textsuperscript))\s*\{\s*$', content[:start])
        suffix = re.match(r'\s*\}', content[end:])
        if prefix and suffix:
            start, end = prefix.start(), end + suffix.end()
        else:
            prefix = re.search(r'\^\s*$', content[:start])
            if prefix:
                start = prefix.start()
            return start, end


class Renderer:
    def __init__(self, index, decisions, registry, references, directory, config=None, heading_layout=None):
        from book_pipeline.ingestion.panels import panel_environments, panel_end_marks
        panels, panel_evidence = panel_environments(index, registry)
        from book_pipeline.publishing.side_notes import side_note_rows
        changed = {c['atom'] for c in decisions['corrections']} | {r['atom'] for r in references}
        self.side_note_rows = side_note_rows(index, panels, registry,
            [*decisions['exclusions'], *decisions.get('page_footnotes', []),
             *(dict(lo=n['reference']['edited_lo'], hi=n['reference']['edited_hi'])
               for n in decisions.get('page_footnotes', []) if n.get('reference'))], changed)
        self.panel_end_marks = panel_end_marks(index, panels, changed)
        from book_pipeline.publishing.native_decorations import native_decorations
        self.terminal_marks, self.source_rules = native_decorations(index, changed)
        from book_pipeline.publishing.raster_decorations import raster_decorations
        raster_panels, raster_rows, raster_rules, raster_evidence = raster_decorations(
            index, registry, [*decisions['exclusions'], *decisions.get('page_footnotes', [])], changed)
        panels.extend(raster_panels)
        self.side_note_rows.extend(raster_rows)
        self.source_rules.update(raster_rules)
        from book_pipeline.publishing.raster_marks import raster_marks
        self.raster_marks, self.raster_labels, _ = raster_marks(
            index, panels, directory,
            [*registry, *decisions['exclusions'], *decisions.get('page_footnotes', [])], changed)
        from book_pipeline.publishing.native_ocr_marks import native_ocr_marks
        self.native_ocr_punctuation, fallback_marks, fallback_evidence = native_ocr_marks(
            index, directory, [*decisions['exclusions'], *decisions.get('page_footnotes', [])],
            changed | set(decisions.get('formula_repairs', {})),
            {**{a:r['latex'] for a,r in decisions.get('native_math_styles', {}).items()},
             **{a:r['text'] for a,r in decisions.get('native_math_text', {}).items()}})
        self.raster_marks.update(fallback_marks)
        from book_pipeline.publishing.raster_inline_ink import raster_inline_ink
        self.raster_inline_ink = raster_inline_ink(index, directory,
            [*decisions['exclusions'], *decisions.get('page_footnotes', [])], changed)
        from book_pipeline.publishing.raster_reference_ink import raster_reference_ink
        for atom, runs in raster_reference_ink(index, directory,
                [*decisions['exclusions'], *decisions.get('page_footnotes', [])], changed).items():
            self.raster_inline_ink.setdefault(atom, []).extend(runs)
            self.raster_inline_ink[atom].sort(key=lambda r: r['start'])
        from book_pipeline.publishing.native_code import recover_native_code
        self.native_code,self.native_code_rows=recover_native_code(index,directory,changed)
        # A paired code/caption row owns its panel inside the main column.
        # Keeping the old outer wrapper would also put the caption in the box.
        panels=[p for p in panels if not any(r['code_range']==[p['lo'],p['hi']]
                                             for r in self.native_code_rows)]
        registry = [*registry, *panels]
        self.index, self.decisions, self.registry = index, decisions, registry
        # A quote-only fragment beside math or a link inherits its source
        # paragraph's script, rather than guessing from the fragment alone.
        self.latin_quote_groups = {group for group, positions in index.groups.items()
                                   if latin_quote_context(''.join(index.atoms[n]['content']
                                       for n in positions if index.atoms[n]['kind'] == 'text'))}
        self.directory = Path(directory)
        self.config = config or {}
        self.typography = validate_typography(self.config.get('typography'))
        self.heading_layout = HeadingLayout([o for o in registry if o['kind'] in HEADINGS], heading_layout)
        self.refs = defaultdict(list)
        self.corrections = defaultdict(list)
        self.issues = []
        self.tex_compatibility=[]
        self.formula_repairs = decisions.get('formula_repairs', {})
        self.table_repairs = decisions.get('table_repairs', {})
        self.source_order_layout=[*panel_evidence, *raster_evidence, *self.side_note_rows]
        self.source_order_layout.extend(fallback_evidence)
        self.source_order_layout.extend(dict(method=l['method'],atom=a,lines=l['lines'],
            page=l['page'],bbox=l['bbox']) for a,l in self.native_code.items())
        self.list_prefixes = {}
        self.crop_prompt_images = set()
        joins, rejected = supported_joins(index, decisions)
        self.issues.extend(rejected)
        self.join_pairs = {(j["left"], j["right"]) for j in joins}
        self.join_left = {left for left, _ in self.join_pairs}
        self.join_right = {right for _, right in self.join_pairs}
        for c in decisions["corrections"]:
            self.corrections[c["atom"]].append(c)
        targets = {o["anchor"]: o for o in registry}
        for r in references:
            if reference_number_mismatch(r, targets):
                self.issues.append({"reason": "explicit_reference_number_mismatch; retained original text without link", "reference": r})
            elif any(r["start_offset"] < c["end_offset"] and r["end_offset"] > c["start_offset"] for c in self.corrections[r["atom"]]):
                self.issues.append({"reason": "reference_overlaps_ocr_correction", "reference": r})
            else:
                self.refs[r["atom"]].append(r)
        self.assets, self.ledger, self.link_targets, self.anchors = {}, [], [], []
        from book_pipeline.publishing.image_search import ImageSearch
        self.image_search = ImageSearch(index, self.directory)
        self.source_anchors = {}
        edited_atoms = {atom for atom, rows in self.corrections.items() if rows} | {atom for atom, rows in self.refs.items() if rows}
        self.recovered_figures = recover_fragmented_figures(index, registry, self.directory, edited_atoms)
        from book_pipeline.publishing.figure_recovery import recover_native_vector_labels
        self.recovered_figures.update(recover_native_vector_labels(index,self.directory,edited_atoms))
        self.image_rows = recover_horizontal_images(index, registry, self.directory, edited_atoms)
        self.image_row_members = {member for row in self.image_rows.values() for member in row['members']}
        self.image_row_groups = {group for row in self.image_rows.values() for group in row['groups']}
        self.image_row_targets = {target for row in self.image_rows.values() for target in row['targets']}
        for atom, row in self.image_rows.items():
            self.source_order_layout.append({'atom': atom, 'method': 'horizontal_source_image_row', **row})
        self.figure_labels = {label for figure in self.recovered_figures.values() for label in figure['labels']}
        for atom, figure in self.recovered_figures.items():
            self.source_order_layout.append({'atom': atom, 'method': 'fragmented_diagram_preserved_from_source_page', **figure})
        self.bold_spans=[index.span(s) for s in decisions.get('bold_spans',[])]
        self.source_styles = decisions.get('source_styles', {})
        self.toc_position = min((e["lo"] for e in decisions["exclusions"] if "original_toc" in e.get("reasons", [e["reason"]])), default=None)
        self.page_footnotes = decisions.get('page_footnotes', [])
        self.rendered_footnotes = []
        self.footnote_bodies = {}
        self.footnote_markers = defaultdict(list)
        for note in self.page_footnotes:
            if note['status'] != 'matched':
                continue
            reference = note['reference']
            self.add_footnote_marker(reference['edited_lo'], reference['edited_hi'], note, True)
            self.add_footnote_marker(note['marker_lo'], note['marker_hi'], note, False)
        from book_pipeline.publishing.margin_assets import margin_assets
        self.margin_assets = margin_assets(index, registry,
            [*decisions['exclusions'], *self.side_note_rows], edited_atoms)
        self.margin_references = {r['reference']:r for r in self.margin_assets}
        self.margin_targets = {t for r in self.margin_assets for t in r['targets']}
        self.source_order_layout.extend(self.margin_assets)

    def margin_asset_tex(self, row):
        label = self.index.atoms[self.index.ids[row['label']]]
        image = self.index.atoms[self.index.ids[row['image']]]
        title = self.source_marker(label) + self.text_fragment(label, 0, label['length'])
        marks = ''.join(self.anchor(t) for t in row['targets']) + self.source_marker(image)
        return ('\\BookMarginAsset{' + title + '}{' + marks
                + '\\includegraphics[width=\\linewidth]{' + self.asset(image) + '}}')

    def add_footnote_marker(self, lo, hi, note, reference):
        if lo == hi and reference and note.get('reference',{}).get('reviewed_missing_callout'):
            address = self.index.address(lo,end=True)
            atom = self.index.atoms[self.index.ids[address['atom']]]
            offset = address['offset']
            if atom['kind'] != 'text' or not 0 < offset <= len(atom['content']):
                raise ValueError('Reviewed missing footnote must follow existing prose')
            if (any(c['start_offset'] < offset < c['end_offset'] for c in self.corrections[atom['id']])
                    or any(m['start'] <= offset <= m['end'] for m in self.footnote_markers[atom['id']])):
                raise ValueError('Reviewed missing footnote overlaps another edit or marker')
            self.footnote_markers[atom['id']].append(dict(start=offset,end=offset,
                tex='\\BookFootnoteReference{'+note['id']+'}'))
            self.anchors.append(note['reference_anchor'])
            note['insert_after'] = lo
            return
        parts = list(self.index.fragments(lo, hi))
        if len(parts) != 1 or parts[0][0]['kind'] not in {'text', 'math'}:
            raise ValueError('Footnote marker must occupy one text or inline math atom: ' + note['id'])
        atom, a, z = parts[0]
        a, z = footnote_marker_span(atom['content'], a, z)
        if any(a < c['end_offset'] and z > c['start_offset'] for c in self.corrections[atom['id']]):
            raise ValueError('Footnote marker overlaps an OCR correction: ' + note['id'])
        if any(a < m['end'] and z > m['start'] for m in self.footnote_markers[atom['id']]):
            raise ValueError('Overlapping matched footnote markers: ' + note['id'])
        command = 'BookFootnoteReference' if reference else 'BookFootnoteLabel'
        self.footnote_markers[atom['id']].append(dict(start=a, end=z, tex='\\' + command + '{' + note['id'] + '}'))
        if reference:
            self.anchors.append(note['reference_anchor'])
            if atom['kind'] == 'text':
                note['insert_after'] = self.index.prefix[self.index.ids[atom['id']]] + z

    def page_footnote_body(self, note):
        if note['id'] in self.footnote_bodies:
            return self.footnote_bodies[note['id']]
        chunks, previous = [], None
        for atom, a, z in self.index.fragments(note['lo'], note['hi']):
            if previous is not None and previous['group'] != atom['group']:
                chunks.append('\\par\n')
            chunks.append(self.fragment(atom, a, z, False))
            previous = atom
        body = ''.join(chunks)
        self.footnote_bodies[note['id']] = body
        return body

    def page_footnote_tex(self, note):
        """Pair page-local matched labels; retain unmatched text and native math."""
        body = self.page_footnote_body(note)
        self.anchors.append(note['anchor'])
        record = {k:note[k] for k in ('id','anchor','reference_anchor','status','page','reason')}
        record.update(original_marker=note['marker']['text'] if note['marker'] else None)
        if note['status'] == 'matched':
            record.update(numbering='rendered_page_from_one', marker_style=note['marker']['style'],
                          reference_order=note['reference']['edited_lo'])
        self.rendered_footnotes.append(record)
        return '\\BookPageFootnote{' + note['anchor'] + '}{' + body + '}\n'

    def asset(self, atom, row=None):
        source = Path(atom["asset"])
        searchable = self.image_search.row_asset(row) if row else self.image_search.asset(atom)
        for row in self.image_search.evidence:
            if row not in self.source_order_layout:
                self.source_order_layout.append(row)
        if searchable:
            relative = str(searchable.relative_to(self.directory))
            self.assets[str(source)] = relative
            return relative
        name = digest(source)[:20] + source.suffix.lower()
        (self.directory / "images").mkdir(exist_ok=True)
        destination = self.directory / "images" / name
        if not destination.exists():
            shutil.copy2(source, destination)
        self.assets[str(source)] = "images/" + name
        return "images/" + name

    def text_fragment(self, atom, lo, hi):
        text_escape = lambda text: escape(text, atom['group'] in self.latin_quote_groups)
        changes = []
        markers = self.footnote_markers[atom['id']]
        for marker in markers:
            start, end = marker['start'], marker['end']
            intersects = lo < start <= hi if start == end else start < hi and end > lo
            if intersects:
                if not lo <= start <= end <= hi:
                    raise ValueError('Rendering boundary splits a matched footnote marker: ' + atom['id'])
                changes.append((start, end, marker['tex']))
        for c in self.corrections[atom["id"]]:
            if lo <= c["start_offset"] and c["end_offset"] <= hi:
                changes.append((c["start_offset"], c["end_offset"], text_escape(c["replacement"])))
        for r in self.refs[atom["id"]]:
            start, end = max(lo, r['start_offset']), min(hi, r['end_offset'])
            spans = [(start, end)] if start < end else []
            for marker in markers:
                spans = [(x, y) for a, z in spans for x, y in
                         ((a, min(z, marker['start'])), (max(a, marker['end']), z)) if x < y]
            for start, end in spans:
                text = r['text'] if (start,end)==(r['start_offset'],r['end_offset']) else atom['content'][start:end]
                body = (self.source_text_fragment(atom, start, end)
                        if self.source_styles.get(atom['id']) and text == atom['content'][start:end] else text_escape(text))
                changes.append((start, end, "\\BookLink{" + r["target"] + "}{" + body + "}"))
                self.link_targets.append(r["target"])
        prefix = self.list_prefixes.get(atom['id'])
        if prefix and lo == 0 and prefix <= hi and not any(a < prefix and b > 0 for a,b,_ in changes):
            # The list environment displays this exact source label once in
            # its margin. Consume its original bytes without duplicating it.
            changes.append((0, prefix, ''))
        label = self.raster_labels.get(atom['id'])
        if (label and lo <= label['start'] < label['end'] <= hi
                and not any(a < label['end'] and label['start'] < z for a,z,_ in changes)):
            rgb = ','.join(f'{c:g}' for c in label['color'])
            changes.append((label['start'], label['end'], '{\\color[rgb]{'+rgb+'}'
                            + text_escape(atom['content'][label['start']:label['text_end']]) + '}'
                            + '\\hspace{'+f"{label['gap_after_bp']:g}"+'bp}'))
            if label not in self.source_order_layout:
                self.source_order_layout.append(label)
        mark = self.terminal_marks.get(atom['id'])
        if (mark and lo <= mark['start'] < mark['end'] <= hi
                and not any(a < mark['end'] and mark['start'] < z for a,z,_ in changes)):
            # Retain the original colored glyph, consuming only its preceding
            # whitespace. The proof symbol in an explanatory note is ordinary
            # prose because it lacks the source right-edge evidence.
            symbol_start = atom['content'].rfind(mark['symbol'], mark['start'], mark['end'])
            glyph = self.source_text_fragment(atom, symbol_start, symbol_start+1)
            changes.append((mark['start'],mark['end'],r'\BookTerminalMark{'+glyph+'}'))
            if mark not in self.source_order_layout:
                self.source_order_layout.append(mark)
        chunks, cursor = [], lo
        for start, end, replacement in sorted(changes):
            chunks.append(self.source_text_fragment(atom, cursor, start))
            chunks.append(replacement)
            cursor = end
        chunks.append(self.source_text_fragment(atom, cursor, hi))
        return "".join(chunks)

    def source_text_fragment(self, atom, lo, hi):
        """Compose independently owned source styles with existing ink runs."""
        from book_pipeline.publishing.raster_styles import METHOD as STYLE_METHOD
        chunks, cursor = [], lo
        previous = 0
        for run in self.source_styles.get(atom['id'], []):
            start, end = run['start'], run['end']
            if (run.get('method') != STYLE_METHOD
                    or type(start) is not int or type(end) is not int
                    or not previous <= start < end <= len(atom['content'])
                    or atom['content'][start:end] != run.get('quote')
                    or type(run.get('italic')) is not bool or type(run.get('bold')) is not bool
                    or not (run['italic'] or run['bold'] or run.get('cjk_family')=='calligraphic')
                    or run.get('cjk_family') not in {None, 'sans', 'calligraphic'}
                    or (run.get('cjk_family') and (run['italic']
                        or (run['cjk_family']=='sans' and not run['bold'])
                        or not re.fullmatch(r'[\u3400-\u9fff]+', run['quote'])))):
                raise ValueError('Invalid source text style range: ' + atom['id'])
            previous = end
            a, z = max(lo, start), min(hi, end)
            if a >= z: continue
            chunks.append(self.source_ink_fragment(atom, cursor, a))
            switches = (r'\itshape' if run['italic'] else '') + (r'\bfseries' if run['bold'] else '')
            if run.get('cjk_family') == 'sans':
                switches = r'\heiti'
            elif run.get('cjk_family') == 'calligraphic':
                switches = r'\BookSourceKaiFont' + (r'\bfseries' if run['bold'] else '')
            # An overlay split can start with a source space. Terminate the
            # font control word explicitly so TeX cannot eat that word boundary.
            terminator = '{}' if atom['content'][a:a+1].isspace() else ' '
            chunks.append('{'+switches+terminator+self.source_ink_fragment(atom, a, z)+'}')
            row = dict(**run, atom=atom['id'], page=atom['page']+1, bbox=atom.get('bbox'))
            if row not in self.source_order_layout: self.source_order_layout.append(row)
            cursor = z
        chunks.append(self.source_ink_fragment(atom, cursor, hi))
        return ''.join(chunks)

    def source_ink_fragment(self, atom, lo, hi):
        """Apply observed ink, preserving text and source offsets."""
        text_escape = lambda text: escape(text, atom['group'] in self.latin_quote_groups)
        chunks, cursor = [], lo
        for run in [*atom.get('source_text_runs', []), *self.raster_inline_ink.get(atom['id'], [])]:
            a, z = max(lo, run['start']), min(hi, run['end'])
            if a >= z: continue
            rgb = run['color']
            if (run.get('method') not in {'native_span_ink_v1', 'neighbor_verified_raster_square_ink_v1',
                                       'exact_cpu_scan_note_reference_ink_v1'}
                    or not 0 <= run['start'] < run['end'] <= len(atom['content'])
                    or a < cursor or len(rgb) != 3 or not all(isinstance(c,(int,float)) and 0 <= c <= 1 for c in rgb)):
                raise ValueError('Invalid source text ink range: ' + atom['id'])
            if run.get('method') in {'neighbor_verified_raster_square_ink_v1', 'exact_cpu_scan_note_reference_ink_v1'} and run not in self.source_order_layout:
                self.source_order_layout.append(run)
            chunks.append(text_escape(atom['content'][cursor:a]))
            if run.get('method') in {'neighbor_verified_raster_square_ink_v1', 'exact_cpu_scan_note_reference_ink_v1'}:
                # XeTeX's color driver uses these balanced specials. Avoid a
                # TeX group around the existing square: grouping changes the
                # following Latin space factor and can move the rest of a line.
                chunks.append(r'\special{color push rgb ' + ' '.join(f'{c:g}' for c in rgb) + '}'
                              + text_escape(atom['content'][a:z]) + r'\special{color pop}')
                cursor = z
                continue
            text = atom['content'][a:z]
            leading = text[:len(text)-len(text.lstrip())]
            trailing = text[len(text.rstrip()):] if text.strip() else ''
            # xcolor deliberately ignores spaces after its command. Source
            # word boundaries must live outside the color group, including
            # a leading space owned by a colored reference such as " 6".
            chunks.append(text_escape(leading))
            chunks.append('{\\color[rgb]{' + ','.join(f'{c:g}' for c in rgb) + '}'
                          + text_escape(text.strip()) + '}')
            chunks.append(text_escape(trailing))
            gap = run.get('gap_after_em', 0)
            if gap and z == run['end']:
                if not isinstance(gap,(int,float)) or not .25 <= gap <= 2:
                    raise ValueError('Invalid source text gap: ' + atom['id'])
                chunks.append('\\hspace{' + f'{gap:g}' + 'em}')
            cursor = z
        chunks.append(text_escape(atom['content'][cursor:hi]))
        return ''.join(chunks)

    def fragment(self, atom, lo, hi, in_box, display_row=False):
        if atom['kind']=='image' and atom.get('recovery_issue')=='formula_source_crop_requires_transcription':
            # A destination before BookImage starts a separate empty paragraph
            # and can remain on the page preceding the actual image.
            marker=self.source_marker(atom)
            keep='\\nobreak' if atom['id'] in self.crop_prompt_images else ''
            return ('\n\\par'+keep+'\n\\BookFormulaImage['+f'{self.image_width(atom):.3f}'+
                    '\\linewidth]{'+self.asset(atom)+'}{'+marker+'}\n')
        # Keep a terminal CJK pair together, including after inline math where
        # xeCJK's look-ahead single-character check has no preceding CJK run.
        # Split source ranges rather than box the text: fonts, ink and link
        # destinations retain their original width and source ownership.
        boundary = hi - 1
        following = self.index.ids[atom['id']] + 1
        terminal = (following == len(self.index.atoms)
                    or self.index.atoms[following]['group'] != atom['group'])
        protected = (atom['kind'] == 'text' and atom.get('role') == 'text'
                     and atom.get('block_type') == 'text' and hi == atom['length']
                     and lo < boundary and terminal and atom['id'] not in self.join_left
                     and re.search(r'[\u3400-\u9fff]{2}$', atom['content'])
                     and not any(c['start_offset'] < boundary < c['end_offset']
                                 for c in self.corrections[atom['id']])
                     and not any(m['start'] < boundary < m['end']
                                 for m in self.footnote_markers[atom['id']])
                     and not (self.list_prefixes.get(atom['id'], 0) > boundary)
                     and not (self.raster_labels.get(atom['id'], {}).get('end', 0) > boundary))
        if protected:
            value = (self.content_fragment(atom, lo, boundary, in_box, display_row=display_row)
                     + r'\nobreak{}'
                     + self.content_fragment(atom, boundary, hi, in_box, display_row=display_row))
        else:
            value = self.content_fragment(atom, lo, hi, in_box, display_row=display_row)
        # Pagination whitespace must not accidentally leave a blank TeX line
        # inside a confirmed paragraph. Source text/coverage remain unchanged.
        if atom["id"] in self.join_left and hi == atom["length"]:
            value = value.rstrip()
        if atom["id"] in self.join_right and lo == 0:
            value = value.lstrip()
        if atom['kind'] == 'text' and hi == atom['length']:
            following = self.index.ids[atom['id']]+1
            if following < len(self.index.atoms):
                next_atom = self.index.atoms[following]
                display = next_atom['kind'] == 'display_math' or (
                    next_atom['kind'] == 'math' and bool(formula_tags(next_atom['content'])))
                if display and atom['group'] != next_atom['group']:
                    # The renderer supplies the display boundary. Retaining
                    # OCR line endings here creates a blank TeX paragraph and
                    # an extra line above the equation, including inside boxes.
                    value = value.rstrip()
        value = self.source_marker(atom) + value
        if atom['id'] in self.margin_references and hi == atom['length']:
            value += self.margin_asset_tex(self.margin_references[atom['id']])
        return value

    def source_marker(self, atom):
        if atom['id'] in self.image_row_members:
            return ''  # All row destinations are emitted together inside the shared image.
        if atom['id'] in self.figure_labels:
            return ''  # These destinations are emitted inside the complete source figure.
        if atom["id"] not in self.source_anchors:
            name = "source-" + atom["id"]
            self.source_anchors[atom["id"]] = name
            # A vertical-mode destination can stay on the preceding page when
            # TeX moves the following paragraph. Attach prose destinations to
            # its first line; displays and image boxes own their own boundaries.
            prefix = "\\leavevmode" if atom['kind'] == 'text' else ""
            return prefix + "\\hypertarget{" + name + "}{}"
        return ""

    def heading_fragment(self, atom, lo, hi):
        value=self.content_fragment(atom,lo,hi,False)
        # Heading macro arguments cannot contain paragraph tokens. Collapse
        # source prose wraps only; inline math's TeX comment newlines must stay.
        return re.sub(r'[\r\n]+',' ',value) if atom['kind']=='text' else value

    def content_fragment(self, atom, lo, hi, in_box, display_row=False):
        if atom['id'] in self.image_rows:
            row = self.image_rows[atom['id']]
            markers = ''
            for member in row['members']:
                name = 'source-' + member
                self.source_anchors[member] = name
                markers += '\\hypertarget{' + name + '}{}'
            for target in row['targets']:
                markers += self.anchor(target)
            column = self.source_column_width(atom['page'])
            width = min(1., (row['bbox'][2]-row['bbox'][0]) / max(.25, column))
            carrier = {**atom, 'asset': row['asset']}
            return '\n' + markers + '\\BookImage[' + f'{width:.3f}' + '\\linewidth]{' + self.asset(carrier,row=row) + '}\n'
        if atom['id'] in self.image_row_members:
            return ''
        if atom['id'] in self.figure_labels:
            return ''
        if atom["kind"] == "text":
            if atom['id'] in self.native_code:
                if lo!=0 or hi!=atom['length']:
                    raise ValueError('Native code requires complete source ownership')
                from book_pipeline.publishing.native_code import code_tex
                return code_tex(self.native_code[atom['id']])
            return self.text_fragment(atom, lo, hi)
        content = atom["content"]
        changes = [(c['start_offset'], c['end_offset'], c['replacement']) for c in self.corrections[atom['id']]]
        changes += [(m['start'], m['end'], r'\mbox{' + m['tex'] + '}') for m in self.footnote_markers[atom['id']]]
        for start, end, replacement in sorted(changes, reverse=True):
            content = content[:start] + replacement + content[end:]
        if atom['id'] in self.formula_repairs:
            replacement = self.formula_repairs[atom['id']]
            self.tex_compatibility.append(dict(atom=atom['id'], original=content, rendered=replacement,
                evidence=['formula image re-read after a compiler error; replacement passed XeLaTeX syntax validation']))
            content = replacement
        if atom['kind'] in {'math','display_math'}:
            compatible,evidence=math_compatibility(content)
            if evidence:
                self.tex_compatibility.append({'atom':atom['id'],'original':content,'rendered':compatible,'evidence':evidence})
            content=compatible
        numbered_inline = atom['kind']=='math' and bool(formula_tags(content))
        from book_pipeline.publishing.prose_math import prose_math_parts
        prose_inline=(atom['kind']=='math' and not numbered_inline
            and atom['id'] not in self.decisions.get('native_math_text',{})
            and atom['id'] not in self.decisions.get('native_math_styles',{})
            and prose_math_parts(content) is not None)
        if prose_inline:
            issue=dict(reason='long_prose_inside_inline_math_requires_source_review',atom=atom['id'])
            if issue not in self.issues:self.issues.append(issue)
        if numbered_inline:
            self.tex_compatibility.append(dict(atom=atom['id'],original=content,rendered=content,
                evidence=['numbered equation in an inline source node rendered in display math; original equation number retained']))
        if atom["kind"] == "math" and not numbered_inline and not prose_inline:
            literal = self.decisions.get('native_math_text', {}).get(atom['id'])
            if literal:
                return escape(literal['text'])
            native = self.decisions.get('native_math_styles', {}).get(atom['id'])
            if native:
                content = native['latex']
            style = r'\displaystyle ' if atom['group'] in getattr(self,'fraction_prose_groups',()) else ''
            return "% source-atom: " + atom["id"] + "\n\\allowbreak{}$" + style + math_tex(content) + "$%\n% end-source-atom: " + atom["id"] + "\n"
        if atom["kind"] == "display_math" or numbered_inline or prose_inline:
            tex = math_tex(content)
            from book_pipeline.publishing.prose_math import prose_math_parts
            prose_parts = prose_math_parts(tex)
            tags = formula_tags(tex)
            image_reason = 'multiple equation tags' if len(tags) > 1 else None
            if len(re.findall(r'\\tag\*?\s*\{', tex)) != len(tags):
                image_reason = 'unbalanced equation tag delimiters'
            if image_reason:
                # OCR can mistake a left-hand item label for a second equation
                # number, or duplicate part of the right-hand number. Neither
                # label can safely be discarded without inspecting the source.
                if not atom.get('asset'):
                    import pymupdf
                    box = atom.get('bbox') or atom.get('block_bbox')
                    if (not isinstance(box,(list,tuple)) or len(box)!=4 or
                            not 0<=box[0]<box[2]<=1 or not 0<=box[1]<box[3]<=1):
                        raise ValueError('Formula has ' + image_reason + ' but no source image: ' + atom['id'])
                    path = self.directory/'source_formula_images'/(atom['id']+'.png')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with pymupdf.open(self.index.source['source_pdf']) as source:
                        page = source[atom['page']]
                        clip = pymupdf.Rect(box[0]*page.rect.width,box[1]*page.rect.height,
                                            box[2]*page.rect.width,box[3]*page.rect.height)
                        page.get_pixmap(dpi=220,clip=clip,alpha=False).save(path)
                    atom = {**atom,'asset':str(path)}
                rendered = self.image_tex(atom)
                self.tex_compatibility.append({'atom': atom['id'], 'original': content, 'rendered': rendered,
                    'evidence': [image_reason + ' preserved using the original formula image; no label or mathematical content discarded']})
                return "\n% source-atom: " + atom['id'] + rendered + "% end-source-atom: " + atom['id'] + "\n"
            # Keep the original equation number outside the measured math
            # box. Only oversized expressions are reduced to fit the column.
            if len(tags) <= 1:
                tag = compatible_equation_tag(tags[0]['tex']) if tags else ""
                if tags and tag != tags[0]['tex']:
                    self.tex_compatibility.append(dict(atom=atom['id'], original=tags[0]['tex'], rendered=tag,
                        evidence=['mathematical equation number wrapped in ensuremath; original number preserved']))
                if tags:
                    tex = tex[:tags[0]['start']] + tex[tags[0]['end']:]
                if prose_parts:
                    tex = r'\BookProseMath{' + ''.join(
                        escape(value, latin_quotes=True) if kind == 'text'
                        else r'$\displaystyle ' + value + '$'
                        for kind, value in prose_parts) + '}'
                else:
                    from book_pipeline.publishing.derivation_style import display_derivation_arrays
                    styled, arrays = display_derivation_arrays(tex)
                    if arrays:
                        self.tex_compatibility.append(dict(atom=atom['id'], original=tex, rendered=styled,
                            evidence=['single-column relation-led derivation rows retain display-sized fractions and operators; mathematical tokens and row breaks unchanged']))
                        tex = styled
                    tex = "\\BookMath[" + ("48pt" if tag else "0pt") + "]{" + tex + "}" + tag
            if display_row:
                return "\n% source-atom: " + atom['id'] + "\n" + tex + "\n% end-source-atom: " + atom['id'] + "\n"
            return "\n% source-atom: " + atom["id"] + "\n\\begin{equation*}\n" + tex + "\n\\end{equation*}\n% end-source-atom: " + atom["id"] + "\n"
        if atom["kind"] == "image":
            return self.image_tex(atom)
        if atom["kind"] == "table":
            if atom['id'] in self.decisions.get('table_math', {}) or atom['id'] in self.decisions.get('table_styles', {}):
                from book_pipeline.publishing.reviewed_tables import annotate_table_math
                ranges = sorted([*self.decisions.get('table_math', {}).get(atom['id'], []),
                                 *self.decisions.get('table_styles', {}).get(atom['id'], [])], key=lambda r:r['start'])
                if any(a['end'] > b['start'] for a,b in zip(ranges, ranges[1:])):
                    raise ValueError('Reviewed table annotations overlap')
                content = annotate_table_math(content, ranges)
                for row in ranges:
                    evidence = dict(row, atom=atom['id'], page=atom['page']+1)
                    if evidence not in self.source_order_layout:
                        self.source_order_layout.append(evidence)
            if atom['id'] in self.table_repairs:
                from book_pipeline.publishing.table_repair import expand_template
                converted = expand_template(content, self.table_repairs[atom['id']], in_box)
                self.tex_compatibility.append(dict(atom=atom['id'], original=content, rendered=converted,
                    evidence=['table layout repaired after compiler feedback; every source cell preserved exactly']))
            else:
                converted = table_tex(content, in_box)
            if converted:
                return '\n% source-atom: '+atom['id']+'\n% source-table-in-box: '+str(int(in_box))+'\n'+converted+'\n% end-source-atom: '+atom['id']+'\n'
            if not atom.get("asset"):
                raise ValueError("Complex table has no source image: " + atom["id"])
            self.issues.append({"reason": "complex_table_preserved_as_image", "atom": atom["id"]})
            return self.image_tex(atom)
        raise ValueError("Unsupported atom kind: " + atom["kind"])

    def image_tex(self, atom):
        markers = ''
        if atom['id'] in self.recovered_figures:
            figure = self.recovered_figures[atom['id']]
            for label in figure['labels']:
                name = 'source-' + label
                self.source_anchors[label] = name
                markers += '\\hypertarget{' + name + '}{}'
            atom = {**atom, 'asset': figure['asset'], 'bbox': figure['bbox']}
        width=self.image_width(atom)
        return "\n" + markers + "\\BookImage[" + f"{width:.3f}" + "\\linewidth]{" + self.asset(atom) + "}\n"

    def image_width(self, atom):
        bbox = atom.get("bbox")
        column_width = self.source_column_width(atom['page'])
        return min(1., max(.15, (bbox[2]-bbox[0])/max(.25,column_width))) if bbox else 1.

    def source_column_width(self, page):
        # MinerU coordinates are normalized against the original page. Estimate
        # the original text column from body blocks, then retain image scale.
        body = [a["bbox"] for a in self.index.atoms if a["page"] == page
                and a["kind"] == "text" and a["block_type"] == "text" and a.get("bbox")
                and not a.get('formula_repair')]
        # Formula review regions may contain many prose leaves, all inheriting
        # the same full-page-width crop. That rectangle describes the review
        # window, rather than the printed text column. Require a matching
        # display region and independent body boxes before discarding it.
        regions = {tuple(a['bbox']) for a in self.index.atoms
                   if a['page'] == page and a['kind'] == 'display_math'
                   and a.get('bbox') and a['bbox'][0] == 0 and a['bbox'][2] == 1}
        measured = [b for b in body if tuple(b) not in regions]
        if len({tuple(b) for b in measured}) >= 2:
            body = measured
        return max((b[2] for b in body), default=.88) - min((b[0] for b in body), default=.12)

    def anchor(self, name):
        self.anchors.append(name)
        return "\\BookAnchor{" + name + "}"

    def object_title(self, item):
        end = self.index.position(item['body_start'])
        parts = list(self.index.fragments(item['lo'], end))
        if item['kind'] in COLOR_KINDS:
            # Trim raw text before escaping TeX. Math atoms, paired name marks
            # and source spans remain intact, including punctuation-only atoms.
            for n in range(len(parts) - 1, -1, -1):
                atom, lo, hi = parts[n]
                if atom['kind'] != 'text':
                    break
                text = trim_title_punctuation(atom['content'][lo:hi])
                parts[n] = atom, lo, lo + len(text)
                if text:
                    break
        return ''.join(self.fragment(atom, lo, hi, False) for atom, lo, hi in parts)

    def wrapped_figure_tex(self, row):
        if row['method']=='explicit_algorithm_source_lines':
            parts=[]
            for atom,lo,hi in self.index.fragments(row['lo'],row['hi']):
                value=self.fragment(atom,lo,hi,True)
                if atom['kind']=='text':
                    # Inline math fragments own TeX comment newlines. Only
                    # source text newlines and indentation become code lines.
                    value=re.sub(r'\n([ \t]*)',lambda m:'\n\\par\\noindent{}'+
                        ('\\hspace*{'+f'{len(m[1].expandtabs(4))*.45:g}'+'em}' if m[1] else ''),value)
                parts.append(value)
            return ('\n\\begin{BookSourcePanel}[fontupper=\\small]{}{black}\n'
                    '\\raggedright\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{2pt}\\noindent\n'+
                    ''.join(parts)+'\n\\par\\end{BookSourcePanel}\n')
        if row['method']=='native_code_side_caption':
            atom=self.index.atoms[self.index.ids[row['code']]]
            layout=self.native_code[atom['id']]
            rgb=','.join(f'{c:g}' for c in layout['panel'].get('background_color',[.95,.95,.95]))
            panel=('\\definecolor{BookNativeCodeBack}{rgb}{'+rgb+'}\n'+
                '\\begin{BookSourcePanel}[colback=BookNativeCodeBack,boxrule=0pt]{}{black}\n'+
                self.source_marker(atom)+self.content_fragment(atom,0,atom['length'],True)+
                '\\end{BookSourcePanel}\n')
            caption=''.join(self.source_marker(a)+self.content_fragment(a,l,h,False)
                for a,l,h in self.index.fragments(*row['caption_range']))
            return ('\n\\begin{BookSideBySide}{'+f"{row['widths'][0]:.5f}"+'}{'+
                f"{row['widths'][1]:.5f}"+'}{1}\n'+panel+
                '\\BookSideSwitch\n\\footnotesize\\raggedright\\BookCaptionParagraph{'+caption+'}\n'+
                '\\end{BookSideBySide}\n')
        """Ship the source figure beside one rejoined, immutable paragraph."""
        if row['method']=='native_marginal_figure':
            paragraphs=[''.join(self.fragment(a,l,h,False)
                for a,l,h in self.index.fragments(*bounds)) for bounds in row['main_ranges']]
            image=self.index.atoms[self.index.ids[row['image']]]
            if image['id'] in self.recovered_figures:
                image={**image,**self.recovered_figures[image['id']]}
            picture=''.join(self.anchor(t) for t in row['targets'])+self.source_marker(image)
            picture+='\\BookImage['+f"{row['image_width_fraction']:.5f}"+'\\linewidth]{'+self.asset(image)+'}\n'
            caption=self.caption_tex(list(self.index.fragments(*row['caption_range'])))
            return ('\n\\begingroup\\renewcommand{\\BookSideGap}{2pt}\n\\begin{BookSideBySide}{'+f"{row['widths'][0]:.5f}"+'}{'+
                f"{row['widths'][1]:.5f}"+'}{'+('1' if row['main_left'] else '0')+'}\n'+paragraphs[0]+
                '\n\\BookSideSwitch\n'+picture+'\\footnotesize\\BookCaptionParagraph{'+caption+'}\n'
                '\\end{BookSideBySide}\n\\endgroup\n'+'\n\\par\n'.join(paragraphs[1:])+'\n\\par\n')
        if row['method']=='native_ruled_table':
            caption=('\\BookCaption{'+self.caption_tex(list(self.index.fragments(*row['caption_range'])))+'}\n'
                     if row.get('caption_range') else '')
            rows=[]
            for cells in row['cells']:
                rows.append(' & '.join('\\leavevmode '+''.join(self.fragment(a,l,h,False)
                    for a,l,h in self.index.fragments(*cell)) for cell in cells)+r' \\')
            spacing=2*(len(row['cells'][0])-1)
            columns='@{}'+''.join(r'>{\raggedright\arraybackslash}p{'+f'{w:.5f}'+
                r'\dimexpr\linewidth-'+str(spacing)+r'\tabcolsep\relax}' for w in row['column_widths'])+'@{}'
            size=self.typography['body_size']*row.get('body_font_ratio',1)
            setup='\\linespread{1}\\fontsize{'+f'{size:g}'+'pt}{'+f'{size*1.35:g}'+r'pt}\selectfont\renewcommand{\arraystretch}{1.15}'+'\n'
            return ('\n\\par\\begin{samepage}\n'+setup+caption+
                r'\noindent\begin{tabular}{'+columns+'}\n\\toprule\n'+
                rows[0]+'\n\\midrule\n'+'\n'.join(rows[1:])+'\n\\bottomrule\n\\end{tabular}\n\\end{samepage}\n')
        if row['method'] in {'native_marginal_glossary','native_small_font_side_note','explicit_geometric_side_note'}:
            main=(self.wrapped_figure_tex(row['table']) if row.get('table') else
                  ''.join(self.fragment(a,l,h,False) for a,l,h in self.index.fragments(*row['main_range'])))
            notes=[]
            for bounds,color in zip(row['note_ranges'],row['note_colors']):
                # Math fragments contain TeX comments and must retain their
                # newlines. Only source prose line endings are side-note breaks.
                text=''.join(self.fragment(a,l,h,False).replace('\n',r'\\ ')
                    if a['kind']=='text' else self.fragment(a,l,h,False)
                    for a,l,h in self.index.fragments(*bounds))
                if color is not None:
                    text='{\\color[rgb]{'+','.join(f'{c:g}' for c in color)+'}'+text+'}'
                notes.append(text)
            note_top=('\\vspace*{'+f"{row['note_top_fraction']:.5f}"+
                '\\dimexpr\\ht\\BookSideFirst+\\dp\\BookSideFirst\\relax}\n'
                if row.get('note_top_fraction') else '')
            compact=row['method']=='native_small_font_side_note'
            return (('\n\\begingroup\\renewcommand{\\BookSideGap}{2pt}\n' if compact else '')+
                '\n\\begin{BookSideBySide}{'+f"{row['widths'][0]:.5f}"+'}{'+
                f"{row['widths'][1]:.5f}"+'}{'+('1' if row['main_left'] else '0')+'}\n'+main+
                '\n\\BookSideSwitch\n\\scriptsize\\raggedright\\setlength{\\parindent}{0pt}\n'+
                note_top+'\\par\\smallskip\\noindent '.join(notes)+'\n\\end{BookSideBySide}\n'+
                ('\\endgroup\n' if compact else ''))
        if row['method']=='native_lateral_caption':
            image=self.index.atoms[self.index.ids[row['image']]]
            if image['id'] in self.recovered_figures:
                image={**image,**self.recovered_figures[image['id']]}
            picture=''.join(self.anchor(t) for t in row['targets'])+self.source_marker(image)
            picture+='\\BookImage[1\\linewidth]{'+self.asset(image)+'}\n'
            caption=self.caption_tex(list(self.index.fragments(*row['caption_range'])))
            before,after='',''
            if self.config.get('float_assets'):
                # This complete image/caption pair owns one float, just like
                # a vertically stacked source figure. A fixed minipage after
                # deferred floats otherwise reserves space before those floats
                # are scheduled and can intrude into the footer.
                before='\\begin{figure}[htp]\n\\centering\n'
                after='\\end{figure}\n'
                self.source_order_layout.append(dict(method='deferred_lateral_figure_float',
                    lo=row['lo'],hi=row['hi'],page=row['page']))
            return ('\n'+before+'\\begin{BookSideBySide}{'+f"{row['widths'][0]:.5f}"+'}{'+
                f"{row['widths'][1]:.5f}"+'}{'+('1' if row['image_left'] else '0')+'}\n'+
                picture+'\n\\BookSideSwitch\n\\BookCaptionParagraph{'+caption+'}\n'
                '\\end{BookSideBySide}\n'+after)
        main = []
        for lo, hi in row['main_ranges']:
            main.append(''.join(self.fragment(a,l,h,False)
                                for a,l,h in self.index.fragments(lo,hi)).strip())
        if row.get('note_range'):
            rgb = ','.join(f'{c:g}' for c in row['color'])
            aside = ('\\definecolor{BookMixedNoteBack}{rgb}{'+rgb+'}\n'
                     '\\begin{BookSourcePanel}[unbreakable,colback=BookMixedNoteBack,'
                     'colframe=BookMixedNoteBack,boxrule=0pt,before skip=0pt,after skip=0pt,'
                     'left=6pt,right=6pt,top=6pt,bottom=6pt]{}{BookMixedNoteBack}\n')
            aside += ''.join(self.fragment(a,l,h,False) for a,l,h in self.index.fragments(*row['note_range']))
            aside += '\n\\end{BookSourcePanel}\n'
        else:
            image = self.index.atoms[self.index.ids[row['image']]]
            aside = ''.join(self.anchor(t) for t in row['targets']) + self.source_marker(image)
            aside += '\\BookImage[1\\linewidth]{' + self.asset(image) + '}\n'
            caption = ''.join(self.fragment(a,l,h,False)
                              for a,l,h in self.index.fragments(*row['image_range']) if a['id'] != image['id'])
            if caption.strip(): aside += '\\BookCaption{' + caption + '}\n'
        return ('\n\\begin{BookWrappedPair}{' + f"{row['note_width']:.5f}" + '}{' +
                f"{row['gutter']:.5f}" + '}{' + ('1' if row['note_left'] else '0') + '}{1}\n' +
                '\n'.join(main) + '\n\\BookWrapSwitch\n' + aside + '\n\\end{BookWrappedPair}\n')

    def caption_tex(self, parts, in_box=False):
        from book_pipeline.publishing.caption_layout import caption_separator
        chunks = []
        for n, (atom, lo, hi) in enumerate(parts):
            separator = caption_separator(parts[n-1][0], atom) if n and lo == 0 else ''
            chunks.append(separator + self.fragment(atom, lo, hi, in_box))
        return ''.join(chunks)

    def observed_captions_tex(self, captions):
        # Each detected line remains a paragraph; source anchors and reviewed
        # typography still pass through the normal fragment renderer.
        return ''.join('\n\\BookCaption{' + self.fragment(a, 0, a['length'], False) + '}\n'
                       for a in captions)

    def render(self):
        from book_pipeline.publishing.native_body_styles import native_minor_headings
        minor_headings=native_minor_headings(self.index,self.registry,
            {c['atom'] for c in self.decisions['corrections']})
        total = self.index.prefix[-1]
        positions = {0, total, *self.index.prefix}
        questions, answers = [], []
        for item in self.registry:
            if item['kind'] != 'example':
                continue
            for field, spans in (('question_ranges', questions), ('answer_ranges', answers)):
                for r in item.get(field, []):
                    a, z = max(item['lo'], r['lo']), min(item['hi'], r['hi'])
                    if a < z:
                        spans.append((a, z)); positions.update((a, z))
        for lo,hi in self.bold_spans:positions.update((lo,hi))
        starts, ends, anchored, skipped = defaultdict(list), defaultdict(list), defaultdict(list), []
        note_insertions = defaultdict(list)
        from book_pipeline.publishing.native_tables import native_ruled_tables
        from book_pipeline.publishing.native_lateral import native_side_notes
        native_tables=native_ruled_tables(self.index,self.registry,self.decisions['exclusions'])
        native_notes,relocated_notes=native_side_notes(self.index,native_tables,self.registry,
            self.decisions['exclusions'],self.page_footnotes,
            {c['atom'] for c in self.decisions['corrections']})
        self.page_footnotes=[n for n in self.page_footnotes if n['id'] not in relocated_notes]
        native_tables=[r for r in native_tables if not any(n.get('table') is r for n in native_notes)]
        from book_pipeline.publishing.side_notes import explicit_side_note_rows
        explicit_notes=explicit_side_note_rows(self.index,self.registry,
            [*self.decisions['exclusions'],*native_tables,*native_notes,*self.page_footnotes],
            {c['atom'] for c in self.decisions['corrections']})
        for row in explicit_notes:
            skipped.extend((lo,hi,'explicit_aside_rendered_beside_owner') for lo,hi in row['note_ranges'])
        from book_pipeline.publishing.native_lateral import marginal_figures
        native_figures=marginal_figures(self.index,self.registry,
            [*self.decisions['exclusions'],*native_tables,*native_notes,*explicit_notes,*self.page_footnotes],
            {c['atom'] for c in self.decisions['corrections']},
            image_boxes={a:r['bbox'] for a,r in self.recovered_figures.items()})
        skipped.extend((r['lo'],r['hi'],'native_table_or_note_unit')
                       for r in [*native_tables,*native_notes,*explicit_notes,*native_figures])
        for note in self.page_footnotes:
            positions.update((note['lo'], note['hi'], note['insert_after']))
            skipped.append((note['lo'], note['hi'], 'page_footnote_rendered_separately'))
            note_insertions[note['insert_after']].append(note)
        associated_captions={}
        relocated_labels=defaultdict(list)
        for row in self.margin_assets:
            for member in ('label','image'):
                n = self.index.ids[row[member]]
                skipped.append((*self.index.prefix[n:n+2], 'margin_asset_moved_to_reference'))
        if self.config.get('float_assets'):
            from book_pipeline.publishing.figure_captions import split_caption_rows
            caption_rows = split_caption_rows(self.index, self.registry, self.decisions['exclusions'],
                {a for a in self.corrections if self.corrections[a]} | {a for a in self.refs if self.refs[a]})
            for row in caption_rows:
                if any(r['lo']<=row['lo'] and row['hi']<=r['hi'] for r in native_figures):continue
                skipped.append((row['lo'],row['hi'],'caption_moved_with_asset'))
                associated_captions[self.index.prefix[self.index.ids[row['image']]]] = (row['lo'],row['hi'],'figure')
            self.source_order_layout.extend(caption_rows)
        for o in self.registry:
            anchored[o["lo"]].append(o)
            positions.add(o["lo"])
            if (self.config.get('float_assets') and o.get('caption_span') and o['kind'] in {'figure','table'}
                    and not any(r['lo']<=o['lo']<o['hi']<=r['hi'] for r in native_figures)):
                a,b=o['caption_span'];positions.update((a,b))
                skipped.append((a,b,'caption_moved_with_asset'))
                associated_captions[o['lo']]=(a,b,o['kind'])
            if o.get('associated_math') and o.get('anchor_only'):
                atom=self.index.atoms[self.index.ids[o['associated_math']]]
                target=self.index.prefix[self.index.ids[atom['id']]]
                if atom['kind']!='display_math' or target>=o['lo']:
                    raise ValueError('Reordered label must follow its associated source formula')
                relocated_labels[target].append(o)
                skipped.append((o['lo'],o['hi'],'numbered_label_moved_before_formula'))
                positions.add(o['hi'])
                self.source_order_layout.append({'label_start':o['start'],'formula':atom['id'],
                                                 'method':'same_line_numbered_label_before_formula',
                                                 'evidence':o['boundary_evidence']})
            if o["kind"] in ("equation", "formula", "figure", "table") or o.get("anchor_only"):
                continue
            positions.add(o["hi"])
            starts[o["lo"]].append(o)
            ends[o["hi"]].append(o)
            if o["kind"] in HEADINGS:
                skipped.append((o["lo"], o["hi"], "heading_transformed"))
            elif o.get("body_start"):
                body = self.index.position(o["body_start"])
                if body > o["lo"]:
                    positions.add(body)
                    skipped.append((o["lo"], body, "object_title_transformed"))
        for e in self.decisions["exclusions"]:
            positions.update((e["lo"], e["hi"]))
            skipped.append((e["lo"], e["hi"], "excluded_" + e["reason"]))
        from book_pipeline.publishing.mixed_notes import mixed_note_rows
        from book_pipeline.publishing.native_lateral import marginal_references,side_caption_rows,marginal_glossaries
        glossaries=marginal_glossaries(self.index,self.registry,
            [dict(lo=a,hi=b) for a,b,_ in skipped],{c['atom'] for c in self.decisions['corrections']})
        marginal_rows=marginal_references(self.index,self.registry,
            [dict(lo=a,hi=b) for a,b,_ in skipped]+glossaries,{c['atom'] for c in self.decisions['corrections']})
        marginal_ends=defaultdict(list)
        for row in marginal_rows:
            marginal_ends[row['attach_after']].append(row)
            skipped.append((row['lo'],row['hi'],'native_marginal_reference'))
        self.source_order_layout.extend(marginal_rows)
        mixed_notes = mixed_note_rows(self.index, self.registry,
            [dict(lo=a,hi=b) for a,b,_ in skipped] + self.side_note_rows,
            {c['atom'] for c in self.decisions['corrections']} | set(self.refs), self.directory)
        from book_pipeline.publishing.wrapped_prose import wrapped_prose
        wrap_joins, wrap_figures = wrapped_prose(self.index, self.registry,
            [dict(lo=a,hi=b) for a,b,_ in skipped] + self.page_footnotes + mixed_notes,
            {c['atom'] for c in self.decisions['corrections']}, self.side_note_rows)
        wrap_figures.extend(mixed_notes)
        wrap_figures.extend(glossaries)
        wrap_figures.extend(native_tables)
        wrap_figures.extend(native_notes)
        wrap_figures.extend(explicit_notes)
        wrap_figures.extend(native_figures)
        wrap_figures.extend(self.native_code_rows)
        from book_pipeline.publishing.algorithm_lines import algorithm_line_rows
        wrap_figures.extend(algorithm_line_rows(self.index,self.registry,
            [dict(lo=a,hi=b) for a,b,_ in skipped]+wrap_figures,
            {c['atom'] for c in self.decisions['corrections']},self.native_code))
        wrap_figures.extend(side_caption_rows(self.index,self.registry,
            [dict(lo=a,hi=b) for a,b,_ in skipped]+wrap_figures,
            {c['atom'] for c in self.decisions['corrections']},
            image_boxes={a:r['bbox'] for a,r in self.recovered_figures.items()}))
        for row in wrap_joins:
            self.join_pairs.add((row['left'],row['right']))
            self.join_left.add(row['left']); self.join_right.add(row['right'])
        wrapped_starts = {row['lo']:row for row in wrap_figures}
        for row in wrap_figures:
            skipped.append((row['lo'],row['hi'],'source_wrap_rendered_as_unit'))
        self.source_order_layout.extend([*wrap_joins,*wrap_figures])
        ordered = sorted(positions)
        side_starts = {row['lo']: row for row in self.side_note_rows}
        side_switches = {row['mid']: row for row in self.side_note_rows}
        side_ends = {row['hi']: row for row in self.side_note_rows}
        stack, chunks, chapters, current = [], [], [], []
        last_atom = None
        caption_open = None
        footnote_previous = {}
        pending_notes = []
        headings = sorted((o for o in self.registry if o["kind"] in HEADINGS), key=lambda o: o["lo"])
        self.body_excerpt = (self.config.get('publication_scope') == 'excerpt'
            and (not headings or headings[0].get('toc_category') == 'body'
                 or ('toc_category' not in headings[0] and bool(headings[0].get('number')))))
        from book_pipeline.structure.pagination import HeadingPagination
        pagination = HeadingPagination(headings, layout=self.heading_layout)
        heading_space = {}
        for i, first in enumerate(headings):
            previous, count = first, 1
            for following in headings[i + 1:]:
                gap = self.index.text(previous["hi"], following["lo"]).strip()
                if gap:
                    break
                count += 1
                previous = following
            if count > 1:
                heading_space[first["lo"]] = 5 * count + 3
        environments = {"definition": "BookDefinition", "notation": "BookDefinition", "theorem": "BookTheorem", "lemma": "BookTheorem", "corollary": "BookTheorem", "proposition": "BookTheorem", "algorithm": "BookAlgorithm", "example": "BookExample", "remark": "BookRemark", "numbered": "BookRemark", "proof": "BookProof", "solution": "BookSolution", "exercise": "BookExercise", "footnote": "BookFootnote", "bibliography_entry": "BookBibliography", "subheading": "BookSubheading", "body": "BookBodySample"}
        colors = validate_colors(self.config.get('block_colors', {}))
        boxed_notes = defaultdict(list)
        for note in self.page_footnotes:
            if note['status'] != 'matched':continue
            parents = [o for o in self.registry if o['kind'] in COLOR_KINDS and
                       o['lo'] <= note['reference']['edited_lo'] < o['hi']]
            if parents:
                boxed_notes[min(parents, key=lambda o:o['lo'])['anchor']].append(note)
        page_breaks = set(self.config.get('page_break_before_anchors', []))
        def environment(item):
            if item['kind'] == 'source_panel':
                return 'BookSourcePanel'
            return 'BookColoredBlock' if item['kind'] != 'example' and selected_color(colors, item) else environments[item['kind']]
        def opening(item, title):
            if item['kind'] == 'source_panel':
                rgb = ','.join(f'{max(0,min(1,c)):g}' for c in item['panel_color'])
                options = []
                definitions = '\\definecolor{BookSourcePanelInk}{rgb}{'+rgb+'}\n'
                if item.get('panel_background'):
                    rgb = ','.join(f'{max(0,min(1,c)):g}' for c in item['panel_background'])
                    definitions += '\\definecolor{BookSourcePanelBack}{rgb}{'+rgb+'}\n'
                    frame_rgb = ','.join(f'{max(0,min(1,c)):g}' for c in item['panel_frame'])
                    definitions += '\\definecolor{BookSourcePanelFrame}{rgb}{'+frame_rgb+'}\n'
                    options += ['colback=BookSourcePanelBack', 'colframe=BookSourcePanelFrame']
                    if not title and not item.get('panel_frame_visible'):
                        options += ['colframe=BookSourcePanelBack', 'boxrule=0pt']
                if item.get('panel_title_color'):
                    rgb = ','.join(f'{max(0,min(1,c)):g}' for c in item['panel_title_color'])
                    definitions += '\\definecolor{BookSourcePanelTitle}{rgb}{'+rgb+'}\n'
                    ink_rgb = ','.join(f'{max(0,min(1,c)):g}' for c in item['panel_title_ink'])
                    definitions += '\\definecolor{BookSourcePanelTitleInk}{rgb}{'+ink_rgb+'}\n'
                    options += ['colbacktitle=BookSourcePanelTitle', 'coltitle=BookSourcePanelTitleInk']
                if item.get('lateral_note'):
                    options += ['unbreakable', 'before skip=0pt', 'after skip=0pt', 'left=6pt', 'right=6pt', 'top=6pt', 'bottom=6pt']
                tail = next((m for m in self.raster_marks.values() if m['panel'] == item['anchor']),None)
                if tail:
                    glyph = '\\includegraphics[height='+f"{tail['height_pt']:g}"+'bp]{'+tail['image']+'}'
                    options += ['overlay unbroken and last={\\node[anchor=south east,inner sep=0pt] at '
                                '([xshift=-'+f"{tail['right_bp']:g}"+'bp,yshift='+f"{tail['bottom_bp']:g}"+
                                'bp]frame.south east){'+glyph+'};}']
                    if tail not in self.source_order_layout:self.source_order_layout.append(tail)
                optional = '['+','.join(options)+']' if options else ''
                return '\n'+definitions+'\\begin{BookSourcePanel}'+optional+'{'+title+'}{BookSourcePanelInk}\n'
            if item['kind'] == 'subheading':
                return '\n\\begin{BookSubheading}{' + f"{subheading_size(self.typography, item):g}" + '}\n'
            color = selected_color(colors, item)
            extra = '{BookColor' + color[1:] + '}' if color and item['kind'] != 'example' else ''
            notes = boxed_notes.get(item['anchor'], [])
            reserve = '\n\\BookFootnoteBoxReserve{' + '\\par\n'.join(self.page_footnote_body(n) for n in notes) + '}\n' if notes else ''
            # The default after-skip permits TeX to ship the box's page before
            # insert_notes emits its page footnotes. Keep that gap unbreakable;
            # the reserved height can then move the box and notes together.
            options = (r'[lines before break=100,shrink break goal=\BookFootnoteReserve,'
                       r'after={\par\nobreak\vskip9pt\nobreak}]') if notes else ''
            return reserve + '\n\\begin{' + environment(item) + '}' + options + '{' + title + '}' + extra + '\n'
        join_pairs = self.join_pairs
        group_limits = {}
        group_atoms = defaultdict(list)
        for i, atom in enumerate(self.index.atoms):
            limits = group_limits.setdefault(atom["group"], [self.index.prefix[i], self.index.prefix[i + 1]])
            limits[1] = self.index.prefix[i + 1]
            group_atoms[atom["group"]].append(i)
        table_captions, moved_anchors, image_groups, table_before_groups = {}, {
            t for row in wrap_figures for t in row['targets']}, {}, {}
        floating_groups={};floating_open=None;pending_reopen=False
        caption_limits = {}
        geometric_captions = {}
        for group, indices in group_atoms.items():
            if group in self.image_row_groups:
                continue
            image_bodies = [i for i in indices if self.index.atoms[i]["kind"] == "image"]
            image_captions = [i for i in indices if self.index.atoms[i]["role"] in ("image_caption", "chart_caption", "image_footnote", "chart_footnote")]
            group_lo, group_hi = group_limits[group]
            all_captions = [i for i in indices if self.index.atoms[i]["role"] in CAPTION_ROLES]
            if all_captions:
                caption_limits[group] = (self.index.prefix[min(all_captions)], self.index.prefix[max(all_captions) + 1])
            safe_group = not any(group_lo < p < group_hi for p in (*starts, *ends)) and not any(
                a < group_hi and b > group_lo for a, b, _ in skipped)
            from book_pipeline.publishing.caption_layout import caption_sides, METHOD as CAPTION_GEOMETRY
            geometry = caption_sides([self.index.atoms[i] for i in indices],
                                     self.decisions.get('caption_boxes', {})) if safe_group else None
            formula_crop = any(self.index.atoms[i].get('recovery_issue') == 'formula_source_crop_requires_transcription' for i in indices)
            if (self.config.get('float_assets') and safe_group and not formula_crop
                    and not any(a < group_hi and group_lo < z and reason == 'margin_asset_moved_to_reference'
                                for a,z,reason in skipped)):
                assets=[i for i in indices if self.index.atoms[i]['kind'] in {'image','table'}]
                only_assets_and_captions=all(self.index.atoms[i]['kind'] in {'image','table'} or
                                            self.index.atoms[i]['role'] in CAPTION_ROLES for i in indices)
                long_table=any(self.index.atoms[i]['kind']=='table' and
                               '\\begin{longtable}' in (table_tex(self.index.atoms[i]['content'],False) or '') for i in assets)
                # Searchable tables participate in the prose's argument. Keep
                # their source position instead of letting later text overtake
                # them at a page boundary. The existing unbreakable table body
                # can still move intact to the next page.
                if assets and only_assets_and_captions and not long_table and not any(
                        self.index.atoms[i]['kind']=='table' for i in assets):
                    float_hi = group_hi
                    # Native extraction gives the figure and its caption
                    # separate source groups. A float must carry an adjacent,
                    # geometrically corroborated caption with it.
                    stop = max(indices) + 1
                    if len(assets) == 1 and self.index.atoms[assets[0]]['kind'] == 'image' and not all_captions and stop < len(self.index.atoms):
                        image = self.index.atoms[assets[0]]
                        following = self.index.atoms[stop]
                        caption_indices = group_atoms[following['group']]
                        caption_atoms = [self.index.atoms[i] for i in caption_indices]
                        a, b = image.get('bbox'), following.get('block_bbox') or following.get('bbox')
                        caption_only = all(atom['role'] in {'image_caption','chart_caption','image_footnote','chart_footnote'}
                                           and atom['page'] == image['page'] for atom in caption_atoms)
                        below = (a and b and -.01 <= b[1]-a[3] <= .08 and min(a[2], b[2]) > max(a[0], b[0]))
                        proposed_hi = self.index.prefix[max(caption_indices) + 1]
                        safe_caption = (not any(group_lo < p < proposed_hi for p in (*starts, *ends))
                                        and not any(x < proposed_hi and z > group_lo for x,z,_ in skipped))
                        if caption_only and below and safe_caption:
                            float_hi = proposed_hi
                            self.source_order_layout.append(dict(method='adjacent_source_caption_kept_with_figure',
                                group=group, caption_group=following['group'], lo=group_lo, hi=float_hi))
                    floating_groups[group]=(group_lo,float_hi,'table' if all(self.index.atoms[i]['kind']=='table' for i in assets) else 'figure')
            if geometry:
                continuation = self.decisions.get('caption_continuations', {}).get(geometry['asset'])
                if continuation:
                    skipped.append((continuation['lo'],continuation['hi'],'reviewed_caption_continuation'))
                    geometry['continuation'] = continuation['atoms']
                    if group in floating_groups:
                        a,_,kind = floating_groups[group]
                        floating_groups[group] = (a,continuation['hi'],kind)
                    self.source_order_layout.append(dict(method='reviewed_caption_continuation_v1',
                        asset=geometry['asset'], lo=continuation['lo'], hi=continuation['hi'],
                        evidence=continuation['evidence']))
                geometric_captions[geometry['asset']] = geometry
                for side in ('before', 'after'):
                    for caption in geometry[side]:
                        n = self.index.ids[caption['id']]
                        skipped.append((*self.index.prefix[n:n+2], 'caption_moved_to_observed_side'))
                        self.source_order_layout.append(dict(method=CAPTION_GEOMETRY, atom=caption['id'],
                            asset=geometry['asset'], side=side, bbox=self.decisions['caption_boxes'][caption['id']]))
                continue
            if len(image_bodies) == 1 and image_captions and min(image_captions) > image_bodies[0] and len(image_bodies) + len(image_captions) == len(indices) and safe_group:
                image_groups[group] = (group_lo, group_hi)
            bodies = [i for i in indices if self.index.atoms[i]["kind"] == "table"]
            captions = [i for i in indices if self.index.atoms[i]["role"] == "table_caption"]
            if len(bodies) == 1 and captions and max(captions) < bodies[0] and len(bodies) + len(captions) == len(indices) and safe_group:
                body_atom = self.index.atoms[bodies[0]]
                table = table_tex(corrected_content(body_atom, self.corrections[body_atom["id"]]), False)
                if table is None or "\\begin{longtable}" not in table:
                    table_before_groups[group] = (group_lo, group_hi)
            if len(bodies) != 1 or not captions or min(captions) < bodies[0]:
                continue
            # Move only the caption of a self-contained table group; never move
            # narrative or content through a semantic boundary or an exclusion.
            if len(bodies) + len(captions) != len(indices) or not safe_group:
                continue
            caption_lo, caption_hi = self.index.prefix[min(captions)], self.index.prefix[max(captions) + 1]
            table_captions[self.index.atoms[bodies[0]]["id"]] = (caption_lo, caption_hi)
            skipped.append((caption_lo, caption_hi, "caption_moved_above_table"))
        # A complete image row is one indivisible asset. Move it outside a
        # breakable example/theorem box just like an ordinary figure, so a tall
        # pair of images cannot extend past the bottom of the physical page.
        for carrier, row in self.image_rows.items():
            start = self.index.ids[carrier]
            stop = max(self.index.ids[member] for member in row['members']) + 1
            a, b = self.index.prefix[start], self.index.prefix[stop]
            members = set(row['members'])
            safe = (all(atom['id'] in members for atom in self.index.atoms[start:stop])
                    and not any(a < p < b for p in (*starts, *ends))
                    and not any(x < b and z > a for x, z, _ in skipped))
            if self.config.get('float_assets') and safe:
                # A native multi-image row has the same caption ownership as
                # a single image. Carry a separate, corroborated caption whole.
                if stop < len(self.index.atoms):
                    following=self.index.atoms[stop]
                    caption_indices=group_atoms[following['group']]
                    captions=[self.index.atoms[i] for i in caption_indices]
                    box=following.get('block_bbox') or following.get('bbox')
                    proposed_hi=self.index.prefix[max(caption_indices)+1]
                    region=row['bbox']
                    if (box and all(c['role'] in {'image_caption','chart_caption','image_footnote','chart_footnote'}
                                    and c['page']==self.index.atoms[start]['page'] for c in captions)
                            and -.01<=box[1]-region[3]<=.08
                            and min(box[2],region[2])>max(box[0],region[0])
                            and not any(a<p<proposed_hi for p in (*starts,*ends))
                            and not any(x<proposed_hi and z>a for x,z,_ in skipped)):
                        b=proposed_hi
                        self.source_order_layout.append(dict(method='adjacent_source_caption_kept_with_image_row',
                            group=self.index.atoms[start]['group'],caption_group=following['group'],lo=a,hi=b))
                floating_groups[self.index.atoms[start]['group']] = (a, b, 'figure')
        # MinerU occasionally treats an exercise/question preceding a table as
        # its caption. A confirmed semantic object owns that paragraph's style.
        semantic_caption_groups = {
            group for group, (a, b) in caption_limits.items()
            if any(o["kind"] not in ("equation", "figure", "table") and a <= o["lo"] < b
                   for o in self.registry)}
        from book_pipeline.publishing.list_layout import list_paragraphs, multiline_list_rows
        rule_region_changed = {c['atom'] for c in self.decisions['corrections']} | set(self.refs)
        list_styles = {group: style for group, style in list_paragraphs(self.index).items()
                       if not any(style['lo'] < p < style['hi'] for p in (*starts, *ends))
                       and not any(a < style['hi'] and b > style['lo'] for a,b,_ in skipped)}
        for group, style in list(list_styles.items()):
            if not style['continuation']:
                first = self.index.atoms[self.index.groups[style['group']][0]]
                prefix = re.match(r'^\s*'+re.escape(style['label'])+r'\s+', first['content'])
                edits = [*self.refs[first['id']], *self.corrections[first['id']]]
                if not prefix or any(e['start_offset'] < prefix.end() for e in edits):
                    del list_styles[group]
                else:
                    self.list_prefixes[first['id']] = prefix.end()
        list_styles = {group: style for group,style in list_styles.items() if style['item_group'] in list_styles}
        multiline_rows = multiline_list_rows(self.index, {*starts, *ends}, skipped,
            corrections=self.decisions['corrections'])
        multiline_rows = [row for row in multiline_rows if row['group'] not in list_styles
            and not any(self.index.prefix[self.index.ids[a]]+ref['start_offset'] < p <
                        self.index.prefix[self.index.ids[a]]+ref['end_offset']
                        for a in self.refs for ref in self.refs[a] for p in (row['lo'],row['hi']))]
        multiline_starts = {row['lo']:row for row in multiline_rows}
        multiline_ends = {row['hi'] for row in multiline_rows}
        multiline_keep = {row['keep_lo']:row for row in multiline_rows if row['keep_lo'] is not None}
        multiline_keep_ends = {row['list_hi'] for row in multiline_keep.values()}
        ordered = sorted(set(ordered) | set(multiline_starts) | multiline_ends | set(multiline_keep))
        self.source_order_layout.extend(multiline_rows)
        for style in list_styles.values():
            self.source_order_layout.append(dict(method='source_list_hanging_indent', **style))
        from book_pipeline.publishing.display_layout import display_runs, prompt_space, list_display_scope, formula_image_prompts, bounded_display_units, colon_introductions, fraction_prose_rows, fraction_prose_units
        from book_pipeline.publishing.math_rows import horizontal_math_rows
        layout_barriers = {*starts, *ends, *(positions-set(self.index.prefix))}
        intro_spaces = {p['lo']:p for p in colon_introductions(self.index,
            layout_barriers | set(note_insertions), skipped)
            if not any(c['atom'] in {a['id'] for a,_,_ in self.index.fragments(p['lo'],p['hi'])}
                       for c in self.decisions['corrections'])}
        self.source_order_layout.extend(intro_spaces.values())
        math_rows = horizontal_math_rows(self.index, self.decisions, layout_barriers | set(note_insertions), skipped)
        math_row_starts = {r['lo']:r for r in math_rows}
        math_row_members = {c['atom']:(r,c) for r in math_rows for c in r['cells']}
        math_skipped = [*skipped, *((r['lo'],r['hi'],'horizontal_math_row') for r in math_rows)]
        self.source_order_layout.extend(math_rows)
        fraction_rows = fraction_prose_rows(self.index, layout_barriers | set(note_insertions), math_skipped)
        fraction_starts = {r['lo'] for r in fraction_rows}
        fraction_ends = {r['hi'] for r in fraction_rows}
        self.fraction_prose_groups = {r['group'] for r in fraction_rows}
        self.source_order_layout.extend(fraction_rows)
        units = bounded_display_units(self.index, layout_barriers | set(note_insertions), math_skipped)
        units += fraction_prose_units(self.index, fraction_rows,
            layout_barriers | set(note_insertions),
            [*math_skipped,*((u['lo'],u['hi'],'existing_display_unit') for u in units)])
        unit_starts = {u['lo'] for u in units}
        unit_ends = {u['hi'] for u in units}
        self.source_order_layout.extend(units)
        runs = display_runs(self.index, self.decisions,
                            barriers=layout_barriers, skipped=math_skipped)
        run_starts = {r['lo']: r for r in runs}
        run_members = {atom: r for r in runs for atom in r['atoms']}
        self.source_order_layout.extend(runs)
        prompt_spaces = {}
        crop_prompts=formula_image_prompts(self.index,{*starts,*ends,*note_insertions},skipped)
        crop_prompt_starts={p['lo'] for p in crop_prompts}
        crop_prompt_ends={p['hi'] for p in crop_prompts}
        self.crop_prompt_images={p['image'] for p in crop_prompts}
        self.source_order_layout.extend(crop_prompts)
        list_ends = set()
        for style in list_styles.values():
            scope = list_display_scope(self.index, style, runs, {*starts, *ends}, self.decisions, math_skipped)
            list_ends.add(scope['hi'] if scope else style['hi'])
            if scope:
                self.source_order_layout.append(dict(method='source_list_includes_derivation',
                    group=style['group'], lo=style['lo'], prompt_hi=style['hi'], hi=scope['hi']))
            space = prompt_space(self.index, style, runs)
            if (space and not any(style['lo'] < b <= style['hi'] for b in (*starts, *ends))
                    and not any(u['lo'] <= style['lo'] < u['hi'] for u in units)):
                # A nested Needspace can force a break inside the enclosing
                # complete unit, leaving its numbered introduction behind.
                prompt_spaces[style['lo']] = space
                self.source_order_layout.append(dict(method='source_prompt_kept_with_display',
                    group=style['group'], lo=style['lo'], hi=style['hi'], baselines=space))
        body_started = self.body_excerpt
        from book_pipeline.publishing.native_rule_regions import native_rule_regions
        rule_regions = native_rule_regions(self.index,self.registry,
            [dict(lo=a,hi=b) for a,b,_ in skipped],
            rule_region_changed)
        rule_region_starts = {r['lo']:r for r in rule_regions}
        rule_region_ends = {r['hi'] for r in rule_regions}
        self.source_order_layout.extend(rule_regions)
        def close_float():
            nonlocal caption_open,floating_open,pending_reopen,last_atom
            if caption_open is not None:current.append('\\par\\endgroup\n');caption_open=None
            current.append('\n\\end{'+floating_open[2]+'}\n')
            # A source-adjacent display must not overtake its preceding figure
            # and leave that figure interrupting the following prose instead.
            stop = bisect_right(self.index.prefix, floating_open[1])-1
            if (floating_open[2] == 'figure' and 0 < stop < len(self.index.atoms)
                    and self.index.prefix[stop] == floating_open[1]
                    and self.index.atoms[stop]['kind'] == 'display_math'
                    and self.index.atoms[stop-1]['page'] == self.index.atoms[stop]['page']):
                current.append('\\FloatBarrier\n')
                self.source_order_layout.append(dict(method='source_adjacent_figure_before_display',
                    lo=floating_open[0],hi=floating_open[1],following=self.index.atoms[stop]['id']))
            floating_open=None;pending_reopen=bool(stack);last_atom=None
        def insert_notes(position):
            for note in note_insertions.pop(position, []):
                pending_notes.append(note)
            # tcolorbox uses minipage footnotes. Emit the page insertion outside
            # captured boxes so it cannot be swallowed or rendered inside them.
            if not stack and floating_open is None:
                for note in pending_notes:
                    current.append(self.page_footnote_tex(note))
                pending_notes.clear()
        for lo, hi in zip(ordered, ordered[1:]):
            if floating_open and lo==floating_open[1]:close_float()
            if caption_open is not None and (starts[lo] or ends[lo] or lo == self.toc_position):
                current.append("\\par\\endgroup\n"); caption_open = None
            if lo == self.toc_position:
                current.append("\n\\clearpage\\tableofcontents\\clearpage\n")
                last_atom = None
            for o in sorted(ends[lo], key=lambda o: -o["lo"]):
                if o["kind"] in HEADINGS:
                    continue
                if not stack or stack[-1]["anchor"] != o["anchor"]:
                    raise ValueError("Unbalanced semantic environments")
                stack.pop()
                if not pending_reopen:current.append("\n\\end{" + environment(o) + "}\n")
                if o["kind"] == "footnote":
                    last_atom = footnote_previous.pop(o["anchor"], None)
            if lo in side_ends:
                current.append('\n\\end{' + ('BookWrappedPair' if side_ends[lo].get('note_width') else 'BookSideBySide') + '}\n')
                last_atom = None
            if lo in side_switches:
                current.append('\n\\BookWrapSwitch\n' if side_switches[lo].get('note_width') else '\n\\BookSideSwitch\n')
                last_atom = None
            if lo in side_starts:
                row = side_starts[lo]
                if row.get('note_width'):
                    current.append('\n\\begin{BookWrappedPair}{' + f"{row['note_width']:.5f}" + '}{' +
                                   f"{row['gutter']:.5f}" + '}{' + ('1' if row['note_left'] else '0') +
                                   '}{' + ('1' if row['main_first'] else '0') + '}\n')
                else:
                    current.append('\n\\begin{BookSideBySide}{'+f"{row['widths'][0]:.5f}"+'}{'+
                                   f"{row['widths'][1]:.5f}"+'}{'+('1' if row['first_left'] else '0')+'}\n')
                last_atom = None
            insert_notes(lo)
            if pending_reopen:
                for parent in stack:
                    continuation='（例题续）' if parent['kind']=='example' else '（续）'
                    current.append(opening(parent, continuation))
                pending_reopen=False
            for o in sorted(starts[lo], key=lambda o: -o["hi"]):
                if o['anchor'] in page_breaks:
                    current.append('\n\\par\\clearpage\n')
                    last_atom = None
                if o["kind"] in HEADINGS:
                    force_start=self.heading_layout.level(o) in self.config.get('start_page_levels',[])
                    current.append('\n\\par\\clearpage\n' if force_start else pagination.before(o))
                    if o["kind"] == "chapter":
                        if current:
                            chapters.append("".join(current)); current = []
                    # An unnumbered preface can legitimately be a part/chapter.
                    # Keep front matter Roman until a numbered body heading, or
                    # an unnumbered body heading after the original contents.
                    begins_body = o.get('toc_category')=='body' if 'toc_category' in o else bool(o.get("number")) or (
                        self.toc_position is not None and lo > self.toc_position)
                    if o["kind"] in ("part", "chapter") and begins_body and not body_started:
                        current.append("\\mainmatter\n"); body_started = True
                    # Render the actual heading atoms, including native inline math.
                    # Rule/review titles never replace the actual source text.
                    if any(a["kind"] not in ("text", "math") for a, _, _ in self.index.fragments(o["lo"], o["hi"])):
                        raise ValueError("Heading contains a display formula/image/table")
                    shown = "".join(self.heading_fragment(a, l, h) for a, l, h in self.index.fragments(o["lo"], o["hi"]))
                    label = self.heading_layout.label_span(self.index.text(o['lo'], o['hi']))
                    if label:
                        prefix_end, body_start = (o['lo'] + offset for offset in label)
                        # A typography gap must never cut an atomic inline formula.
                        try:
                            self.index.position(self.index.address(prefix_end))
                            self.index.position(self.index.address(body_start))
                        except ValueError:
                            pass
                        else:
                            prefix = ''.join(self.heading_fragment(a,l,h) for a,l,h in self.index.fragments(o['lo'],prefix_end))
                            body = ''.join(self.heading_fragment(a,l,h) for a,l,h in self.index.fragments(body_start,o['hi']))
                            shown = prefix + r'\BookHeadingGap{}' + body
                    # Claim the anchor here; heading macro inserts it at the correct location.
                    self.anchors.append(o["anchor"])
                    level = self.heading_layout.level(o)
                    markers = "".join(self.source_marker(a) for a, _, _ in self.index.fragments(o["lo"], o["hi"]))
                    if lo in heading_space and not self.heading_layout.standalone(o):
                        current.append("\n\\Needspace{" + str(heading_space[lo]) + "\\baselineskip}\n")
                    toc_title = re.sub(r'\{\\BookUnicodeFont\\char"24(?:6[0-9A-F]|7[0-3])\}', "", shown)
                    toc_title = re.sub(r'\\BookFootnoteReference\{([^{}]+)\}',
                                       lambda m: r'\textsuperscript{\BookFootnoteLabel{' + m[1] + '}}', toc_title)
                    toc_title = r'\texorpdfstring{'+toc_title+'}{'+bookmark_title(self.index,o['lo'],o['hi'])+'}'
                    if o['anchor'] in minor_headings:
                        current.append('\n\\BookMinorHeading{'+str(level)+'}{'+shown+'}{'+o['anchor']+'}{'+markers+'}{'+toc_title+'}\n')
                        self.source_order_layout.append(minor_headings[o['anchor']])
                    else:
                        current.append("\n\\BookHeading{" + str(level) + "}{" + shown + "}{" + o["anchor"] + "}[" + markers + "][{" + toc_title + "}]\n")
                else:
                    if o["kind"] == "footnote":
                        footnote_previous[o["anchor"]] = last_atom
                        last_atom = None
                    if o.get("body_start") and self.index.position(o["body_start"]) > o["lo"]:
                        title = self.object_title(o)
                    else:
                        # Without a separately confirmed title span the original
                        # label remains in the body. Adding a generated label
                        # would repeat "例2.1", "解", or "证明" in the PDF.
                        title = ""
                    current.append("\n" + self.anchor(o["anchor"]) + opening(o, title))
                    stack.append(o)
            for o in relocated_labels[lo]:
                shown=''.join(self.source_marker(a)+self.content_fragment(a,l,h,False)
                              for a,l,h in self.index.fragments(o['lo'],o['hi']))
                current.append('\n'+self.anchor(o['anchor'])+shown+'\n')
                moved_anchors.add(o['anchor']);last_atom=None
            if lo in wrapped_starts:
                current.append(self.wrapped_figure_tex(wrapped_starts[lo]))
                last_atom = None
            group=self.index.atoms[self.index.ids[self.index.address(lo)['atom']]]['group']
            if group in floating_groups and lo==floating_groups[group][0]:
                if caption_open is not None:current.append('\\par\\endgroup\n');caption_open=None
                for parent in reversed(stack):current.append('\n\\end{'+environment(parent)+'}\n')
                floating_open=floating_groups[group]
                current.append('\n\\par\\begin{'+floating_open[2]+'}[htp]\n\\centering\n')
                self.source_order_layout.append({'group':group,'method':'deferred_asset_float','placement':'htp',
                                                'lo':lo,'hi':floating_open[1],'surrounding_modules':[p['anchor'] for p in stack]})
                last_atom=None
            if lo in run_starts:
                current.append('\n\\begin{gather*}\n')
            elif self.index.atoms[self.index.ids[self.index.address(lo)['atom']]]['id'] in run_members:
                current.append('\\\\\n')
            math_cell = math_row_members.get(self.index.address(lo)['atom'])
            if lo in unit_starts:
                current.append('\n\\begin{BookDisplayUnit}\n')
            if lo in fraction_starts:
                current.append('\n\\par\\addvspace{6pt}\\noindent\\begin{minipage}{\\linewidth}\\centering\n')
            if lo in math_row_starts:
                current.append('\n\\par\\addvspace{6pt}\\noindent'
                               '\\begin{minipage}{\\linewidth}\\noindent%\n')
            if math_cell:
                _, cell = math_cell
                current.append('\\hspace*{' + f"{cell['gap']:.6f}" + '\\linewidth}'
                               '\\begin{minipage}[t]{' + f"{cell['width']:.6f}" + '\\linewidth}'
                               '\\vspace{0pt}\\BookMathCellSetup\\noindent%\n')
            for o in anchored[lo]:
                if o['anchor'] in self.image_row_targets or o['anchor'] in self.margin_targets:
                    continue
                if (o["kind"] in ("equation", "formula", "figure", "table") or o.get("anchor_only")) and o["anchor"] not in moved_anchors:
                    current.append(self.anchor(o["anchor"]))
            actions = [action for a, b, action in skipped if a <= lo and hi <= b]
            action = actions[0] if actions else "rendered"
            self.ledger.append({"start": lo, "end": hi, "action": action})
            if actions:
                if caption_open is not None:
                    current.append("\\par\\endgroup\n"); caption_open = None
                if action in ("heading_transformed", "object_title_transformed"):
                    last_atom = None
                continue
            for atom, start, end in self.index.fragments(lo, hi):
                if lo in rule_region_starts and start == 0:
                    region = rule_region_starts[lo]
                    rgb = ','.join(f'{c:g}' for c in region['color'])
                    current.append('\n\\begin{BookSourceInset}{'+f"{region['left_fraction']:.5f}"+
                        '}{'+f"{region['right_fraction']:.5f}"+'}{'+
                        f"{region['thickness_pt']:g}"+'}{'+rgb+'}\n')
                    last_atom = None
                # Even a fully deleted text atom owns a source anchor and may
                # close a list or another layout environment. Its empty text
                # replacement must pass through the same boundary handling.
                rule = self.source_rules.get((atom['id'], 'before'))
                if rule and start == 0:
                    rgb = ','.join(f'{c:g}' for c in rule['color'])
                    current.append('\n\\BookSourceRuleBefore{' + f"{rule['thickness_pt']:g}" + '}{' + rgb + '}\n')
                    self.source_order_layout.append(rule)
                display_run = run_members.get(atom['id'])
                if display_run or math_cell:
                    pass  # gather rows already have explicit, non-paragraph separators.
                elif last_atom is not None and last_atom["group"] != atom["group"] and (last_atom["id"], atom["id"]) not in join_pairs:
                    # A display ends the preceding paragraph itself. A blank
                    # paragraph before its source anchor inserts an empty line
                    # and lets a proof introduction separate from its formula.
                    display = atom['kind'] == 'display_math' or (atom['kind']=='math' and formula_tags(atom['content']))
                    current.append("\n" if display else "\n\n")
                elif last_atom is not None and (last_atom["id"], atom["id"]) in join_pairs:
                    current.append("\n")
                elif (last_atom is not None and last_atom['group']==atom['group']
                        and last_atom['page']!=atom['page']
                        and last_atom['kind']==atom['kind']=='text'
                        and re.search(r'[A-Za-z0-9]$',last_atom['content'])
                        and re.match(r'[A-Za-z0-9]',atom['content'])):
                    # Saved cross-page units keep their original text atoms.
                    # A page boundary between Latin words still needs a space.
                    current.append('\n')
                if (atom.get('block_type') == 'paragraph_title' and start == 0
                        and lo == group_limits[atom['group']][0]):
                    # A recognized object title can remain ordinary source
                    # content after structural classification. It still needs
                    # body lines below it, even without a semantic box.
                    current.append('\n\\par\\Needspace{4\\baselineskip}\n')
                list_style = list_styles.get(atom['group'])
                if lo in intro_spaces:
                    current.append('\n\\par\\Needspace{4\\baselineskip}\n')
                boundary=self.decisions.get('native_paragraph_starts',{}).get(atom['id'])
                if boundary and start==0:
                    current.append('\n\n')
                    self.source_order_layout.append(dict(atom=atom['id'],**boundary))
                if lo in multiline_keep:
                    current.append('\n\\par\\Needspace{'+str(multiline_keep[lo]['keep_baselines'])+
                                   '\\baselineskip}\\begin{samepage}\n')
                if lo in crop_prompt_starts and start==0:
                    current.append('\n\\par\\begingroup\\interlinepenalty=10000\n')
                if list_style and lo == list_style['lo']:
                    if lo in prompt_spaces:
                        current.append('\n\\par\\Needspace{'+str(prompt_spaces[lo])+'\\baselineskip}\n')
                    macro = 'BookListContinuationBegin' if list_style['continuation'] else 'BookListBegin'
                    current.append('\n\\'+macro+'{'+str(list_style['indent_em'])+'}{'+escape(list_style['label'])+'}\n')
                if lo in multiline_starts:
                    current.append('\n\\par\\begingroup\\leftskip=1em\\hangindent=1.4em'
                                   '\\hangafter=1\\noindent\n')
                if atom["group"] in image_groups and atom['group'] not in floating_groups and lo == image_groups[atom["group"]][0]:
                    current.append("\n\\par\\noindent\\begin{minipage}{\\linewidth}\n")
                if atom["group"] in table_before_groups and atom['group'] not in floating_groups and lo == table_before_groups[atom["group"]][0]:
                    current.append("\n\\par\\noindent\\begin{minipage}{\\linewidth}\n")
                if atom["id"] in table_captions:
                    table = table_tex(corrected_content(atom, self.corrections[atom["id"]]), bool(stack))
                    keep_table = table is None or "\\begin{longtable}" not in table
                    current.append("\n\\par\\noindent\\begin{minipage}{\\linewidth}\n" if keep_table else "\n\\Needspace{6\\baselineskip}\n")
                    caption_lo, caption_hi = table_captions[atom["id"]]
                    for position, objects in anchored.items():
                        if caption_lo <= position < caption_hi:
                            for o in objects:
                                current.append(self.anchor(o["anchor"]))
                                moved_anchors.add(o["anchor"])
                    caption = self.caption_tex(list(self.index.fragments(caption_lo, caption_hi)), bool(stack))
                    current.append("\n\\BookCaption{" + caption + "}\n")
                cap=associated_captions.get(lo)
                if cap and cap[2]=='table':
                    caption=self.caption_tex(list(self.index.fragments(cap[0],cap[1])))
                    current.append('\n\\BookCaption{'+caption+'}\n')
                value = self.fragment(atom, start, end, bool(stack), display_row=bool(display_run or math_cell))
                if end == atom['length'] and atom['id'] in self.native_ocr_punctuation:
                    value += self.native_ocr_punctuation[atom['id']]['text']
                if end==atom['length']:
                    for note in marginal_ends.get(atom['id'],[]):
                        value+='\\BookMarginText{'+self.caption_tex(list(
                            self.index.fragments(note['lo'],note['hi'])))+'}'
                geometry = geometric_captions.get(atom['id'])
                if geometry:
                    value = (self.observed_captions_tex(geometry['before']) + value
                             + ('\n\\nopagebreak\n' if geometry['after'] else '')
                             + self.observed_captions_tex(geometry['after']))
                    if geometry.get('continuation'):
                        parts = [(a,0,a['length']) for a in geometry['continuation']]
                        value += '\n\\BookCaptionParagraph{'+self.caption_tex(parts)+'}\n'
                    if atom['kind'] == 'table':
                        parser = TableParser()
                        parser.feed(atom['content'])
                        # Bounded tables should not strand a few rows at the
                        # page foot. Longer grids retain longtable pagination.
                        reserve = len(parser.rows) + 2 + 2*len(geometry['before']+geometry['after'])
                        if table_columns(parser) and reserve <= 24:
                            value = '\n\\par\\Needspace{'+str(reserve)+'\\baselineskip}\n'+value
                if math_cell:
                    value = '$\\displaystyle\n' + value + '$\\par\\end{minipage}%\n'
                mark = self.panel_end_marks.get(atom['id'])
                if (mark and lo == mark['lo'] and hi == mark['hi']
                        and any(o['anchor'] == 'source-' + mark['panel'] for o in stack)):
                    rgb = ','.join(f'{max(0,min(1,c)):g}' for c in mark['color'])
                    value = '\\BookPanelEndMark{' + rgb + '}{' + value + '}\n'
                    self.source_order_layout.append(mark)
                if any(a <= lo and hi <= z for a, z in answers):
                    value = '{\\BookExampleAnswerFont ' + value + '}'
                elif any(a <= lo and hi <= z for a, z in questions):
                    value = '{\\BookExampleQuestionFont ' + value + '}'
                if cap and cap[2]=='figure':
                    caption=self.caption_tex(list(self.index.fragments(cap[0],cap[1])))
                    value+='\n\\BookCaption{'+caption+'}\n'
                if any(a<=lo and hi<=b for a,b in self.bold_spans):
                    value='{\\bfseries '+value+'}'
                elif atom.get('block_type') == 'paragraph_title':
                    value='{\\bfseries '+value+'}'
                is_caption = (atom["role"] in CAPTION_ROLES and atom["group"] not in semantic_caption_groups
                              and atom['id'] not in self.image_row_members)
                if caption_open is not None and (not is_caption or caption_open != atom["group"]):
                    current.append("\\par\\endgroup\n"); caption_open = None
                if is_caption:
                    # A caption can alternate between text and inline formulas.
                    # Set its font per fragment, but break paragraphs only once.
                    group_lo, group_hi = caption_limits[atom["group"]]
                    value = "{\\BookCaptionFont " + value + "}"
                    if caption_open is not None and last_atom is not None and start == 0:
                        from book_pipeline.publishing.caption_layout import caption_separator
                        value = caption_separator(last_atom, atom) + value
                    if caption_open is None:
                        value = "\n\\par\\begingroup\\centering\\noindent " + value
                        caption_open = atom["group"]
                    elif start == 0 and atom["kind"] == "text" and ((atom["role"].endswith("_footnote") and (last_atom is None or last_atom["role"] != atom["role"])) or re.match(r"\s*(?:[图表]\s*\d|\([a-z]\))", atom["content"])):
                        value = "\\par\\noindent " + value
                    if hi == group_hi:
                        value += "\\par\\endgroup\n"
                        caption_open = None
                current.append(value)
                if hi in fraction_ends:
                    current.append('\\par\\end{minipage}\\par\\addvspace{6pt}\n')
                if math_cell and hi == math_cell[0]['hi']:
                    current.append('\\end{minipage}\\par\\addvspace{6pt}\n')
                if hi in crop_prompt_ends:
                    current.append('\n\\par\\endgroup\n')
                if display_run and hi == display_run['hi']:
                    current.append('\\end{gather*}\n')
                scan_mark = self.raster_marks.get(atom['id'])
                if scan_mark and not scan_mark['panel'] and end == atom['length']:
                    glyph = '\\includegraphics[height='+f"{scan_mark['height_pt']:g}"+'bp]{'+scan_mark['image']+'}'
                    if scan_mark.get('inline'):
                        current.append('\\unskip\\nobreak\\hspace{'+f"{scan_mark['gap_bp']:g}"+
                            'bp}\\raisebox{'+f"{-scan_mark['depth_bp']:g}"+'bp}{'+glyph+'}\n')
                    else:
                        current.append('\\BookTerminalMark{'+glyph+'}\n')
                    if scan_mark not in self.source_order_layout:
                        self.source_order_layout.append(scan_mark)
                if hi in list_ends:
                    current.append('\\BookListEnd{}\n')
                if hi in multiline_ends:
                    current.append('\n\\par\\endgroup\n')
                if hi in multiline_keep_ends:
                    current.append('\n\\end{samepage}\n')
                if hi in unit_ends:
                    current.append('\\end{BookDisplayUnit}\n')
                if atom["id"] in table_captions and keep_table:
                    current.append("\n\\end{minipage}\\par\n")
                if atom["group"] in image_groups and atom['group'] not in floating_groups and hi == image_groups[atom["group"]][1]:
                    current.append("\n\\end{minipage}\\par\n")
                if atom["group"] in table_before_groups and atom['group'] not in floating_groups and hi == table_before_groups[atom["group"]][1]:
                    current.append("\n\\end{minipage}\\par\n")
                rule = self.source_rules.get((atom['id'], 'after'))
                if rule and end == atom['length']:
                    rgb = ','.join(f'{c:g}' for c in rule['color'])
                    current.append('\n\\BookSourceRule{' + f"{rule['thickness_pt']:g}" + '}{' + rgb + '}\n')
                    self.source_order_layout.append(rule)
                last_atom = atom
            if hi in rule_region_ends:
                current.append('\n\\end{BookSourceInset}\n')
                last_atom = None
        if floating_open:close_float()
        if caption_open is not None:
            current.append("\\par\\endgroup\n")
        for o in sorted(ends[total], key=lambda o: -o["lo"]):
            if o["kind"] not in HEADINGS:
                if not stack or stack[-1]["anchor"] != o["anchor"]:
                    raise ValueError("Unbalanced final environments")
                stack.pop()
                if not pending_reopen:current.append("\n\\end{" + environment(o) + "}\n")
        if total in side_ends:
            current.append('\n\\end{' + ('BookWrappedPair' if side_ends[total].get('note_width') else 'BookSideBySide') + '}\n')
        insert_notes(total)
        if pending_notes or note_insertions:
            raise ValueError('A page footnote was not rendered')
        if stack:
            raise ValueError("Unclosed semantic environment")
        if current:
            chapters.append("".join(current))
        if len(self.anchors) != len(set(self.anchors)) or set(self.link_targets) - (set(self.anchors) | set(self.source_anchors.values())):
            raise ValueError("Duplicate anchor or missing hyperlink destination")
        if sum(v["end"] - v["start"] for v in self.ledger) != total or any(a["end"] != b["start"] for a, b in zip(self.ledger, self.ledger[1:])):
            raise ValueError("Unexplained source loss/duplication")
        return chapters


def write_project(index, decisions, registry, references, metadata, config, directory, *, heading_layout=None, sample_mode=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("elegantbook.cls", "book-style.sty", "PROVENANCE.md"):
        shutil.copy2(ROOT / "book_templates" / name, directory / name)
    (directory / "chapters").mkdir(exist_ok=True)
    # Remove obsolete generated chapters only, never any source or user-authored file.
    for path in (directory / "chapters").glob("generated-*.tex"):
        path.unlink()
    renderer = Renderer(index, decisions, registry, references, directory, config, heading_layout)
    chapters = renderer.render()
    inputs = []
    for i, content in enumerate(chapters, 1):
        name = f"chapters/generated-{i:03d}.tex"
        (directory / name).write_text(content, encoding="utf-8")
        inputs.append("\\input{" + name + "}")
    font = Path(config["caption_font_file"]).expanduser()
    if not font.is_absolute():
        font = ROOT / font
    if not font.is_file():
        raise ValueError("Caption font missing: " + str(font))
    (directory / "fonts").mkdir(exist_ok=True)
    shutil.copy2(font, directory / "fonts/caption.ttf")
    license_file = font.parent / "OFL.txt"
    if license_file.is_file():
        shutil.copy2(license_file, directory / "fonts/OFL.txt")
    title, author = metadata.get("title") or "技术书籍", metadata.get("author") or ""
    typography = renderer.typography
    body = typography['body_size']
    # Keep the established CJK leading and note geometry. Latin prose needs
    # only the explicit 1.56 baseline, without the class's Chinese multiplier.
    from book_pipeline.ingestion.translation import language
    body_language = language(''.join(a['content'] for a in index.atoms
        if a['kind']=='text' and a.get('role') not in {'header','footer','page_number'}))
    body_spread = r'\linespread{1}' if body_language == 'en' else ''
    from book_pipeline.publishing.native_code import font_commands
    preamble = "\n".join([
        # Matched note labels are paired explicitly with physical page records.
        # There is no automatic footnotemark to pair with footnotetext.
        r"\PassOptionsToPackage{hyperfootnotes=false}{hyperref}",
        r"\documentclass[lang=cn,chinesefont=nofont,10pt,twoside,openany,math=newtx]{elegantbook}",
        "\\setCJKmainfont{" + escape(config["main_font"]) + "}",
        "\\setCJKsansfont[Scale=1,AutoFakeBold=2]{" + escape(config["heading_font"]) + "}",
        "\\setCJKfamilyfont{zhhei}[Scale=1,AutoFakeBold=2]{" + escape(config["heading_font"]) + "}",
        r"\newfontfamily\BookSubheadingLatinFont[Scale=1,Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]{texgyreheros}",
        r"\providecommand{\heiti}{\CJKfamily{zhhei}}",
        r"\newCJKfontfamily\BookCaptionFont[Path=fonts/,AutoFakeBold=2]{caption.ttf}",
        r"\newfontfamily\BookUnicodeFont[Path=fonts/,AutoFakeBold=2]{caption.ttf}",
        r"\newCJKfontfamily\BookExampleAnswerCJKFont[Scale=1]{FandolFang-Regular.otf}",
        r"\newCJKfontfamily\BookSourceKaiFont[Scale=1,AutoFakeBold=2]{FandolKai-Regular.otf}",
        font_commands(renderer.native_code),
        r"\usepackage{book-style}",
        color_definitions(validate_colors(config.get('block_colors', {}))),
        r'\colorlet{BookExampleInk}{BookColor' + validate_colors(config['block_colors'])['example'][1:] + '}' if config.get('block_colors', {}).get('example') else '',
        r'\renewcommand{\BookExampleQuestionFont}{\BookCaptionFont}',
        r'\renewcommand{\BookExampleAnswerFont}{\normalfont\rmfamily\BookExampleAnswerCJKFont}',
        r'\AtBeginDocument{' + body_spread + r'\fontsize{' + f'{body:g}' + 'pt}{' + f'{body * 1.56:g}' + r'pt}\selectfont}',
        r'\renewcommand{\BookFootnoteFont}{\normalfont\BookCaptionFont\fontsize{' + f'{body*.8:g}' + 'pt}{' + f'{body*1.12:g}' + r'pt}\selectfont}',
        renderer.heading_layout.tex_setup(),
        r"\pdfstringdefDisableCommands{\def\hypertarget#1#2{#2}\def\BookHeadingGap{ }}",
        "\\renewcommand{\\BookFooter}{" + escape(title + ("　" + author if author else "")) + "}",
        "\\hypersetup{pdftitle={" + escape(title) + "},pdfauthor={" + escape(author) + "}}",
        r"\begin{document}",
        '\n'.join(r'\BookDeclareFootnote{' + note['id'] + '}{' + note['marker']['style'] + '}'
                  for note in sorted((n for n in renderer.page_footnotes if n['status']=='matched'),
                                     key=lambda n: n['reference']['edited_lo'])),
        r"\pagestyle{empty}" if sample_mode else (r"\mainmatter" if renderer.body_excerpt else r"\frontmatter"),
        r"\tableofcontents" if not sample_mode and config.get('publication_scope','book') != 'excerpt'
            and renderer.toc_position is None and any(o['kind'] in HEADINGS for o in registry) else "",
        "\n".join(inputs), r"\end{document}", ""])
    (directory / "main.tex").write_text(preamble, encoding="utf-8")
    (directory / "README.md").write_text("# 编译\n\n安装 XeLaTeX、latexmk、Noto Serif/Sans CJK SC 后，在本目录运行：\n\n```bash\nlatexmk -xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex\n```\n\n模板固定于 ElegantBook v4.7；中文楷体字体与授权随工程提供。结构与疑点报告位于上一级 reports/。\n", encoding="utf-8")
    audit = {"source_units": index.prefix[-1], "accounted_units": sum(v["end"] - v["start"] for v in renderer.ledger),
             "coverage_complete": True, "ledger": renderer.ledger, "anchors": renderer.anchors,
             "source_anchors": renderer.source_anchors,
             "toc_source_position": renderer.toc_position,
             "publication_scope": config.get('publication_scope','book'),
             "heading_layout": renderer.heading_layout.describe(),
             "heading_levels": [{'anchor':o['anchor'],'level':renderer.heading_layout.level(o),
                                  'lo':o['lo'],'hi':o['hi'],'standalone':renderer.heading_layout.standalone(o)}
                                 for o in registry if o['kind'] in HEADINGS],
             "link_targets": renderer.link_targets, "assets": renderer.assets, "issues": renderer.issues,
             "tex_compatibility":renderer.tex_compatibility,"source_order_layout":renderer.source_order_layout}
    audit['footnotes'] = renderer.rendered_footnotes
    save(directory.parent / "reports/coverage.json", audit)
    return audit
