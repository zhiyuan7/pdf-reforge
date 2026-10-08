"""Deterministic rendering, original-number anchors and exact source accounting."""
from __future__ import annotations

from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil

from book_pipeline.config import ROOT, digest, save
from book_pipeline.structure.semantics import HEADINGS, LABELS
from book_pipeline.structure.numbered import CAPTION_ROLES
from book_pipeline.ingestion.source import corrected_content
from book_pipeline.ingestion.math_syntax import math_compatibility, formula_tags
from book_pipeline.structure.paragraphs import supported_joins
from book_pipeline.structure.heading_layout import HeadingLayout

ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def literal_text(text):
    return "".join(unicode_symbol(c) or ESCAPES.get(c, c) if ord(c) >= 32 or c == "\n" else "" for c in text)


URL_TEXT = re.compile(r"(?:https?://|www\.)[^\s<>(){}，。；：、]+")


def escape(text):
    # URLs remain literal source text, with invisible break opportunities so a
    # bibliography address cannot extend beyond the physical page.
    chunks, cursor = [], 0
    for match in URL_TEXT.finditer(text):
        chunks.append(literal_text(text[cursor:match.start()]))
        chunks.append(r"\allowbreak{}".join(literal_text(c) for c in match[0]))
        cursor = match.end()
    chunks.append(literal_text(text[cursor:]))
    return "".join(chunks)


def unicode_symbol(char, math=False):
    # Circled footnote markers occur inside superscripts as well as ordinary text.
    # NewTX's OT1 math text font cannot represent these Unicode characters.
    code = ord(char)
    if char in '₀₁₂₃₄₅₆₇₈₉':
        subscript='_{'+str('₀₁₂₃₄₅₆₇₈₉'.index(char))+'}'
        return subscript if math else r'\ensuremath{{}'+subscript+'}'
    cjk_math = math and (0x2E80 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF
                        or 0xFF01 <= code <= 0xFF60 or 0x20000 <= code <= 0x3134F)
    if (0x2460 <= code <= 0x24FF or 0x400 <= code <= 0x52F or 0x2160 <= code <= 0x2188
            or char in "∞☉↗↘∼" or cjk_math):
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
    return "".join(unicode_symbol(c, math=True) or c for c in text)


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
    text=''.join(values)
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
        self.complex = False

    def handle_starttag(self, tag, attrs):
        if tag not in ("table", "tbody", "thead", "tr", "td", "th", "eq", "br"):
            self.complex = True
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            if any(k in ("rowspan", "colspan") and v != "1" for k, v in attrs):
                self.complex = True
            self.in_cell, self.cell = True, ""
        if tag == "eq":
            self.in_math = True
            self.cell += "$"
        if tag == "br":
            self.cell += " "

    def handle_endtag(self, tag):
        if tag == "eq":
            self.in_math = False
            self.cell += "$"
        if tag in ("td", "th"):
            if self.row is None:
                self.complex = True
            else:
                self.row.append(self.cell)
            self.in_cell = False
        if tag == "tr":
            if self.row:
                self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.in_cell:
            self.cell += math_tex(data) if self.in_math else escape(data)


def table_tex(html, in_box=False):
    parser = TableParser()
    parser.feed(html)
    rows = parser.rows
    if parser.complex or not rows or len({len(r) for r in rows}) != 1:
        return None
    columns = len(rows[0])
    if not 1 <= columns <= 10:
        return None
    # Layer names such as Conv2.1,Conv2.2 must be able to wrap in a
    # longtable's fixed-width columns, without changing the visible text.
    content = "\n".join(" & ".join(cell.replace(",", r",\allowbreak{}") for cell in row) + r" \\" for row in rows)
    if len(rows) > 12 and not in_box:
        column = "p{\\dimexpr\\linewidth/" + str(columns) + "-2\\tabcolsep\\relax}"
        # Narrow plain-text tables need natural columns: equal fixed columns
        # needlessly split short Chinese headers. Long/math cells keep wrapping.
        widths=[max(sum(10.5 if ord(c)>127 else 5.3 for c in row[n]) for row in rows) for n in range(columns)]
        short=all('\\' not in cell and '$' not in cell for row in rows for cell in row) and sum(widths)+12*columns<375
        specification='c'*columns if short else column*columns
        return "\n{\\setlength{\\LTleft}{\\fill}\\setlength{\\LTright}{\\fill}\n\\begin{longtable}{" + specification + "}\n\\toprule\n" + content + "\n\\bottomrule\n\\end{longtable}}\n"
    if len(rows) > 12:
        return None
    return "\n\\par\\noindent\\makebox[\\linewidth][c]{%\n\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{" + "c" * columns + "}\n\\toprule\n" + content + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}}\\par\n"


class Renderer:
    def __init__(self, index, decisions, registry, references, directory, config=None, heading_layout=None):
        self.index, self.decisions, self.registry = index, decisions, registry
        self.directory = Path(directory)
        self.config = config or {}
        self.heading_layout = HeadingLayout([o for o in registry if o['kind'] in HEADINGS], heading_layout)
        self.refs = defaultdict(list)
        self.corrections = defaultdict(list)
        self.issues = []
        self.tex_compatibility=[]
        self.source_order_layout=[]
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
        self.source_anchors = {}
        self.bold_spans=[index.span(s) for s in decisions.get('bold_spans',[])]
        self.toc_position = min((e["lo"] for e in decisions["exclusions"] if "original_toc" in e.get("reasons", [e["reason"]])), default=None)

    def asset(self, atom):
        source = Path(atom["asset"])
        name = digest(source)[:20] + source.suffix.lower()
        (self.directory / "images").mkdir(exist_ok=True)
        destination = self.directory / "images" / name
        if not destination.exists():
            shutil.copy2(source, destination)
        self.assets[str(source)] = "images/" + name
        return "images/" + name

    def text_fragment(self, atom, lo, hi):
        changes = []
        for c in self.corrections[atom["id"]]:
            if lo <= c["start_offset"] and c["end_offset"] <= hi:
                changes.append((c["start_offset"], c["end_offset"], escape(c["replacement"])))
        for r in self.refs[atom["id"]]:
            if lo <= r["start_offset"] and r["end_offset"] <= hi:
                changes.append((r["start_offset"], r["end_offset"], "\\BookLink{" + r["target"] + "}{" + escape(r["text"]) + "}"))
                self.link_targets.append(r["target"])
        chunks, cursor = [], lo
        for start, end, replacement in sorted(changes):
            chunks.append(escape(atom["content"][cursor:start]))
            chunks.append(replacement)
            cursor = end
        chunks.append(escape(atom["content"][cursor:hi]))
        return "".join(chunks)

    def fragment(self, atom, lo, hi, in_box):
        value = self.content_fragment(atom, lo, hi, in_box)
        # Pagination whitespace must not accidentally leave a blank TeX line
        # inside a confirmed paragraph. Source text/coverage remain unchanged.
        if atom["id"] in self.join_left and hi == atom["length"]:
            value = value.rstrip()
        if atom["id"] in self.join_right and lo == 0:
            value = value.lstrip()
        return self.source_marker(atom) + value

    def source_marker(self, atom):
        if atom["id"] not in self.source_anchors:
            name = "source-" + atom["id"]
            self.source_anchors[atom["id"]] = name
            return "\\hypertarget{" + name + "}{}"
        return ""

    def content_fragment(self, atom, lo, hi, in_box):
        if atom["kind"] == "text":
            return self.text_fragment(atom, lo, hi)
        content = atom["content"]
        for c in sorted(self.corrections[atom["id"]], key=lambda c: -c["start_offset"]):
            content = content[:c["start_offset"]] + c["replacement"] + content[c["end_offset"]:]
        if atom['kind'] in {'math','display_math'}:
            compatible,evidence=math_compatibility(content)
            if evidence:
                self.tex_compatibility.append({'atom':atom['id'],'original':content,'rendered':compatible,'evidence':evidence})
            content=compatible
        if atom["kind"] == "math":
            return "% source-atom: " + atom["id"] + "\n\\allowbreak $" + math_tex(content) + "$%\n% end-source-atom: " + atom["id"] + "\n"
        if atom["kind"] == "display_math":
            tex = math_tex(content)
            tags = formula_tags(tex)
            # Keep the original equation number outside the measured math
            # box. Only oversized expressions are reduced to fit the column.
            if len(tags) <= 1:
                tag = tags[0]['tex'] if tags else ""
                if tags:
                    tex = tex[:tags[0]['start']] + tex[tags[0]['end']:]
                tex = "\\BookMath[" + ("48pt" if tag else "0pt") + "]{" + tex + "}" + tag
            return "\n% source-atom: " + atom["id"] + "\n\\begin{equation*}\n" + tex + "\n\\end{equation*}\n% end-source-atom: " + atom["id"] + "\n"
        if atom["kind"] == "image":
            return self.image_tex(atom)
        if atom["kind"] == "table":
            converted = table_tex(content, in_box)
            if converted:
                return converted
            if not atom.get("asset"):
                raise ValueError("Complex table has no source image: " + atom["id"])
            self.issues.append({"reason": "complex_table_preserved_as_image", "atom": atom["id"]})
            return self.image_tex(atom)
        raise ValueError("Unsupported atom kind: " + atom["kind"])

    def image_tex(self, atom):
        bbox = atom.get("bbox")
        # MinerU coordinates are normalized against the original page. Estimate
        # the original text column from body blocks, then retain image scale.
        body = [a["bbox"] for a in self.index.atoms if a["page"] == atom["page"]
                and a["kind"] == "text" and a["block_type"] == "text" and a.get("bbox")]
        column_width = max((b[2] for b in body), default=.88) - min((b[0] for b in body), default=.12)
        width = min(1., max(.15, (bbox[2] - bbox[0]) / max(.25, column_width))) if bbox else 1.
        return "\n\\BookImage[" + f"{width:.3f}" + "\\linewidth]{" + self.asset(atom) + "}\n"

    def anchor(self, name):
        self.anchors.append(name)
        return "\\BookAnchor{" + name + "}"

    def render(self):
        total = self.index.prefix[-1]
        positions = {0, total, *self.index.prefix}
        for lo,hi in self.bold_spans:positions.update((lo,hi))
        starts, ends, anchored, skipped = defaultdict(list), defaultdict(list), defaultdict(list), []
        associated_captions={}
        relocated_labels=defaultdict(list)
        for o in self.registry:
            anchored[o["lo"]].append(o)
            positions.add(o["lo"])
            if self.config.get('float_assets') and o.get('caption_span') and o['kind'] in {'figure','table'}:
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
        ordered = sorted(positions)
        stack, chunks, chapters, current = [], [], [], []
        last_atom = None
        caption_open = None
        footnote_previous = {}
        headings = sorted((o for o in self.registry if o["kind"] in HEADINGS), key=lambda o: o["lo"])
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
        environments = {"definition": "BookDefinition", "notation": "BookDefinition", "theorem": "BookTheorem", "lemma": "BookTheorem", "corollary": "BookTheorem", "proposition": "BookTheorem", "algorithm": "BookAlgorithm", "example": "BookExample", "remark": "BookRemark", "numbered": "BookRemark", "proof": "BookProof", "solution": "BookSolution", "exercise": "BookExercise", "footnote": "BookFootnote", "bibliography_entry": "BookBibliography"}
        join_pairs = self.join_pairs
        group_limits = {}
        group_atoms = defaultdict(list)
        for i, atom in enumerate(self.index.atoms):
            limits = group_limits.setdefault(atom["group"], [self.index.prefix[i], self.index.prefix[i + 1]])
            limits[1] = self.index.prefix[i + 1]
            group_atoms[atom["group"]].append(i)
        table_captions, moved_anchors, image_groups, table_before_groups = {}, set(), {}, {}
        floating_groups={};floating_open=None;pending_reopen=False
        caption_limits = {}
        for group, indices in group_atoms.items():
            image_bodies = [i for i in indices if self.index.atoms[i]["kind"] == "image"]
            image_captions = [i for i in indices if self.index.atoms[i]["role"] in ("image_caption", "chart_caption", "image_footnote", "chart_footnote")]
            group_lo, group_hi = group_limits[group]
            all_captions = [i for i in indices if self.index.atoms[i]["role"] in CAPTION_ROLES]
            if all_captions:
                caption_limits[group] = (self.index.prefix[min(all_captions)], self.index.prefix[max(all_captions) + 1])
            safe_group = not any(group_lo < p < group_hi for p in (*starts, *ends)) and not any(
                a < group_hi and b > group_lo for a, b, _ in skipped)
            if self.config.get('float_assets') and safe_group:
                assets=[i for i in indices if self.index.atoms[i]['kind'] in {'image','table'}]
                only_assets_and_captions=all(self.index.atoms[i]['kind'] in {'image','table'} or
                                            self.index.atoms[i]['role'] in CAPTION_ROLES for i in indices)
                long_table=any(self.index.atoms[i]['kind']=='table' and
                               '\\begin{longtable}' in (table_tex(self.index.atoms[i]['content'],False) or '') for i in assets)
                if assets and only_assets_and_captions and not long_table:
                    floating_groups[group]=(group_lo,group_hi,'table' if all(self.index.atoms[i]['kind']=='table' for i in assets) else 'figure')
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
        # MinerU occasionally treats an exercise/question preceding a table as
        # its caption. A confirmed semantic object owns that paragraph's style.
        semantic_caption_groups = {
            group for group, (a, b) in caption_limits.items()
            if any(o["kind"] not in ("equation", "figure", "table") and a <= o["lo"] < b
                   for o in self.registry)}
        body_started = False
        def close_float():
            nonlocal caption_open,floating_open,pending_reopen,last_atom
            if caption_open is not None:current.append('\\par\\endgroup\n');caption_open=None
            current.append('\n\\end{'+floating_open[2]+'}\n')
            floating_open=None;pending_reopen=bool(stack);last_atom=None
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
                if not pending_reopen:current.append("\n\\end{" + environments[o["kind"]] + "}\n")
                if o["kind"] == "footnote":
                    last_atom = footnote_previous.pop(o["anchor"], None)
            if pending_reopen:
                for parent in stack:
                    continuation='（例题续）' if parent['kind']=='example' else '（续）'
                    current.append('\n\\begin{'+environments[parent['kind']]+'}{'+continuation+'}\n')
                pending_reopen=False
            for o in sorted(starts[lo], key=lambda o: -o["hi"]):
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
                    shown = "".join(self.content_fragment(a, l, h, False) for a, l, h in self.index.fragments(o["lo"], o["hi"]))
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
                            prefix = ''.join(self.content_fragment(a,l,h,False) for a,l,h in self.index.fragments(o['lo'],prefix_end))
                            body = ''.join(self.content_fragment(a,l,h,False) for a,l,h in self.index.fragments(body_start,o['hi']))
                            shown = prefix + r'\BookHeadingGap{}' + body
                    # Claim the anchor here; heading macro inserts it at the correct location.
                    self.anchors.append(o["anchor"])
                    level = self.heading_layout.level(o)
                    markers = "".join(self.source_marker(a) for a, _, _ in self.index.fragments(o["lo"], o["hi"]))
                    if lo in heading_space and not self.heading_layout.standalone(o):
                        current.append("\n\\Needspace{" + str(heading_space[lo]) + "\\baselineskip}\n")
                    toc_title = re.sub(r'\{\\BookUnicodeFont\\char"24(?:6[0-9A-F]|7[0-3])\}', "", shown)
                    toc_title = r'\texorpdfstring{'+toc_title+'}{'+bookmark_title(self.index,o['lo'],o['hi'])+'}'
                    current.append("\n\\BookHeading{" + str(level) + "}{" + shown + "}{" + o["anchor"] + "}[" + markers + "][{" + toc_title + "}]\n")
                else:
                    if o["kind"] == "footnote":
                        footnote_previous[o["anchor"]] = last_atom
                        last_atom = None
                    if o.get("body_start") and self.index.position(o["body_start"]) > o["lo"]:
                        title = "".join(self.fragment(a, l, h, False) for a, l, h in self.index.fragments(o["lo"], self.index.position(o["body_start"])))
                    else:
                        # Without a separately confirmed title span the original
                        # label remains in the body. Adding a generated label
                        # would repeat "例2.1", "解", or "证明" in the PDF.
                        title = ""
                    current.append("\n" + self.anchor(o["anchor"]) + "\\begin{" + environments[o["kind"]] + "}{" + title + "}\n")
                    stack.append(o)
            for o in relocated_labels[lo]:
                shown=''.join(self.source_marker(a)+self.content_fragment(a,l,h,False)
                              for a,l,h in self.index.fragments(o['lo'],o['hi']))
                current.append('\n'+self.anchor(o['anchor'])+shown+'\n')
                moved_anchors.add(o['anchor']);last_atom=None
            group=self.index.atoms[self.index.ids[self.index.address(lo)['atom']]]['group']
            if group in floating_groups and lo==floating_groups[group][0]:
                if caption_open is not None:current.append('\\par\\endgroup\n');caption_open=None
                for parent in reversed(stack):current.append('\n\\end{'+environments[parent['kind']]+'}\n')
                floating_open=floating_groups[group]
                current.append('\n\\par\\begin{'+floating_open[2]+'}[htp]\n\\centering\n')
                self.source_order_layout.append({'group':group,'method':'deferred_asset_float','placement':'htp',
                                                'lo':lo,'hi':floating_open[1],'surrounding_modules':[p['anchor'] for p in stack]})
                last_atom=None
            for o in anchored[lo]:
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
                if atom["kind"] == "text" and start == 0 and end == atom["length"] and any(
                        c["start_offset"] == 0 and c["end_offset"] == atom["length"] and c["replacement"] == "" for c in self.corrections[atom["id"]]):
                    continue
                if last_atom is not None and last_atom["group"] != atom["group"] and (last_atom["id"], atom["id"]) not in join_pairs:
                    current.append("\n\n")
                elif last_atom is not None and (last_atom["id"], atom["id"]) in join_pairs:
                    current.append("\n")
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
                    caption = "".join(self.fragment(a, l, h, bool(stack)) for a, l, h in self.index.fragments(caption_lo, caption_hi))
                    current.append("\n\\BookCaption{" + caption + "}\n")
                cap=associated_captions.get(lo)
                if cap and cap[2]=='table':
                    caption=''.join(self.fragment(a,l,h,False) for a,l,h in self.index.fragments(cap[0],cap[1]))
                    current.append('\n\\BookCaption{'+caption+'}\n')
                value = self.fragment(atom, start, end, bool(stack))
                if cap and cap[2]=='figure':
                    caption=''.join(self.fragment(a,l,h,False) for a,l,h in self.index.fragments(cap[0],cap[1]))
                    value+='\n\\BookCaption{'+caption+'}\n'
                if any(a<=lo and hi<=b for a,b in self.bold_spans):
                    value='{\\bfseries '+value+'}'
                is_caption = atom["role"] in CAPTION_ROLES and atom["group"] not in semantic_caption_groups
                if caption_open is not None and (not is_caption or caption_open != atom["group"]):
                    current.append("\\par\\endgroup\n"); caption_open = None
                if is_caption:
                    # A caption can alternate between text and inline formulas.
                    # Set its font per fragment, but break paragraphs only once.
                    group_lo, group_hi = caption_limits[atom["group"]]
                    value = "{\\BookCaptionFont " + value + "}"
                    if caption_open is None:
                        value = "\n\\par\\begingroup\\centering\\noindent " + value
                        caption_open = atom["group"]
                    elif start == 0 and atom["kind"] == "text" and ((atom["role"].endswith("_footnote") and (last_atom is None or last_atom["role"] != atom["role"])) or re.match(r"\s*(?:[图表]\s*\d|\([a-z]\))", atom["content"])):
                        value = "\\par\\noindent " + value
                    if hi == group_hi:
                        value += "\\par\\endgroup\n"
                        caption_open = None
                current.append(value)
                if atom["id"] in table_captions and keep_table:
                    current.append("\n\\end{minipage}\\par\n")
                if atom["group"] in image_groups and atom['group'] not in floating_groups and hi == image_groups[atom["group"]][1]:
                    current.append("\n\\end{minipage}\\par\n")
                if atom["group"] in table_before_groups and atom['group'] not in floating_groups and hi == table_before_groups[atom["group"]][1]:
                    current.append("\n\\end{minipage}\\par\n")
                last_atom = atom
        if floating_open:close_float()
        if caption_open is not None:
            current.append("\\par\\endgroup\n")
        for o in sorted(ends[total], key=lambda o: -o["lo"]):
            if o["kind"] not in HEADINGS:
                if not stack or stack[-1]["anchor"] != o["anchor"]:
                    raise ValueError("Unbalanced final environments")
                stack.pop()
                if not pending_reopen:current.append("\n\\end{" + environments[o["kind"]] + "}\n")
        if stack:
            raise ValueError("Unclosed semantic environment")
        if current:
            chapters.append("".join(current))
        if len(self.anchors) != len(set(self.anchors)) or set(self.link_targets) - (set(self.anchors) | set(self.source_anchors.values())):
            raise ValueError("Duplicate anchor or missing hyperlink destination")
        if sum(v["end"] - v["start"] for v in self.ledger) != total or any(a["end"] != b["start"] for a, b in zip(self.ledger, self.ledger[1:])):
            raise ValueError("Unexplained source loss/duplication")
        return chapters


def write_project(index, decisions, registry, references, metadata, config, directory, *, heading_layout=None):
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
    preamble = "\n".join([
        # BookFootnote uses original labels and explicit destinations. There
        # is no automatic footnotemark for hyperref to pair with footnotetext.
        r"\PassOptionsToPackage{hyperfootnotes=false}{hyperref}",
        r"\documentclass[lang=cn,chinesefont=nofont,10pt,twoside,openany,math=newtx]{elegantbook}",
        "\\setCJKmainfont{" + escape(config["main_font"]) + "}",
        "\\setCJKsansfont{" + escape(config["heading_font"]) + "}",
        "\\setCJKfamilyfont{zhhei}{" + escape(config["heading_font"]) + "}",
        r"\providecommand{\heiti}{\CJKfamily{zhhei}}",
        r"\newCJKfontfamily\BookCaptionFont[Path=fonts/,AutoFakeBold=2]{caption.ttf}",
        r"\newfontfamily\BookUnicodeFont[Path=fonts/,AutoFakeBold=2]{caption.ttf}",
        r"\usepackage{book-style}",
        renderer.heading_layout.tex_setup(),
        r"\pdfstringdefDisableCommands{\def\hypertarget#1#2{#2}\def\BookHeadingGap{ }}",
        "\\renewcommand{\\BookFooter}{" + escape(title + ("　" + author if author else "")) + "}",
        "\\hypersetup{pdftitle={" + escape(title) + "},pdfauthor={" + escape(author) + "}}",
        r"\begin{document}", r"\frontmatter", r"\tableofcontents" if renderer.toc_position is None else "", "\n".join(inputs), r"\end{document}", ""])
    (directory / "main.tex").write_text(preamble, encoding="utf-8")
    (directory / "README.md").write_text("# 编译\n\n安装 XeLaTeX、latexmk、Noto Serif/Sans CJK SC 后，在本目录运行：\n\n```bash\nlatexmk -xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex\n```\n\n模板固定于 ElegantBook v4.7；中文楷体字体与授权随工程提供。结构与疑点报告位于上一级 reports/。\n", encoding="utf-8")
    audit = {"source_units": index.prefix[-1], "accounted_units": sum(v["end"] - v["start"] for v in renderer.ledger),
             "coverage_complete": True, "ledger": renderer.ledger, "anchors": renderer.anchors,
             "source_anchors": renderer.source_anchors,
             "toc_source_position": renderer.toc_position,
             "heading_layout": renderer.heading_layout.describe(),
             "heading_levels": [{'anchor':o['anchor'],'level':renderer.heading_layout.level(o),
                                  'lo':o['lo'],'hi':o['hi'],'standalone':renderer.heading_layout.standalone(o)}
                                 for o in registry if o['kind'] in HEADINGS],
             "link_targets": renderer.link_targets, "assets": renderer.assets, "issues": renderer.issues,
             "tex_compatibility":renderer.tex_compatibility,"source_order_layout":renderer.source_order_layout}
    save(directory.parent / "reports/coverage.json", audit)
    return audit
