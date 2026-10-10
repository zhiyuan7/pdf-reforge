"""Literal, page-local footnote labels and reversible source relationships."""
from collections import Counter, defaultdict
from copy import deepcopy
import re
import unicodedata

METHOD = 'page_footnotes_same_page_unique_v2'
FORMULA_DEDUP_METHOD = 'page_footnote_overlapping_formula_copy_v1'
MARGINALS = {'header', 'footer', 'page_number'}
CIRCLED = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳➀➁➂➃➄➅➆➇➈➉❶❷❸❹❺❻❼❽❾❿'
SUPERSCRIPTS = '⁰¹²³⁴⁵⁶⁷⁸⁹'
SYMBOLS = '*＊†‡※●'
TOKEN = rf'(?:\\(?:textcircled|circled)\s*\{{\s*[0-9０-９]+\s*\}}|[{CIRCLED}]|[{SUPERSCRIPTS}]+|[0-9０-９]+|[{re.escape(SYMBOLS)}])'
LABEL = re.compile(rf'\s*(?:\^\s*\{{\s*(?P<sup>{TOKEN})\s*\}}|\^\s*(?P<bare>{TOKEN})|(?P<plain>{TOKEN}))')
TOKENS = re.compile(TOKEN)


def marker_key(token):
    circled = re.fullmatch(r'\\(?:textcircled|circled)\s*\{\s*([0-9０-９]+)\s*\}', token)
    if circled:
        return 'circled', str(int(unicodedata.normalize('NFKC', circled[1])))
    if token in CIRCLED:
        number = unicodedata.numeric(token)
        return 'circled', str(int(number))
    if token[0] in SYMBOLS:
        return 'symbol', token.replace('＊', '*')
    return 'number', str(int(unicodedata.normalize('NFKC', token)))


def extract_marker(text):
    match = LABEL.match(text)
    if not match:
        return None
    group = next(name for name in ('sup', 'bare', 'plain') if match[name] is not None)
    token = match[group]
    # A formula beginning "10^{\\circ}" is not a numbered annotation.
    if group == 'plain' and token[0] in '0123456789０１２３４５６７８９' and match.end() < len(text):
        following = text[match.end()]
        if following not in ' \t\r\n.．、:：)）]］' and not '\u3400' <= following <= '\u9fff':
            return None
    style, number = marker_key(token)
    return dict(text=token, style=style, number=number,
                start=match.start(group), end=match.end(group))


def marker_occurrences(text, marker, kind):
    for match in TOKENS.finditer(text):
        if marker_key(match[0]) != (marker['style'], marker['number']):
            continue
        if marker['style'] == 'number':
            before, after = text[:match.start()], text[match.end():]
            if (before and (before[-1].isalnum() and ord(before[-1]) < 128 or before[-1] in '.．_')) or (
                    after and (after[0].isalnum() and ord(after[0]) < 128 or after[0] in '.．_')):
                continue
            # Numbers inside ordinary mathematics are not textual note marks.
            # Accept only an isolated superscript marker, never x^1 or 1+2.
            if kind != 'text' and not re.fullmatch(r'\s*(?:\{\s*\}\s*)?\^\s*(?:\{\s*'+re.escape(match[0])+r'\s*\}|'+re.escape(match[0])+r')\s*', text):
                continue
        yield match.start(), match.end()


def formula_tokens(text):
    """Compare OCR spellings only; the retained formula is never rewritten."""
    text = text.replace('π', r'\pi').replace('′', "'").replace('’', "'")
    text = re.sub(r'\\(?:left|right)(?![A-Za-z])', '', text)
    text = re.sub(r'\^\s*\{\s*((?:\\prime\s*)+)\}',
                  lambda m: "'" * m[1].count(r'\prime'), text)
    text = re.sub(r'\^\s*\\prime(?![A-Za-z])', "'", text)
    text = re.sub(r'\\(?:pmb|mathbf|boldsymbol)\s*\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:varepsilon|epsilon)(?![A-Za-z])', lambda m: r'\epsilon', text)
    text = re.sub(r'\\(?:dots|cdots|ldots)(?![A-Za-z])', lambda m: r'\dots', text)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r'([_^])\s*\{\s*([^{}\s]|\\[A-Za-z]+)\s*\}', r'\1\2', text)
    # Tokenize control words before discarding whitespace: \alpha b and
    # \alphab must remain distinct. Sentence punctuation can attach to a copy.
    return tuple(re.findall(r'\\[A-Za-z]+|\\.|[^\s]', text.rstrip().rstrip(',.;，。；')))


def formula_copy_overlap(copy, retained):
    """Require the copied glyph region to lie inside the retained OCR region."""
    if not copy or not retained or len(copy) != 4 or len(retained) != 4:
        return False
    area = (copy[2]-copy[0]) * (copy[3]-copy[1])
    if copy[2] <= copy[0] or copy[3] <= copy[1]:
        return False
    width = max(0, min(copy[2], retained[2])-max(copy[0], retained[0]))
    height = max(0, min(copy[3], retained[3])-max(copy[1], retained[1]))
    return width * height >= .8 * area


def duplicate_footnote_formulas(index, plan):
    """Drop extra OCR blocks covered by formulas in a complete, labeled note."""
    notes = [b for b in plan['blocks'] if b['source_type'] == 'page_footnote'
             and not b['heading'] and b['page'] not in plan['excluded_pages']]
    retained_by_page = defaultdict(list)
    for block in notes:
        if not extract_marker(block['text']):
            continue
        formulas = []
        for n, part in enumerate(block['parts']):
            if part['kind'] not in {'math', 'display_math'}:
                continue
            atom = index.atoms[index.ids[part['atom']]]
            following = block['parts'][n+1] if n+1 < len(block['parts']) else None
            punctuation = re.match(r'\s*([,.;()，。；（）]*)', following['text'])[1] if following and following['kind'] == 'text' else ''
            formulas.append((part, atom.get('bbox'), formula_tokens(part['text']+punctuation)))
        if formulas:
            retained_by_page[block['page']].append((block, formulas))
    deletions = []
    for block in notes:
        # Complete footnotes, including repeated mathematical arguments inside
        # their text, remain intact. Only separate unlabelled OCR copies qualify.
        if extract_marker(block['text']) or any(p['kind'] not in {'text', 'math', 'display_math'} for p in block['parts']):
            continue
        parts = [p for p in block['parts'] if p['text'].strip()]
        if not parts:
            continue
        for retained, formulas in retained_by_page[block['page']]:
            matches = []
            for part in parts:
                atom = index.atoms[index.ids[part['atom']]]
                tokens = formula_tokens(part['text'])
                match = next((p for p, bbox, value in formulas if tokens
                    and formula_copy_overlap(atom.get('bbox'), bbox)
                    and any(tokens == value[n:n+len(tokens)] for n in range(len(value)-len(tokens)+1))), None)
                if match is None:
                    break
                matches.append(dict(source_atom=part['atom'], retained_atom=match['atom'],
                                    text=part['text'], retained_formula=match['text']))
            if len(matches) == len(parts):
                deletions.append(dict(b=block['b'], lo=block['lo'], hi=block['hi'],
                    start=block['start'], end=block['end'], page=block['page'],
                    reason='footnote_duplicate_formula', method=FORMULA_DEDUP_METHOD,
                    why='Extra OCR formula block overlaps the same formula inside a labeled page footnote',
                    removed=block['text'], retained_block=retained['b'], matches=matches, outside_scope=False))
                break
    return deletions


def classify_footnotes(index, plan):
    """Classify before any model input; multiple note/body candidates never pair."""
    plan = deepcopy(plan)
    duplicates = duplicate_footnote_formulas(index, plan)
    duplicate_blocks = {d['b'] for d in duplicates}
    by_page = defaultdict(list)
    owners = {bid: s['id'] for s in plan['sections'] for bid in s['blocks']}
    notes = []
    for block in plan['blocks']:
        is_note = block['source_type'] == 'page_footnote' and not block['heading']
        block['content_role'] = 'page_footnote' if is_note else 'body'
        if block['page'] in plan['excluded_pages']:
            continue
        if is_note:
            if block['b'] in duplicate_blocks:
                continue
            marker = extract_marker(block['text'])
            notes.append(dict(id='footnote-'+block['b'], block=block['b'], page=block['page'],
                              section=owners.get(block['b']), lo=block['lo'], hi=block['hi'],
                              start=block['start'], end=block['end'], marker=marker,
                              text=block['text'], status='unmatched', candidates=[], reference=None,
                              reason='marker_missing' if marker is None else 'reference_missing'))
        elif block['source_type'] not in MARGINALS and (block['heading'] or block['b'] in owners):
            by_page[block['page']].append(block)
    labels = Counter((n['page'], n['marker']['style'], n['marker']['number']) for n in notes if n['marker'])
    for note in notes:
        marker = note['marker']
        if marker is None:
            continue
        for block in by_page[note['page']]:
            for part in block['parts']:
                if part['kind'] not in {'text', 'math'}:
                    continue
                for a, z in marker_occurrences(part['text'], marker, part['kind']):
                    lo, hi = part['lo']+a, part['lo']+z
                    atom = index.atoms[index.ids[part['atom']]]
                    note['candidates'].append(dict(block=block['b'], atom=atom['id'], lo=lo, hi=hi,
                        start_offset=part['offset']+a, end_offset=part['offset']+z,
                        insert_after=hi if atom['kind']=='text' else index.prefix[index.ids[atom['id']]+1],
                        kind=part['kind'], text=part['text'][a:z], page=block['page']))
        if labels[note['page'], marker['style'], marker['number']] > 1:
            note['reason'] = 'duplicate_note_marker'
        elif len(note['candidates']) == 1:
            note.update(status='matched', reference=note['candidates'][0], reason='unique_same_page_marker')
        elif note['candidates']:
            note['reason'] = 'ambiguous_reference'
    result = dict(method=METHOD, match_policy='same_original_page_unique_marker', items=notes,
                  duplicate_formulas=duplicates,
                  statistics=dict(blocks=len(notes), matched=sum(n['status']=='matched' for n in notes),
                                  unmatched=sum(n['status']=='unmatched' for n in notes),
                                  duplicate_formula_blocks=len(duplicates),
                                  reasons=dict(Counter(n['reason'] for n in notes))))
    plan['footnotes'] = result
    return plan


def source_ranges(index, block):
    """Exact retained intervals, including paragraphs joined around footnotes."""
    spans = []
    for part in block['parts']:
        base = index.prefix[index.ids[part.get('source_atom', part.get('atom'))]]
        lo = base+part.get('start_offset', part.get('offset', 0))
        hi = base+part['end_offset']
        if spans and spans[-1]['hi'] == lo:
            spans[-1]['hi'] = hi
        else:
            spans.append(dict(lo=lo, hi=hi))
    return spans
