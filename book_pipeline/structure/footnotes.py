"""Literal, page-local footnote labels and reversible source relationships."""
from collections import Counter, defaultdict
from copy import deepcopy
import re
import unicodedata

METHOD = 'page_footnotes_same_page_unique_v1'
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


def classify_footnotes(index, plan):
    """Classify before any model input; multiple note/body candidates never pair."""
    plan = deepcopy(plan)
    by_page = defaultdict(list)
    owners = {bid: s['id'] for s in plan['sections'] for bid in s['blocks']}
    notes = []
    for block in plan['blocks']:
        is_note = block['source_type'] == 'page_footnote' and not block['heading']
        block['content_role'] = 'page_footnote' if is_note else 'body'
        if block['page'] in plan['excluded_pages']:
            continue
        if is_note:
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
                  statistics=dict(blocks=len(notes), matched=sum(n['status']=='matched' for n in notes),
                                  unmatched=sum(n['status']=='unmatched' for n in notes),
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
