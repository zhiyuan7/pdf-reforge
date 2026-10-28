"""Keep rounding delimiters when CPU OCR and a blind image reading agree.

Only paired square brackets may become ceilings/floors. Every other token of
the complete, uniquely owned formula must match, including grouping and tags.
The baseline's spelling, spacing, order and numbering remain byte-for-byte.
"""
from copy import deepcopy
import re

METHOD = 'complete_cpu_blind_scan_rounding_agreement_v1'
TOKENS = re.compile(r'\\[A-Za-z]+|\\.|[^\s]')
OPEN = {'[': ']', r'\lceil': r'\rceil', r'\lfloor': r'\rfloor'}
CLOSE = set(OPEN.values())
PRESENTATION = {r'\left', r'\right', r'\quad', r'\qquad', r'\,', r'\;', r'\!', '\\ '}


def reading(text):
    if not isinstance(text, str) or re.search(r'\\(?:text\w*|operatorname)\b|[%$#&]', text):
        return None
    tokens = [(m[0], m.start(), m.end()) for m in TOKENS.finditer(text) if m[0] not in PRESENTATION]
    stack = []
    for token, _, _ in tokens:
        if token == '{':
            stack.append(token)
        elif token == '}':
            if not stack:
                return None
            stack.pop()
    if stack:
        return None
    # Braces around one TeX atom are presentation. Compound argument groups
    # stay significant: x_{ab} must never corroborate x_ab.
    while True:
        discard = {k for n in range(len(tokens)-2)
                   if tokens[n][0] == '{' and tokens[n+2][0] == '}'
                   and tokens[n+1][0] not in {'{', '}'} for k in (n, n+2)}
        if not discard:
            break
        tokens = [t for n, t in enumerate(tokens) if n not in discard]
    pairs, stack = [], []
    for n, (token, _, _) in enumerate(tokens):
        if token in OPEN:
            stack.append((n, token))
        elif token in CLOSE:
            if not stack or OPEN[stack[-1][1]] != token:
                return None
            start, opener = stack.pop()
            pairs.append((start, n, opener))
    if stack:
        return None
    aliases = {r'\le': r'\leq', r'\ge': r'\geq'}
    identity = tuple(aliases.get(t, t) for t, _, _ in tokens)
    key = tuple('ROUND_OPEN' if t in OPEN else 'ROUND_CLOSE' if t in CLOSE else t for t in identity)
    return dict(key=key, identity=identity, tokens=tokens, pairs=pairs)


def source_formulas(index, band):
    """Use only complete OCR displays inside the current band's owners."""
    return [dict(group=a['group'], latex=a['content']) for a in index.atoms
            if a['group'] in band['groups'] and a['kind'] == 'display_math'
            and a.get('extraction_method') == 'ocr']


def candidates(answer, originals):
    result = {}
    parsed = [(n, reading(i['latex'])) for n, i in enumerate(answer['items']) if i['kind'] == 'formula']
    sources = [(s, reading(s['latex'])) for s in originals]
    for n, base in parsed:
        if base is None or sum(p is not None and p['key'] == base['key'] for _, p in parsed) != 1:
            continue
        matches = [(s, p) for s, p in sources if p is not None and p['key'] == base['key']]
        if len(matches) != 1:
            continue
        source, other = matches[0]
        edits = []
        for start, end, opener in base['pairs']:
            target = other['tokens'][start][0]
            if opener == target:
                continue
            if opener != '[' or target not in {r'\lceil', r'\lfloor'}:
                break
            for k in (start, end):
                before, lo, hi = base['tokens'][k]
                after = other['tokens'][k][0]
                if hi < len(answer['items'][n]['latex']) and answer['items'][n]['latex'][hi].isalpha():
                    after += ' '
                edits.append(dict(start=lo, end=hi, before=before, after=after))
        else:
            if edits:
                result[n] = dict(source_group=source['group'], reading=other, edits=edits)
    return result


def restore_rounding(answer, witness, originals):
    result, changes = deepcopy(answer), []
    reads = [(i.get('number', ''), reading(i['latex'])) for i in witness['items'] if i['kind'] == 'formula']
    for n, proposal in candidates(answer, originals).items():
        other = proposal['reading']
        matches = [(number, p) for number, p in reads if p is not None and p['key'] == other['key']]
        if (len(matches) != 1 or matches[0][0] != answer['items'][n].get('number', '')
                or matches[0][1]['identity'] != other['identity']):
            continue
        text = result['items'][n]['latex']
        for edit in sorted(proposal['edits'], key=lambda e: e['start'], reverse=True):
            text = text[:edit['start']]+edit['after']+text[edit['end']:]
        result['items'][n]['latex'] = text
        changes.append(dict(item=n, source_group=proposal['source_group'], edits=proposal['edits']))
    return result, changes
