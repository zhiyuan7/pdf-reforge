"""Reuse two blind complete scan readings for a narrow inline-letter repair.

Only one Latin denominator letter may change in an otherwise identical item.
Presentation boundaries and comma/period separators may differ in the readers;
the baseline's punctuation, grouping, digits, operators and prose stay fixed.
This never infers a variable from the mathematical meaning of the sentence.
"""
import re

METHOD = 'two_complete_scan_readings_inline_denominator_letter_v3'
SPACING = re.compile(r'\\(?:quad\b|qquad\b|[,;! ])')
DENOMINATOR = re.compile(r'(?<![A-Za-z])([A-Za-z]{1,4})/([A-Za-z]{1,4})(?=\s*[=<>])')


def reading(text):
    """Flatten simple inline notation while retaining source character offsets.

    A comma and a period each occupy one separator position. Decimal points
    retain their identity; no separator is dropped or used to edit punctuation.
    Unsupported TeX, non-ASCII text and unbalanced math abstain.
    """
    if (not isinstance(text, str) or not text.isascii() or text.count('$') % 2
            or not re.search(r'[A-Za-z]{3,}', re.sub(r'\$[^$]*\$', '', text))):
        return None
    spans = list(re.finditer(r'\$([^$]+)\$', text))
    if not spans:
        return None
    denominators = set()
    for span in spans:
        for match in DENOMINATOR.finditer(span[1]):
            denominators.update(range(span.start(1)+match.start(2), span.start(1)+match.end(2)))
    key, offsets = [], []
    n = 0
    while n < len(text):
        char = text[n]
        if char == '\\':
            space = SPACING.match(text, n)
            if space is None:
                return None
            n = space.end()
            continue
        if char in '{}':
            return None
        if not char.isspace() and char != '$':
            decimal = (char == '.' and n > 0 and n+1 < len(text)
                       and text[n-1].isdigit() and text[n+1].isdigit())
            key.append('\x1f' if char in ',.' and not decimal else char)
            offsets.append(n)
        n += 1
    return ''.join(key), offsets, denominators


def ownership(text):
    parsed = reading(text)
    if parsed is None:
        return None
    key, offsets, denominators = parsed
    return (''.join('#' if offset in denominators else char for char, offset in zip(key, offsets)),
            prose_words(text))


def prose_words(text):
    # Whitespace inside math is presentation; ordinary word boundaries are
    # evidence. "is not" and "isn ot" must never establish the same owner.
    return tuple(re.findall(r'[A-Za-z]+', re.sub(r'\$[^$]*\$', ' ', text)))


def propose_inline_edits(answer, witness):
    """A unique complete item proposes one letter, never a partial math match."""
    baseline = [i['text'] for i in answer['items'] if i['kind'] == 'text']
    readings = [(i['text'], reading(i['text'])) for i in witness['items'] if i['kind'] == 'text']
    edits = []
    for n, item in enumerate(answer['items']):
        if item['kind'] != 'text':
            continue
        text = item['text']
        parsed = reading(text)
        if parsed is None or sum(ownership(t) == ownership(text) for t in baseline) != 1:
            continue
        key, offsets, denominators = parsed
        # An unchanged independent reading or duplicate candidate vetoes repair.
        if any(p and p[0] == key for _, p in readings):
            continue
        candidates = []
        for other, parsed_other in readings:
            if (parsed_other is None or len(parsed_other[0]) != len(key)
                    or prose_words(other) != prose_words(text)):
                continue
            changed = [k for k, (a, b) in enumerate(zip(key, parsed_other[0])) if a != b]
            if len(changed) != 1:
                continue
            k = changed[0]
            offset = offsets[k]
            other_offset = parsed_other[1][k]
            if (offset not in denominators or other_offset not in parsed_other[2]
                    or not key[k].isalpha() or not parsed_other[0][k].isalpha()
                    or key[k].isupper() != parsed_other[0][k].isupper()):
                continue
            candidates.append((offset, parsed_other[0][k]))
        if len(candidates) != 1:
            continue
        offset, letter = candidates[0]
        start, end = offset, offset+1
        while text.count(text[start:end]) != 1 and end-start <= 64:
            if start:
                start -= 1
            elif end < len(text):
                end += 1
            else:
                break
        if end-start > 64 or text.count(text[start:end]) != 1:
            continue
        edits.append(dict(item=n, field='text', before=text[start:end],
                          after=text[start:offset]+letter+text[offset+1:end],
                          reason='Two blind complete scan readings agree on one inline denominator letter'))
    return edits


def agreed_inline_edits(answer, first, second, band):
    from book_pipeline.ingestion.formula_bands import apply_detail_edits
    proposed = propose_inline_edits(answer, first)
    corroborated = propose_inline_edits(answer, second)
    accepted = [e for e in proposed if e in corroborated]
    rejected = [e for e in proposed if e not in corroborated]
    corrected = apply_detail_edits(answer, dict(edits=accepted, unresolved=[]), band)
    return corrected, accepted, rejected
