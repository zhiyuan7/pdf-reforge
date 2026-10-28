"""Recover repeated letters only from two complete, blind scan readings.

Baseline-guided review can silently retain modernized historical spellings.
This narrow fallback restores literal repeated-letter differences in ordinary
English paragraphs. Math, labels, punctuation, word order and case are fixed.
"""
import re

METHOD = 'independent_scan_prose_repeated_letters_v1'
READING_METHOD = 'independent_scan_prose_reading_v1'
PROMPT = '''
这是一次独立的原图逐字复读，不提供任何已有转录或候选修改。
尤其核对普通英文正文中的每个词：保留出版时的拼写，即使它不符合现代拼写习惯也不能改成现代写法。
不要根据数学常识、常见表达或词典补字、删字；完整保留行内数学、标点和段落。
普通段落合并自然换行；公式前后连接词仍与原图位置相符。看不清写[不清楚]，不要猜测。'''


def plain_prose(text):
    return (isinstance(text, str)
            and re.fullmatch(r'''[A-Za-z\s,;:.'’“”"()?!–—-]+''', text) is not None
            and len(re.findall(r'[A-Za-z]+', text)) >= 8)


def compact(text):
    return re.sub(r'\s+', '', text)


def repeated_letter_variant(before, after):
    # Preserve every letter and its case; only repeat counts may differ.
    collapse = lambda s: re.sub(r'([A-Za-z])\1+', r'\1', s)
    return (before != after and min(len(before), len(after)) >= 4
            and 0 < abs(len(before)-len(after)) <= 2
            and collapse(before) == collapse(after))


def paragraph_key(text):
    return re.sub(r'([A-Za-z])\1+', r'\1', compact(text))


def paragraph_edits(before, after, item):
    """Align complete plain paragraphs without a fuzzy text or math match."""
    if not plain_prose(before) or not plain_prose(after):
        return []
    words = list(re.finditer(r'[A-Za-z]+', before))
    alternatives = list(re.finditer(r'[A-Za-z]+', after))
    if len(words) != len(alternatives):
        return []
    skeleton = lambda s: re.sub(r'[A-Za-z]+', '#', compact(s))
    if skeleton(before) != skeleton(after):
        return []
    changes = [(n, a, b) for n, (a, b) in enumerate(zip(words, alternatives)) if a[0] != b[0]]
    if not 1 <= len(changes) <= 4 or any(not repeated_letter_variant(a[0], b[0]) for _, a, b in changes):
        return []
    edits = []
    for n, a, b in changes:
        start, end = a.span()
        # A repeated word needs literal surrounding context, never a guessed
        # occurrence. Contexts that overlap another edit are rejected below.
        while before.count(before[start:end]) != 1 and end-start <= 64:
            if start:
                start -= 1
            elif end < len(before):
                end += 1
            else:
                return []
        if end-start > 64:
            return []
        replacement = before[start:a.start()]+b[0]+before[a.end():end]
        if len(replacement) > 64:
            return []
        edits.append(dict(item=item, field='text', before=before[start:end], after=replacement,
                          reason='Two independent complete source readings agree on repeated letters'))
    spans = sorted((before.index(e['before']), before.index(e['before'])+len(e['before'])) for e in edits)
    return [] if any(a[1] > b[0] for a, b in zip(spans, spans[1:])) else edits


def propose_prose_edits(answer, witness):
    """A first blind reading proposes only uniquely aligned whole paragraphs."""
    readings = [i['text'] for i in witness['items'] if i['kind'] == 'text']
    edits = []
    for n, item in enumerate(answer['items']):
        if item['kind'] != 'text' or not plain_prose(item['text']):
            continue
        # Duplicate baseline paragraphs are not independently owned.
        if sum(i['kind'] == 'text' and paragraph_key(i['text']) == paragraph_key(item['text']) for i in answer['items']) != 1:
            continue
        if any(compact(text) == compact(item['text']) for text in readings):
            continue
        candidates = [paragraph_edits(item['text'], text, n) for text in readings]
        candidates = [c for c in candidates if c]
        if len(candidates) == 1:
            edits.extend(candidates[0])
    return edits


def agreed_prose_edits(answer, first, second, band):
    """Require a unique complete candidate paragraph in each blind reading."""
    from book_pipeline.ingestion.formula_bands import apply_detail_edits
    proposed = propose_prose_edits(answer, first)
    review = dict(edits=proposed, unresolved=[])
    candidate = apply_detail_edits(answer, review, band)
    fields = {e['item'] for e in proposed}
    accepted_items = set()
    corroborated = propose_prose_edits(answer, second)
    for n in fields:
        text = compact(candidate['items'][n]['text'])
        if ([e for e in proposed if e['item'] == n] == [e for e in corroborated if e['item'] == n]
                and all(sum(i['kind'] == 'text' and compact(i['text']) == text for i in w['items']) == 1
               for w in (first, second))):
            accepted_items.add(n)
    accepted = [e for e in proposed if e['item'] in accepted_items]
    rejected = [e for e in proposed if e['item'] not in accepted_items]
    return apply_detail_edits(answer, dict(edits=accepted, unresolved=[]), band), accepted, rejected
