"""Locate left statement labels without rereading mathematical content.

CPU coordinates establish label placement. Numeric labels and reference dots
require corroborating ordinary words from their observed physical row.
Variables, operators, item order and reference digits stay fixed.
"""
from copy import deepcopy
import csv
from io import StringIO
from pathlib import Path
import re
import subprocess

from book_pipeline.config import digest, fingerprint, read, save

METHOD = 'cpu_scan_labels_and_reference_dots_v2'
LABEL = re.compile(r'^\((\d{1,4})\)$')


def policy_signature():
    from book_pipeline.publishing.latin_word_ocr import runtime_signature
    from book_pipeline.ingestion.scan_reference_punctuation import policy_signature as reference_policy
    from book_pipeline.ingestion.scan_reference_audit import METHOD as audit_method, MIN_CONFIDENCE as audit_confidence
    return dict(method=METHOD, cpu=runtime_signature(), row_gap=4, label_confidence=.90,
                reference_policy=reference_policy(), reference_audit=dict(method=audit_method, confidence=audit_confidence,
                                                                         scales=[1, 2, 3], mode=7))


def physical_rows(image, directory):
    """Image-only row segmentation and words; optional CPU reader fails closed."""
    from book_pipeline.publishing.latin_word_ocr import runtime_signature
    signature = runtime_signature()
    if signature is None:
        return []
    from PIL import Image
    import cv2
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = fingerprint(dict(image=digest(image), policy=policy_signature()))
    cache = directory/'rows.json'
    if cache.is_file():
        saved = read(cache)
        if saved.get('key') == stamp and all(
                Path(r['image']).is_file() and digest(r['image']) == r['image_sha256']
                for r in saved['rows']):
            return saved['rows']
    with Image.open(image) as original:
        gray = np.array(original.convert('L'))
        h, w = gray.shape
        if w <= 1024 or h > .5*w:
            return []  # This reader handles wide horizontal scan bands only.
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        active = np.flatnonzero((ink > 0).sum(axis=1) > max(5, w*.004))
        runs = []
        for y in active:
            if not runs or y-runs[-1][-1] > 4:
                runs.append([])
            runs[-1].append(int(y))
        rows = []
        try:
            for ys in runs:
                if len(ys) < 8:
                    continue
                lo, hi = max(0, ys[0]-4), min(h, ys[-1]+5)
                path = directory/f'row-{len(rows):03d}.png'
                original.crop((0, lo, w, hi)).save(path)
                result = subprocess.run(['tesseract', str(path), 'stdout', '-l', 'eng', '--psm', '7', 'tsv'],
                                        capture_output=True, check=True, timeout=10)
                words = []
                for token in csv.DictReader(StringIO(result.stdout.decode()), delimiter='\t'):
                    if token['level'] == '5':
                        words.append(dict(text=token['text'], score=float(token['conf'])/100,
                                          bbox=[int(token[k]) for k in ('left', 'top', 'width', 'height')]))
                rows.append(dict(id=len(rows), image=str(path), image_sha256=digest(path),
                                 pixel_bbox=[0, lo, w, hi], words=words))
        except (OSError, subprocess.SubprocessError, UnicodeError, KeyError, ValueError, csv.Error):
            return []
    save(cache, dict(key=stamp, rows=rows))
    return rows


def label_rows(rows):
    result = {}
    for row in rows:
        words = row['words']
        if len(words) < 2:
            continue
        first, following = words[:2]
        match = LABEL.fullmatch(first['text'])
        width = row['pixel_bbox'][2]
        if (match and first['score'] >= .90 and 0 < first['bbox'][0] < .4*width
                and following['bbox'][0] > first['bbox'][0]+first['bbox'][2]):
            result.setdefault(match[1], []).append(row)
    return {number: entries[0] for number, entries in result.items() if len(entries) == 1}


def statement_text(latex, number):
    """Split only flat literal TeX text; retain every non-presentation symbol."""
    if re.search(r'\$|\\(?:begin|end|tag|label)\b|\\\\', latex):
        return None
    pieces, end = [], 0
    matches = list(re.finditer(r'\\text\{([^{}]*)\}', latex))
    if len(matches) != len(re.findall(r'\\text\b', latex)):
        return None
    def math_piece(value):
        value = re.sub(r'^(?:\s|\\(?:quad|qquad)\b)+|(?:\s|\\(?:quad|qquad)\b)+$', '', value)
        return '$'+value+'$' if value else ''
    for match in matches:
        pieces.extend((math_piece(latex[end:match.start()]), match[1]))
        end = match.end()
    pieces.append(math_piece(latex[end:]))
    return '('+number+') '+' '.join(p.strip() for p in pieces if p.strip())


def restore_left_labels(answer, rows):
    from book_pipeline.ingestion.formula_bands import comparable_item
    result, changes = deepcopy(answer), []
    labels = label_rows(rows)
    numbers = [i['number'].strip().strip('()') for i in answer['items'] if i['kind'] == 'formula']
    for n, item in enumerate(answer['items']):
        if item['kind'] != 'formula':
            continue
        number = item['number'].strip().strip('()')
        row = labels.get(number)
        if row is None or numbers.count(number) != 1:
            continue
        # Independent ordinary words corroborate ownership; CPU formula guesses
        # and the label itself cannot authorize changes to variables or reasons.
        prose = ' '.join(re.findall(r'\\text\{([^{}]*)\}', item['latex']))
        anchors = {w['text'].strip('.,;:()').lower() for w in row['words'][1:]
                   if w['score'] >= .80 and re.fullmatch(r'[A-Za-z]{2,}[.,;:()]?', w['text'])}
        supported = anchors & set(re.findall(r'[a-z]{2,}', prose.lower()))
        if len(supported) < 2:
            continue
        text = statement_text(item['latex'], number)
        if text is None or comparable_item(text) != '('+number+')'+comparable_item(item['latex']):
            continue
        result['items'][n] = dict(kind='text', text=text)
        changes.append(dict(item=n, number=number, row=row['id'], anchors=sorted(supported),
                            method='independent_cpu_left_label_no_content_change'))
    return result, changes



def recover_anchors(answer, image, directory):
    """CPU-only placement and ink-supported dot recovery; no API requests."""
    from book_pipeline.ingestion.formula_bands import validate_answer
    from book_pipeline.ingestion.scan_reference_punctuation import observe_references, restore_reference_dots
    evidence = dict(method=METHOD, labels=[], reference_dots=[], reference_conflicts=[])
    references = any(i['kind'] == 'formula' and re.search(r'\bby\s+\(\d{2,8}\)', i['latex'])
                     for i in answer['items'])
    dotted = any(i['kind'] == 'formula' and re.search(r'\bby\s+\(\d{1,4}\.\d{1,4}\)', i['latex'])
                 for i in answer['items'])
    if not references and not dotted and not any(i['kind'] == 'formula' and i.get('number') for i in answer['items']):
        return answer, evidence
    rows = physical_rows(image, directory)
    output, evidence['labels'] = restore_left_labels(answer, rows)
    if references:
        observations = observe_references(rows, Path(directory)/'reference_dots')
        output, evidence['reference_dots'] = restore_reference_dots(output, rows, observations)
        evidence['reference_observations'] = observations
    if dotted:
        from book_pipeline.ingestion.scan_reference_audit import observe_reasons, reference_conflicts
        observations = observe_reasons(rows, Path(directory)/'reference_audit')
        evidence['reference_conflicts'] = reference_conflicts(output, rows, observations)
    evidence['rows'] = str(Path(directory)/'rows.json') if rows else None
    if not any(i['kind'] == 'preserve' for i in output['items']):
        validate_answer(output, dict(preserve=[]))
    return output, evidence
