"""Restore a missing reference dot without changing any printed digit.

Blind local OCR must agree across two views, and a separate connected-ink
check must see the dot between digit-sized glyphs. Unique ordinary prose owns
the physical row; mathematical guesses cannot authorize a content rewrite.
"""
from copy import deepcopy
from pathlib import Path
import re

from book_pipeline.config import digest

METHOD = 'cpu_reference_dot_ink_witness_v1'
MIN_CONFIDENCE = .96
REFERENCE = re.compile(r'by\s+\((\d{1,4}\.\d{1,4})\)[.,;:]?')


def policy_signature():
    from book_pipeline.ingestion.formula_text_anchors import policy_signature as reader_policy
    return dict(method=METHOD, reader=reader_policy(), scales=[2, 3], confidence=MIN_CONFIDENCE)


def dot_ink(gray, reference):
    """Require one baseline dot in the OCR-observed gap, independently of OCR.

    Project connected ink onto x after dropping tiny dust. This reunites a
    damaged parenthesis without treating its separated fragments as digits.
    """
    import cv2
    import numpy as np
    first, second = reference.split('.')
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    clean = np.zeros_like(ink)
    for n, (_, _, _, _, area) in enumerate(stats[1:], 1):
        if area >= 4:
            clean[labels == n] = 1
    xs = np.flatnonzero(clean.any(0))
    runs = []
    for x in xs:
        if not runs or x-runs[-1][-1] > 2:
            runs.append([])
        runs[-1].append(int(x))
    boxes = []
    for run in runs:
        x, z = run[0], run[-1]+1
        ys = np.flatnonzero(clean[:, x:z].any(1))
        boxes.append([x, int(ys[0]), z, int(ys[-1])+1])
    if len(boxes) < 5:
        return None
    height = max(b[3]-b[1] for b in boxes)
    tall = [b for b in boxes if b[3]-b[1] >= .5*height]
    # Parentheses and every digit must have their own observed column.
    if len(tall) != len(first+second)+2:
        return None
    a, b = tall[len(first)], tall[len(first)+1]
    baseline = (a[3]+b[3])/2
    dots = [c for c in boxes if a[2] <= c[0] < c[2] <= b[0]
            and .04*height <= c[3]-c[1] <= .22*height
            and .4 <= (c[2]-c[0])/(c[3]-c[1]) <= 2.5
            and baseline-.18*height <= (c[1]+c[3])/2 <= baseline+.15*height]
    return dict(dot_bbox=dots[0], glyph_boxes=tall) if len(dots) == 1 else None


def observe_references(rows, directory):
    """Read only right-side reasons, with no transcript or proposed reference."""
    from PIL import Image, ImageOps
    import numpy as np
    from book_pipeline.publishing.image_search import local_ocr
    candidates = []
    for row in rows:
        words = row['words']
        by = [w for w in words if w['text'] == 'by' and w['score'] >= .8
              and w['bbox'][0] > .65*row['pixel_bbox'][2]]
        if len(by) != 1:
            continue
        word = by[0]
        suffix = [w for w in words if w['bbox'][0] > word['bbox'][0]+word['bbox'][2]]
        if len(suffix) == 1:
            candidates.append((row, word, suffix[0]))
    if not candidates:
        return []
    runtime = local_ocr()
    if runtime is None:
        return []
    model, signature = runtime
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for row, by, suffix in candidates:
        if not Path(row['image']).is_file() or digest(row['image']) != row['image_sha256']:
            continue
        readings = []
        with Image.open(row['image']) as original:
            x, y, w, h = suffix['bbox']
            if not (0 <= by['bbox'][0]-4 < x < x+w+4 <= original.width
                    and 0 <= y < y+h <= original.height):
                continue
            crop = original.crop((by['bbox'][0]-4, 0, x+w+4, original.height))
            for scale in (2, 3):
                view = ImageOps.expand(crop.resize((crop.width*scale, crop.height*scale)),
                                       border=12*scale, fill='white')
                image = directory/f"row-{row['id']:03d}-s{scale}.png"
                view.save(image)
                observed = model.ocr(np.array(view.convert('RGB'))[:, :, ::-1].copy())[0] or []
                if len(observed) != 1:
                    break
                _, (text, score) = observed[0]
                match = REFERENCE.fullmatch(text.strip())
                if match is None or not MIN_CONFIDENCE <= float(score) <= 1:
                    break
                readings.append(dict(reference=match[1], text=text, score=float(score),
                                     image=str(image), image_sha256=digest(image)))
            if len(readings) != 2 or readings[0]['reference'] != readings[1]['reference']:
                continue
            reference = readings[0]['reference']
            witness = dot_ink(np.array(original.crop((x-4, 0, x+w+4, original.height)).convert('L')), reference)
            if witness is not None:
                records.append(dict(row=row['id'], reference=reference, readings=readings,
                                    reader_signature=signature, ink=witness))
    return records


def prose_words(item):
    if item['kind'] == 'formula':
        text = ' '.join(re.findall(r'\\text\{([^{}]*)\}', item['latex']))
    elif item['kind'] == 'text':
        text = re.sub(r'\$[^$]*\$', '', item['text'])
    else:
        return set()
    return set(re.findall(r'\b[a-z]{4,}\b', text.lower()))


def restore_reference_dots(answer, rows, observations):
    """Only insert a dot: never replace, delete or permute digits or symbols."""
    result, changes = deepcopy(answer), []
    words = [prose_words(item) for item in answer['items']]
    row_words = [{w['text'].strip('.,;:()').lower() for w in row['words']
                  if w['score'] >= .95 and re.fullmatch(r'[A-Za-z]{4,}[.,;:]?', w['text'])}
                 for row in rows]
    for n, item in enumerate(answer['items']):
        if item['kind'] != 'formula':
            continue
        # References must be ordinary prose inside a flat TeX text span.
        spans = [m for text in re.finditer(r'\\text\{[^{}]*\}', item['latex'])
                 for m in re.finditer(r'\bby\s+\((\d{2,8})\)', item['latex'][text.start():text.end()])]
        if len(spans) != 1:
            continue
        old = spans[0][1]
        supported = []
        for observation in observations:
            reference = observation['reference']
            if (not re.fullmatch(r'\d{1,4}\.\d{1,4}', reference)
                    or reference.replace('.', '') != old or not observation.get('reader_signature')
                    or not observation.get('ink') or len(observation.get('readings', [])) != 2
                    or any(r['reference'] != reference or not MIN_CONFIDENCE <= r['score'] <= 1
                           for r in observation['readings'])):
                continue
            owners = [k for k, row in enumerate(rows) if row['id'] == observation['row']]
            if len(owners) != 1:
                continue
            k = owners[0]
            anchors = {w for w in words[n] & row_words[k]
                       if sum(w in ws for ws in words) == 1 and sum(w in ws for ws in row_words) == 1}
            if anchors:
                supported.append((observation, anchors))
        before = '('+old+')'
        if len(supported) != 1 or item['latex'].count(before) != 1:
            continue
        observation, anchors = supported[0]
        result['items'][n]['latex'] = item['latex'].replace(before, '('+observation['reference']+')')
        changes.append(dict(item=n, before=old, after=observation['reference'],
                            row=observation['row'], anchors=sorted(anchors), method=METHOD))
    return result, changes
