"""Render caption nodes using independently observed page geometry.

Inherited OCR asset boxes cannot establish caption order. Reuse only exact,
unique line locations from the current source's CPU typography observations.
The source stream and its code-point addresses are never rewritten.
"""
from pathlib import Path
import re

from book_pipeline.config import read
from book_pipeline.structure.headings import valid_box
from book_pipeline.structure.numbered import CAPTION_ROLES

METHOD = 'cpu_observed_caption_geometry_v1'


def caption_separator(left, right):
    if left['group'] != right['group']:
        return ' '
    # Independent label and description nodes are not a split word. Inline
    # math, punctuation, and arbitrary adjacent text retain their exact joins.
    if (left['kind'] == right['kind'] == 'text'
            and re.fullmatch(r'\s*(?:Fig(?:ure)?\.?|Table|图|表)\s*\d+(?:[.\-]\d+)*[.:：]?\s*',
                             left['content'], re.I)
            and right['content'] and not right['content'][0].isspace()
            and not left['content'][-1:].isspace()):
        return ' '
    return ''


def observed_caption_boxes(atoms, observations, size):
    normalize = lambda s: re.sub(r'\s+', '', s)
    result = {}
    for atom in atoms:
        if atom['kind'] != 'text' or atom['role'] not in CAPTION_ROLES:
            continue
        hits = []
        for row in observations:
            if row.get('score', 0) < .95 or normalize(row['text']) != normalize(atom['content']):
                continue
            b = [v/d for v,d in zip(row['bbox'], size*2)]
            if valid_box(b):
                hits.append(b)
        # The collector reads several completed versions of the same line.
        # Accept that cluster, but never select among repeated page locations.
        if hits and max(b[1] for b in hits)-min(b[1] for b in hits) < .008:
            result[atom['id']] = [min(b[0] for b in hits), min(b[1] for b in hits),
                                  max(b[2] for b in hits), max(b[3] for b in hits)]
    return result


def recover_caption_boxes(index, original, body_report, excluded=()):
    import math
    import pymupdf
    from book_pipeline.publishing.raster_body_styles import DPI
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return {}
    candidates = []
    for atom in index.atoms:
        old = original.atoms[original.ids[atom['original_atom']]] if atom.get('original_atom') in original.ids else None
        if (atom['id'] not in excluded and atom.get('extraction_method') == 'ocr' and old
                and old['content'][atom['original_start_offset']:atom['original_end_offset']] == atom['content']):
            candidates.append(atom)
    result = {}
    with pymupdf.open(source) as pdf:
        for record in body_report.get('records', []):
            number = record['page']-1
            cache = Path(record.get('cache', ''))
            if not cache.is_file() or not 0 <= number < len(pdf):
                continue
            page = pdf[number]
            size = [math.ceil(v*DPI/72) for v in (page.rect.width, page.rect.height)]
            result.update(observed_caption_boxes([a for a in candidates if a['page'] == number], read(cache), size))
    return result


def caption_sides(atoms, boxes):
    assets = [a for a in atoms if a['kind'] in {'image', 'table'}]
    captions = [a for a in atoms if a['role'] in CAPTION_ROLES]
    if len(assets) != 1 or not captions or len(assets)+len(captions) != len(atoms):
        return None
    asset = assets[0]
    b = asset.get('bbox')
    if not valid_box(b):
        return None
    sides = dict(before=[], after=[])
    for caption in captions:
        c = boxes.get(caption['id'])
        if (caption['kind'] != 'text' or caption['page'] != asset['page'] or not valid_box(c)
                or min(b[2], c[2]) <= max(b[0], c[0])):
            return None
        if 0 <= b[1]-c[3] <= .10:
            sides['before'].append(caption)
        elif 0 <= c[1]-b[3] <= .10:
            sides['after'].append(caption)
        else:
            return None
    for side in sides.values():
        side.sort(key=lambda a: (round(boxes[a['id']][1]/.008), boxes[a['id']][0]))
    return dict(asset=asset['id'], **sides)


def reviewed_caption_continuations(index, review, boxes, registry=(), exclusions=()):
    """Attach reviewed, unchanged prose immediately below a located label."""
    from book_pipeline.config import fingerprint
    if review.get('source_sha256') != fingerprint(index.source) or not review.get('evidence'):
        raise ValueError('Caption review requires current source and visual evidence')
    result = {}
    rows = review.get('caption_continuations', [])
    if not isinstance(rows, list):
        raise ValueError('Caption continuations require a list')
    claimed = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('asset'), str)
                or row['asset'] not in index.ids or not isinstance(row.get('group'), str)
                or row['group'] not in index.groups or row['group'] in claimed):
            raise ValueError('Caption continuation must name one existing asset and prose group')
        asset = index.atoms[index.ids[row['asset']]]
        geometry = caption_sides([index.atoms[n] for n in index.groups[asset['group']]], boxes)
        members = index.groups[row['group']]
        atoms = [index.atoms[n] for n in members]
        lo, hi = index.prefix[min(members)], index.prefix[max(members)+1]
        if (asset['kind'] != 'image' or asset['id'] in result or not geometry or not geometry['after']
                or min(members) != max(index.groups[asset['group']])+1
                or any(a['page'] != asset['page'] or a['kind'] not in {'text','math'}
                       or a['role'] != 'text' or a.get('source_panel') for a in atoms)
                or row.get('quote') != index.text(lo,hi) or hi-lo > 2400
                or any(o['lo'] < hi and lo < o['hi'] for o in [*registry,*exclusions])):
            raise ValueError('Caption continuation crosses a source or semantic boundary')
        b = atoms[0].get('block_bbox') or atoms[0].get('bbox')
        label = boxes[geometry['after'][-1]['id']]
        if (not valid_box(b) or not -.008 <= b[1]-label[3] <= .025
                or min(b[2],label[2]) <= max(b[0],label[0])):
            raise ValueError('Caption continuation is not adjacent to its observed source label')
        result[asset['id']] = dict(atoms=atoms,lo=lo,hi=hi,evidence=review['evidence'])
        claimed.add(row['group'])
    return result
