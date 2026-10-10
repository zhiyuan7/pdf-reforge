"""Preserve figure-only source pages when OCR splits diagram labels into math."""
from collections import defaultdict
from pathlib import Path
import re

from book_pipeline.ingestion.math_syntax import formula_tags
from book_pipeline.structure.headings import valid_box
from book_pipeline.structure.numbered import CAPTION_ROLES


def recover_fragmented_figures(index, registry, directory, edited_atoms=()):
    source_pdf = index.source.get('source_pdf')
    if not source_pdf:
        return {}
    by_page = defaultdict(list)
    for atom in index.atoms:
        by_page[atom['page']].append(atom)
    result = {}
    for page, atoms in by_page.items():
        images = [a for a in atoms if a['kind'] == 'image']
        targets = [r for r in registry if r.get('page') == page + 1]
        if len(images) != 1 or len(targets) != 1 or targets[0]['kind'] != 'figure':
            continue
        image = images[0]
        labels = [a for a in atoms if a['id'] != image['id'] and a['role'] not in CAPTION_ROLES]
        # Restrict recovery to a complete diagram page: no prose, corrections,
        # linked text, or independent objects can be hidden by this operation.
        if not labels or any(a['id'] in edited_atoms for a in atoms):
            continue
        if any(a['kind'] not in {'text', 'math', 'display_math'} or len(a['content']) > 40
               or re.search(r'[。！？；]', a['content']) for a in labels):
            continue
        if not any(len(re.findall(r'\\tag\*?\s*\{', a['content'])) != len(formula_tags(a['content']))
                   for a in labels if a['kind'] in {'math', 'display_math'}):
            continue
        if not all(valid_box(a.get('bbox')) for a in [image, *labels]):
            continue
        box = image['bbox']
        if not all(a['bbox'][0] >= box[0] - .15 and a['bbox'][2] <= box[2] + .15
                   and a['bbox'][1] >= box[1] - .06 and a['bbox'][3] <= box[3] + .06 for a in labels):
            continue
        boxes = [a['bbox'] for a in [image, *labels]]
        region = [max(0, min(b[0] for b in boxes) - .008), max(0, min(b[1] for b in boxes) - .008),
                  min(1, max(b[2] for b in boxes) + .008), min(1, max(b[3] for b in boxes) + .008)]
        import pymupdf
        path = Path(directory) / 'images' / f'source-figure-page-{page+1:06d}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(source_pdf) as document:
            source = document[page]
            clip = pymupdf.Rect(region[0] * source.rect.width, region[1] * source.rect.height,
                                region[2] * source.rect.width, region[3] * source.rect.height)
            source.get_pixmap(dpi=180, clip=clip, alpha=False).save(path)
        result[image['id']] = dict(asset=str(path.resolve()), bbox=region, page=page+1,
                                  labels=[a['id'] for a in labels])
    return result
