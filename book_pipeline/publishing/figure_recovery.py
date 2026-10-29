"""Preserve figure-only source pages when OCR splits diagram labels into math."""
from collections import defaultdict
from pathlib import Path
import re

from book_pipeline.ingestion.math_syntax import formula_tags
from book_pipeline.structure.headings import valid_box
from book_pipeline.structure.numbered import CAPTION_ROLES


def recover_native_vector_labels(index,directory,edited_atoms=()):
    """Include nearby outlined plot labels without crossing owned text.

    Native plots can encode tick/axis labels as small vector paths. They have
    no text spans, so a layout box tightly enclosing the axes can lose them.
    Only bounded, unassigned source paths can enlarge the rendered crop.
    """
    source=index.source.get('source_pdf')
    if not source or not Path(source).is_file():return {}
    import pymupdf
    result={}
    with pymupdf.open(source) as pdf:
        for atom in index.atoms:
            if (atom['kind']!='image' or atom.get('extraction_method')!='native'
                    or atom['id'] in edited_atoms or atom.get('recovery_issue')
                    or not valid_box(atom.get('bbox')) or not 0<=atom['page']<len(pdf)):
                continue
            page=pdf[atom['page']];b=atom['bbox'];region=list(b)
            blockers=[a['bbox'] for a in index.atoms if a['page']==atom['page']
                      and a['id']!=atom['id'] and valid_box(a.get('bbox'))]
            overlaps=lambda a,c:min(a[2],c[2])>max(a[0],c[0]) and min(a[3],c[3])>max(a[1],c[1])
            for path in page.get_drawings():
                rect=path['rect']
                c=[rect.x0/page.rect.width,rect.y0/page.rect.height,
                   rect.x1/page.rect.width,rect.y1/page.rect.height]
                if (not valid_box(c) or c[2]-c[0]>.08 or c[3]-c[1]>.08
                        or c[0]<b[0]-.05 or c[1]<b[1]-.05
                        or c[2]>b[2]+.05 or c[3]>b[3]+.05
                        or any(overlaps(c,other) for other in blockers)):
                    continue
                proposed=[min(region[0],c[0]),min(region[1],c[1]),
                          max(region[2],c[2]),max(region[3],c[3])]
                if not any(overlaps(proposed,other) for other in blockers):region=proposed
            # Substantial external ink is required; tiny edge differences can
            # be stroke overshoot or an unrelated neighboring fragment.
            if max(abs(x-y) for x,y in zip(region,b))<.01:continue
            # A small antialiasing border is clipped to the same text barriers.
            padded=[max(0,region[0]-.001),max(0,region[1]-.001),
                    min(1,region[2]+.001),min(1,region[3]+.001)]
            if not any(overlaps(padded,other) for other in blockers):region=padded
            path=Path(directory)/'images'/('native-vector-labels-'+atom['id']+'.png')
            path.parent.mkdir(parents=True,exist_ok=True)
            clip=pymupdf.Rect([v*d for v,d in zip(region,[page.rect.width,page.rect.height]*2)])
            page.get_pixmap(dpi=240,clip=clip,alpha=False).save(path)
            result[atom['id']]=dict(asset=str(path.resolve()),bbox=region,page=atom['page']+1,
                labels=[],method='bounded_unassigned_native_vector_labels',original_bbox=b)
    return result


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
