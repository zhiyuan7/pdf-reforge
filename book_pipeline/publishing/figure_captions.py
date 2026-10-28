"""Recover adjacent caption rows that OCR classified as separate headings."""
import re

from book_pipeline.structure.headings import valid_box


def split_caption_rows(index, registry, exclusions=(), changed=()):
    rows = []
    for n, image in enumerate(index.atoms[:-1]):
        if (image['kind'] != 'image' or not valid_box(image.get('bbox'))
                or image.get('recovery_issue') or image['id'] in changed
                or len(index.groups[image['group']]) != 1):
            continue
        label = index.atoms[n+1]
        if not re.fullmatch(r'\s*(?:Fig(?:ure)?\.?|图)\s*\d+(?:[.\-]\d+)*\s*[.:：]?\s*',
                            label['content'], flags=re.I):
            continue
        caption = []
        for atom in index.atoms[n+1:n+4]:
            b = atom.get('block_bbox') or atom.get('bbox')
            if (atom['kind'] != 'text' or atom.get('block_type') != 'paragraph_title'
                    or atom['page'] != image['page'] or atom['id'] in changed
                    or not valid_box(b) or len(index.groups[atom['group']]) != 1
                    or atom.get('source_panel') or len(atom['content']) > 180):
                break
            if caption:
                prev = caption[-1].get('block_bbox') or caption[-1]['bbox']
                if abs(b[1]-prev[1]) > .005 or not .005 <= b[0]-prev[2] <= .06:
                    break
            elif not 0 <= b[1]-image['bbox'][3] <= .035:
                break
            caption.append(atom)
        if len(caption) != 2: continue
        lo, hi = index.prefix[n+1], index.prefix[n+3]
        if any(r['lo'] < hi and lo < r['hi'] for r in [*registry,*exclusions]): continue
        if any(r.get('caption_span') and r['lo'] == index.prefix[n] for r in registry): continue
        rows.append(dict(method='same_line_source_caption_row', image=image['id'],
                         lo=lo, hi=hi, page=image['page']+1))
    return rows
