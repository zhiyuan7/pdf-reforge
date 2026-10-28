"""Bounded raster furniture with independent OCR ownership.

Only flat gray side notes and isolated full-width rules are recovered. Source
text, OCR geometry and extraction caches are immutable; ambiguous mixed blocks,
tables and photographs retain their existing flow.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

from book_pipeline.ingestion.panels import intersection_fraction, panel_environments
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.side_notes import side_note_rows

DPI = 150
METHOD = 'raster_gray_note_and_isolated_rules_v1'


def page_pixels(page):
    import numpy as np
    import pymupdf
    pix = page.get_pixmap(dpi=DPI, alpha=False, colorspace=pymupdf.csRGB)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)


def gray_regions(rgb):
    """A note needs a large, uniform light-gray interior, not just gray ink."""
    import cv2
    import numpy as np
    h, w = rgb.shape[:2]
    neutral = rgb.max(2).astype(int)-rgb.min(2).astype(int) <= 6
    mask = (neutral & (rgb.min(2) >= 190) & (rgb.max(2) <= 246)).astype('uint8')
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    regions = []
    for x, y, cw, ch, count in stats[1:]:
        if not (.15 <= cw/w <= .52 and .018 <= ch/h <= .36 and count/(cw*ch) >= .72):
            continue
        pixels = rgb[y:y+ch, x:x+cw]
        background = np.median(pixels.reshape(-1, 3), axis=0)
        uniform = (np.abs(pixels.astype(int)-background).max(2) <= 5).mean()
        if uniform < .75:
            continue
        regions.append(dict(bbox=[float(x/w), float(y/h), float((x+cw)/w), float((y+ch)/h)],
                            color=[round(float(c/255), 4) for c in background],
                            uniform_coverage=float(uniform)))
    return sorted(regions, key=lambda r: (r['bbox'][1], r['bbox'][0]))


def horizontal_rules(rgb, page):
    import cv2
    import numpy as np
    h, w = rgb.shape[:2]
    black = ((rgb.max(2) < 180) &
             (rgb.max(2).astype(int)-rgb.min(2).astype(int) <= 8)).astype('uint8')
    lines = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((1, round(.65*w)), 'uint8'))
    _, _, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    result = []
    for x, y, cw, ch, count in stats[1:]:
        thickness = ch*page.rect.height/h
        if not (.6 <= thickness <= 4 and cw >= .65*w and count/(cw*ch) >= .95):
            continue
        gray = float(np.median(rgb[y:y+ch, x:x+cw])/255)
        result.append(dict(bbox=[float(x/w), float(y/h), float((x+cw)/w), float((y+ch)/h)],
                           thickness_pt=round(float(thickness), 4), color=[round(gray, 4)]*3))
    return [r for r in result if not any(s is not r and
            abs(s['bbox'][0]-r['bbox'][0]) < .003 and
            abs(s['bbox'][2]-r['bbox'][2]) < .003 and
            abs(s['bbox'][1]-r['bbox'][1]) < .045 for s in result)]


def group_box(atoms):
    boxes = [a.get('block_bbox') or a.get('bbox') for a in atoms]
    if not all(b and len(b) == 4 for b in boxes):
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def raster_decorations(index, registry, exclusions=(), changed=()):
    """Return render-only panels, lateral rows, rules and ownership evidence."""
    import numpy as np
    import pymupdf
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return [], [], {}, []
    by_page = OrderedDict()
    for atom in index.atoms:
        by_page.setdefault(atom['page'], OrderedDict()).setdefault(atom['group'], []).append(atom)
    panels, rows, rules, evidence = [], [], {}, []
    with pymupdf.open(source) as pdf:
        for number, groups in by_page.items():
            if (not 0 <= number < len(pdf) or pdf[number].get_text().strip()
                    or not any(a.get('extraction_method') == 'ocr' for aa in groups.values() for a in aa)):
                continue
            page = pdf[number]; rgb = page_pixels(page)
            boxes = {g: group_box(aa) for g, aa in groups.items()}
            for n, region in enumerate(gray_regions(rgb), 1):
                frame = region['bbox']
                overlaps = [g for g, b in boxes.items() if intersection_fraction(b, frame) > .05]
                owned = [g for g in overlaps if intersection_fraction(boxes[g], frame) >= .9]
                row = dict(method=METHOD, page=number+1, bbox=frame, status='rejected')
                evidence.append(row)
                if len(owned) != 1 or owned != overlaps:
                    row['reason'] = 'requires_one_complete_ocr_group'; continue
                atoms = groups[owned[0]]
                if (sum(len(a['content'].strip()) for a in atoms) < 20 or any(
                        a['kind'] not in {'text', 'math'} or a.get('source_panel') or a['id'] in changed
                        or a.get('block_type') not in {'text', 'paragraph'}
                        or a['role'] not in {'text', 'paragraph', 'equation_inline'}
                        or a.get('extraction_method') != 'ocr' for a in atoms)):
                    row['reason'] = 'unsupported_or_edited_ocr_content'; continue
                pid = f'p{number+1:06d}-gray-note-{n:03d}'
                panel = dict(id=pid, bbox=frame, color=region['color'],
                             background_color=region['color'], method=METHOD)
                # A private geometry view supplies ownership to the existing
                # pairing checks. It never changes the saved source index.
                view = deepcopy(index.source)
                ids = {a['id'] for a in atoms}
                for atom in view['atoms']:
                    if atom['id'] in ids: atom['source_panel'] = panel
                proxy = SourceIndex(view)
                candidates, _ = panel_environments(proxy, registry)
                candidates = [p for p in candidates if p['anchor'] == 'source-'+pid]
                paired = side_note_rows(proxy, candidates, registry, exclusions, changed)
                if not paired or any(r['lo'] < paired[0]['hi'] and paired[0]['lo'] < r['hi'] for r in rows):
                    row['reason'] = 'no_unambiguous_adjacent_main_paragraph'; continue
                panels.extend(candidates); rows.extend(paired)
                row.update(status='rendered', group=owned[0], uniform_coverage=region['uniform_coverage'])
            h, w = rgb.shape[:2]
            for rule in horizontal_rules(rgb, page):
                frame = rule['bbox']; y = (frame[1]+frame[3])/2
                candidates, crossing = [], False
                for group, atoms in groups.items():
                    box = boxes[group]
                    if not box or min(box[2],frame[2]) <= max(box[0],frame[0]): continue
                    eligible = (all(a.get('extraction_method') == 'ocr' and a['id'] not in changed
                        and not a.get('source_panel') and a['kind'] in {'text','math','display_math'}
                        and a['role'] not in {'header','footer','page_number'} for a in atoms)
                        and all(a.get('block_type') in {'text','paragraph','paragraph_title','equation'} for a in atoms))
                    bounds = box
                    # Formula review uses padded full-width bands. A rule can
                    # lie in their blank tail; require actual pixels to prove
                    # that every remaining formula row is above the rule.
                    if eligible and box[1] < y < box[3] and all(a['kind']=='display_math' for a in atoms):
                        x0,x1=max(0,round(box[0]*w)),min(w,round(box[2]*w))
                        y0,y1=max(0,round(box[1]*h)),min(h,round(box[3]*h))
                        ink = rgb[y0:y1,x0:x1].max(2) < 180
                        a,z=max(0,int(frame[1]*h)-y0-2),min(y1-y0,int(frame[3]*h)-y0+3)
                        ink[a:z] = False
                        ys = np.flatnonzero(ink.sum(1) >= 3)
                        if len(ys): bounds = [box[0], (ys[0]+y0)/h, box[2], (ys[-1]+y0+1)/h]
                    if bounds[1] <= y <= bounds[3]: crossing = True
                    if not eligible: continue
                    # A padded formula may span the page; ordinary prose must
                    # fit inside the observed rule's horizontal interval.
                    if not (all(a['kind']=='display_math' for a in atoms) or
                            frame[0]-.003 <= box[0] < box[2] <= frame[2]+.003): continue
                    if max(rule['color']) <= .1 and 0 < frame[1]-bounds[3] < .025:
                        candidates.append((frame[1]-bounds[3], atoms[-1], 'after'))
                    if atoms[0].get('block_type') == 'paragraph_title' and 0 < box[1]-frame[3] < .025:
                        candidates.append((box[1]-frame[3], atoms[0], 'before'))
                if crossing or not candidates: continue
                _, atom, placement = min(candidates, key=lambda x:x[0])
                lo = index.prefix[index.ids[atom['id']]]
                hi = lo+atom['length']
                if any(r['lo'] < hi and lo < r['hi'] for r in [*registry,*exclusions]): continue
                if (atom['id'],placement) in rules: continue
                rules[atom['id'],placement] = dict(rule, method=METHOD, atom=atom['id'],
                    bbox=[v*d for v,d in zip(frame,[page.rect.width,page.rect.height]*2)],
                    placement=placement, page=number+1)
    return panels, rows, rules, evidence
