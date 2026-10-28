"""Render-only recovery of isolated colored scan furniture.

Pixels prove the mark and its color; OCR geometry proves its owner. Small
source crops preserve the actual shape instead of guessing a Unicode glyph.
No text, formula, extraction cache or semantic registry is rewritten.
"""
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
import re

from book_pipeline.ingestion.panels import intersection_fraction
from book_pipeline.publishing.raster_decorations import page_pixels, group_box

METHOD = 'raster_isolated_colored_furniture_v1'


def black_terminal_squares(rgb, page, eligible, page_groups):
    """An isolated square must follow a visible period on the final ink line.

    OCR can omit black example end marks anywhere along a short last line.
    Shape alone cannot distinguish them from a formula or an illustration:
    require unique unchanged prose ownership, the actual terminal dot and
    blank space to the right and below within the owning source rectangle.
    """
    import cv2
    import numpy as np
    h, w = rgb.shape[:2]
    black = ((rgb.max(2) < 160) &
             (rgb.max(2).astype(int)-rgb.min(2).astype(int) < 35)).astype('uint8')
    _, _, stats, _ = cv2.connectedComponentsWithStats(black, 8)
    result = {}
    for x, y, cw, ch, area in stats[1:]:
        width, height = cw*page.rect.width/w, ch*page.rect.height/h
        if not (3 <= width <= 8 and 3 <= height <= 8
                and .85 <= width/height <= 1.15 and area/(cw*ch) >= .9):
            continue
        box = [float(x/w), float(y/h), float((x+cw)/w), float((y+ch)/h)]
        owners = [g for g, atoms in page_groups.items() if group_box(atoms)
                  and intersection_fraction(box, group_box(atoms)) > .05]
        if len(owners) != 1 or owners[0] not in eligible:
            continue
        atoms, bounds, _, _ = eligible[owners[0]]
        if (atoms[-1]['kind'] != 'text' or not atoms[-1]['content'].rstrip().endswith('.')
                or any(a.get('source_panel') or a['kind'] == 'display_math'
                       or re.search('[■□▪]', a['content']) for a in atoms)
                or intersection_fraction(box, bounds) < .99):
            continue
        # A tiny detached period directly before the square, aligned at its
        # baseline. No inferred punctuation or matching mathematical token.
        dots = [(dx, dy, dw, dh) for dx, dy, dw, dh, da in stats[1:]
                if .3 <= dw*page.rect.width/w <= 2
                and .3 <= dh*page.rect.height/h <= 2 and da/(dw*dh) >= .45
                and 1 <= (x-dx-dw)*page.rect.width/w <= 10
                and abs(dy+dh-y-ch)*page.rect.height/h <= 1.5]
        if len(dots) != 1:
            continue
        dx, dy, dw, dh = dots[0]
        x0, y0 = max(0, round(bounds[0]*w)), max(0, round(bounds[1]*h))
        x1, y1 = min(w, round(bounds[2]*w)), min(h, round(bounds[3]*h))
        # The square and period must end this group's last visible line.
        # A bullet inside a sentence or a math square before more text fails.
        if (black[max(y0, y-ch//2):min(y1, y+ch+2), x+cw+2:x1].any()
                or black[y+ch+max(3, round(ch*.8)):y1, x0:x1].any()
                or black[max(y, dy):min(y+ch, dy+dh), dx+dw+1:x-1].any()):
            continue
        candidate = dict(bbox=box, pixels=[int(x),int(y),int(x+cw),int(y+ch)],
                         width_pt=float(width), height_pt=float(height), color=[0.,0.,0.],
                         inline=True, gap_bp=float((x-dx-dw-1)*page.rect.width/w),
                         depth_bp=float((y+ch+1-dy-dh)*page.rect.height/h))
        result.setdefault(atoms[-1]['id'], []).append(candidate)
    return {atom: matches[0] for atom, matches in result.items() if len(matches) == 1}


def colored_components(rgb, page):
    import cv2
    import numpy as np
    h, w = rgb.shape[:2]
    mask = ((rgb.max(2).astype(int)-rgb.min(2).astype(int) > 70)
            & (rgb.min(2) < 160)).astype('uint8')
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for x, y, cw, ch, count in stats[1:]:
        width, height = cw*page.rect.width/w, ch*page.rect.height/h
        if not (2 <= width <= 24 and 2 <= height <= 14 and count >= 6):
            continue
        pixels = rgb[y:y+ch,x:x+cw][mask[y:y+ch,x:x+cw].astype(bool)]
        components.append(dict(bbox=[float(x/w),float(y/h),float((x+cw)/w),float((y+ch)/h)],
            pixels=[int(x),int(y),int(x+cw),int(y+ch)], width_pt=float(width),height_pt=float(height),
            density=float(count/(cw*ch)),color=[float(c/255) for c in np.median(pixels,axis=0)]))
    return components


def raster_marks(index, wrappers, directory, exclusions=(), changed=()):
    import numpy as np
    import pymupdf
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return {}, {}, []
    groups = OrderedDict()
    for atom in index.atoms:
        groups.setdefault(atom['page'], OrderedDict()).setdefault(atom['group'], []).append(atom)
    marks, labels, evidence = {}, {}, []
    # Open once for all OCR pages, never inspect live-text pages as scans.
    with pymupdf.open(source) as pdf:
        for number, page_groups in groups.items():
            from book_pipeline.ingestion.scan_panels import raster_page
            if not 0 <= number < len(pdf) or not raster_page(pdf[number]):
                continue
            page = pdf[number]
            eligible = {}
            for group, atoms in page_groups.items():
                lo = index.prefix[index.ids[atoms[0]['id']]]
                hi = index.prefix[index.ids[atoms[-1]['id']]+1]
                if (any(a.get('extraction_method') != 'ocr' or a['id'] in changed
                        or a['kind'] not in {'text','math','display_math'}
                        or a.get('block_type') not in {'text','paragraph','paragraph_title','equation'}
                        or a.get('role') in {'header','footer','page_number'} for a in atoms)
                        or any(r['lo'] < hi and lo < r['hi'] for r in exclusions)):
                    continue
                box = group_box(atoms)
                if box: eligible[group] = (atoms,box,lo,hi)
            if not eligible: continue
            rgb = page_pixels(page);h,w = rgb.shape[:2]
            for atom, candidate in black_terminal_squares(rgb, page, eligible, page_groups).items():
                last = index.atoms[index.ids[atom]]
                mark = make_mark(candidate, rgb, directory, last, None)
                mark['method'] = 'raster_period_owned_black_terminal_square_v1'
                mark.update({k:candidate[k] for k in ('inline','gap_bp','depth_bp')})
                marks[atom] = mark
                evidence.append(mark)
            components = colored_components(rgb,page)
            claimed = set()
            for wrapper in wrappers:
                last = index.atoms[index.ids[index.address(wrapper['hi'],end=True)['atom']]]
                if last['page'] != number or last['group'] not in eligible: continue
                panel = last.get('source_panel',{});frame = panel.get('bbox')
                if not frame or wrapper['anchor'] != 'source-'+panel.get('id',''): continue
                # Every member must remain unchanged and fully owned.
                members = [a for a in index.atoms if wrapper['lo'] <= index.prefix[index.ids[a['id']]] < wrapper['hi']]
                if any(a['group'] not in eligible for a in members): continue
                candidates = [c for c in components if 3 <= c['width_pt'] <= 10
                    and 3 <= c['height_pt'] <= 10 and .18 <= c['density'] <= 1
                    and intersection_fraction(c['bbox'],frame) > .99
                    and c['bbox'][0] > frame[0]+.94*(frame[2]-frame[0])
                    and c['bbox'][1] > frame[1]+.5*(frame[3]-frame[1])
                    and .001 < frame[3]-c['bbox'][3] < .015
                    and max(abs(a-b) for a,b in zip(c['color'],panel['color'])) < .3]
                # Never duplicate an already recognized decoration or a source
                # crop/image. Ordinary math symbols are not furniture owners.
                if len(candidates)!=1 or any(re.search('[♣♧♠♤♥♡♦♢■□▪◇◆]',a['content']) for a in members): continue
                candidate = candidates[0]
                # A body OCR text box touching the crop makes ownership unsafe;
                # full-width display bands are padded, so test nearby black ink.
                x,y,z,t = candidate['pixels']
                vicinity = rgb[max(0,y-3):min(h,t+3),max(0,x-3):min(w,z+3)]
                dark = (vicinity.max(2)<140) & (vicinity.max(2).astype(int)-vicinity.min(2).astype(int)<35)
                if dark.any(): continue
                marks[last['id']] = make_mark(candidate,rgb,directory,last,wrapper['anchor'])
                marks[last['id']].update(
                    right_bp=max(1.,(frame[2]-candidate['bbox'][2]-1/w)*page.rect.width),
                    bottom_bp=max(1.,(frame[3]-candidate['bbox'][3]-1/h)*page.rect.height))
                evidence.append(marks[last['id']]);claimed.add(tuple(candidate['pixels']))
            for group,(atoms,box,lo,hi) in eligible.items():
                if any(a.get('source_panel') for a in atoms): continue
                # A solid colored square at the right end of a prose group's
                # final source line. Inline explanations, diagrams and formulas
                # cannot acquire this attachment.
                last = atoms[-1]
                if last['kind'] != 'text' or any(a['kind']=='display_math' for a in atoms): continue
                candidates = [c for c in components if tuple(c['pixels']) not in claimed
                    and 3 <= c['width_pt'] <= 10 and 3 <= c['height_pt'] <= 10
                    and .85 <= c['width_pt']/c['height_pt'] <= 1.15 and c['density'] >= .9
                    and c['bbox'][0] >= .8 and intersection_fraction(c['bbox'],box) > .95
                    and abs(c['bbox'][3]-box[3]) < .012]
                if len(candidates)==1 and not any(re.search('[■□▪]',a['content']) for a in atoms):
                    c=candidates[0]
                    # No other group may own this source square.
                    overlaps = [g for g, aa in page_groups.items() if group_box(aa)
                        and intersection_fraction(c['bbox'],group_box(aa)) > .05]
                    if overlaps == [group]:
                        marks[last['id']] = make_mark(c,rgb,directory,last,None)
                        evidence.append(marks[last['id']])
                # A proof prefix is styled only when source color is confined
                # to precisely its expected first-line character width.
                first=atoms[0];m=re.match(r'^(证明|Proof)(?=\s|[:：])',first['content'])
                if first['kind']!='text' or not m or box[3]-box[1]>.04: continue
                label=m.group();cjk=label=='证明'
                label_width=(box[3]-box[1])*page.rect.height/page.rect.width*(len(label) if cjk else 1.7)
                prefix=[c for c in components if box[0]-.002 <= c['bbox'][0]
                    and c['bbox'][2] <= box[0]+label_width*1.35
                    and box[1] <= c['bbox'][1] < c['bbox'][3] <= box[3]]
                if not prefix: continue
                x0=min(c['bbox'][0] for c in prefix);x1=max(c['bbox'][2] for c in prefix)
                if not .55*label_width <= x1-x0 <= 1.25*label_width or x0-box[0] > .015: continue
                color=np.median([c['color'] for c in prefix],axis=0).tolist()
                if any(max(abs(a-b) for a,b in zip(c['color'],color)) > .2 for c in prefix): continue
                # Any additional colored ink on the line invalidates the
                # prefix mapping (e.g. a fully colored sentence or title).
                if any(c not in prefix and intersection_fraction(c['bbox'],box)>.5 for c in components): continue
                # Observe the gap to the following black text, consuming the
                # OCR separator once. CJK typesetting otherwise discards it.
                y0,y1=max(0,round(box[1]*h)),min(h,round(box[3]*h))
                x0pix,x1pix=min(w,round(x1*w)+2),min(w,round(box[2]*w))
                following=rgb[y0:y1,x0pix:x1pix]
                black=(following.max(2)<130)&(following.max(2).astype(int)-following.min(2).astype(int)<35)
                columns=np.flatnonzero(black.sum(0)>=2)
                gap=(float(columns[0])+x0pix-x1*w)*page.rect.width/w if len(columns) else 0
                if not 2 <= gap <= 20: continue
                end=len(label)
                while end<len(first['content']) and first['content'][end].isspace():end+=1
                labels[first['id']]=dict(method=METHOD,atom=first['id'],start=0,end=end,text_end=len(label),gap_after_bp=gap,
                    color=color,page=number+1,kind='label',bbox=[x0,box[1],x1,box[3]])
                evidence.append(labels[first['id']])
    return marks,labels,evidence


def make_mark(candidate,rgb,directory,atom,panel):
    import pymupdf
    x,y,z,t=candidate['pixels']
    # Keep the faint antialiased fringe outside the saturation component.
    h,w=rgb.shape[:2]
    x,y,z,t=max(0,x-1),max(0,y-1),min(w,z+1),min(h,t+1)
    crop=rgb[y:t,x:z].copy()
    png=pymupdf.Pixmap(pymupdf.csRGB,z-x,t-y,crop.tobytes(),False).tobytes('png')
    relative='images/raster-mark-'+sha256(png).hexdigest()[:20]+'.png'
    target=Path(directory)/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(png)
    return dict(method=METHOD,kind='mark',atom=atom['id'],page=atom['page']+1,panel=panel,
        bbox=candidate['bbox'],color=candidate['color'],
        height_pt=candidate['height_pt']*(t-y)/(candidate['pixels'][3]-candidate['pixels'][1]),image=relative)
