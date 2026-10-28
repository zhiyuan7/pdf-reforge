"""Locate short gray notes embedded in an OCR paragraph's final text atom.

Independent local OCR must match the note's exact letters and the neighboring
main line. Only rendering moves immutable source intervals; symbols and words
are never replaced with the local OCR transcription.
"""
from collections import OrderedDict
from pathlib import Path
import re

from book_pipeline.config import digest, fingerprint, read, save
from book_pipeline.publishing.raster_decorations import gray_regions, page_pixels, group_box

METHOD = 'exact_local_ocr_mixed_gray_note_v1'


def letters(text):
    return ''.join(c for c in text if c.isalnum())


def observed_lines(rgb, model):
    import cv2
    h, w = rgb.shape[:2]
    crop = cv2.resize(rgb[:, :, ::-1].copy(), None, fx=2, fy=2)
    crop = cv2.copyMakeBorder(crop, 30, 30, 30, 30, cv2.BORDER_REPLICATE)
    rows = model.ocr(crop)[0] or []
    return [dict(text=text, score=float(score),
                 bbox=[(min(p[0] for p in poly)-30)/2, (min(p[1] for p in poly)-30)/2,
                       (max(p[0] for p in poly)-30)/2, (max(p[1] for p in poly)-30)/2])
            for poly, (text, score) in rows]


def mixed_note_rows(index, registry, exclusions, changed, directory):
    import pymupdf
    from book_pipeline.publishing.image_search import local_ocr
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file(): return []
    groups = OrderedDict()
    for n, a in enumerate(index.atoms): groups.setdefault(a['group'], []).append(n)
    groups = list(groups.values())
    result = []
    runtime = None
    with pymupdf.open(source) as pdf:
        for number in sorted({a['page'] for a in index.atoms if a.get('extraction_method') == 'ocr'}):
            if not 0 <= number < len(pdf) or pdf[number].get_text().strip(): continue
            rgb = page_pixels(pdf[number]); h, w = rgb.shape[:2]
            for region in gray_regions(rgb):
                frame = region['bbox']
                if not .012 <= frame[3]-frame[1] <= .045: continue
                for positions, following in zip(groups, groups[1:]):
                    atoms = [index.atoms[n] for n in positions]
                    tail = [index.atoms[n] for n in following]
                    all_atoms = atoms+tail
                    box, after = group_box(atoms), group_box(tail)
                    if (any(a['page'] != number or a.get('extraction_method') != 'ocr'
                            or a['id'] in changed or a.get('source_panel') or a.get('recovery_issue')
                            or a['kind'] not in {'text','math'} or a['role'] not in {'text','paragraph','equation_inline'}
                            or a.get('block_type') not in {'text','paragraph'} for a in all_atoms)
                            or positions+following != list(range(positions[0],following[-1]+1))
                            or not box or not after or atoms[-1]['kind'] != 'text'
                            or not (box[0]+.2 < frame[0] < box[2] <= frame[2]+.01
                                    and abs(box[1]-frame[1]) < .008 and abs(box[3]-frame[3]) < .008
                                    and 0 <= after[1]-box[3] < .015
                                    and abs(after[0]-box[0]) < .07 and after[2] >= frame[2]-.015)):
                        continue
                    lo, hi = index.prefix[positions[0]], index.prefix[following[-1]+1]
                    if hi-lo > 1600 or any(r['lo'] < hi and lo < r['hi'] for r in [*registry,*exclusions,*result]): continue
                    runtime = runtime or local_ocr()
                    if runtime is None: return result
                    model, signature = runtime
                    x0,y0,x1,y1 = [round(v*d) for v,d in zip(frame,[w,h,w,h])]
                    key = fingerprint(dict(method=METHOD, pdf=digest(source), page=number,
                                           frame=frame, main=box, ocr=signature))[:24]
                    cache = Path(directory)/'mixed-note-ocr'/f'{key}.json'
                    if cache.is_file(): observed = read(cache)
                    else:
                        observed = dict(note=observed_lines(rgb[y0:y1,x0:x1],model),
                            main=observed_lines(rgb[max(0,y0-3):min(h,y1+3),round(box[0]*w):x0],model))
                        save(cache,observed)
                    note, main = observed['note'], observed['main']
                    if len(note) != 1 or len(main) != 1 or min(note[0]['score'],main[0]['score']) < .99: continue
                    literal = atoms[-1]['content']; word = letters(note[0]['text'])
                    if not 6 <= len(word) <= 80: continue
                    matches = [n for n,c in enumerate(literal) if c.isalnum() and letters(literal[n:]) == word]
                    if len(matches) != 1: continue
                    cut = matches[0]
                    prefix = [a['content'] if a['kind']=='text' else '' for a in atoms[:-1]]+[literal[:cut]]
                    anchors = [letters(t) for t in prefix if letters(t)]
                    observed_main = letters(main[0]['text'])
                    if (not anchors or len(anchors[0]) < 6 or len(anchors[-1]) < 3
                            or not observed_main.startswith(anchors[0]) or not observed_main.endswith(anchors[-1])
                            or re.search(r'[.!?。！？:：;；][\s”’）)]*$',literal[:cut].rstrip())
                            or not re.match(r'[a-z\u3400-\u9fff]',tail[0]['content'].lstrip())):
                        continue
                    main_end = (round(box[0]*w)+main[0]['bbox'][2])/w
                    gap = frame[0]-main_end
                    extent = max(after[2],frame[2])-min(box[0],after[0])
                    if not .008 <= gap <= .12: continue
                    note_lo = index.prefix[positions[-1]]+cut
                    result.append(dict(method=METHOD,lo=lo,hi=hi,page=number+1,
                        main_ranges=[[lo,note_lo],[index.prefix[positions[-1]+1],hi]],
                        note_range=[note_lo,index.prefix[positions[-1]+1]],targets=[],
                        note_width=(frame[2]-frame[0])/extent,gutter=gap/extent,note_left=False,
                        color=region['color'],source_boxes=[box,frame,after],
                        observed=observed,cache=str(cache)))
    return result
