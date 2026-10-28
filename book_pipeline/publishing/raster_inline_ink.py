"""Color an existing OCR square only after its two neighbors match the scan.

The source glyph, text offsets and layout remain intact. Geometry and pixels
identify one solid colored square; independent CPU OCR proves the surrounding
letters. This is deliberately separate from missing terminal-mark recovery.
"""
from collections import OrderedDict
from pathlib import Path
import re

from book_pipeline.config import digest, fingerprint, read, save
from book_pipeline.ingestion.panels import intersection_fraction
from book_pipeline.publishing.mixed_notes import letters, observed_lines
from book_pipeline.publishing.raster_decorations import gray_regions, group_box, page_pixels
from book_pipeline.publishing.raster_marks import colored_components

METHOD = 'neighbor_verified_raster_square_ink_v1'


def raster_inline_ink(index, directory, exclusions=(), changed=()):
    import pymupdf
    from book_pipeline.publishing.image_search import local_ocr
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file(): return {}
    pages = OrderedDict()
    for atom in index.atoms:
        pages.setdefault(atom['page'], OrderedDict()).setdefault(atom['group'], []).append(atom)
    result, runtime, pdf_sha = {}, None, None
    with pymupdf.open(source) as pdf:
        for number, groups in pages.items():
            if not 0 <= number < len(pdf) or pdf[number].get_text().strip(): continue
            eligible = []
            for group, atoms in groups.items():
                box = group_box(atoms)
                lo = index.prefix[index.ids[atoms[0]['id']]]
                hi = index.prefix[index.ids[atoms[-1]['id']]+1]
                hits = [(a, m.start()) for a in atoms if a['kind'] == 'text'
                        for m in re.finditer('[■▪]', a['content'])]
                if (len(hits) != 1 or not box or not 0 < box[3]-box[1] <= .04
                        or any(a.get('extraction_method') != 'ocr' or a['id'] in changed
                            or a.get('recovery_issue') or a.get('source_text_runs')
                            or a['kind'] not in {'text', 'math'}
                            or a.get('block_type') not in {'text', 'paragraph'}
                            or a.get('role') not in {'text', 'paragraph', 'equation_inline'} for a in atoms)
                        or any(r['lo'] < hi and lo < r['hi'] for r in exclusions)):
                    continue
                eligible.append((group, atoms, box, *hits[0]))
            if not eligible: continue
            page = pdf[number]; rgb = page_pixels(page); h, w = rgb.shape[:2]
            squares = [c for c in colored_components(rgb, page)
                       if 3 <= c['width_pt'] <= 10 and 3 <= c['height_pt'] <= 10
                       and .85 <= c['width_pt']/c['height_pt'] <= 1.15 and c['density'] >= .9]
            notes = gray_regions(rgb)
            for group, atoms, box, atom, offset in eligible:
                candidates = [c for c in squares if intersection_fraction(c['bbox'], box) > .98]
                if len(candidates) != 1: continue
                c = candidates[0]
                # Shared/overlapping OCR groups cannot claim the same pixels.
                if any(g != group and intersection_fraction(c['bbox'], group_box(aa)) > .05
                       for g, aa in groups.items()): continue
                frames = [n['bbox'] for n in notes if intersection_fraction(c['bbox'], n['bbox']) > .99]
                if len(frames) > 1: continue
                frame = frames[0] if frames else box
                if frame[3]-frame[1] > .045: continue
                k = atoms.index(atom)
                before = letters(''.join(a['content'] for a in atoms[:k])+atom['content'][:offset])
                after = letters(atom['content'][offset+1:]+''.join(a['content'] for a in atoms[k+1:]))
                if min(len(before), len(after)) < 2: continue
                runtime = runtime or local_ocr()
                if runtime is None: return result
                model, signature = runtime
                pdf_sha = pdf_sha or digest(source)
                key = fingerprint(dict(method=METHOD, pdf=pdf_sha, page=number, frame=frame,
                    square=c['pixels'], before=before, after=after, ocr=signature))[:24]
                cache = Path(directory)/'inline-ink-ocr'/f'{key}.json'
                if cache.is_file(): observed = read(cache)
                else:
                    x, y, z, t = c['pixels']
                    x0, y0, x1, y1 = [round(v*d) for v,d in zip(frame, [w,h,w,h])]
                    # Exclude the square's antialias fringe from each neighbor.
                    if x-x0 < 8 or x1-z < 8: continue
                    observed = dict(left=observed_lines(rgb[max(0,y0-2):min(h,y1+2),max(0,x0):x-1], model),
                                    right=observed_lines(rgb[max(0,y0-2):min(h,y1+2),z+1:min(w,x1)], model))
                    save(cache, observed)
                left, right = observed['left'], observed['right']
                if len(left) != 1 or len(right) != 1: continue
                a, b = letters(left[0]['text']), letters(right[0]['text'])
                if (min(left[0]['score'], right[0]['score']) < .99 or min(len(a),len(b)) < 2
                        or not before.endswith(a) or not after.startswith(b)):
                    continue
                result[atom['id']] = [dict(method=METHOD, atom=atom['id'], page=number+1,
                    start=offset, end=offset+1, color=c['color'], bbox=c['bbox'],
                    frame=frame, observed=observed, cache=str(cache))]
    return result
