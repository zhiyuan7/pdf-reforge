"""Restore OCR fallback end marks using visible native glyph ownership.

Only complete, unchanged prose groups can anchor mathematical gaps. The live
PDF proves punctuation and any isolated bitmap square; OCR and formula bytes
remain untouched. Hidden text on scans supplies no evidence here.
"""
from pathlib import Path
import re

from book_pipeline.ingestion.panels import intersection_fraction
from book_pipeline.publishing.native_body_styles import prose_letters
from book_pipeline.publishing.native_lateral import native_groups

METHOD = 'complete_live_prose_owned_ocr_terminal_marks_v1'


def group_ending(atoms, glyphs):
    """Return a uniquely owned terminal period, never infer a formula."""
    if not glyphs or glyphs[-1]['c'] != '.':
        return None
    if any(re.search(r'[■□▪]|\\(?:blacksquare|square|Box|qed|qedhere)\b', a['content']) for a in atoms):
        return None
    if sum(len(prose_letters(a['content'])) for a in atoms if a['kind'] == 'text') < 30:
        return None
    pieces = []
    for atom in atoms:
        if atom['kind'] == 'text':
            pieces.append(re.escape(prose_letters(atom['content'])))
        elif atom['kind'] == 'math':
            if not pieces or pieces[-1] != '(.+?)':
                pieces.append('(.+?)')
        else:
            return None
    last = atoms[-1]
    missing = last['kind'] == 'math'
    if missing:
        # Never double a period already inside a math wrapper, or overwrite
        # another printed punctuation mark. Existing ellipses also abstain.
        if re.search(r'(?:[.,;:!?]|\\(?:dots|ldots|cdots))\s*[}\s]*$', last['content']):
            return None
        pieces.append(r'\.')
    elif not last['content'].rstrip().endswith('.'):
        return None
    pattern = ''.join(pieces)
    observed = prose_letters(''.join(g['c'] for g in glyphs))
    first = re.fullmatch(pattern, observed)
    final = re.fullmatch(pattern.replace('(.+?)', '(.+)'), observed)
    if not first or not final or first.regs != final.regs:
        return None
    return dict(missing=missing, glyph=glyphs[-1])


def native_ocr_marks(index, directory, exclusions=(), changed=(), rendered_math=None):
    import pymupdf
    from book_pipeline.publishing.raster_marks import make_mark
    from book_pipeline.publishing.raster_decorations import page_pixels

    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return {}, {}, []
    groups = native_groups(index, allow_ocr=True)
    punctuation, marks, evidence = {}, {}, []
    with pymupdf.open(source) as pdf:
        cached = {}
        for group in groups:
            atoms = group['atoms']; number = group['page']
            if (not 0 <= number < len(pdf)
                    or any(a.get('extraction_method') != 'ocr' or a['id'] in changed
                           or a.get('recovery_issue') or a['kind'] not in {'text', 'math'}
                           or a.get('block_type') not in {'text', 'paragraph'}
                           or a.get('role') not in {'text', 'paragraph'} for a in atoms)
                    or any(r['lo'] < group['hi'] and group['lo'] < r['hi'] for r in exclusions)):
                continue
            page = pdf[number]
            if number not in cached:
                trace = page.get_texttrace()
                # A scan's hidden OCR must never authorize visible additions.
                if not trace or any(s['type'] == 3 or s.get('opacity', 1) == 0 for s in trace):
                    cached[number] = None
                else:
                    glyphs = [c for b in page.get_text('rawdict')['blocks']
                              for line in b.get('lines', []) if line.get('dir') == (1., 0.)
                              for span in line['spans'] for c in span['chars']]
                    cached[number] = (glyphs, page_pixels(page))
            if cached[number] is None:
                continue
            glyphs, rgb = cached[number]
            x, y, z, t = group['bbox']; width, height = page.rect.width, page.rect.height
            selected = [g for g in glyphs if x*width <= g['origin'][0] <= z*width
                        and y*height <= g['origin'][1] <= t*height and g['c'].strip()]
            ending = group_ending(atoms, selected)
            if not ending:
                continue
            dot = ending['glyph']; last = atoms[-1]
            h, w = rgb.shape[:2]
            # The visible dot must have ink in its own glyph rectangle.
            dx, dy, dz, dt = dot['bbox']
            dot_crop = rgb[max(0, int(dy/height*h)):min(h, int(dt/height*h)+1),
                           max(0, int(dx/width*w)):min(w, int(dz/width*w)+1)]
            if not dot_crop.size or not (dot_crop.max(2) < 160).any():
                continue
            rendered = (rendered_math or {}).get(last['id'], '')
            if ending['missing'] and not re.search(r'[.,;:!?]\s*[}\s]*$', rendered):
                row = dict(method=METHOD, kind='punctuation', atom=last['id'],
                           text='.', bbox=list(dot['bbox']), page=number+1)
                punctuation[last['id']] = row; evidence.append(row)
            # Small bitmap squares occur in otherwise native PDFs. Require a
            # black filled image at the right edge of the final source line.
            candidates = []
            for image in page.get_image_info():
                b = image['bbox']; iw, ih = b[2]-b[0], b[3]-b[1]
                box = [b[0]/width, b[1]/height, b[2]/width, b[3]/height]
                if (not 3 <= iw <= 9 or not 3 <= ih <= 9 or not .95 <= iw/ih <= 1.05
                        or z-x < .3 or b[0] < dz+2 or abs(b[2]-z*width) > 4
                        or abs(b[3]-dot['origin'][1]) > 1.5
                        or intersection_fraction(box, group['bbox']) < .98
                        or any(a['group'] != group['group'] and a['page'] == number
                               and a.get('bbox') and intersection_fraction(box, a['bbox']) > .05
                               for a in index.atoms)):
                    continue
                px = [round(box[0]*w), round(box[1]*h), round(box[2]*w), round(box[3]*h)]
                crop = rgb[px[1]+1:px[3]-1, px[0]+1:px[2]-1]
                if not crop.size or (crop.max(2) < 100).mean() < .98:
                    continue
                candidates.append(dict(bbox=box, pixels=px, height_pt=ih, color=[0., 0., 0.]))
            if len(candidates) == 1:
                mark = make_mark(candidates[0], rgb, directory, last, None)
                mark['method'] = METHOD
                marks[last['id']] = mark; evidence.append(mark)
    return punctuation, marks, evidence
