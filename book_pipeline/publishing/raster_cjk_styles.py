"""CPU evidence for local Heiti emphasis in calligraphic scan notes.

Two OCR readings own a complete CJK line, a third independently owns its
emphasis phrase. Blank-column glyph boundaries and competing installed font
families establish the style. No native counterpart or expected source word
is used to recognize the scan.
"""
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
import re

from book_pipeline.config import ROOT, digest, fingerprint, read, save
from book_pipeline.ingestion.panels import intersection_fraction
from book_pipeline.publishing.raster_styles import METHOD as STYLE_METHOD, eligible_items

METHOD = 'cpu_gray_note_cjk_family_contrast_v1'
DPI = 300
NORMAL = ('FandolKai-Regular', 'FandolFang-Regular', 'FandolSong-Regular')
EMPHASIS = ('FandolHei-Regular', 'FandolHei-Bold')
FONTDIR = ROOT/'.tools/.TinyTeX/texmf-dist/fonts/opentype/public/fandol'


@lru_cache(maxsize=512)
def glyph_templates(char):
    import numpy as np
    import pymupdf
    from book_pipeline.publishing.raster_body_styles import trim_ink
    result = {}
    for name in (*NORMAL, *EMPHASIS):
        font = pymupdf.Font(fontfile=str(FONTDIR/(name+'.otf')))
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=130, height=130)
            writer = pymupdf.TextWriter(page.rect)
            writer.append((15, 90), char, font=font, fontsize=60)
            writer.write_text(page)
            pix = page.get_pixmap(colorspace=pymupdf.csGRAY)
            result[name] = trim_ink(255-np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width))
    return result


def glyph_grid(mask, count):
    """A bounded equal-advance hypothesis must meet real blank columns."""
    import numpy as np
    pitch = mask.shape[1]/count
    if min(mask.shape) < 12 or pitch < 12:
        return []
    cuts = np.where(~(mask > 0).any(0))[0]
    if not len(cuts):
        return []
    edges = [0]
    for n in range(1, count):
        cut = int(cuts[np.argmin(np.abs(cuts-n*pitch))])
        if abs(cut-n*pitch) > .18*pitch or not .65*pitch <= cut-edges[-1] <= 1.35*pitch:
            return []
        edges.append(cut)
    edges.append(mask.shape[1])
    return list(zip(edges, edges[1:])) if .65*pitch <= edges[-1]-edges[-2] <= 1.35*pitch else []


def glyph_scores(mask, char):
    import cv2
    import numpy as np
    from book_pipeline.publishing.raster_body_styles import trim_ink
    mask = trim_ink(mask)
    if mask is None or min(mask.shape) < 8:
        return {}
    return {name: float(cv2.matchTemplate(mask.astype(np.float32),
                cv2.resize(template, (mask.shape[1], mask.shape[0])).astype(np.float32),
                cv2.TM_CCOEFF_NORMED)[0, 0])
            for name, template in glyph_templates(char).items() if template is not None}


def emphasis_spans(scores):
    """Every heavy glyph competes with regular print, beside proven Kai ink."""
    heavy, ordinary = [], []
    for row in scores:
        if set(row) != set((*NORMAL, *EMPHASIS)):
            heavy.append(False); ordinary.append(False)
            continue
        sans = max(row[name] for name in EMPHASIS)
        heavy.append(sans >= .75 and sans-max(row[name] for name in NORMAL) >= .18)
        ordinary.append(row['FandolKai-Regular'] >= .65 and row['FandolKai-Regular']-sans >= .18)
    if sum(ordinary) < 2:
        return []
    runs = []
    start = None
    for n, active in enumerate([*heavy, False]):
        if active and start is None:
            start = n
        elif not active and start is not None:
            if 2 <= n-start <= 20 and any(ordinary[k] for k in range(max(0,start-2), min(len(ordinary),n+2))):
                runs.append((start, n))
            start = None
    return runs


def analyze_note(rgb, model):
    import cv2
    import numpy as np
    from book_pipeline.publishing.raster_body_styles import complete_line_box, prose_ink
    rows = []
    for polygon, (detected, score) in model.ocr(rgb[:, :, ::-1].copy())[0] or []:
        if score < .99 or not re.fullmatch(r'[\u3400-\u9fff]{6,60}[.。]?', detected):
            continue
        h, w = rgb.shape[:2]
        x, y = [max(0, round(min(p[k] for p in polygon))-2) for k in (0, 1)]
        z, t = [min(d, round(max(p[k] for p in polygon))+2) for k, d in ((0,w),(1,h))]
        x, y, z, t = complete_line_box(rgb, [x,y,z,t])
        crop = rgb[y:t, x:z]
        reread = model.ocr(crop[:, :, ::-1].copy(), det=False)[0] or []
        if (len(reread) != 1 or reread[0][1] < .99
                or not re.fullmatch(r'[\u3400-\u9fff]{6,60}[.。]?', reread[0][0])
                or reread[0][0].rstrip('.。') != detected.rstrip('.。')):
            continue
        text, confidence = reread[0]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        iy, ix = np.where(mask > 0)
        if not len(ix):
            continue
        left, top = int(ix.min()), int(iy.min())
        tight = mask[top:int(iy.max())+1, left:int(ix.max())+1]
        letters = text
        if text[-1] in '.。':
            tight, stripped = prose_ink(tight, text[:-1]+'.')
            if stripped.endswith('.'):
                continue
            letters = text[:-1]
        grid = glyph_grid(tight, len(letters))
        if not grid:
            continue
        scores = [glyph_scores(tight[:,a:b], char) for char, (a,b) in zip(letters, grid)]
        words = []
        for a, b in emphasis_spans(scores):
            lo, hi = grid[a][0]+left, grid[b-1][1]+left
            # Read the independently segmented phrase from original pixels;
            # neither the source atom nor the line reading supplies its text.
            answer = model.ocr(crop[:,lo:hi,::-1].copy(), det=False)[0] or []
            if len(answer) == 1 and answer[0][1] >= .99 and answer[0][0] == letters[a:b]:
                words.append(dict(text=answer[0][0], score=float(answer[0][1]), start=a, end=b,
                    bbox=[x+lo,y,x+hi,t], glyph_scores=scores[a:b]))
        if words:
            rows.append(dict(text=text, score=float(confidence), detection_score=float(score),
                             bbox=[x,y,z,t], words=words, glyph_scores=scores))
    return rows


def owned_note_ranges(observations, items, note, dimensions):
    from book_pipeline.publishing.raster_body_styles import literal_span
    result = {}
    for line in observations:
        if min(line['score'], line['detection_score']) < .99:
            continue
        box = [v/d for v,d in zip(line['bbox'], dimensions)]
        if intersection_fraction(box, note) < .95:
            continue
        owners = [(i, literal_span(i['text'], line['text'])) for i in items
                  if intersection_fraction(box, i['bbox']) >= .9]
        owners = [(i, span) for i,span in owners if span is not None]
        if len(owners) != 1:
            continue
        item, (lo, hi) = owners[0]
        confirmed = emphasis_spans(line['glyph_scores'])
        for word in line['words']:
            a, b = word['start'], word['end']
            if (word['score'] < .99 or line['text'][a:b] != word['text']
                    or (a,b) not in confirmed or word['glyph_scores'] != line['glyph_scores'][a:b]
                    or intersection_fraction(word['bbox'],line['bbox']) < .99):
                continue
            if any(set(s) != set((*NORMAL,*EMPHASIS)) or max(s[f] for f in EMPHASIS) < .75
                   or max(s[f] for f in EMPHASIS)-max(s[f] for f in NORMAL) < .18 for s in word['glyph_scores']):
                continue
            span = literal_span(item['text'][lo:hi], word['text'])
            if span is None:
                continue
            start, end = (n+lo for n in span)
            row = dict(method=STYLE_METHOD, start=start, end=end, quote=item['text'][start:end],
                       italic=False, bold=True, cjk_family='sans', observed_bbox=[v/d for v,d in zip(word['bbox'], dimensions)],
                       glyph_scores=word['glyph_scores'], evidence=METHOD)
            if row not in result.get(item['atom'], []):
                result.setdefault(item['atom'], []).append(row)
    return {a: sorted(r, key=lambda v:v['start']) for a,r in result.items()}


def recover_cjk_note_styles(index, original, directory, excluded=()):
    import numpy as np
    import pymupdf
    from book_pipeline.publishing.image_search import local_ocr
    from book_pipeline.publishing.raster_decorations import gray_regions, page_pixels
    candidates = eligible_items(index, original, excluded)
    candidates = {p:items for p,items in candidates.items() if any(re.search(r'[\u3400-\u9fff]',i['text']) for i in items)}
    report = dict(method=METHOD, status='not_needed', records=[])
    source = index.source.get('source_pdf')
    if not candidates or not source or not Path(source).is_file() or not all((FONTDIR/(f+'.otf')).is_file() for f in (*NORMAL,*EMPHASIS)):
        return {}, report
    runtime = local_ocr()
    if runtime is None:
        report['status'] = 'unavailable_local_ocr'
        return {}, report
    model, signature = runtime
    pdf_sha = digest(source)
    fonts = [digest(FONTDIR/(f+'.otf')) for f in (*NORMAL,*EMPHASIS)]
    styles = {}
    with pymupdf.open(source) as pdf:
        for number, items in candidates.items():
            if not 0 <= number < len(pdf) or pdf[number].get_text().strip():
                continue
            page = pdf[number]
            pix = page.get_pixmap(dpi=DPI,alpha=False,colorspace=pymupdf.csRGB)
            for region in gray_regions(page_pixels(page)):
                note = region['bbox']
                batch = [i for i in items if intersection_fraction(i['bbox'],note) >= .9]
                if not batch or any(.05 < intersection_fraction(i['bbox'],note) < .9 for i in items):
                    continue
                key = fingerprint(dict(method=METHOD,pdf=pdf_sha,page=number,note=note,dpi=DPI,ocr=signature,fonts=fonts,
                                       pymupdf=version('PyMuPDF'),opencv=__import__('cv2').__version__))[:24]
                cache = Path(directory)/'cjk-note-ocr'/(key+'.json')
                if cache.is_file():
                    observed = read(cache)
                else:
                    rgb = np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
                    x,y,z,t = [round(v*d) for v,d in zip(note,[pix.width,pix.height]*2)]
                    observed = analyze_note(rgb[y:t,x:z], model)
                    for line in observed:
                        for record in [line,*line['words']]:
                            record['bbox'] = [v+delta for v,delta in zip(record['bbox'],[x,y,x,y])]
                    save(cache,observed)
                dimensions = [pix.width,pix.height]*2
                ranges = owned_note_ranges(observed,batch,note,dimensions)
                styles.update(ranges)
                report['records'].append(dict(page=number+1,note_bbox=note,cache=str(cache),ranges=ranges))
    if styles:
        report['status'] = 'requires_visual_review'
    return styles, report
