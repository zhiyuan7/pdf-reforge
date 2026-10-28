"""Recover isolated colored note references using pixels and exact CPU OCR.

Both the full observed line and its isolated colored token must agree with one
unchanged text interval. No recognition output is substituted into the source.
"""
from pathlib import Path
import re

from book_pipeline.config import digest, fingerprint, read, save
from book_pipeline.ingestion.panels import intersection_fraction
from book_pipeline.publishing.mixed_notes import observed_lines
from book_pipeline.publishing.raster_decorations import gray_regions, page_pixels
from book_pipeline.publishing.raster_marks import colored_components

METHOD = 'exact_cpu_scan_note_reference_ink_v1'
DPI = 300


def reference_range(text, line, token):
    """Map a unique literal OCR line and its unique terminal token to offsets."""
    if (line['score'] < .99 or token['score'] < .99
            or sum(c.isalnum() for c in line['text']) < 8
            or not re.fullmatch(r'[A-Za-z0-9]{1,8}', token['text'])):
        return None
    # Whitespace may differ in OCR; all other characters must match exactly.
    pattern = r'\s*'.join(re.escape(c) for c in re.sub(r'\s+', '', line['text']))
    if not pattern: return None
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1: return None
    a, z = matches[0].span()
    hits = list(re.finditer(r'(?<!\w)' + re.escape(token['text']) + r'(?!\w)', text[a:z]))
    if len(hits) != 1: return None
    start, end = hits[0].span()
    # Bound scope to isolated terminal references; inline math is ineligible.
    if not re.fullmatch(r'[\s.!?。！？,;:）)\]]*', text[a+end:z]): return None
    return a+start, a+end


def raster_reference_ink(index, directory, exclusions=(), changed=()):
    import pymupdf
    from book_pipeline.publishing.image_search import local_ocr
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file() or index.source.get('translation'): return {}
    result, runtime, sha = {}, None, None
    with pymupdf.open(source) as pdf:
        for number in sorted({a['page'] for a in index.atoms if a.get('extraction_method') == 'ocr'}):
            if not 0 <= number < len(pdf) or pdf[number].get_text().strip(): continue
            page = pdf[number]
            regions = gray_regions(page_pixels(page))
            if not regions: continue
            import numpy as np
            pix = page.get_pixmap(dpi=DPI,alpha=False,colorspace=pymupdf.csRGB)
            rgb = np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
            h, w = rgb.shape[:2]
            components = colored_components(rgb, page)
            if not components: continue
            for region in regions:
                frame = region['bbox']
                colors = [c for c in components if intersection_fraction(c['bbox'], frame) > .99
                          and .1 <= c['density'] <= .85]
                # Several colored components may be a formula or illustration.
                if len(colors) != 1: continue
                c = colors[0].copy()
                x,y,z,t = c['pixels']
                pixels = rgb[y:t,x:z].reshape(-1,3)
                chroma = pixels.max(1).astype(int)-pixels.min(1).astype(int)
                # Antialias fringes mix ink with gray paper. Sample the most
                # saturated quarter to estimate the original colored ink.
                saturated = chroma[chroma > 70]
                core = pixels[chroma >= np.quantile(saturated,.75)]
                c['color'] = [float(v/255) for v in np.median(core,axis=0)]
                owners = [a for a in index.atoms if a['page'] == number and a.get('bbox')
                          and intersection_fraction(c['bbox'], a['bbox']) > .05]
                # OCR splits inline math into atoms sharing the same box. The
                # exact line match below, rather than box order, decides ownership.
                if (not owners or any(a.get('extraction_method') != 'ocr' or a['id'] in changed
                        or a.get('recovery_issue') or a.get('source_text_runs')
                        or a['kind'] not in {'text', 'math'}
                        or a.get('role') not in {'text','paragraph','equation_inline'}
                        or a.get('block_type') not in {'text', 'paragraph'}
                        or intersection_fraction(a['bbox'],frame) < .9 for a in owners)):
                    continue
                if any(r['lo'] < index.prefix[index.ids[a['id']]+1]
                       and index.prefix[index.ids[a['id']]] < r['hi'] for a in owners for r in exclusions):
                    continue
                runtime = runtime or local_ocr()
                if runtime is None: return result
                model, signature = runtime; sha = sha or digest(source)
                key = fingerprint(dict(method=METHOD, dpi=DPI, pdf=sha, page=number, frame=frame,
                    component=c, owners=[(a['id'],a['content'],a['bbox']) for a in owners], ocr=signature))[:24]
                cache = Path(directory)/'reference-ink-ocr'/f'{key}.json'
                if cache.is_file(): observed = read(cache)
                else:
                    x,y,z,t = c['pixels']
                    x0,y0,x1,y1 = [round(v*d) for v,d in zip(frame,[w,h,w,h])]
                    # The component owns the token. The independent row owns its
                    # surrounding words; neither can assign a nearby equation.
                    pad = max(3, round((t-y)*.4))
                    observed = dict(line=observed_lines(rgb[max(y0,y-pad):min(y1,t+pad),x0:x1],model),
                        token=observed_lines(rgb[max(y0,y-2):min(y1,t+2),max(x0,x-2):min(x1,z+2)],model))
                    save(cache, observed)
                if len(observed['line']) != 1 or len(observed['token']) != 1: continue
                line, token = observed['line'][0], observed['token'][0]
                # The token must also be at the observed line's right edge.
                if abs((round(frame[0]*w)+line['bbox'][2])-c['pixels'][2]) > 1.5*(c['pixels'][3]-c['pixels'][1]): continue
                hits = [(a, reference_range(a['content'],line,token)) for a in owners if a['kind'] == 'text']
                hits = [(a,r) for a,r in hits if r is not None]
                if len(hits) != 1: continue
                atom, (start,end) = hits[0]
                result.setdefault(atom['id'],[]).append(dict(method=METHOD,atom=atom['id'],page=number+1,
                    start=start,end=end,color=c['color'],bbox=c['bbox'],frame=frame,
                    quote=atom['content'][start:end],observed=observed,cache=str(cache)))
    return result
