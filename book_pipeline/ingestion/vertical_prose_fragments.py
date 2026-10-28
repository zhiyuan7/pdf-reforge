"""Image-only short vertical runs, used only within a complete prose witness.

Fragment strings cannot authorize transcription edits or formula ownership by
themselves. Adjacent single-glyph detections are reread together without text
hints; their initial recognition confidence does not establish agreement.
"""
import re

METHOD = 'blind_vertical_fragment_join_v1'
PROSE = re.compile(r'[\u3400-\u9fff]{2,60}')


def observe_fragments(rgb):
    import numpy as np
    from PIL import Image, ImageOps
    from book_pipeline.publishing.image_search import local_ocr
    runtime = local_ocr()
    if runtime is None:
        return []
    model, signature = runtime
    raw = model.ocr(rgb[:, :, ::-1].copy())[0] or []
    rows, glyphs = [], []
    h, w = rgb.shape[:2]
    for polygon, (text, score) in raw:
        p = np.asarray(polygon, dtype=float)
        if p.shape != (4, 2) or not np.isfinite(p).all():
            continue
        x, y = p.min(0); z, t = p.max(0)
        if not (0 <= x < z <= w and 0 <= y < t <= h):
            continue
        box = [float(x), float(y), float(z), float(t)]
        if PROSE.fullmatch(text) and .99 <= float(score) <= 1:
            pitch = (t-y)/len(text)
            if t-y >= 1.2*(z-x) and .4*(z-x) <= pitch <= 1.8*(z-x):
                rows.append(dict(text=text, score=float(score), bbox=box, reader_signature=signature))
        elif re.fullmatch(r'[\u3400-\u9fff]', text) and .75 <= (t-y)/(z-x) <= 1.4:
            glyphs.append(box)
    # Build maximal runs from geometry only, never from an expected phrase.
    chains = []
    for box in sorted(glyphs, key=lambda b: b[1]):
        a,b,c,d = box
        parents = []
        for chain in chains:
            x,y,z,t = chain[-1]
            pitch = ((d-b)+(t-y))/2
            if (.75 <= (d-b)/(t-y) <= 1.33 and .75 <= (c-a)/(z-x) <= 1.33
                    and abs((a+c-x-z)/2) <= .25*pitch and -.35*pitch <= b-t <= .5*pitch):
                parents.append(chain)
        if len(parents) == 1:
            parents[0].append(box)
        else:
            chains.append([box])
    for chain in chains:
        if not 2 <= len(chain) <= 5:
            continue
        box = [min(b[0] for b in chain),min(b[1] for b in chain),
               max(b[2] for b in chain),max(b[3] for b in chain)]
        x,y,z,t = [round(v) for v in box]
        crop = Image.fromarray(rgb[y:t,x:z])
        factor = max(1, round(210/crop.width))
        view = ImageOps.expand(crop.resize((crop.width*factor,crop.height*factor), Image.Resampling.NEAREST),
                               border=40,fill='white')
        readings = model.ocr(np.array(view)[:,:,::-1].copy())[0] or []
        readings = [r for r in readings if r[1][0]]
        if len(readings) != 1:
            continue
        _, (text, score) = readings[0]
        if PROSE.fullmatch(text) and len(text) == len(chain) and .99 <= float(score) <= 1:
            rows.append(dict(text=text,score=float(score),bbox=box,reader_signature=signature,
                             fragment_method=METHOD))
    return rows
