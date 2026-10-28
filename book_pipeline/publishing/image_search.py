"""A conservative, invisible search layer for otherwise unextracted captions.

Only standalone bottom prose in unchanged OCR image atoms is eligible. Math
labels, composite source crops and already extracted captions are left alone.
The PDF wrapper embeds the original raster stream without painting new text.
"""
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
import re

from book_pipeline.config import digest, fingerprint, read, save

METHOD = 'isolated_raster_caption_search_v2'
ROW_METHOD = 'source_bound_horizontal_row_caption_search_v1'


@lru_cache(maxsize=1)
def local_ocr():
    # Rendering must never download models or require a GPU/API credential.
    try:
        from mineru.model.registry import MINERU_4_MODELS_ONNX as repo
        from mineru.model.ocr.resources import PPOCRV6_DICT_PATH
        from mineru.model.ocr import pp_ocr_v6_onnx as runtime
    except ImportError:
        return None
    paths = [repo.ocr_det.local_path(), repo.ocr_rec.local_path(), Path(PPOCRV6_DICT_PATH)]
    if not all(p.is_file() for p in paths):
        return None
    from book_pipeline.ingestion.ocr_geometry import adapt_detector, METHOD as CROP_GEOMETRY
    signature = fingerprint(dict(models=[digest(p) for p in paths], crop_geometry=CROP_GEOMETRY,
                                 mineru=version('mineru'), onnxruntime=version('onnxruntime')))
    model = runtime.PPOCRv6ONNX(str(paths[0]), str(paths[1]), device='cpu',
                       enable_merge_det_boxes=False)
    adapt_detector(model)
    return model, signature


def caption_rows(rgb, rows):
    """Require plain prose, upright geometry and a blank strip above the row."""
    import numpy as np
    h, w = rgb.shape[:2]
    accepted = []
    for poly, (text, score) in rows or []:
        text = text.strip()
        prose = (re.fullmatch(r"[A-Za-z]+(?:[ '’-][A-Za-z]+)+[.!?]?", text)
                 or re.fullmatch(r'[\u3400-\u9fff]{3,40}[.。！？]?', text))
        p = np.asarray(poly, dtype=float)
        if not prose or not .995 <= float(score) <= 1 or p.shape != (4, 2) or not np.isfinite(p).all():
            continue
        x0, y0 = p.min(axis=0); x1, y1 = p.max(axis=0)
        if not (0 <= x0 < x1 <= w and .60*h <= y0 < y1 <= h
                and .05*h <= y1-y0 <= .24*h and x1-x0 >= .15*w
                and abs((x0+x1)/2-w/2) <= .25*w
                and abs(p[1, 1]-p[0, 1]) <= .15*(y1-y0)
                and abs(p[2, 1]-p[3, 1]) <= .15*(y1-y0)):
            continue
        gap = max(2, round(.025*h))
        strip = rgb[max(0, int(y0)-gap):int(y0)]
        if strip.size == 0 or (strip.min(axis=2) < 200).any():
            continue
        accepted.append(dict(text=text, score=float(score), bbox=[float(x0), float(y0), float(x1), float(y1)]))
    # Multi-line or ambiguous captions need a different ownership model.
    return accepted if len(accepted) == 1 else []


def searchable_raster(source, target, rows):
    """Write one image page; invisible glyphs retain their own observed bounds."""
    import pymupdf
    pix = pymupdf.Pixmap(source)
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=pix.width, height=pix.height)
        page.insert_image(page.rect, filename=str(source))
        for row in rows:
            text = row['text']; box = pymupdf.Rect(row['bbox'])
            fontname = 'china-s' if re.search(r'[\u3400-\u9fff]', text) else 'helv'
            font = pymupdf.Font(fontname)
            size = box.height/(font.ascender-font.descender)
            natural = font.text_length(text, fontsize=size)
            point = pymupdf.Point(box.x0, box.y0+font.ascender*size)
            # TextWriter embeds this exact font. insert_text's built-in CJK
            # alias uses different Latin widths and would extend the search
            # rectangle beyond punctuation at the observed caption boundary.
            writer = pymupdf.TextWriter(page.rect)
            writer.append(point, text, font=font, fontsize=size)
            writer.write_text(page, render_mode=3,
                              morph=(point, pymupdf.Matrix(box.width/natural, 1)))
        # A short CJK caption needs a few glyphs, not an entire font per image.
        pdf.subset_fonts()
        pdf.save(target, garbage=4, deflate=True)


def source_row_caption_rows(index, row, rgb, observations, model):
    """Locate already owned plain captions, without inventing any text.

    Unlike image-only prose recovery, immutable caption text supplies an exact
    second witness. Detection and a fresh crop reading must both agree with it;
    only captions below their own source illustration can acquire a layer.
    Mathematical labels remain opaque pixels.
    """
    from book_pipeline.structure.numbered import CAPTION_ROLES
    from book_pipeline.structure.headings import valid_box
    import numpy as np
    normalize = lambda text: re.sub(r'\s+', '', text)
    members = [index.atoms[index.ids[a]] for a in row['members']]
    images = [a for a in members if a['kind']=='image' and valid_box(a.get('bbox'))]
    candidates = [a for a in members if a['kind']=='text' and a['role'] in CAPTION_ROLES
                  and a.get('extraction_method')=='ocr' and not a.get('recovery_issue')
                  and 3 <= len(a['content'].strip()) <= 180]
    height,width=rgb.shape[:2]
    result=[]
    for atom in candidates:
        quote=normalize(atom['content'])
        if sum(normalize(a['content'])==quote for a in candidates)!=1:
            continue
        owned=[a['bbox'] for a in images if a['group']==atom['group']]
        if not owned:
            continue
        floor=max(b[3] for b in owned)
        hits=[]
        for poly,(text,score) in observations or []:
            points=np.asarray(poly,dtype=float)
            if (normalize(text)!=quote or not .90 <= float(score) <= 1
                    or points.shape!=(4,2) or not np.isfinite(points).all()):
                continue
            x,y=points.min(0);z,t=points.max(0)
            source_top=row['bbox'][1]+y/height*(row['bbox'][3]-row['bbox'][1])
            if (not 0 <= x < z <= width or not 0 <= y < t <= height
                    or source_top < floor-.004 or t-y > .25*height):
                continue
            hits.append((x,y,z,t,float(score)))
        if len(hits)!=1:
            continue
        x,y,z,t,score=hits[0]
        crop=rgb[max(0,int(y)-2):min(height,int(t)+3),max(0,int(x)-2):min(width,int(z)+3)]
        reread=model.ocr(crop,det=False)[0] or []
        if len(reread)!=1 or normalize(reread[0][0])!=quote or not .90 <= float(reread[0][1]) <= 1:
            continue
        result.append(dict(text=atom['content'].strip(),bbox=[float(v) for v in (x,y,z,t)],
                           atom=atom['id'],score=score,reread_score=float(reread[0][1])))
    return result


class ImageSearch:
    def __init__(self, index, directory):
        self.index = index
        self.directory = Path(directory)
        self.results = {}
        self.evidence = []

    def row_asset(self, row):
        from book_pipeline.structure.numbered import CAPTION_ROLES
        members=[self.index.atoms[self.index.ids[a]] for a in row['members']]
        if not any(a['kind']=='text' and a['role'] in CAPTION_ROLES
                   and a.get('extraction_method')=='ocr' and not a.get('recovery_issue')
                   and 3 <= len(a['content'].strip()) <= 180 for a in members):
            return None
        runtime=local_ocr()
        if runtime is None:
            return None
        model,signature=runtime
        source=Path(row['asset'])
        key=fingerprint(dict(method=ROW_METHOD,image=digest(source),ocr=signature,
            pymupdf=version('PyMuPDF'),bbox=row['bbox'],members=members))[:24]
        cache=self.directory/'images'/('row-caption-search-'+key+'.json')
        cache.parent.mkdir(parents=True,exist_ok=True)
        if cache.is_file():
            rows=read(cache)['rows']
        else:
            import cv2
            rgb=cv2.imread(str(source))
            rows=source_row_caption_rows(self.index,row,rgb,model.ocr(rgb)[0],model)
            save(cache,dict(method=ROW_METHOD,rows=rows,ocr_signature=signature))
        if not rows:
            return None
        target=cache.with_suffix('.pdf')
        if not target.is_file():
            searchable_raster(source,target,rows)
        self.evidence.append(dict(method=ROW_METHOD,status='requires_visual_review',
            source_sha256=digest(source),image=str(target.relative_to(self.directory)),rows=rows))
        return target

    def asset(self, atom):
        original = self.index.atoms[self.index.ids[atom['id']]]
        if (atom['kind'] != 'image' or atom.get('extraction_method') != 'ocr'
                or atom.get('recovery_issue') or atom.get('block_type') != 'image'
                or original.get('asset') != atom.get('asset')
                or len(self.index.groups[atom['group']]) != 1
                or Path(atom['asset']).suffix.lower() not in {'.png', '.jpg', '.jpeg'}):
            return None
        source = Path(atom['asset'])
        owner = atom['id'], str(source)
        if owner in self.results:
            return self.results[owner]
        runtime = local_ocr()
        if runtime is None:
            self.evidence.append(dict(method=METHOD, atom=atom['id'], status='unavailable_local_ocr'))
            self.results[owner] = None
            return None
        model, signature = runtime
        key = fingerprint(dict(method=METHOD, image=digest(source), ocr=signature,
                               pymupdf=version('PyMuPDF')))[:24]
        cache = self.directory/'images'/('caption-search-'+key+'.json')
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.is_file():
            rows = read(cache)['rows']
        else:
            import cv2
            rgb = cv2.imread(str(source))
            rows = caption_rows(rgb, model.ocr(rgb)[0])
            save(cache, dict(method=METHOD, rows=rows, signature=signature))
        # Even a separate caption with an inaccurate inherited bbox must not
        # acquire a duplicate hidden copy. Compare complete text groups.
        normalize = lambda t: re.sub(r'\W', '', t).casefold()
        known = {normalize(''.join(self.index.atoms[n]['content'] for n in positions))
                 for positions in self.index.groups.values()
                 if self.index.atoms[positions[0]]['page'] == atom['page']}
        rows = [r for r in rows if normalize(r['text']) not in known]
        target = cache.with_suffix('.pdf')
        result = None
        if rows:
            if not target.is_file():
                searchable_raster(source, target, rows)
            result = target
            self.evidence.append(dict(method=METHOD, atom=atom['id'], status='rendered',
                source_sha256=digest(source), ocr_signature=signature,
                image=str(target.relative_to(self.directory)), rows=rows))
        self.results[owner] = result
        return result
