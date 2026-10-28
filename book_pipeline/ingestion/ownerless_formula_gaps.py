"""Recover source ink in a ruled cell whose formula was omitted entirely.

An exact, unique long suffix supplies the reading-order anchor. The preceding
run is witnessed by independent geometry, including its original glyph count;
its transcription is explicitly unverified and is never edited by this path.
This fallback is restricted to cells with one plain paragraph and no images.
"""
from copy import deepcopy
import hashlib
import re

from book_pipeline.ingestion.ruled_prose import page_box

METHOD = 'ownerless_cell_unique_suffix_and_two_scale_geometry_v1'
PROSE = re.compile(r'[\u3400-\u9fff]{6,60}')


def observe_layout(rgb):
    """Observe all rectangles and readings, with no source-text hints."""
    import numpy as np
    from book_pipeline.publishing.image_search import local_ocr
    runtime = local_ocr()
    if runtime is None:
        return []
    model, signature = runtime
    h,w = rgb.shape[:2]
    rows = []
    for poly,(text,score) in model.ocr(rgb[:,:,::-1].copy())[0] or []:
        p = np.asarray(poly,dtype=float)
        if p.shape != (4,2) or not np.isfinite(p).all():
            continue
        x,y = p.min(0);z,t = p.max(0)
        if 0 <= x < z <= w and 0 <= y < t <= h:
            rows.append(dict(text=text,score=float(score),bbox=[float(x),float(y),float(z),float(t)],
                             reader_signature=signature))
    return rows


def suffix_witness(text, rows, boxes):
    """An exact long suffix and a single preceding glyph-sized vertical run.

    No approximate text match is accepted. The unread prefix has to occupy the
    expected number of physical glyph slots, in the suffix's column. A short,
    tall, shifted or ambiguous predecessor cannot authorize a source crop.
    """
    matches = []
    for row in rows:
        suffix = row['text']
        if (not PROSE.fullmatch(suffix) or not .99 <= row['score'] <= 1
                or not row.get('reader_signature') or not text.endswith(suffix)):
            continue
        prefix = text[:-len(suffix)]
        if not PROSE.fullmatch(prefix):
            continue
        a,b,c,d = row['bbox'];pitch=(d-b)/len(suffix)
        if not (.4*(c-a) <= pitch <= 1.8*(c-a) and d-b >= 3*(c-a)):
            continue
        predecessors = []
        for box in boxes:
            x,y,z,t = box
            if (t-y >= 3*(z-x) and .6*pitch <= z-x <= 1.8*pitch
                    and abs((x+z-a-c)/2) <= .3*pitch
                    and .2*pitch <= b-t <= 3*pitch
                    and .85 <= (t-y)/(len(prefix)*pitch) <= 1.15):
                predecessors.append(dict(bbox=list(box),text=prefix,transcription_verified=False))
        if len(predecessors) == 1:
            matches.append((predecessors[0],row,len(prefix),pitch))
    return matches[0] if len(matches) == 1 else None


def gap_bounds(rgb, boxes, first, second, pitch):
    import cv2
    import numpy as np
    from book_pipeline.ingestion.formula_crops import ink_bounds
    from book_pipeline.ingestion.prose_ink_exclusions import exclusions
    boxes=[[round(v) for v in box] for box in boxes]
    a,b,c,d = first['bbox'];x,y,z,t = second['bbox'];center=(a+c+x+z)/4
    selected = [n for n,(l,u,r,v) in enumerate(boxes)
                if d-.2*pitch <= (u+v)/2 <= y+.2*pitch and v-u <= 2.7*pitch
                and d-.4*pitch <= u < v <= y+.4*pitch
                and abs((l+r)/2-center) <= 4.5*pitch]
    # A one-pixel stroke clipped at the cell edge becomes a two-pixel primary
    # component when enlarged. It cannot determine the formula's vertical
    # extent. Ignore only these clipped slivers for extent analysis; every
    # original source pixel inside the final rectangle is still exported.
    _,ink=cv2.threshold(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),0,255,
                        cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    _,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
    edge_ids=[n for n,(x,y,w,h,area) in enumerate(stats[1:],1)
              if w <= max(1,round(.002*rgb.shape[1])) and h >= .5*pitch
              and (x==0 or x+w==rgb.shape[1])]
    analysis=rgb.copy() if edge_ids else rgb
    if edge_ids:
        analysis[np.isin(labels,edge_ids)]=255
    bounds = ink_bounds(analysis,boxes,selected) if selected else None
    if bounds is None:
        return None
    # A neighboring glyph's disconnected descender can enter detector padding
    # at one scale. Remove only its witnessed component before comparing the
    # formula's observed extent. The final crop records the same mask if needed.
    mask=exclusions(rgb,bounds,first,second,pitch)
    if mask:
        cleaned=analysis.copy()
        _,ink=cv2.threshold(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),0,255,
                            cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        _,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
        components={tuple(b) for b in mask['components']}
        ids=[n for n,(x,y,w,h,area) in enumerate(stats[1:],1) if (x,y,x+w,y+h) in components]
        if len(ids)!=len(components):
            return None
        # Extent calculation needs the entire witnessed component removed,
        # including pixels outside the provisional rectangle. The saved asset
        # still applies exclusions only within its final source rectangle.
        component_mask=cv2.dilate(np.isin(labels,ids).astype(np.uint8),np.ones((3,3),np.uint8))
        cleaned[component_mask>0]=mask['fill_rgb']
        bounds=ink_bounds(cleaned,boxes,selected)
        if bounds is None:
            return None
    l,u,r,v = bounds
    if not (d-.4*pitch <= u < v <= y+.4*pitch and r-l <= 9*pitch and v-u <= 3*pitch):
        return None
    # An additional vertical prose run inside the crop is a hard veto.
    def anchored(box):
        x,y,z,t=box
        return any(max(0,min(z,c)-max(x,a))*max(0,min(t,d)-max(y,b)) >= .8*(z-x)*(t-y)
                   for a,b,c,d in (first['bbox'],second['bbox']))
    if any(n not in selected and not anchored(box) and min(r,box[2])>max(l,box[0])
           and min(v,box[3])>max(u,box[1]) for n,box in enumerate(boxes)):
        return None
    return bounds


def recover_cell(rgb, cell, answer, output, page_size, rows):
    """Return original-index replacements; all other cell paths remain intact."""
    import numpy as np
    from PIL import Image
    if cell.get('image_blocks') != [] or not cell.get('regions'):
        return {},[]
    absolute = cell['pixel_bbox'];left,top,right,bottom=absolute
    owners=[]
    for n,block in enumerate(answer['blocks']):
        x,y,z,t=block['bbox'];cx=(x+z)*page_size[0]/2;cy=(y+t)*page_size[1]/2
        if left <= cx < right and top <= cy < bottom:
            owners.append((n,block))
    if len(owners)!=1:
        return {},[]
    n,block=owners[0]
    if block['type']!='text' or any(p['kind']!='text' for p in block['parts']):
        return {},[]
    text=''.join(p['content'] for p in block['parts'])
    witness=suffix_witness(text,rows,cell['regions'])
    if witness is None:
        return {},[]
    first,second,offset,pitch=witness
    suffix=text[offset:]
    if sum(''.join(p['content'] for p in b['parts']).count(suffix) for b in answer['blocks'])!=1:
        return {},[]
    local=rgb[top:bottom,left:right]
    scaled=np.array(Image.fromarray(local).resize((local.shape[1]*2,local.shape[0]*2),Image.Resampling.NEAREST))
    observed=observe_layout(scaled)
    second_boxes=[r['bbox'] for r in observed]
    corroboration=suffix_witness(text,observed,second_boxes)
    if corroboration is None or corroboration[2]!=offset:
        return {},[]
    for a,b in zip((first,second),corroboration[:2]):
        if max(abs(x-y/2) for x,y in zip(a['bbox'],b['bbox'])) > .2*pitch:
            return {},[]
    bounds=gap_bounds(local,cell['regions'],first,second,pitch)
    second_bounds=gap_bounds(scaled,second_boxes,*corroboration[:2],corroboration[3])
    if bounds is None or second_bounds is None or max(abs(x-y/2) for x,y in zip(bounds,second_bounds)) > .2*pitch:
        return {},[]
    h,w=local.shape[:2]
    def mapped(box):
        return page_box([box[0]/w,box[1]/h,box[2]/w,box[3]/h],absolute,page_size)
    box=mapped(bounds)
    if any(b['type']=='image' and min(box[2],b['bbox'][2])>max(box[0],b['bbox'][0])
           and min(box[3],b['bbox'][3])>max(box[1],b['bbox'][1]) for b in output['blocks']):
        return {},[]
    original=deepcopy(output['blocks'][n])
    before=dict(original,bbox=mapped(first['bbox']),parts=[dict(kind='text',content=text[:offset])])
    after=dict(original,bbox=mapped(second['bbox']),parts=[dict(kind='text',content=text[offset:])])
    image=dict(type='image',bbox=box,parts=[],reason='Source formula bounded by unique suffix and independent prefix geometry; transcription unverified')
    from book_pipeline.ingestion.prose_ink_exclusions import exclusions
    mask=exclusions(local,bounds,first,second,pitch)
    if mask:
        l,u,r,v=bounds
        mask.update(pixel_bbox=[left+l,top+u,left+r,top+v],page_pixel_size=page_size,
                    page_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        image['prose_ink_exclusions']=mask
    return {n:[before,image,after]},[dict(method=METHOD,action='inserted_ownerless_gap',block=n,cell=cell['cell'],
        first=first,second=second,second_scale=corroboration[:2],local_pixel_bbox=bounds,bbox=box,
        original_bbox=block['bbox'],text_unchanged=True,prefix_transcription_verified=False,requires_visual_review=True)]
