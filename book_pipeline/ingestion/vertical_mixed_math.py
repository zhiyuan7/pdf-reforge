"""Replace an unverified mixed inline span with independently located source ink.

No mathematical reading is used as evidence. Complete ordinary-text readings
at two scales locate the source interval; the original mixed transcription is
retained in review evidence, including any ambiguous interstitial character.
"""
from copy import deepcopy
import re

METHOD = 'two_scale_mixed_vertical_math_source_fallback_v1'
TIGHT_METHOD = 'blind_projection_complete_component_prose_rereading_v2'
MIN_CONFIDENCE = .99


def complete_prose_bounds(bounds, stats, width, shape):
    """Close a projected strip over whole supported source components.

    Projection thresholds can cut faint first/last strokes while recognition
    still returns the entire expected row at high confidence. Finish geometry
    before asking the reader; remote dust and other columns supply no bounds.
    A large completion is ambiguous and must be rejected, not silently clipped.
    """
    x,y,z,t = bounds
    members = [[int(l),int(u),int(l+w),int(u+h)] for l,u,w,h,area in stats
               if area >= .008*width*width and max(w,h) <= 1.8*width
               and x-.2*width <= l < l+w <= z+.2*width
               and max(0,min(t,u+h)-max(y,u)) >= .2*h]
    if not members:
        return None
    box = [min(b[0] for b in members),min(b[1] for b in members),
           max(b[2] for b in members),max(b[3] for b in members)]
    if (y-box[1] > .25*width or box[3]-t > .25*width
            or not 0 <= box[0] < box[2] <= shape[1]
            or not 0 <= box[1] < box[3] <= shape[0]):
        return None
    return box


def tight_prose(rgb):
    """Reread image-derived subregions to remove detector padding into math.

    Candidate strips come only from ink projection. The CPU reader never gets
    an expected string. A shorter strip is usable only if it independently
    reads the same complete text as the original, with no extra detected row.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    from book_pipeline.ingestion.vertical_prose_fragments import observe_fragments
    from book_pipeline.publishing.image_search import local_ocr
    rows = observe_fragments(rgb)
    runtime = local_ocr()
    if runtime is None:
        return []
    model, signature = runtime
    _, ink = cv2.threshold(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _,_,stats,_ = cv2.connectedComponentsWithStats(ink,8)
    result = []
    for row in rows:
        if len(row['text']) < 5:
            result.append(row)
            continue
        x,y,z,t = [round(v) for v in row['bbox']]
        width = z-x
        active = (ink[y:t,x:z] > 0).sum(1) >= .12*width
        active = cv2.morphologyEx(active.astype('uint8')[:,None], cv2.MORPH_CLOSE,
                                 np.ones((max(1,round(.12*width)),1),np.uint8)).ravel() > 0
        edges = np.flatnonzero(np.diff(np.r_[False,active,False]))
        runs = [(int(a+y),int(b+y)) for a,b in zip(edges[::2],edges[1::2])]
        candidates = []
        for i,(a,_) in enumerate(runs):
            for j,(_,b) in enumerate(runs[i:],i):
                if (not 2*width <= b-a <= 1.4*width*len(row['text'])
                        or any(runs[k+1][0]-runs[k][1] > .5*width for k in range(i,j))):
                    continue
                box = complete_prose_bounds([x,a,z,b],stats[1:],width,rgb.shape)
                if box is not None and box not in candidates:
                    candidates.append(box)
        # Bound work on noisy pages. Failing closed is preferable to an
        # unbounded sequence of CPU recognitions or a guessed boundary.
        if len(candidates) > 24:
            continue
        matches = []
        for left,a,right,b in sorted(candidates,key=lambda v:v[3]-v[1]):
            crop = Image.fromarray(rgb[a:b,left:right])
            view = ImageOps.expand(crop,border=max(1,round(.2*width)),fill='white')
            readings = model.ocr(np.asarray(view)[:,:,::-1].copy())[0] or []
            if (len(readings) != 1 or readings[0][1][0] != row['text']
                    or not MIN_CONFIDENCE <= float(readings[0][1][1]) <= 1):
                continue
            matches.append(dict(row,bbox=[left,a,right,b],tight_method=TIGHT_METHOD,
                                reread_score=float(readings[0][1][1]),reader_signature=signature))
            break  # Shortest complete blind reading; wider strips add no proof.
        if matches:
            result.extend(matches)
    return result


def math_spans(parts):
    """A short glyph between math parts is uncertain, never ordinary prose proof."""
    n = 1
    while n < len(parts)-1:
        if parts[n]['kind'] != 'math' or parts[n-1]['kind'] != 'text':
            n += 1
            continue
        end = n+1
        while end < len(parts)-1:
            if parts[end]['kind'] == 'math':
                end += 1
            elif (parts[end]['kind'] == 'text' and re.fullmatch(r'[\u3400-\u9fff]',parts[end]['content'])
                  and parts[end+1]['kind'] == 'math'):
                end += 2
            else:
                break
        if end < len(parts) and parts[end]['kind'] == 'text':
            yield n,end
        n = end+1


def anchored_span(before, after, rows):
    from book_pipeline.ingestion.vertical_formula_gaps import canonical
    left = [r for r in rows if len(r['text']) >= 5 and r.get('reader_signature')
            and r['score'] >= MIN_CONFIDENCE and r.get('tight_method') == TIGHT_METHOD
            and canonical(before).endswith(canonical(r['text']))]
    right = [r for r in rows if len(r['text']) >= 5 and r.get('reader_signature')
             and r['score'] >= MIN_CONFIDENCE and canonical(after).startswith(canonical(r['text']))]
    if len(left) != 1 or len(right) != 1:
        return None
    first,second = left[0],right[0]
    a,b,c,d = first['bbox']; x,y,z,t = second['bbox']
    p,q = (d-b)/len(first['text']), (t-y)/len(second['text'])
    pitch = (p+q)/2
    if (p <= 0 or q <= 0 or not .65 <= p/q <= 1.5
            or abs((a+c-x-z)/2) > .5*pitch or not .2*pitch <= y-d <= 3*pitch):
        return None
    return first,second,pitch


def complete_bounds(rgb, bounds, first, second, pitch):
    """Keep whole nearby ink inside the independently witnessed prose gap.

    Coarse detector padding can claim an exponent or a fraction-bar tip for
    ordinary text. Only this tightly witnessed interval may override that
    claim; a component crossing either prose boundary remains ineligible.
    """
    import cv2
    _, ink = cv2.threshold(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),0,255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _,_,stats,_ = cv2.connectedComponentsWithStats(ink,8)
    l,u,r,v = bounds
    kept = []
    for x,y,w,h,area in stats[1:]:
        if (area < 6 or max(w,h) > 1.8*pitch
                or not first['bbox'][3] <= y < y+h <= second['bbox'][1]
                or not l-.3*pitch <= x < x+w <= r+.3*pitch):
            continue
        dx,dy = max(0,l-x-w,x-r),max(0,u-y-h,y-v)
        if dx <= .25*pitch and dy <= .25*pitch:
            kept.append([int(x),int(y),int(x+w),int(y+h)])
    if not kept:
        return bounds,[]
    expanded = [max(0,min([l]+[b[0]-2 for b in kept])),max(0,min([u]+[b[1]-2 for b in kept])),
                min(rgb.shape[1],max([r]+[b[2]+2 for b in kept])),
                min(rgb.shape[0],max([v]+[b[3]+2 for b in kept]))]
    return expanded,kept


def recover_mixed_math(rgb, cells, answer, page_size):
    import numpy as np
    from PIL import Image
    from book_pipeline.ingestion.vertical_formula_gaps import compact_bounds
    from book_pipeline.ingestion.ruled_prose import page_box
    from book_pipeline.ingestion.prose_ink_exclusions import exclusions
    from book_pipeline.ingestion.formula_gap_ink import boundary_ink_clear
    output, evidence, replacements = deepcopy(answer), [], {}
    for cell in cells:
        if not cell.get('regions'):
            continue
        left,top,right,bottom = cell['pixel_bbox']
        eligible = []
        for n,block in enumerate(answer['blocks']):
            if block['type'] != 'text' or n in replacements:
                continue
            x,y,z,t = [v*page_size[k%2] for k,v in enumerate(block['bbox'])]
            if not (left <= (x+z)/2 < right and top <= (y+t)/2 < bottom):
                continue
            spans = list(math_spans(block['parts']))
            if spans:
                eligible.append((n,block,spans))
        if not eligible:
            continue
        local = rgb[top:bottom,left:right]
        rows = tight_prose(local)
        scaled = np.asarray(Image.fromarray(local).resize((local.shape[1]*2,local.shape[0]*2),Image.Resampling.NEAREST))
        secondary = tight_prose(scaled)
        for row in secondary:
            row['bbox'] = [v/2 for v in row['bbox']]
        for n,block,spans in eligible:
            for start,end in spans:
                before,after = block['parts'][start-1]['content'],block['parts'][end]['content']
                witness = anchored_span(before,after,rows)
                second_witness = anchored_span(before,after,secondary)
                if witness is None or second_witness is None:
                    continue
                first,second,pitch = witness
                if any(a['text'] != b['text'] or max(abs(v-w) for v,w in zip(a['bbox'],b['bbox'])) > .2*pitch
                       for a,b in zip(witness[:2],second_witness[:2])):
                    continue
                # Old geometry must own this column, even if its vertical box
                # was coarse. Independent text is not a license to move a span.
                center = left+(first['bbox'][0]+first['bbox'][2])/2
                if not block['bbox'][0]*page_size[0]-.5*pitch <= center <= block['bbox'][2]*page_size[0]+.5*pitch:
                    continue
                bounds = compact_bounds(local,cell['regions'],first,second,pitch,wide=True)
                if bounds is None:
                    continue
                bounds,components = complete_bounds(local,bounds,first,second,pitch)
                f,s = deepcopy(second_witness[:2])
                for r in (f,s):
                    r['bbox'] = [v*2 for v in r['bbox']]
                scaled_bounds = compact_bounds(scaled,[[v*2 for v in b] for b in cell['regions']],
                                               f,s,second_witness[2]*2,wide=True)
                if scaled_bounds is not None:
                    scaled_bounds,_ = complete_bounds(scaled,scaled_bounds,f,s,second_witness[2]*2)
                if scaled_bounds is None or max(abs(v-w/2) for v,w in zip(bounds,scaled_bounds)) > .2*pitch:
                    continue
                mask = exclusions(local,bounds,first,second,pitch)
                l,u,r,v = bounds
                # Padding can overlap a witnessed prose box; actual prose ink
                # cannot. Other independently read prose is always protected.
                protected = False
                for row in rows:
                    a,b,c,d = row['bbox']
                    overlap = max(0,min(r,c)-max(l,a))*max(0,min(v,d)-max(u,b))
                    if overlap > .08*(c-a)*(d-b):
                        if row not in (first,second) or not boundary_ink_clear(local,bounds,row,mask):
                            protected = True
                if protected:
                    continue
                mapped = page_box([l/local.shape[1],u/local.shape[0],r/local.shape[1],v/local.shape[0]],
                                  cell['pixel_bbox'],page_size)
                if any(b['type']=='image' and max(0,min(mapped[2],b['bbox'][2])-max(mapped[0],b['bbox'][0]))*
                       max(0,min(mapped[3],b['bbox'][3])-max(mapped[1],b['bbox'][1])) > 0 for b in answer['blocks']):
                    continue
                image = dict(type='image',bbox=mapped,parts=[],
                             reason='Unverified inline transcription replaced by independently anchored source ink')
                if mask:
                    import hashlib
                    mask.update(pixel_bbox=[left+l,top+u,left+r,top+v],page_pixel_size=page_size,
                                page_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                    image['prose_ink_exclusions'] = mask
                # Preserve all parts outside the fallback span verbatim. In
                # particular, a partial ordinary-text witness cannot edit the
                # rest of its paragraph or its historical spelling.
                prefix = dict(deepcopy(block),parts=deepcopy(block['parts'][:start]))
                suffix = dict(deepcopy(block),parts=deepcopy(block['parts'][end:]))
                prefix['bbox'] = [block['bbox'][0],block['bbox'][1],block['bbox'][2],
                                  (top+first['bbox'][3])/page_size[1]]
                suffix['bbox'] = [block['bbox'][0],(top+second['bbox'][1])/page_size[1],
                                  block['bbox'][2],block['bbox'][3]]
                if prefix['bbox'][1] >= prefix['bbox'][3] or suffix['bbox'][1] >= suffix['bbox'][3]:
                    continue
                replacements[n] = [prefix,image,suffix]
                evidence.append(dict(method=METHOD,block=n,cell=cell['cell'],first=first,second=second,
                    second_scale=second_witness[:2],bbox=mapped,local_pixel_bbox=bounds,
                    boundary_components=components,original_parts=deepcopy(block['parts'][start:end]),
                    transcription_verified=False,requires_visual_review=True))
                break  # One independently witnessed span per unchanged block.
    output['blocks'] = [b for n,block in enumerate(output['blocks']) for b in replacements.get(n,[block])]
    return output,evidence
