"""Rejoin witnessed prose at the reading boundary of two ruled registers.

Only the last column above and first column below are eligible. Two image-only
readings locate complete, unique source strings; they never authorize edits.
"""
from copy import deepcopy
import re

from book_pipeline.ingestion.vertical_formula_gaps import canonical

METHOD = 'two_scale_ruled_register_prose_continuation_v2'
MIN_CONFIDENCE = .98


def plain(block, minimum, maximum):
    if block['type'] != 'text' or any(p['kind'] != 'text' for p in block['parts']):
        return None
    text = ''.join(p['content'] for p in block['parts'])
    return text if re.fullmatch(r'[\u3400-\u9fff]{%d,%d}' % (minimum, maximum), text) else None


def witness(text, rows):
    matches = [r for r in rows if canonical(r['text']) == canonical(text)
               and MIN_CONFIDENCE <= r['score'] <= 1 and r.get('reader_signature')]
    if len(matches) != 1:
        return None
    row = matches[0]
    a,b,c,d = row['bbox']
    pitch = (d-b)/len(text)
    if not (.4*(c-a) <= pitch <= 1.8*(c-a) and d-b >= 1.5*(c-a)):
        return None
    return row, pitch


def join_registers(rgb, cells, answer, page_size):
    """Return a copy and evidence; retain literal parts and formula assets."""
    import numpy as np
    from PIL import Image
    from book_pipeline.ingestion.ownerless_formula_gaps import observe_layout

    registers = {}
    for cell in cells:
        l,t,r,b = cell['pixel_bbox']
        registers.setdefault((t,b), []).append(cell)
    registers = sorted(registers.items())
    replacements, evidence, occupied = {}, [], set()
    texts = [''.join(p['content'] for p in b['parts']) for b in answer['blocks']]
    for ((top,bottom),upper), ((next_top,next_bottom),lower) in zip(registers,registers[1:]):
        if (bottom != next_top or min(len(upper),len(lower)) < 2
                or min(c['pixel_bbox'][0] for c in upper) != min(c['pixel_bbox'][0] for c in lower)
                or max(c['pixel_bbox'][2] for c in upper) != max(c['pixel_bbox'][2] for c in lower)):
            continue
        last = min(upper,key=lambda c:c['pixel_bbox'][0])
        first = max(lower,key=lambda c:c['pixel_bbox'][2])
        def belongs(block, cell):
            a,b,c,d = block['bbox'];l,t,r,v = cell['pixel_bbox']
            return l <= (a+c)*page_size[0]/2 < r and t <= (b+d)*page_size[1]/2 < v
        candidates = [(n,plain(a,6,60),plain(b,2,5))
                      for n,(a,b) in enumerate(zip(answer['blocks'],answer['blocks'][1:]))
                      if belongs(a,last) and belongs(b,first)]
        candidates = [(n,a,b) for n,a,b in candidates if a and b and n not in occupied and n+1 not in occupied
                      and all(sum(canonical(s).count(canonical(t)) for s in texts)==1 for t in (a,b))]
        if len(candidates) != 1:
            continue
        n,before,after = candidates[0]
        observations = []
        for cell,text,end in ((last,before,True),(first,after,False)):
            l,t,r,b = cell['pixel_bbox'];local = rgb[t:b,l:r]
            rows = observe_layout(local)
            initial = witness(text,rows)
            if initial is None:
                break
            row,pitch = initial;a,y,c,d = row['bbox']
            if (end and not 0 <= b-t-d <= .8*pitch) or (not end and not 0 <= y <= 2.5*pitch):
                break
            # An intervening detected object in this column defeats continuity.
            center = (a+c)/2
            if any(abs((x+z)/2-center) <= .5*pitch and
                   ((end and u >= d) or (not end and v <= y))
                   for x,u,z,v in cell.get('regions',[])):
                break
            scaled = np.asarray(Image.fromarray(local).resize((local.shape[1]*2,local.shape[0]*2),
                                                               Image.Resampling.NEAREST))
            second = witness(text,observe_layout(scaled))
            if (second is None or second[0]['reader_signature'] != row['reader_signature']
                    or max(abs(x-y/2) for x,y in zip(row['bbox'],second[0]['bbox'])) > .15*pitch):
                break
            observations.append(dict(cell=cell['cell'],text=text,first=row,second_scale=second[0],pitch=pitch))
        if len(observations) != 2 or not .75 <= observations[0]['pitch']/observations[1]['pitch'] <= 1.33:
            continue
        a,b = answer['blocks'][n:n+2]
        joined = deepcopy(a)
        joined['parts'] += deepcopy(b['parts'])
        joined['source_prose_end_bbox'] = deepcopy(b.get('source_prose_end_bbox',b['bbox']))
        joined['bbox'] = [min(a['bbox'][0],b['bbox'][0]),min(a['bbox'][1],b['bbox'][1]),
                          max(a['bbox'][2],b['bbox'][2]),max(a['bbox'][3],b['bbox'][3])]
        replacements[n],replacements[n+1] = [joined],[]
        occupied.update((n,n+1))
        evidence.append(dict(method=METHOD,blocks=[n,n+1],observations=observations,
                             source_boxes=[a['bbox'],b['bbox']],text_unchanged=True,requires_visual_review=True))
    result = deepcopy(answer)
    result['blocks'] = [b for n,block in enumerate(result['blocks']) for b in replacements.get(n,[block])]
    return result,evidence
