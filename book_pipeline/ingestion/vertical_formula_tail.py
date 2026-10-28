"""Bound a multirow formula with two independently observed prose boundaries.

The caller supplies a complete, unchanged paragraph witness at two scales.
A short terminal run cannot establish ownership by itself. It must agree at
both scales, match the paragraph's physical glyph size and finish this cell.
"""
from copy import deepcopy
import re

from book_pipeline.ingestion.vertical_formula_gaps import canonical

METHOD = 'two_scale_terminal_prose_formula_bounds_v4'


def bounded_tail(rgb, boxes, preceding, rows, second_rows, pitch, original, source_texts):
    from book_pipeline.ingestion.formula_crops import ink_bounds
    from book_pipeline.ingestion.prose_ink_exclusions import exclusions

    a,b,c,d = preceding['bbox']
    center = (a+c)/2
    if not original[0]-.3*pitch <= center <= original[2]+.3*pitch:
        return None
    candidates = []
    for terminal in rows:
        text = terminal['text']
        if (not re.fullmatch(r'[\u3400-\u9fff]{2,5}',text) or not terminal.get('reader_signature')
                or terminal.get('score',0) < .99
                or any(canonical(text) in canonical(s) for s in source_texts)):
            continue
        x,y,z,t = terminal['bbox']
        terminal_pitch = (t-y)/len(text)
        if (abs((x+z)/2-center) > .5*pitch or not .75 <= terminal_pitch/pitch <= 1.33
                or not 1*pitch <= y-d <= 5*pitch or not 0 <= rgb.shape[0]-t <= .8*pitch):
            continue
        matches = [r for r in second_rows if canonical(r['text']) == canonical(text)
                   and r.get('reader_signature') == terminal['reader_signature'] and r.get('score',0) >= .99
                   and max(abs(v-w) for v,w in zip(r['bbox'],terminal['bbox'])) <= .15*pitch]
        if len(matches) == 1:
            candidates.append((terminal,matches[0]))
    if len(candidates) != 1:
        return None
    terminal, corroboration = candidates[0]
    x,y,z,t = terminal['bbox']
    chosen = [n for n,(l,u,r,v) in enumerate(boxes)
              if d <= (u+v)/2 <= y and v-u <= 2.2*pitch
              and d-.5*pitch <= u < v <= y+.5*pitch
              and abs((l+r)/2-center) <= 4.5*pitch]
    bounds = ink_bounds(rgb,boxes,chosen) if chosen else None
    if bounds is None:
        return None
    l,u,r,v = bounds
    overlap = max(0,min(r,original[2])-max(l,original[0])) * max(0,min(v,original[3])-max(u,original[1]))
    if (not d-.5*pitch <= u < v <= y+.6*pitch or not l <= center <= r
            or r-l > 9*pitch or v-u > 5*pitch
            or overlap < .15*(original[2]-original[0])*(original[3]-original[1])):
        return None
    # Both boundary rows can overlap a little padding, but any other independently
    # read prose inside the formula defeats this interpretation.
    for row in rows:
        if row is preceding or row is terminal:
            continue
        e,f,g,h = row['bbox']
        if max(0,min(r,g)-max(l,e))*max(0,min(v,h)-max(u,f)) > .08*(g-e)*(h-f):
            return None
    mask = exclusions(rgb,bounds,preceding,terminal,pitch)
    return dict(bounds=bounds, terminal=deepcopy(terminal), second_scale=deepcopy(corroboration),
                mask=mask, method=METHOD, requires_visual_review=True)


def continue_in_next_column(terminal, following, cell_box, page_size, pitch):
    """Reflow an unpunctuated cell-end fragment into adjacent column prose.

    The following source text is unchanged. Its original column must start at
    this register's top immediately to the left, with matching glyph scale.
    Titles, mathematics, distant columns and paragraph endings are barriers.
    """
    if (following['type'] != 'text' or any(p['kind'] != 'text' for p in following['parts'])):
        return None
    text=''.join(p['content'] for p in following['parts'])
    if not re.fullmatch(r'[\u3400-\u9fff]{2,16}',text):
        return None
    left,top,right,bottom=cell_box
    a,b,c,d=[v*page_size[i%2] for i,v in enumerate(following['bbox'])]
    if (not .3*pitch <= left-(a+c)/2 <= 2*pitch or c > left+.2*pitch
            or not 0 <= b-top <= 1.25*pitch or c-a > 2.5*pitch
            or not .75 <= (d-b)/len(text)/pitch <= 1.33):
        return None
    result=deepcopy(following)
    result['parts']=[*deepcopy(terminal['parts']),*result['parts']]
    result['source_prose_end_bbox']=deepcopy(following.get('source_prose_end_bbox',following['bbox']))
    boxes=[terminal['bbox'],following['bbox']]
    result['bbox']=[min(q[0] for q in boxes),min(q[1] for q in boxes),
                    max(q[2] for q in boxes),max(q[3] for q in boxes)]
    return result
