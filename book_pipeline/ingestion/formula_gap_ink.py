"""Complete detector-missed scripts between witnessed vertical prose runs."""

METHOD = 'bounded_vertical_script_components_v1'


def extend_scripts(rgb, boxes, selected, bounds, first, second, pitch):
    """Admit a nearby above/below component, never a different OCR owner.

    Prose boundaries supply a tight vertical band. A missing component must
    sit directly beside selected mathematical ink and outside witnessed prose.
    This does not extend the horizontal envelope or follow chains of dust.
    """
    import cv2
    import numpy as np

    _,ink = cv2.threshold(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),0,255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _,labels,stats,_ = cv2.connectedComponentsWithStats(ink,8)
    l,u,r,v = bounds
    seeds, candidates = [], []
    def fraction(n, area, box):
        a,b,c,d = [round(q) for q in box]
        a,b,c,d = max(0,a),max(0,b),min(rgb.shape[1],c),min(rgb.shape[0],d)
        return np.count_nonzero(labels[b:d,a:c] == n)/area if a<c and b<d else 0
    for n,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area < 6 or any(fraction(n,area,q['bbox']) >= .9 for q in (first,second)):
            continue
        selected_share = max((fraction(n,area,boxes[k]) for k in selected),default=0)
        other = max((fraction(n,area,b) for k,b in enumerate(boxes) if k not in selected),default=0)
        box = [int(x),int(y),int(x+w),int(y+h)]
        if selected_share >= .1 and selected_share > other+.05:
            seeds.append(box)
        elif (other < .1 and area >= .01*pitch*pitch and max(w,h) <= 1.8*pitch
                and l <= x < x+w <= r and (y < u or y+h > v)
                and first['bbox'][3]-.6*pitch <= y < y+h <= second['bbox'][1]+.6*pitch):
            candidates.append(box)
    tails = []
    for x,y,z,t in candidates:
        for a,b,c,d in seeds:
            horizontal = max(0,a-z,x-c)
            vertical = max(0,b-t,y-d)
            if horizontal <= .25*pitch and vertical <= .25*pitch:
                tails.append([x,y,z,t])
                break
    if not tails:
        return bounds, []
    return [l,max(0,min([u]+[b[1]-2 for b in tails])),r,
            min(rgb.shape[0],max([v]+[b[3]+2 for b in tails]))], tails


def boundary_ink_clear(rgb, bounds, row, mask):
    """A prose box may include padding; no remaining prose ink may enter it."""
    import cv2
    import numpy as np

    if mask is None:
        return False
    _,ink = cv2.threshold(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),0,255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    l,u,r,v = bounds
    for y,x,z in mask['runs']:
        ink[u+y,l+x:l+z] = 0
    a,b,c,d = [round(q) for q in row['bbox']]
    a,b,c,d = max(l,a),max(u,b),min(r,c),min(v,d)
    if a>=c or b>=d:
        return True
    _,_,stats,_ = cv2.connectedComponentsWithStats(ink[b:d,a:c],8)
    return not any(area >= 6 for _,_,_,_,area in stats[1:])
