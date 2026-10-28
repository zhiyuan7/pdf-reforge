"""Exclude independently witnessed neighboring prose from a formula crop.

    Rectangular padding can include a prose descender while mathematics starts
    elsewhere on the same row. Keep the full rectangle and remove only the
    connected prose component; original source images are never modified.
"""
import hashlib

METHOD = 'witnessed_neighbor_prose_component_exclusion_v2'
SUPPORTED_METHODS = {METHOD, 'witnessed_neighbor_prose_component_exclusion_v1'}


def exclusions(rgb, bounds, first, second, pitch):
    import cv2
    import numpy as np
    threshold,ink=cv2.threshold(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),0,255,
                                cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    _,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
    mask=np.zeros(ink.shape,np.uint8)
    l,u,r,v=bounds
    components=[]
    witnessed_parts=[]
    for n,(x,y,w,h,area) in enumerate(stats[1:],1):
        if (area < .02*pitch*pitch or not .25*pitch <= w <= 1.8*pitch
                or not .5*pitch <= h <= 1.8*pitch
                or min(r,x+w)<=max(l,x) or min(v,y+h)<=max(u,y)):
            continue
        witnessed=False
        side=None
        for k,row in enumerate((first,second)):
            # A component entirely inside the formula is never ordinary prose
            # merely because an OCR box reaches into it.
            if not (y < u-.5*pitch if k==0 else y+h > v+.5*pitch):
                continue
            a,b,c,d=[round(q) for q in row['bbox']]
            a,b,c,d=max(a,x),max(b,y),min(c,x+w),min(d,y+h)
            if a<c and b<d and np.count_nonzero(labels[b:d,a:c]==n)>=.9*area:
                witnessed=True
                side=k
        if not witnessed:
            continue
        member=(labels==n).astype(np.uint8)
        expanded=cv2.dilate(member,np.ones((3,3),np.uint8))
        # Antialias padding must not touch another ink component.
        if np.any((expanded>0)&(labels!=0)&(labels!=n)):
            continue
        mask|=expanded
        components.append([int(x),int(y),int(x+w),int(y+h)])
        witnessed_parts.append((side,int(x),int(y),int(w),int(h)))
    # A CJK glyph may have a narrow disconnected radical. It is eligible only
    # beside an already witnessed ordinary-size component on the same glyph
    # row, within the same OCR prose run and outside the same crop boundary.
    # This does not relax the safeguards for an isolated mathematical stroke.
    for n,(x,y,w,h,area) in enumerate(stats[1:],1):
        if (area < .01*pitch*pitch or not .1*pitch <= w < .25*pitch
                or not .5*pitch <= h <= 1.8*pitch
                or min(r,x+w)<=max(l,x) or min(v,y+h)<=max(u,y)):
            continue
        member=(labels==n).astype(np.uint8)
        supported=False
        for side,a,b,c,d in witnessed_parts:
            outside = y < u-.35*pitch if side==0 else y+h > v+.35*pitch
            overlap=max(0,min(y+h,b+d)-max(y,b))
            gap=max(0,a-(x+w),x-(a+c))
            row=(first,second)[side]
            e,f,g,j=[round(q) for q in row['bbox']]
            e,f,g,j=max(e,x),max(f,y),min(g,x+w),min(j,y+h)
            if (outside and overlap >= .7*min(h,d) and gap <= .4*pitch
                    and e<g and f<j and np.count_nonzero(labels[f:j,e:g]==n) >= .9*area):
                supported=True
                break
        if not supported:
            continue
        expanded=cv2.dilate(member,np.ones((3,3),np.uint8))
        if np.any((expanded>0)&(labels!=0)&(labels!=n)):
            continue
        mask|=expanded
        components.append([int(x),int(y),int(x+w),int(y+h)])
    local=mask[u:v,l:r]
    if not local.any():
        return None
    ring=cv2.dilate(mask,np.ones((7,7),np.uint8))&(~mask)
    paper=rgb[(ring>0)&(ink==0)]
    if not len(paper):
        return None
    fill=np.median(paper,axis=0).round().astype(int).tolist()
    runs=[]
    for y,row in enumerate(local):
        edges=np.flatnonzero(np.diff(np.r_[0,row,0]))
        runs.extend([y,int(a),int(b)] for a,b in zip(edges[::2],edges[1::2]))
    return dict(method=METHOD,runs=runs,fill_rgb=fill,components=components,
                excluded_pixels=int(local.sum()))


def render_crop(page, dpi, record):
    """Validate the exact source raster before applying the reviewed mask."""
    import pymupdf
    from PIL import Image
    pix=page.get_pixmap(dpi=dpi,alpha=False,colorspace=pymupdf.csRGB)
    if (record['method'] not in SUPPORTED_METHODS or record['page_pixel_size']!=[pix.width,pix.height]
            or record['page_rgb_sha256']!=hashlib.sha256(pix.samples).hexdigest()):
        raise ValueError('Formula prose exclusion source raster changed')
    box=record['pixel_bbox']
    if (len(box)!=4 or any(type(v) is not int for v in box)
            or not 0<=box[0]<box[2]<=pix.width or not 0<=box[1]<box[3]<=pix.height):
        raise ValueError('Invalid formula prose exclusion bounds')
    fill=record['fill_rgb']
    if len(fill)!=3 or any(type(v) is not int or not 0<=v<=255 for v in fill):
        raise ValueError('Invalid formula prose exclusion paper color')
    crop=Image.frombytes('RGB',[pix.width,pix.height],pix.samples).crop(box)
    for run in record['runs']:
        if (len(run)!=3 or any(type(v) is not int for v in run)
                or not 0<=run[0]<crop.height or not 0<=run[1]<run[2]<=crop.width):
            raise ValueError('Invalid formula prose exclusion pixel run')
        y,x,z=run
        crop.paste(tuple(fill),(x,y,z,y+1))
    return crop
