"""Exclude a tall OCR ghost only where the pixels contain a bare binding stroke."""
import numpy as np

METHOD='pixel_owned_outer_binding_ghost_v1'


def binding_groups(index,page,gray):
    import cv2
    height,width=gray.shape
    ink=(gray<110).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
    strokes=[]
    for label,(x,y,w,h,area) in enumerate(stats[1:],1):
        if (h<height*.7 or w>width*.04 or area<height*2
                or not (x+w<width*.06 or x>width*.94)):continue
        strokes.append((label,[x/width,y/height,(x+w)/width,(y+h)/height]))
    if not strokes:return None
    groups=[[index.atoms[n] for n in ns] for ns in index.groups.values()
            if index.atoms[ns[0]]['page']==page]
    excluded=[];evidence=[]
    for atoms in groups:
        box=atoms[0].get('block_bbox')
        if (not box or box[3]-box[1]<.65 or box[2]-box[0]>.06
                or not (box[2]<.06 or box[0]>.94)
                or any(a.get('extraction_method')!='ocr' or a.get('writing_direction')
                    or a['kind'] not in {'text','math'}
                    or a['role'] not in {'text','paragraph','aside_text'} for a in atoms)):continue
        # Other body groups independently establish a separate readable column.
        body=[g for g in groups if g is not atoms and g[0].get('block_bbox')
              and g[0]['block_bbox'][2]-g[0]['block_bbox'][0]>.3]
        if len(body)<3:continue
        if (box[0]>.94 and any(g[0]['block_bbox'][2]>box[0]-.02 for g in body)
                or box[2]<.06 and any(g[0]['block_bbox'][0]<box[2]+.02 for g in body)):continue
        x0,x1=max(0,int(box[0]*width)),min(width,int(np.ceil(box[2]*width)))
        y0,y1=max(0,int(box[1]*height)),min(height,int(np.ceil(box[3]*height)))
        crop=labels[y0:y1,x0:x1];total=np.count_nonzero(ink[y0:y1,x0:x1])
        if not total:continue
        for label,bounds in strokes:
            if not bounds[0]-3/width<=box[0]<box[2]<=bounds[2]+3/width:continue
            owned=np.count_nonzero(crop==label)/total
            # Visible letters beside the edge create separate ink components.
            # No script/language guess or recognized words influence removal.
            if owned<.95:continue
            other=crop[(crop!=0)&(crop!=label)]
            if other.size:
                parts,counts=np.unique(other,return_counts=True)
                if any(n>=height*width*.000025 or stats[k,3]>=height*.008
                       for k,n in zip(parts,counts)):continue
            excluded.append(atoms)
            evidence.append(dict(group=atoms[0]['group'],method=METHOD,
                stroke_bbox=bounds,ocr_bbox=box,stroke_ink_fraction=float(owned)))
            break
    if (not excluded or sum(a['length'] for g in excluded for a in g)>
            .08*sum(a['length'] for g in groups for a in g)):return None
    return dict(excluded_groups=[g[0]['group'] for g in excluded],
        excluded_text=[''.join(a['content'] for a in g) for g in excluded],
        crossing_groups=[],binding_artifacts=evidence,
        binding_recognition_bounds=[max([0]+[e['stroke_bbox'][2]+2/width for e in evidence if e['stroke_bbox'][2]<.06]),
            min([1]+[e['stroke_bbox'][0]-2/width for e in evidence if e['stroke_bbox'][0]>.94])])
