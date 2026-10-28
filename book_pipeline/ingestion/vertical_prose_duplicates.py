"""Merge a redundant image owner only when source ink is already preserved.

A repeated literal paragraph, two-scale complete source witness and incompatible
old prose geometry are all required. An ungrounded image alone is not deletable.
"""
METHOD = 'covered_image_duplicate_prose_owner_v1'


def redundant_owner(rgb, blocks, conflict, paragraph, full_text, first, second, pitch,
                    cell_box, page_size, preserved_bounds):
    import cv2
    import numpy as np
    from book_pipeline.ingestion.vertical_formula_gaps import canonical

    # Only a separate, earlier text/image pair can be a duplicate of the
    # witnessed paragraph and its following formula.
    if not 0 < conflict < paragraph-1 or blocks[conflict]['type'] != 'image':
        return None
    duplicate = blocks[conflict-1]
    if (duplicate['type'] != 'text' or any(p['kind'] != 'text' for p in duplicate['parts'])
            or canonical(''.join(p['content'] for p in duplicate['parts'])) != canonical(full_text)):
        return None
    copies = [b for b in blocks if b['type'] == 'text' and
              canonical(''.join(p['content'] for p in b['parts'])) == canonical(full_text)]
    if len(copies) != 2:
        return None
    left,top,right,bottom = cell_box
    h,w = rgb.shape[:2]
    def local(box):
        return [v*page_size[i%2]-(left if i%2 == 0 else top) for i,v in enumerate(box)]
    text_box, image_box = local(duplicate['bbox']), local(blocks[conflict]['bbox'])
    if any(not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)
           for box in (text_box,image_box)):
        return None
    # The other box cannot hold the complete paragraph at its independently
    # observed physical glyph scale, and it does not cover the actual column.
    if (text_box[3]-text_box[1] >= .8*len(full_text)*pitch
            or any(max(0,min(text_box[2],row['bbox'][2])-max(text_box[0],row['bbox'][0])) > .1*pitch
                   for row in (first,second))):
        return None
    _,ink=cv2.threshold(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),0,255,
                        cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    _,labels,stats,_=cv2.connectedComponentsWithStats(ink,8)
    preserved=np.zeros((h,w),bool)
    for box in preserved_bounds:
        a,b,c,d=[round(v) for v in box]
        preserved[max(0,b):min(h,d),max(0,a):min(w,c)]=True
    a,b,c,d=[round(v) for v in image_box]
    old=np.zeros((h,w),bool);old[b:d,a:c]=True
    checked=0; ignored=0
    for label,(x,y,cw,ch,area) in enumerate(stats[1:],1):
        member=labels[y:y+ch,x:x+cw]==label
        overlap=member & old[y:y+ch,x:x+cw]
        count=int(overlap.sum())
        if not count:
            continue
        # Rules at this cell's boundary and isolated paper specks cannot
        # establish another formula owner. All glyph-sized ink must survive.
        rule=((cw >= .9*w and ch <= .2*pitch and (y <= .2*pitch or y+ch >= h-.2*pitch))
              or (ch >= .9*h and cw <= .2*pitch and (x <= .2*pitch or x+cw >= w-.2*pitch)))
        if rule or area < 6 or cw+ch < .5*pitch:
            ignored += count
            continue
        if np.any(overlap & ~preserved[y:y+ch,x:x+cw]):
            return None
        checked += count
    if checked < .1*pitch*pitch:
        return None
    return dict(method=METHOD,removed_blocks=[conflict-1,conflict],retained_paragraph=paragraph,
                duplicate_text=full_text,source_image_bbox=image_box,
                preserved_bounds=preserved_bounds,covered_ink_pixels=checked,
                ignored_rule_or_speck_pixels=ignored,requires_visual_review=True)
