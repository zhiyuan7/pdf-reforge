"""Recover an omitted source image between independently located prose runs.

Only an entire, unchanged plain-text block matched by two image-only scales
can be split. Mathematical OCR is never used as a transcription witness.
"""
from copy import deepcopy

from book_pipeline.ingestion.formula_text_anchors import observe_prose
from book_pipeline.ingestion.ruled_prose import page_box

METHOD = 'two_scale_vertical_prose_formula_gap_v6'
# These are traditional glyph forms of the same character. Matching may use
# either form; the book's existing transcription is always retained verbatim.
GLYPH_FORMS = str.maketrans({'爲': '為'})


def canonical(text):
    return text.translate(GLYPH_FORMS)


def displaced_unique_owner(block, text, witness, blocks, absolute, page_size):
    """A complete unique paragraph can locate a slightly displaced source box.

    This supplies only ownership; both scales and all ink/prose vetoes still
    have to pass. A long literal prefix, same ruled cell, vertical overlap and
    at most one glyph of horizontal displacement are required.
    """
    first, second, offset, pitch = witness
    if offset < 6 or sum(canonical(''.join(p['content'] for p in b['parts'])).count(canonical(text))
                         for b in blocks if b['type'] != 'image') != 1:
        return False
    left, top, right, bottom = absolute
    x,y,z,t = [v*page_size[k%2] for k,v in enumerate(block['bbox'])]
    a,b,c,d = first['bbox']; l,u,r,v = second['bbox']
    center = (a+c+l+r)/4 + left
    overlap = max(0,min(t,top+v)-max(y,top+b))
    return (left <= (x+z)/2 < right and top <= (y+t)/2 < bottom
            and max(0,x-center,center-z) <= pitch
            and overlap >= .5*min(t-y,v-b))


def split_witness(text, rows):
    """Require unique complete prefix/suffix, in one vertical column."""
    if not 6 <= len(text) <= 60:
        return None
    matches = []
    for first in rows:
        for second in rows:
            prefix, suffix = first['text'], second['text']
            if len(prefix) < 2 or len(suffix) < 2:
                continue
            if canonical(prefix + suffix) != canonical(text):
                continue
            a, b, c, d = first['bbox']; x, y, z, t = second['bbox']
            pitch = ((d-b)/len(prefix) + (t-y)/len(suffix))/2
            if (abs((a+c-x-z)/2) <= .5*pitch
                    and .2*pitch <= y-d <= 1.5*pitch
                    and .75 <= ((d-b)/len(prefix))/((t-y)/len(suffix)) <= 1.33):
                matches.append((first, second, len(prefix), pitch))
    return matches[0] if len(matches) == 1 else None


def compact_bounds(rgb, boxes, first, second, pitch, wide=False, evidence=None):
    """Select only compact ink in the observed gap, inside its prose column."""
    from book_pipeline.ingestion.formula_crops import ink_bounds
    a, b, c, d = first['bbox']; x, y, z, t = second['bbox']
    center = (a+c+x+z)/4
    chosen = [n for n, (l, u, r, v) in enumerate(boxes)
              if d-(.6 if wide else 0)*pitch <= (u+v)/2 <= y and v-u <= (2.2 if wide else 1.7)*pitch
              and abs((l+r)/2-center) <= (4.5 if wide else 1.6)*pitch
              and d-(.6 if wide else .4)*pitch <= u < v <= y+(.6 if wide else .4)*pitch]
    if not chosen:
        return None
    bounds = ink_bounds(rgb, boxes, chosen)
    if bounds is None:
        return None
    if wide:
        from book_pipeline.ingestion.formula_gap_ink import extend_scripts
        bounds,tails = extend_scripts(rgb,boxes,chosen,bounds,first,second,pitch)
        if evidence is not None and tails:
            evidence['script_components'] = tails
    l, u, r, v = bounds
    if (not d-(.6 if wide else .4)*pitch <= u < v <= y+(.6 if wide else .4)*pitch
            or r-l > (9 if wide else 4)*pitch or v-u > (3 if wide else 2)*pitch):
        return None
    return bounds


def preceding_bounds(rgb, boxes, first, pitch, original):
    """Tighten an existing horizontal formula before a proved prose split.

    The complete paragraph provides ownership. Its short prefix alone is not
    an anchor. Require substantial overlap with the existing image and select
    only horizontal detector regions, excluding ordinary vertical glyphs.
    """
    from book_pipeline.ingestion.formula_crops import ink_bounds
    a,b,c,d = first['bbox']
    center = (a+c)/2
    if not original[0]-.3*pitch <= center <= original[2]+.3*pitch:
        return None
    selected = []
    for n,(l,u,r,v) in enumerate(boxes):
        overlap = max(0,min(r,original[2])-max(l,original[0])) * max(0,min(v,original[3])-max(u,original[1]))
        if (r-l >= 1.6*(v-u) and v-u <= 1.7*pitch
                and b-4*pitch <= (u+v)/2 <= b+.2*pitch
                and abs((l+r)/2-center) <= 4.5*pitch
                and overlap >= .2*(r-l)*(v-u)):
            selected.append(n)
    bounds = ink_bounds(rgb,boxes,selected) if selected else None
    if bounds is None:
        return None
    l,u,r,v = bounds
    if (not -.2*pitch <= b-v <= .7*pitch or v-u > 4*pitch or r-l > 9*pitch
            or not l <= center <= r):
        return None
    return bounds


def recover_gaps(rgb, region_cells, answer, output, page_size):
    """Append review evidence and return a new answer; never change input."""
    import numpy as np
    from PIL import Image
    replacements, evidence = {}, []
    for cell in region_cells:
        if not cell.get('regions'):
            continue
        absolute = cell['pixel_bbox']; left, top, right, bottom = absolute
        local = rgb[top:bottom, left:right]
        from book_pipeline.ingestion.vertical_prose_fragments import observe_fragments
        long_rows = observe_prose(local)
        fragment_rows = None
        # Long-run matching remains the first choice. Short runs are eligible
        # only as parts of a complete unchanged paragraph of six or more glyphs.
        candidates = []
        for n, block in enumerate(answer['blocks']):
            if block['type'] != 'text' or any(p['kind'] != 'text' for p in block['parts']):
                continue
            text = ''.join(p['content'] for p in block['parts'])
            if not 6 <= len(text) <= 60:
                continue
            # A long witness in another paragraph must not suppress this
            # paragraph's short-run fallback. Prefer the long witness for each
            # complete source string, without mixing readings between modes.
            rows, fragments = long_rows, False
            witness = split_witness(text, rows)
            if witness is None:
                if fragment_rows is None:
                    fragment_rows = observe_fragments(local)
                rows, fragments = fragment_rows, True
                witness = split_witness(text, rows)
            if n not in replacements and witness:
                # Original geometry must still own this column.
                f, s, offset, pitch = witness
                center = (f['bbox'][0]+f['bbox'][2])/2 + left
                owners = [b for b in answer['blocks'] if b['type'] != 'image'
                          and canonical(''.join(p['content'] for p in b['parts'])).count(canonical(text))
                          and b['bbox'][0]*page_size[0]-.3*pitch <= center <= b['bbox'][2]*page_size[0]+.3*pitch]
                if ((len(owners) == 1 and owners[0] is block)
                        or (not owners and displaced_unique_owner(block,text,witness,
                            answer['blocks'],absolute,page_size))):
                    candidates.append((n, text, witness, rows, fragments))
        if not candidates:
            from book_pipeline.ingestion.ownerless_formula_gaps import recover_cell
            recovered, details = recover_cell(rgb,cell,answer,output,page_size,
                                             fragment_rows if fragment_rows is not None else long_rows)
            replacements.update(recovered)
            evidence.extend(details)
            continue
        scaled = np.asarray(Image.fromarray(local).resize((local.shape[1]*2, local.shape[0]*2),
                                                         Image.Resampling.NEAREST))
        second_by_mode = {}
        for n, text, witness, rows, fragments in candidates:
            if fragments not in second_by_mode:
                second_rows = observe_fragments(scaled) if fragments else observe_prose(scaled)
                for row in second_rows:
                    row['bbox'] = [v/2 for v in row['bbox']]
                second_by_mode[fragments] = second_rows
            second_rows = second_by_mode[fragments]
            first, second, offset, pitch = witness
            corroboration = split_witness(text, second_rows)
            if not corroboration or corroboration[2] != offset:
                continue
            if any(max(abs(x-y) for x,y in zip(a['bbox'], b['bbox'])) > .15*pitch
                   for a,b in zip((first, second), corroboration[:2])):
                continue
            script_evidence = {}
            bounds = compact_bounds(local, cell['regions'], first, second, pitch, wide=fragments,
                                    evidence=script_evidence)
            if bounds is None:
                continue
            if script_evidence:
                # Detector-missed scripts also need independent scale geometry.
                scaled_first,scaled_second = deepcopy(corroboration[:2])
                for row in (scaled_first,scaled_second):
                    row['bbox'] = [v*2 for v in row['bbox']]
                scaled_bounds = compact_bounds(scaled,[[v*2 for v in b] for b in cell['regions']],
                    scaled_first,scaled_second,corroboration[3]*2,wide=True)
                if scaled_bounds is None or max(abs(v-w/2) for v,w in zip(bounds,scaled_bounds)) > .15*pitch:
                    continue
                script_evidence['second_scale_bounds'] = scaled_bounds
            l,u,r,v = bounds
            mask = None
            if fragments:
                from book_pipeline.ingestion.prose_ink_exclusions import exclusions
                mask = exclusions(local,bounds,first,second,pitch)
            from book_pipeline.ingestion.formula_gap_ink import boundary_ink_clear
            if any(max(0,min(r,row['bbox'][2])-max(l,row['bbox'][0])) *
                   max(0,min(v,row['bbox'][3])-max(u,row['bbox'][1])) >
                   .08*(row['bbox'][2]-row['bbox'][0])*(row['bbox'][3]-row['bbox'][1])
                   and not (row in (first,second) and boundary_ink_clear(local,bounds,row,mask)) for row in rows):
                continue
            def mapped(box):
                h,w = local.shape[:2]
                return page_box([box[0]/w,box[1]/h,box[2]/w,box[3]/h],absolute,page_size)
            box = mapped(bounds)
            if any(b['type']=='image' and
                   max(0,min(box[2],b['bbox'][2])-max(box[0],b['bbox'][0])) *
                   max(0,min(box[3],b['bbox'][3])-max(box[1],b['bbox'][1])) >
                   .1*(box[2]-box[0])*(box[3]-box[1])
                   for b in output['blocks']):
                continue
            original = deepcopy(output['blocks'][n])
            before = dict(original, bbox=mapped(first['bbox']), parts=[dict(kind='text', content=text[:offset])])
            after = dict(original, bbox=mapped(second['bbox']), parts=[dict(kind='text', content=text[offset:])])
            image = dict(type='image',bbox=box,parts=[],reason='Independently located source ink between complete prose runs; transcription unverified')
            if fragments:
                import hashlib
                if mask:
                    mask.update(pixel_bbox=[left+l,top+u,left+r,top+v],page_pixel_size=page_size,
                                page_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                    image['prose_ink_exclusions']=mask
            replacements[n] = [before,image,after]
            evidence.append(dict(method=METHOD,action='inserted_gap',block=n,cell=cell['cell'],first=first,second=second,
                                 second_scale=corroboration[:2],bbox=box,local_pixel_bbox=bounds,
                                 script_geometry=script_evidence,
                                 text_unchanged=True,requires_visual_review=True))
            if fragments and n > 0 and output['blocks'][n-1]['type'] == 'image':
                owner = output['blocks'][n-1]
                original_box = [v*page_size[i%2]-(left if i%2 == 0 else top)
                                for i,v in enumerate(owner['bbox'])]
                previous = preceding_bounds(local,cell['regions'],first,pitch,original_box)
                if previous is not None:
                    l,u,r,v = previous
                    safe = not any(max(0,min(r,row['bbox'][2])-max(l,row['bbox'][0])) *
                                   max(0,min(v,row['bbox'][3])-max(u,row['bbox'][1])) >
                                   .08*(row['bbox'][2]-row['bbox'][0])*(row['bbox'][3]-row['bbox'][1])
                                   for row in rows)
                    previous_box = mapped(previous)
                    conflict = any(j != n-1 and b['type'] == 'image' and
                        max(0,min(previous_box[2],b['bbox'][2])-max(previous_box[0],b['bbox'][0])) *
                        max(0,min(previous_box[3],b['bbox'][3])-max(previous_box[1],b['bbox'][1])) > 0
                        for j,b in enumerate(output['blocks']))
                    if safe and not conflict and n-1 not in replacements:
                        replacements[n-1] = [dict(deepcopy(owner),bbox=previous_box)]
                        evidence.append(dict(method=METHOD,action='tightened_preceding_image',block=n-1,
                            cell=cell['cell'],following_complete_prose=text,first=first,second=second,
                            second_scale=corroboration[:2],original_bbox=owner['bbox'],bbox=previous_box,
                            local_pixel_bbox=previous,text_unchanged=True,requires_visual_review=True))
            # The same complete paragraph witness can locate its existing
            # following image. It may not acquire another column's formula.
            if n+1 >= len(output['blocks']) or output['blocks'][n+1]['type'] != 'image':
                continue
            owner = output['blocks'][n+1]
            a,b,c,d = second['bbox']
            if not owner['bbox'][0]*page_size[0]-.3*pitch <= (a+c)/2+left <= owner['bbox'][2]*page_size[0]+.3*pitch:
                continue
            # A short terminal prose run at both scales can bound a longer
            # existing formula. Its ownership comes from this entire paragraph,
            # its original image column and its original rectangle overlap.
            from book_pipeline.ingestion.vertical_formula_tail import bounded_tail
            original_box = [v*page_size[i%2]-(left if i%2 == 0 else top)
                            for i,v in enumerate(owner['bbox'])]
            enclosed = bounded_tail(local,cell['regions'],second,rows,second_rows,pitch,original_box,
                [''.join(p['content'] for p in b['parts']) for b in output['blocks']])
            if enclosed:
                l,u,r,v = enclosed['bounds']
                tail_box = mapped(enclosed['bounds'])
                conflicts = [j for j,b in enumerate(output['blocks']) if j != n+1 and b['type'] == 'image' and
                    max(0,min(tail_box[2],b['bbox'][2])-max(tail_box[0],b['bbox'][0])) *
                    max(0,min(tail_box[3],b['bbox'][3])-max(tail_box[1],b['bbox'][1])) > 0]
                duplicate = None
                if len(conflicts) == 1:
                    from book_pipeline.ingestion.vertical_prose_duplicates import redundant_owner
                    duplicate = redundant_owner(local,output['blocks'],conflicts[0],n,text,first,second,pitch,
                        absolute,page_size,[enclosed['bounds'],enclosed['terminal']['bbox']])
                    if duplicate and any(j in replacements for j in duplicate['removed_blocks']):
                        duplicate = None
                if not conflicts or duplicate:
                    image = dict(deepcopy(owner),bbox=tail_box)
                    mask = enclosed['mask']
                    if mask:
                        import hashlib
                        mask.update(pixel_bbox=[left+l,top+u,left+r,top+v],page_pixel_size=page_size,
                                    page_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                        image['prose_ink_exclusions'] = mask
                    terminal = enclosed['terminal']
                    recovered = dict(type='text',bbox=mapped(terminal['bbox']),
                                     parts=[dict(kind='text',content=terminal['text'])])
                    replacements[n+1] = [image,recovered]
                    if n+2 < len(output['blocks']) and n+2 not in replacements:
                        from book_pipeline.ingestion.vertical_formula_tail import continue_in_next_column
                        continuation = continue_in_next_column(recovered,output['blocks'][n+2],absolute,page_size,pitch)
                        if continuation:
                            replacements[n+1] = [image]
                            replacements[n+2] = [continuation]
                            evidence.append(dict(method=enclosed['method'],action='continued_terminal_prose_in_next_column',
                                cell=cell['cell'],block=n+2,recovered_text=terminal['text'],
                                following_text=''.join(p['content'] for p in output['blocks'][n+2]['parts']),
                                source_boxes=[recovered['bbox'],output['blocks'][n+2]['bbox']],
                                requires_visual_review=True))
                    if duplicate:
                        for j in duplicate['removed_blocks']:
                            replacements[j] = []
                        evidence.append(dict(duplicate,action='merged_covered_duplicate_owner',cell=cell['cell']))
                    evidence.append(dict(method=enclosed['method'],action='bounded_following_image_and_terminal_prose',
                        block=n+1,cell=cell['cell'],preceding_complete_prose=text,preceding=second,
                        terminal=terminal,second_scale=enclosed['second_scale'],
                        original_bbox=owner['bbox'],bbox=tail_box,local_pixel_bbox=enclosed['bounds'],
                        recovered_text=terminal['text'],requires_visual_review=True))
                    continue
            tail = compact_bounds(local,cell['regions'],second,
                                  dict(bbox=[a,d+2*pitch,c,d+3*pitch]),pitch)
            if tail is None:
                continue
            l,u,r,v = tail
            if local.shape[0]-v > pitch or not -.2*pitch <= u-d <= .7*pitch:
                continue
            if any(top_ > v and abs((left_+right_)/2-(a+c)/2) <= 1.6*pitch
                   for left_,top_,right_,bottom_ in cell['regions']):
                continue
            if any(max(0,min(r,row['bbox'][2])-max(l,row['bbox'][0])) *
                   max(0,min(v,row['bbox'][3])-max(u,row['bbox'][1])) >
                   .08*(row['bbox'][2]-row['bbox'][0])*(row['bbox'][3]-row['bbox'][1]) for row in rows):
                continue
            tail_box = mapped(tail)
            if any(j != n+1 and b['type']=='image' and
                   max(0,min(tail_box[2],b['bbox'][2])-max(tail_box[0],b['bbox'][0])) *
                   max(0,min(tail_box[3],b['bbox'][3])-max(tail_box[1],b['bbox'][1])) > 0
                   for j,b in enumerate(output['blocks'])):
                continue
            replacements[n+1] = [dict(deepcopy(owner),bbox=tail_box)]
            evidence.append(dict(method=METHOD,action='relocated_following_image',block=n+1,
                                 cell=cell['cell'],preceding=second,second_scale=corroboration[1],
                                 original_bbox=owner['bbox'],bbox=tail_box,local_pixel_bbox=tail,
                                 text_unchanged=True,requires_visual_review=True))
    result = deepcopy(output)
    result['blocks'] = [item for n,b in enumerate(result['blocks']) for item in replacements.get(n,[b])]
    return result, evidence
