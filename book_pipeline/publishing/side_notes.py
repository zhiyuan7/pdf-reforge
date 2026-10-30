"""Bounded lateral notes supported by adjacent, disjoint source geometry."""
from collections import OrderedDict
from pathlib import Path


def explicit_side_note_rows(index, registry=(), exclusions=(), changed=()):
    """Attach explicitly classified asides to a unique disjoint paragraph.

    OCR often emits all asides after the main column. Source geometry, rather
    than that serialization order or English glyph identity, owns the note;
    this also preserves the relationship after translation.
    """
    from book_pipeline.publishing.native_lateral import native_groups
    groups = native_groups(index, allow_ocr=True)
    def available(group):
        return (not any(a['id'] in changed or a.get('recovery_issue') for a in group['atoms'])
                and not any(r['lo'] < group['hi'] and group['lo'] < r['hi']
                            for r in [*registry, *exclusions]))
    mains = [g for g in groups if available(g)
             and all(a['kind'] in {'text', 'math'} and a['role'] in {'text', 'paragraph'}
                     for a in g['atoms'])
             and .3 <= g['bbox'][2]-g['bbox'][0] <= .8
             and .015 <= g['bbox'][3]-g['bbox'][1] <= .28
             and 40 <= g['hi']-g['lo'] <= 1800]
    owned = OrderedDict()
    for note in groups:
        c = note['bbox']
        if (not available(note) or not all(a['kind'] in {'text', 'math'} and a['role']=='aside_text'
                                          for a in note['atoms'])
                or not .035 <= c[2]-c[0] <= .22 or not 0 < c[3]-c[1] <= .15
                or not 4 <= note['hi']-note['lo'] <= 240):
            continue
        candidates = []
        for main in mains:
            b = main['bbox']
            overlap = min(b[3],c[3])-max(b[1],c[1])
            if (main['page']==note['page']
                    and (.008 <= c[0]-b[2] <= .12 or .008 <= b[0]-c[2] <= .12)
                    and overlap >= .5*min(b[3]-b[1],c[3]-c[1])):
                candidates.append(main)
        if len(candidates)==1:
            main=candidates[0]
            owned.setdefault(main['group'], (main, []))[1].append(note)
    rows = []
    for main, notes in owned.values():
        b=main['bbox'];notes.sort(key=lambda n:n['bbox'][1])
        # Mixed sides or overlapping notes have no unambiguous column layout.
        if (len({n['bbox'][0]>b[2] for n in notes})!=1
                or any(a['bbox'][3]>z['bbox'][1] for a,z in zip(notes,notes[1:]))):
            continue
        c=[min(n['bbox'][0] for n in notes),min(n['bbox'][1] for n in notes),
           max(n['bbox'][2] for n in notes),max(n['bbox'][3] for n in notes)]
        extent=max(b[2],c[2])-min(b[0],c[0])
        widths=[(b[2]-b[0])/extent,(c[2]-c[0])/extent]
        if not .07 <= widths[1] <= .3:continue
        rows.append(dict(method='explicit_geometric_side_note',lo=main['lo'],hi=main['hi'],
            main_range=[main['lo'],main['hi']],note_ranges=[[n['lo'],n['hi']] for n in notes],
            note_colors=[None]*len(notes),widths=widths,main_left=b[0]<c[0],
            page=main['page']+1,main_bbox=b,note_bbox=c,targets=[]))
    return rows


def side_note_rows(index, panels, registry, exclusions=(), changed=()):
    groups = OrderedDict()
    for n, atom in enumerate(index.atoms):
        groups.setdefault(atom['group'], []).append(n)
    rows, occupied = [], []
    group_list = list(groups.values())
    for panel in panels:
        if panel.get('body_start'):
            continue
        members = [n for n, a in enumerate(index.atoms)
                   if (a.get('source_panel') or {}).get('id') == panel['anchor'][7:]]
        if not members:
            continue
        frame = index.atoms[members[0]]['source_panel']['bbox']
        # Full-width declarations and tall notes retain their breakable flow.
        if not .15 <= frame[2]-frame[0] <= .52 or frame[3]-frame[1] > .36:
            continue
        for positions in group_list:
            lo, hi = index.prefix[positions[0]], index.prefix[positions[-1]+1]
            if hi != panel['lo'] and lo != panel['hi']:
                continue
            atoms = [index.atoms[n] for n in positions]
            if (any(a['page'] != index.atoms[members[0]]['page'] or a.get('source_panel')
                    or a['kind'] not in {'text', 'math'} or a['role'] not in {'text', 'paragraph', 'equation_inline'}
                    or a.get('block_type') not in {'text', 'paragraph'} for a in atoms)):
                continue
            boxes = [a.get('block_bbox') or a.get('bbox') for a in atoms]
            if not all(b and len(b) == 4 for b in boxes):
                continue
            box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                   max(b[2] for b in boxes), max(b[3] for b in boxes)]
            wrapping = wrapped_geometry(index, atoms, box, frame, changed)
            original_box = box
            if wrapping:
                box = wrapping['narrow_bbox']
            overlap = min(frame[3], box[3])-max(frame[1], box[1])
            if (min(frame[3]-frame[1], box[3]-box[1]) <= 0
                    or overlap < .65*min(frame[3]-frame[1], box[3]-box[1])
                    or abs(frame[1]-box[1]) > .05 or box[3]-box[1] > .36):
                continue
            left, right = sorted([box, frame], key=lambda b:b[0])
            gap = right[0]-left[2]
            if not .008 <= gap <= .12 or left[2]-left[0] < .15:
                continue
            start, stop = min(lo, panel['lo']), max(hi, panel['hi'])
            if stop-start > 1400 or any(start < z and a < stop for a, z in occupied):
                continue
            # Boxes cannot straddle another environment, footnote, or exclusion.
            if any(o['lo'] < stop and start < o['hi'] for o in [*registry, *exclusions]):
                continue
            extent = right[2]-left[0]
            widths = [(b[2]-b[0])/extent for b in (box, frame)]
            if min(widths) < .28:
                continue
            main_first = lo < panel['lo']
            first_left = (box if main_first else frame)[0] == left[0]
            rows.append(dict(method='bounded_adjacent_source_side_note', panel=panel['anchor'],
                             lo=start, mid=hi if main_first else panel['hi'], hi=stop,
                             widths=widths if main_first else widths[::-1], first_left=first_left,
                             main_bbox=box, note_bbox=frame, page=atoms[0]['page']+1))
            if wrapping:
                rows[-1].update(wrapping, method='native_wrapped_source_side_note',
                                main_first=main_first, main_bbox=original_box)
            occupied.append((start, stop))
            panel['lateral_note'] = True
            break
    return rows


def wrapped_geometry(index, atoms, box, frame, changed=()):
    """Accept a narrowing only when live source lines prove the wrap.

    A paragraph's union bbox overlaps the note even though its first lines do
    not. Read those lines from the original PDF without changing extraction or
    text. Raster/edited/ambiguous groups keep their ordinary flow.
    """
    if (box[2] <= frame[0] or frame[2] <= box[0]
            or any(a.get('extraction_method') != 'native' or a['id'] in changed for a in atoms)
            or abs(box[1]-frame[1]) > .015 or frame[3] >= box[3]
            or box[3]-box[1] > .36):
        return None
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return None
    import pymupdf
    with pymupdf.open(source) as pdf:
        if not 0 <= atoms[0]['page'] < len(pdf):
            return None
        page = pdf[atoms[0]['page']]
        dimensions = [page.rect.width, page.rect.height]*2
        clip = pymupdf.Rect([v*d for v,d in zip(box,dimensions)]) + (-.2,-.2,.2,.2)
        lines = []
        for block in page.get_text('dict', clip=clip)['blocks']:
            for line in block.get('lines', []):
                if not ''.join(s['text'] for s in line['spans']).strip():
                    continue
                b = [v/d for v,d in zip(line['bbox'],dimensions)]
                # Other content in the block's union can only be the note.
                if (frame[0]-.001 <= b[0] < b[2] <= frame[2]+.001
                        and frame[1]-.001 <= b[1] < b[3] <= frame[3]+.001):
                    continue
                lines.append(b)
    early = [b for b in lines if b[1] < frame[3] and frame[1] < b[3]]
    later = [b for b in lines if b[1] >= frame[3]]
    if not early or not later or abs(min(b[1] for b in early)-box[1]) > .003:
        return None
    left = all(b[2]+.008 <= frame[0] for b in early)
    right = all(frame[2]+.008 <= b[0] for b in early)
    if not (left or right):
        return None
    narrow = [box[0] if left else min(b[0] for b in early), min(b[1] for b in early),
              max(b[2] for b in early) if left else box[2], max(b[3] for b in early)]
    # At least one following line must actually return across the note column.
    if not any(b[0] < frame[2]-.03 and frame[0]+.03 < b[2] for b in later):
        return None
    extent = max(box[2], frame[2])-min(box[0], frame[0])
    gap = frame[0]-narrow[2] if left else narrow[0]-frame[2]
    if not .008 <= gap <= .12 or not .15 <= narrow[2]-narrow[0] <= .52:
        return None
    return dict(narrow_bbox=narrow, note_left=right,
                note_width=(frame[2]-frame[0])/extent, gutter=gap/extent,
                source_narrow_lines=len(early), source_full_lines=len(later))
