"""Native small-font margin geometry remains evidence after translation."""
from collections import defaultdict
from pathlib import Path
from statistics import median

from book_pipeline.publishing.native_lateral import native_groups


def observed_note_colors(index, rows):
    """Read ink for already-owned notes, including OCR on native-text pages.

    This supplies palette evidence only; it never infers or moves a note.
    Mixed ink, missing glyphs and body-size text retain the existing palette.
    """
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return rows
    from book_pipeline.structure.headings import valid_box
    import pymupdf
    result = []
    pages = {}
    with pymupdf.open(source) as pdf:
        for row in rows:
            if not row.get('note_ranges') or not 0 <= row.get('page', 0)-1 < len(pdf):
                result.append(row); continue
            number = row['page']-1
            page = pdf[number]
            if number not in pages:
                pages[number] = [dict(c, size=s['size'], color=s['color'])
                    for b in page.get_text('rawdict')['blocks'] for line in b.get('lines', [])
                    if line.get('dir') == (1., 0.) for s in line['spans']
                    for c in s['chars'] if c['c'].strip()]
            def observed(box):
                if not valid_box(box):
                    return []
                clip = pymupdf.Rect(*(v*(page.rect.width if n%2==0 else page.rect.height)
                                     for n,v in enumerate(box)))+(-.6,-.6,.6,.6)
                return [g for g in pages[number] if clip.contains(pymupdf.Point(
                    (g['bbox'][0]+g['bbox'][2])/2, (g['bbox'][1]+g['bbox'][3])/2))]
            body = observed(row.get('main_bbox'))
            size = median(g['size'] for g in body) if body else 0
            colors = list(row['note_colors'])
            evidence = []
            for n, span in enumerate(row['note_ranges']):
                if colors[n] is not None:
                    continue
                atoms = [a for a,_,_ in index.fragments(*span)]
                if any(a['page'] != number or not valid_box(a.get('bbox')) for a in atoms):
                    continue
                boxes = [a['bbox'] for a in atoms]
                box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                       max(b[2] for b in boxes), max(b[3] for b in boxes)]
                glyphs = observed(box)
                inks = {g['color'] for g in glyphs}
                if not glyphs or len(inks) != 1 or any(g['size'] > .93*size for g in glyphs):
                    continue
                ink = inks.pop()
                colors[n] = [(ink>>k & 255)/255 for k in (16,8,0)]
                evidence.append(dict(method='native_owned_note_ink', range=span, bbox=box,
                    source_text=''.join(g['c'] for g in glyphs), color=colors[n],
                    max_font_size=max(g['size'] for g in glyphs), body_font_size=size))
            result.append(dict(row, note_colors=colors, note_ink_evidence=evidence) if evidence else row)
    return result


def native_margin_evidence(index, *, allow_fragmented_owners=False):
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return {}
    groups = native_groups(index)
    allowed = {'text','paragraph','paragraph_title','index','aside_text','page_footnote'}
    body = [g for g in groups if .3 <= g['bbox'][2]-g['bbox'][0] <= .8
            and .009 <= g['bbox'][3]-g['bbox'][1] <= .28
            and all(a['kind'] in {'text','math'} and a['role'] in {'text','paragraph','paragraph_title'}
                    for a in g['atoms'])]
    candidates = defaultdict(list)
    for note in groups:
        c = note['bbox']
        if (not .035 <= c[2]-c[0] <= .22 or not 0 < c[3]-c[1] <= .12
                or not 3 <= note['hi']-note['lo'] <= 240
                or not all(a['kind'] in {'text','math'} and a['role'] in allowed
                           and not a.get('recovery_issue') for a in note['atoms'])):
            continue
        owners = [g for g in body if g['page']==note['page']
            and (.008 <= c[0]-g['bbox'][2] <= .12 or .008 <= g['bbox'][0]-c[2] <= .12)
            and min(c[3],g['bbox'][3])-max(c[1],g['bbox'][1]) >=
                .35*min(c[3]-c[1],g['bbox'][3]-g['bbox'][1])]
        if len(owners)==1 or allow_fragmented_owners and owners:
            candidates[note['page']].append((note,owners))
    if not candidates:
        return {}
    import pymupdf
    result = {}
    with pymupdf.open(source) as pdf:
        for number, pairs in candidates.items():
            if not 0 <= number < len(pdf):
                continue
            page = pdf[number]
            glyphs = [dict(c,size=s['size'],color=s['color'])
                for b in page.get_text('rawdict')['blocks'] for line in b.get('lines',[])
                if line.get('dir') == (1.,0.) for s in line['spans']
                for c in s['chars'] if c['c'].strip()]
            def observed(box):
                clip = pymupdf.Rect(*(v*(page.rect.width if n%2==0 else page.rect.height)
                                     for n,v in enumerate(box)))+(-.6,-.6,.6,.6)
                return [g for g in glyphs if clip.contains(pymupdf.Point(
                    (g['bbox'][0]+g['bbox'][2])/2,(g['bbox'][1]+g['bbox'][3])/2))]
            for note, owners in pairs:
                small = observed(note['bbox'])
                main = [g for owner in owners for g in observed(owner['bbox'])]
                if not small or not main:
                    continue
                size = median(g['size'] for g in main)
                if any(g['size'] > .85*size for g in small):
                    continue
                colors = {g['color'] for g in small}
                color = next(iter(colors)) if len(colors)==1 else None
                result[note['group']] = dict(method='native_small_margin_glyphs',
                    owner=owners[0]['group'] if len(owners)==1 else None,
                    owners=[o['group'] for o in owners], page=number+1, bbox=note['bbox'],
                    color=([(color>>k & 255)/255 for k in (16,8,0)] if color is not None else None),
                    source_text=''.join(g['c'] for g in small),
                    max_font_size=max(g['size'] for g in small), body_font_size=size)
    return result
