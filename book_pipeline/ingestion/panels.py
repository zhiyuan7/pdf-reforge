"""Source geometry for text panels, independent of semantic keyword counts.

Only vector-backed panels are recovered here. Raster scans require separate
visual evidence; no synthetic panel is inferred from a declaration's wording.
"""
from collections import Counter, defaultdict


METHOD = 'vector_text_panels_v3'


def visible_fill(drawings, box, title=None):
    """Composite broad interior layers in PDF paint order, excluding title strips."""
    body = type(box)(box)
    if title is not None:
        body.y0 = max(body.y0, title.y1)
    inner = body + (body.width*.08, body.height*.2, -body.width*.08, -body.height*.2)
    rgb = [1., 1., 1.]
    for path in drawings:
        if ((path['rect'] & inner).get_area() < inner.get_area()*.98
                or path['rect'].get_area() > box.get_area()*1.05):
            continue
        opacity = path.get('fill_opacity', 1.)
        rgb = [c*opacity + old*(1-opacity) for c, old in zip(path['fill'], rgb)]
    return [round(v, 4) for v in rgb]


def title_ink(page, box):
    colors = Counter()
    for block in page.get_text('dict', clip=box)['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                colors[span['color']] += len(span['text'].strip())
    color = colors.most_common(1)[0][0] if colors else 0
    return [round(((color >> shift) & 255)/255, 4) for shift in (16, 8, 0)]


def text_panels(page):
    """Recover flat rectangular/rounded backgrounds containing live prose."""
    area = page.rect.get_area()
    drawings = [d for d in page.get_drawings() if d.get('fill') is not None
                and 1 <= len(d['items']) <= 12
                and all(item[0] in {'l', 'c', 're', 'qu'} for item in d['items'])
                and d['rect'].width > 70 and d['rect'].height > 8
                and area*.002 < d['rect'].get_area() < area*.8]
    colored = [d for d in drawings if min(d['fill']) < .985]
    selected = []
    for d in sorted(colored, key=lambda d: -d['rect'].get_area()):
        box = d['rect']
        if any((box & old['rect']).get_area() >= box.get_area()*.95 for old in selected):
            continue
        if len(page.get_text(clip=box).strip()) >= 12:
            selected.append(d)
    panels = []
    title_paths = set()
    for d in selected:
        box = d['rect']
        # A title strip can straddle the upper border, or sit inside it.
        strips = [s for s in colored if s is not d and s['rect'].height < min(40, box.height*.75)
                  and abs(s['rect'].y0-box.y0) <= s['rect'].height
                  and (s['rect'] & box).width >= box.width*.7
                  and max(s['fill'])-min(s['fill']) > .15
                  and len(page.get_text(clip=s['rect']).strip()) >= 3]
        strip = max(strips, key=lambda s: s['rect'].width, default=None)
        if strip:
            title_paths.add(id(strip))
        full = box | strip['rect'] if strip else box
        rgb = strip['fill'] if strip else d['fill']
        panels.append(dict(path=id(d), bbox=[round(v/page.rect.width if k%2==0 else v/page.rect.height, 5)
                                            for k,v in enumerate(full)],
                           title_bbox=([round(v/page.rect.width if k%2==0 else v/page.rect.height, 5)
                                        for k,v in enumerate(strip['rect'])] if strip else None),
                           color=[round(v, 4) for v in rgb],
                           background_color=visible_fill(drawings, box, strip['rect'] if strip else None),
                           frame_color=[round(v, 4) for v in (d.get('color') or d['fill'])],
                           title_color=([round(v, 4) for v in strip['fill']] if strip else None),
                           title_text_color=(title_ink(page, strip['rect']) if strip else None),
                           method=METHOD))
    panels = [p for p in panels if p.pop('path') not in title_paths]
    for n,panel in enumerate(sorted(panels, key=lambda p:(p['bbox'][1], p['bbox'][0]))):
        panel['id'] = f'p{page.number+1:06d}-panel-{n+1:03d}'
    return panels


def intersection_fraction(box, region):
    if not box or not region:
        return 0
    area = max(0,box[2]-box[0])*max(0,box[3]-box[1])
    overlap = max(0,min(box[2],region[2])-max(box[0],region[0]))*max(0,min(box[3],region[3])-max(box[1],region[1]))
    return overlap/area if area else 0


def annotate_panels(page, blocks):
    for panel in text_panels(page):
        for block in blocks:
            if block['type'] not in {'text','paragraph','paragraph_title','doc_title','equation'}:
                continue
            if intersection_fraction(block.get('bbox'),panel['bbox']) >= .9:
                block['source_panel'] = panel
                block['source_panel_title'] = intersection_fraction(block.get('bbox'),panel['title_bbox']) >= .8


def panel_environments(index, registry):
    """Return wrappers only for contiguous, fully owned, nonoverlapping panels.

    A mixed source block or intervening side note is never pulled into a box.
    Semantic objects keep priority. Rejected geometry is explicitly auditable.
    """
    groups = defaultdict(list)
    for n,atom in enumerate(index.atoms):
        if atom.get('source_panel'):
            groups[atom['source_panel']['id']].append(n)
    wrappers, evidence = [], []
    for pid,positions in groups.items():
        first,last = min(positions),max(positions)
        panel = index.atoms[first]['source_panel']
        lo,hi = index.prefix[first],index.prefix[last+1]
        row = dict(method=panel.get('method', METHOD), panel=pid, bbox=panel['bbox'], lo=lo, hi=hi,
                   page=index.atoms[first]['page']+1, color=panel['color'])
        if positions != list(range(first,last+1)):
            evidence.append({**row,'status':'rejected','reason':'noncontiguous_source_ownership'})
            continue
        overlaps = [o for o in registry if o['lo'] < hi and lo < o['hi']
                    and o['kind'] not in {'equation','formula','figure','table'} and not o.get('anchor_only')]
        if overlaps:
            evidence.append({**row,'status':'rejected','reason':'existing_semantic_environment'})
            continue
        title_end = first
        while title_end <= last and index.atoms[title_end].get('source_panel_title'):
            title_end += 1
        wrapper = dict(kind='source_panel',lo=lo,hi=hi,anchor='source-'+pid,
                       panel_color=panel['color'],number='',name='',label='')
        if panel.get('background_color'):
            wrapper['panel_background'] = panel['background_color']
            wrapper['panel_frame'] = panel.get('frame_color', panel['color'])
            wrapper['panel_title_color'] = panel.get('title_color')
            wrapper['panel_title_ink'] = panel.get('title_text_color', [0, 0, 0])
        if panel.get('frame_visible'):
            wrapper['panel_frame_visible'] = True
        if first < title_end <= last:
            wrapper['body_start'] = index.address(index.prefix[title_end])
        wrappers.append(wrapper)
        evidence.append({**row,'status':'rendered','title_separated':'body_start' in wrapper})
    return wrappers,evidence


def panel_end_marks(index, wrappers, edited_atoms=()):
    """Align an existing isolated end mark only when the source owns it.

    The final, single-atom group must be a small pictographic mark at a
    recovered panel's lower right. Symbols in prose/formulas, uncertain
    ownership and edited marks retain their ordinary rendering. Missing
    decorations are never inferred from a panel's type or color.
    """
    marks = {}
    for wrapper in wrappers:
        last = index.ids[index.address(wrapper['hi'], end=True)['atom']]
        atom = index.atoms[last]
        panel = atom.get('source_panel', {})
        box, frame = atom.get('bbox'), panel.get('bbox')
        if (atom['id'] in edited_atoms or atom['kind'] != 'text'
                or atom['content'].strip() not in '♣♧♠♤♥♡♦♢■□▪◇◆'
                or len(atom['content'].strip()) != 1
                or atom.get('source_panel_title')
                or index.groups[atom['group']] != [last]
                or wrapper['anchor'] != 'source-' + panel.get('id', '')
                or not box or not frame or intersection_fraction(box, frame) < .9):
            continue
        width, height = frame[2]-frame[0], frame[3]-frame[1]
        glyph_width, glyph_height = box[2]-box[0], box[3]-box[1]
        if (width <= 0 or height <= 0 or not 0 < glyph_width <= .08*width
                or not 0 < glyph_height <= .4*height
                or box[0] < frame[0]+.8*width or box[1] < frame[1]+.5*height
                or frame[2]-box[2] > .08*width
                or frame[3]-box[3] > .6*glyph_height):
            continue
        marks[atom['id']] = dict(method='source_panel_lower_right_end_mark',
            atom=atom['id'], panel=panel['id'], bbox=box, panel_bbox=frame,
            lo=index.prefix[last], hi=index.prefix[last+1], color=panel['color'])
    return marks
