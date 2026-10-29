"""Source geometry for text panels, independent of semantic keyword counts.

Only vector-backed panels are recovered here. Raster scans require separate
visual evidence; no synthetic panel is inferred from a declaration's wording.
"""
from collections import Counter, defaultdict


METHOD = 'rectilinear_text_panels_v5'


def flat_panel_path(path):
    """Require four straight sides, allowing curved corners but not ellipses."""
    box = path['rect']
    if box.width <= 0 or box.height <= 0 or not 1 <= len(path['items']) <= 12:
        return False
    edges = []
    for item in path['items']:
        if item[0] == 're':
            r = item[1]
            edges += [(r.tl, r.tr), (r.tr, r.br), (r.br, r.bl), (r.bl, r.tl)]
        elif item[0] == 'l':
            edges.append(item[1:3])
        elif item[0] == 'qu':
            q = item[1]
            edges += [(q.ul, q.ur), (q.ur, q.lr), (q.lr, q.ll), (q.ll, q.ul)]
        elif item[0] != 'c':
            return False
    tolerance = max(.1, min(box.width, box.height)*.002)
    sides = set()
    for a, b in edges:
        if abs(a.y-b.y) <= tolerance and abs(a.x-b.x) >= box.width*.5:
            for name, y in (('top', box.y0), ('bottom', box.y1)):
                if abs(a.y-y) <= tolerance and abs(b.y-y) <= tolerance:
                    sides.add(name)
        elif abs(a.x-b.x) <= tolerance and abs(a.y-b.y) >= box.height*.5:
            for name, x in (('left', box.x0), ('right', box.x1)):
                if abs(a.x-x) <= tolerance and abs(b.x-x) <= tolerance:
                    sides.add(name)
    return len(sides) == 4


def panel_outline(path, drawings):
    """Find a visible matching stroke, including a separately painted outline."""
    for stroke in reversed(drawings):
        if (stroke.get('color') is not None and (stroke.get('width') or 0) > 0
                and stroke.get('stroke_opacity', 1) > 0 and flat_panel_path(stroke)
                and stroke.get('seqno', 0) >= path.get('seqno', 0)
                and max(abs(a-b) for a,b in zip(stroke['rect'],path['rect']))
                    <= max(.5, (stroke.get('width') or 0)/2)):
            return stroke
    return None


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
    paths = page.get_drawings()
    drawings = [d for d in paths if d.get('fill') is not None and flat_panel_path(d)
                and d['rect'].width > 70 and d['rect'].height > 8
                and area*.002 < d['rect'].get_area() < area*.8]
    colored = [d for d in drawings if min(d['fill']) < .985 or panel_outline(d, paths)]
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
        strips = [s for s in drawings if s is not d and s['rect'].height < min(40, box.height*.75)
                  and abs(s['rect'].y0-box.y0) <= s['rect'].height
                  and box.x0-.5 <= s['rect'].x0 < s['rect'].x1 <= box.x1+.5
                  and (s['rect'] & box).width >= max(35, box.width*.15)
                  and (s['fill'] != d['fill'] or panel_outline(d, paths))
                  and len(page.get_text(clip=s['rect']).strip()) >= 3]
        strip = max(strips, key=lambda s: s['rect'].width, default=None)
        if strip:
            title_paths.add(id(strip))
        full = box | strip['rect'] if strip else box
        rgb = strip['fill'] if strip else d['fill']
        outline = panel_outline(d, paths)
        panels.append(dict(path=id(d), bbox=[round(v/page.rect.width if k%2==0 else v/page.rect.height, 5)
                                            for k,v in enumerate(full)],
                           title_bbox=([round(v/page.rect.width if k%2==0 else v/page.rect.height, 5)
                                        for k,v in enumerate(strip['rect'])] if strip else None),
                           color=[round(v, 4) for v in rgb],
                           background_color=visible_fill(drawings, box, strip['rect'] if strip else None),
                           frame_color=[round(v, 4) for v in (outline['color'] if outline else d['fill'])],
                           frame_visible=outline is not None,
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
    # Saved native extractions may predate explicit frame evidence. Recheck
    # the same live panel; never infer an outline from its semantic type.
    live = {}
    from pathlib import Path
    source = index.source.get('source_pdf')
    if groups and source and Path(source).is_file():
        import pymupdf
        with pymupdf.open(source) as pdf:
            for page in {index.atoms[p[0]]['page'] for p in groups.values()}:
                if 0 <= page < len(pdf):live[page] = text_panels(pdf[page])
    for pid,positions in groups.items():
        first,last = min(positions),max(positions)
        panel = index.atoms[first]['source_panel']
        matches = [p for p in live.get(index.atoms[first]['page'],[])
                   if max(abs(a-b) for a,b in zip(p['bbox'],panel['bbox'])) < .0001]
        if len(matches) == 1:
            panel = {**panel,'frame_visible':matches[0]['frame_visible']}
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
