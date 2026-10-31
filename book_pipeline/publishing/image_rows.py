"""Keep overlapping source-image rows together at their original proportions."""
from collections import defaultdict
from pathlib import Path
from unicodedata import east_asian_width

from book_pipeline.structure.headings import valid_box
from book_pipeline.structure.numbered import CAPTION_ROLES


def embedded_panel_label_rows(index, occupied=()):
    """Keep short OCR panel labels at their source positions above one image.

    Only complete, geometrically isolated image groups qualify. The long
    caption remains searchable prose; the panel labels belong to the pixels.
    """
    groups = defaultdict(list)
    for atom in index.atoms:
        groups[atom['group']].append(atom)
    rows = []
    for atoms in groups.values():
        images = [a for a in atoms if a['kind'] == 'image']
        if (len(images) != 1 or images[0]['id'] in occupied
                or any(a.get('recovery_issue') or not valid_box(a.get('bbox')) for a in atoms)
                or len({a['page'] for a in atoms}) != 1):
            continue
        image = images[0]; b = image['bbox']
        labels = [a for a in atoms if index.ids[a['id']] < index.ids[image['id']]]
        tail = [a for a in atoms if index.ids[a['id']] > index.ids[image['id']]]
        if (len(labels) < 2 or b[2]-b[0] < .2
                or any(a['kind'] != 'text' or a['role'] not in CAPTION_ROLES
                       or not b[0]-.004 <= a['bbox'][0] < a['bbox'][2] <= b[2]+.004
                       or not b[1]-.045 <= a['bbox'][1] < b[1]
                       or a['bbox'][3] > b[1]+.004
                       or a['bbox'][3]-a['bbox'][1] > .035
                       or a['bbox'][2]-a['bbox'][0] > .30
                       or len(a['content'].strip()) > 48 for a in labels)
                or not tail or any(a['role'] not in CAPTION_ROLES
                       or a['bbox'][1] < b[3] or a['bbox'][1] > b[3]+.08 for a in tail)):
            continue
        region = [min(a['bbox'][0] for a in [image, *labels]),
                  min(a['bbox'][1] for a in labels),
                  max(a['bbox'][2] for a in [image, *labels]), b[3]]
        # An independently owned object intersecting the crop makes the
        # attribution ambiguous, even when its OCR reading order is elsewhere.
        if any(a['page'] == image['page'] and a['group'] != image['group']
               and valid_box(a.get('bbox'))
               and min(region[2], a['bbox'][2]) > max(region[0], a['bbox'][0])
               and min(region[3], a['bbox'][3]) > max(region[1], a['bbox'][1])
               for a in index.atoms):
            continue
        rows.append(dict(page=image['page'], images=images, embedded_panel_labels=True))
    return rows


def detect_image_rows(index):
    pages = defaultdict(list)
    for atom in index.atoms:
        if (atom['kind'] == 'image' and valid_box(atom.get('bbox'))
                and atom.get('recovery_issue') != 'formula_source_crop_requires_transcription'):
            pages[atom['page']].append(atom)
    rows = []
    for page, images in pages.items():
        remaining = {atom['id']: atom for atom in images}
        while remaining:
            _, first = remaining.popitem()
            members = [first]
            changed = True
            while changed:
                changed = False
                for identifier, atom in list(remaining.items()):
                    if any(min(atom['bbox'][3], other['bbox'][3]) > max(atom['bbox'][1], other['bbox'][1])
                           for other in members):
                        members.append(atom)
                        del remaining[identifier]
                        changed = True
            if len(members) > 1:
                rows.append(dict(page=page, images=sorted(members, key=lambda a: a['bbox'][0])))
    # A shared-caption multirow figure is one source image region. Individual
    # row crops can include the neighboring row's labels, even when the image
    # rectangles themselves are disjoint. Keep these original positions once.
    stacks, pending = [], []
    for atom in index.atoms:
        if (atom['kind']=='image' and atom.get('extraction_method')=='native'
                and not atom.get('recovery_issue') and valid_box(atom.get('bbox'))):
            if pending and atom['page']!=pending[-1]['page']:
                pending=[]
            pending.append(atom)
            continue
        if (len(pending)>=2 and atom['role'] in CAPTION_ROLES
                and atom.get('extraction_method')=='native' and not atom.get('recovery_issue')
                and atom['page']==pending[0]['page'] and valid_box(atom.get('bbox'))
                and atom['group'] not in {a['group'] for a in pending}):
            bands=[]
            for image in sorted(pending,key=lambda a:a['bbox'][1]):
                b=image['bbox']
                if bands and b[1]<bands[-1][3]:
                    a=bands[-1];bands[-1]=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
                else:
                    bands.append(list(b))
            caption=atom['bbox'];bottom=max(a['bbox'][3] for a in pending)
            independent_caption=False
            for other in index.atoms:
                if (other['page']!=atom['page'] or other['role'] not in CAPTION_ROLES
                        or other['group']==atom['group'] or not valid_box(other.get('bbox'))):
                    continue
                c=other['bbox']
                for image in pending:
                    b=image['bbox'];vertical=min(b[3],c[3])-max(b[1],c[1])
                    horizontal=min(b[2],c[2])-max(b[0],c[0])
                    side=(vertical>=.5*min(b[3]-b[1],c[3]-c[1]) and
                          (.008<=b[0]-c[2]<=.12 or .008<=c[0]-b[2]<=.12))
                    above=(0<=b[1]-c[3]<=.06 and horizontal>0)
                    if side or above:
                        independent_caption=True
            if (len(bands)>=2 and bottom-bands[0][1]<=.74
                    and not independent_caption
                    and -.01<=caption[1]-bottom<=.08
                    and min(caption[2],bands[-1][2])>max(caption[0],bands[-1][0])
                    and all(0<=b[1]-a[3]<=.06 and min(a[2],b[2])-max(a[0],b[0])
                            >=.7*min(a[2]-a[0],b[2]-b[0]) for a,b in zip(bands,bands[1:]))):
                stacks.append(dict(page=pending[0]['page'],images=list(pending),shared_caption_stack=True))
        pending=[]
    stacked={a['id'] for row in stacks for a in row['images']}
    rows = [r for r in rows if not any(a['id'] in stacked for a in r['images'])]+stacks
    occupied = {a['id'] for row in rows for a in row['images']}
    return rows + embedded_panel_label_rows(index, occupied)


def recover_horizontal_images(index, registry, directory, edited_atoms=(), image_boxes=None):
    """Use a source crop to preserve spacing, tiny labels and image scale.

    Each original atom/target retains an anchor; only its printed representation
    moves into the shared row. Unrelated content is never silently hidden.
    """
    if not index.source.get('source_pdf'):
        return {}
    import pymupdf
    result = {}
    for n, row in enumerate(detect_image_rows(index), 1):
        images = row['images']
        if row.get('shared_caption_stack') or row.get('embedded_panel_labels'):
            owned_atoms = ([a for a in index.atoms if a['group'] == images[0]['group']]
                           if row.get('embedded_panel_labels') else images)
            lo=min(index.prefix[index.ids[a['id']]] for a in owned_atoms)
            hi=max(index.prefix[index.ids[a['id']]+1] for a in owned_atoms)
            owned=[r for r in registry if r.get('lo',hi)<hi and lo<r.get('hi',lo)]
            if any(r['kind']!='figure' for r in owned) or len(owned)>1:
                continue
        groups = {atom['group'] for atom in images}
        members = [atom for atom in index.atoms if atom['page'] == row['page']
                   and (atom['id'] in {a['id'] for a in images}
                        or atom['group'] in groups and atom['role'] in CAPTION_ROLES)]
        # Translation changes the text, not the source PDF pixels. A trailing
        # prose caption must remain typeset instead of being replaced with its
        # original-language raster. Short in-panel labels stay with the images.
        retained = []
        if index.source.get('translation') or row.get('embedded_panel_labels'):
            floor = max(a['bbox'][3] for a in images)
            below = [a for a in members if a['role'] in CAPTION_ROLES and valid_box(a.get('bbox'))
                     and floor-.005 <= a['bbox'][1] <= floor+.08]
            # A rich caption can start with a short figure number, then
            # alternate tiny text fragments and math. Judge the whole tail
            # and retain its prefix too, not only its longest text fragment.
            prose_width = sum(2 if east_asian_width(c) in {'W','F'} else 1
                              for a in below for c in a['content'].strip())
            if prose_width > 16 or row.get('embedded_panel_labels'):
                start = min(index.ids[a['id']] for a in below)
                retained = [a for a in members if index.ids[a['id']] >= start]
                if any(a['role'] not in CAPTION_ROLES or not valid_box(a.get('bbox'))
                       or a['bbox'][1] < floor-.005 for a in retained):
                    continue
                members = [a for a in members if a not in retained]
        # Native vector-label recovery can extend an image past its extraction
        # box (panel letters and axis labels). A shared row must preserve those
        # verified bounds too, instead of cropping them a second time.
        boxes = [(image_boxes or {}).get(atom['id'], atom.get('bbox')) for atom in members]
        boxes = [b for b in boxes if valid_box(b)]
        region = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                  max(b[2] for b in boxes), max(b[3] for b in boxes)]
        # Some OCR caption leaves inherit the image-body box, which excludes
        # the actual digits underneath. Nearby short labels supply a lower
        # boundary; leave a little room for inherited caption boxes as well.
        nearby_labels = [a for a in index.atoms if a['page'] == row['page']
                         and a['kind'] in {'text', 'math'} and valid_box(a.get('bbox'))
                         and len(a['content'].strip()) <= 16
                         and region[0] <= a['bbox'][0] <= a['bbox'][2] <= region[2]
                         and region[3] <= a['bbox'][1] <= region[3] + .03]
        lower = max([region[3], *(a['bbox'][3] for a in nearby_labels)])
        if not row.get('embedded_panel_labels') and any(a['role'] in CAPTION_ROLES for a in members):
            lower = max(lower, region[3] + .024)
        next_prose = min((a['bbox'][1] for a in index.atoms if a['page'] == row['page']
                         and a['kind'] == 'text' and len(a['content']) > 16
                         and valid_box(a.get('bbox')) and a['bbox'][1] >= region[3]), default=1.)
        if not row.get('embedded_panel_labels') and any(a['role'] in CAPTION_ROLES for a in members):
            # A shared figure number can sit below both subfigure labels, even
            # when all three caption leaves inherit the image-body rectangle.
            # Use the free gap before the next source object, bounded so that
            # another image row or a short prose block is never included.
            known = {a['id'] for a in members + nearby_labels}
            boundary = min((a['bbox'][1] for a in index.atoms if a['page'] == row['page']
                            and a['id'] not in known and a['block_type'] not in {'header', 'footer', 'page_number'}
                            and valid_box(a.get('bbox')) and a['bbox'][1] >= region[3]), default=1.)
            lower = max(lower, min(region[3] + .08, boundary - .003))
        region[3] = min(lower, next_prose - .003) if next_prose > region[3] else region[3]
        if retained:
            boundary = min(a['bbox'][1] for a in retained)
            if boundary < max(a['bbox'][3] for a in images):
                continue
            region[3] = min(region[3], boundary)
        # Separate short labels (e.g. the digits below the counting rods) can
        # be ordinary text atoms. Include complete contained source groups.
        for atom in index.atoms:
            if atom['page'] != row['page'] or atom in members or atom in retained or not valid_box(atom.get('bbox')):
                continue
            if row.get('embedded_panel_labels'):
                continue
            b = atom['bbox']
            contained = b[0] >= region[0] - .004 and b[2] <= region[2] + .004 and b[1] >= region[1] - .004 and b[3] <= region[3] + .004
            if contained and atom['kind'] in {'text', 'math', 'display_math'}:
                members.append(atom)
        if any(atom['id'] in edited_atoms for atom in members):
            # A clickable or edited caption must remain native. Native row
            # support can be used instead of replacing those strings by pixels.
            continue
        members.sort(key=lambda atom: index.ids[atom['id']])
        if not members:
            continue
        # Place the row after source prose physically above it, even if OCR's
        # sequence inserted that prose between the first two image objects.
        start, end = index.ids[members[0]['id']], index.ids[members[-1]['id']]
        intervening = [atom for atom in index.atoms[start:end+1] if atom not in members]
        above = [atom for atom in intervening if valid_box(atom.get('bbox')) and atom['bbox'][3] <= region[1] + .001]
        carrier = next((atom for atom in members if index.ids[atom['id']] > max(
            (index.ids[a['id']] for a in above), default=start-1)), members[0])
        clip_box = [max(0, region[0]-.002), max(0, region[1]-.002),
                    min(1, region[2]+.002), min(1, region[3]+.002)]
        if retained:
            clip_box[3] = min(clip_box[3], min(a['bbox'][1] for a in retained))
        path = Path(directory) / 'images' / f'source-image-row-p{row["page"]+1:06d}-{n:03d}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(index.source['source_pdf']) as document:
            page = document[row['page']]
            clip = pymupdf.Rect(clip_box[0]*page.rect.width, clip_box[1]*page.rect.height,
                clip_box[2]*page.rect.width, clip_box[3]*page.rect.height)
            # A complete shared-caption stack also owns the ink between rows.
            # Its guarded source rectangle preserves that ink at one position.
            composite = (any(a['id'] in (image_boxes or {}) for a in images)
                         and all(a['kind']=='image' for a in members)
                         and not row.get('shared_caption_stack'))
            if composite:
                # Different panels' labels can overlap in vertical extent.
                # A full-width rectangular crop would duplicate the next row's
                # panel letters. Compose only each image's verified rectangle,
                # leaving their original relative positions and scale intact.
                matrix = pymupdf.Matrix(240/72, 240/72)
                pix = pymupdf.Pixmap(pymupdf.csRGB, (clip*matrix).irect, False)
                pix.clear_with(255)
                for box in boxes:
                    rect = pymupdf.Rect([v*d for v,d in zip(box,[page.rect.width,page.rect.height]*2)])
                    patch = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
                    pix.copy(patch, patch.irect)
                pix.save(path)
            else:
                page.get_pixmap(dpi=240, clip=clip, alpha=False).save(path)
        # Source anchors and object destinations are emitted inside the shared
        # row instead of left behind at the old stacked positions.
        member_ids = {atom['id'] for atom in members}
        targets = [target['anchor'] for target in registry if target['kind'] == 'figure'
                   and (target.get('source_atom') in member_ids or target.get('start', {}).get('atom') in member_ids)]
        result[carrier['id']] = dict(asset=str(path.resolve()), bbox=clip_box, page=row['page']+1,
            members=[atom['id'] for atom in members], images=[atom['id'] for atom in images],
            groups=sorted(groups), targets=targets,
            native_caption_members=[a['id'] for a in retained],
            crop_method='verified_image_bounds_composite' if composite else 'source_rectangle',
            embedded_panel_labels=row.get('embedded_panel_labels',False),
            shared_caption_stack=row.get('shared_caption_stack',False))
    return result
