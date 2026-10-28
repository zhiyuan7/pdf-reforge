"""Keep overlapping source-image rows together at their original proportions."""
from collections import defaultdict
from pathlib import Path

from book_pipeline.structure.headings import valid_box
from book_pipeline.structure.numbered import CAPTION_ROLES


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
    return rows


def recover_horizontal_images(index, registry, directory, edited_atoms=()):
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
        groups = {atom['group'] for atom in images}
        members = [atom for atom in index.atoms if atom['page'] == row['page']
                   and (atom['id'] in {a['id'] for a in images}
                        or atom['group'] in groups and atom['role'] in CAPTION_ROLES)]
        boxes = [atom['bbox'] for atom in members if valid_box(atom.get('bbox'))]
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
        if any(a['role'] in CAPTION_ROLES for a in members):
            lower = max(lower, region[3] + .024)
        next_prose = min((a['bbox'][1] for a in index.atoms if a['page'] == row['page']
                         and a['kind'] == 'text' and len(a['content']) > 16
                         and valid_box(a.get('bbox')) and a['bbox'][1] >= region[3]), default=1.)
        if any(a['role'] in CAPTION_ROLES for a in members):
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
        # Separate short labels (e.g. the digits below the counting rods) can
        # be ordinary text atoms. Include complete contained source groups.
        for atom in index.atoms:
            if atom['page'] != row['page'] or atom in members or not valid_box(atom.get('bbox')):
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
        path = Path(directory) / 'images' / f'source-image-row-p{row["page"]+1:06d}-{n:03d}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(index.source['source_pdf']) as document:
            page = document[row['page']]
            page.get_pixmap(dpi=240, clip=pymupdf.Rect(clip_box[0]*page.rect.width, clip_box[1]*page.rect.height,
                clip_box[2]*page.rect.width, clip_box[3]*page.rect.height), alpha=False).save(path)
        # Source anchors and object destinations are emitted inside the shared
        # row instead of left behind at the old stacked positions.
        member_ids = {atom['id'] for atom in members}
        targets = [target['anchor'] for target in registry if target['kind'] == 'figure'
                   and (target.get('source_atom') in member_ids or target.get('start', {}).get('atom') in member_ids)]
        result[carrier['id']] = dict(asset=str(path.resolve()), bbox=clip_box, page=row['page']+1,
            members=[atom['id'] for atom in members], images=[atom['id'] for atom in images],
            groups=sorted(groups), targets=targets)
    return result
