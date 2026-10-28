"""Conservative, page-local pairing of numbered margin QR codes and citations."""
from pathlib import Path
import re


def margin_assets(index, registry, exclusions=(), changed=()):
    """Only relocate a complete aside label and a detected QR code.

    Geometry and an exact, unique superscript must agree. The QR payload is
    never decoded or followed. Ambiguous, edited or boxed content stays put.
    """
    from book_pipeline.structure.headings import valid_box
    blocked = [*exclusions, *(r for r in registry if r['kind'] not in {'figure', 'table', 'equation'})]

    def available(atom):
        n = index.ids[atom['id']]; lo, hi = index.prefix[n:n+2]
        return (atom['id'] not in changed and not atom.get('source_panel')
                and len(index.groups[atom['group']]) == 1
                and not any(r['lo'] < hi and lo < r['hi'] for r in blocked))

    labels = [a for a in index.atoms if a['kind'] == 'text' and a['role'] == 'aside_text'
              and re.fullmatch(r'\d{1,3}', a['content'].strip())
              and valid_box(a.get('bbox')) and available(a)]
    images = [a for a in index.atoms if a['kind'] == 'image' and valid_box(a.get('bbox'))
              and available(a) and a.get('asset') and Path(a['asset']).is_file()
              and not a.get('recovery_issue')]
    proposed = []
    for label in labels:
        lb = label['bbox']; marker = label['content'].strip()
        for image in images:
            if image['page'] != label['page']: continue
            box = image['bbox']; width, height = box[2]-box[0], box[3]-box[1]
            if (not .015 <= width <= .085 or not .01 <= height <= .085
                    or min(lb[3], box[3]) <= max(lb[1], box[1])
                    or not (0 <= box[0]-lb[2] <= .025 or 0 <= lb[0]-box[2] <= .025)):
                continue
            references = []
            for atom in index.atoms:
                if (atom['page'] != label['page'] or atom['kind'] != 'math'
                        or atom['id'] in changed or atom.get('source_panel')
                        or not re.fullmatch(r'\s*\^\s*\{\s*'+re.escape(marker)+r'\s*\}\s*', atom['content'])):
                    continue
                group = [index.atoms[n] for n in index.groups[atom['group']]]
                bounds = atom.get('block_bbox') or atom.get('bbox')
                if not valid_box(bounds): continue
                left = box[2] <= bounds[0] and bounds[0]-box[2] < .035
                right = bounds[2] <= box[0] and box[0]-bounds[2] < .035
                n = index.ids[atom['id']]; lo, hi = index.prefix[n:n+2]
                if (not (left or right) or not bounds[1] <= (box[1]+box[3])/2 <= bounds[3]+.015
                        or not any(a['kind']=='text' and len(a['content'])>30 for a in group)
                        or any(a['kind'] not in {'text','math'} or a.get('source_panel')
                               or a['id'] in changed for a in group)
                        or any(r['lo'] < index.prefix[index.groups[atom['group']][-1]+1]
                               and index.prefix[index.groups[atom['group']][0]] < r['hi']
                               for r in [*blocked, *registry])):
                    continue
                references.append(atom)
            if len(references) != 1: continue
            # OpenCV detection supplies independent evidence that this tiny
            # marginal image is a code, rather than a diagram or equation.
            import cv2
            pixels = cv2.imread(image['asset'])
            if pixels is None: continue
            pixels = cv2.copyMakeBorder(pixels, 16,16,16,16, cv2.BORDER_CONSTANT, value=[255]*3)
            if not cv2.QRCodeDetector().detect(pixels)[0]: continue
            targets = [r['anchor'] for r in registry if r['kind']=='figure'
                       and r['lo'] == index.prefix[index.ids[image['id']]]
                       and r['hi'] <= index.prefix[index.ids[image['id']]+1]]
            proposed.append(dict(method='numbered_source_margin_qr', page=label['page']+1,
                                 label=label['id'], image=image['id'], reference=references[0]['id'],
                                 targets=targets, bbox=box, marker=marker))
    # Repeated labels or competing assets cannot silently acquire ownership.
    return [p for p in proposed if all(sum(q[k]==p[k] for q in proposed)==1
                                     for k in ('label','image','reference'))]
