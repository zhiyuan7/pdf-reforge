"""Rejoin page-local prose split where an observed obstacle ends.

This is a rendering plan over immutable source intervals. A width change alone
does not establish a continuation: a separate figure or a recovered side note
must explain the narrow lines, and semantic objects remain hard boundaries.
"""
from collections import OrderedDict
import re

from book_pipeline.structure.headings import valid_box


def group_bounds(atoms):
    boxes = [a.get('block_bbox') or a.get('bbox') for a in atoms]
    if not all(valid_box(b) for b in boxes):
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def continuation(left, right, obstacle):
    """Narrow lines must resume at full width immediately below the obstacle."""
    a, b, c = left['bbox'], right['bbox'], obstacle
    if not a or not b or not valid_box(c): return False
    text, following = left['text'].rstrip(), right['text'].lstrip()
    if (not text or not following or re.search(r'[.!?。！？:：;；][\s\"\'”’）)]*$', text)
            or not re.match(r'[a-z\u3400-\u9fff]', following)):
        return False
    if (not 0 <= b[1]-a[3] <= .015 or not .005 <= b[1]-c[3] <= .035
            or b[2]-b[0] < a[2]-a[0]+.18
            or abs(b[0]-min(a[0],c[0])) > .065
            or b[0] > c[0]+.005 or b[2] < c[2]-.015):
        return False
    # The obstacle can flank either side. Its last row must overlap the
    # narrow paragraph, with a bounded gutter rather than a second column.
    left_side = .008 <= c[0]-a[2] <= .12
    right_side = .008 <= a[0]-c[2] <= .12
    return (left_side or right_side) and min(a[3],c[3]) > max(a[1],c[1])


def wrapped_prose(index, registry, exclusions=(), changed=(), side_notes=()):
    groups = OrderedDict()
    for n, atom in enumerate(index.atoms):
        groups.setdefault(atom['group'], []).append(n)
    rows = []
    for group, positions in groups.items():
        atoms = [index.atoms[n] for n in positions]
        rows.append(dict(group=group, atoms=atoms, lo=index.prefix[positions[0]],
                         hi=index.prefix[positions[-1]+1], page=atoms[0]['page'],
                         contiguous=positions == list(range(positions[0],positions[-1]+1)),
                         bbox=group_bounds(atoms), text=''.join(a['content'] for a in atoms)))

    def prose(row):
        return (row['contiguous'] and row['bbox'] and len(row['text'].strip()) >= 10
                and len({a['page'] for a in row['atoms']}) == 1
                and all(a['kind'] in {'text','math'} and a['id'] not in changed
                        and not a.get('source_panel') and not a.get('recovery_issue')
                        and a['role'] in {'text','paragraph','equation_inline'}
                        and a['block_type'] in {'text','paragraph'} for a in row['atoms']))

    def blocked(lo, hi, allowed=()):
        return any(r['lo'] < hi and lo < r['hi'] and r.get('anchor') not in allowed
                   for r in [*registry,*exclusions])

    joins, figures, occupied = [], [], []
    for n, left in enumerate(rows[:-1]):
        if not prose(left): continue
        right = rows[n+1]
        # Adjacent OCR groups can split a word as the note column ends.
        if (prose(right) and right['page'] == left['page'] and left['hi'] == right['lo']
                and right['hi']-left['lo'] <= 1800 and not blocked(left['lo'],right['hi'])):
            obstacles = [s for s in side_notes if s['page'] == left['page']+1
                         and continuation(left,right,s['note_bbox'])]
            if len(obstacles) == 1:
                joins.append(dict(method='source_obstacle_prose_continuation',
                    left=left['atoms'][-1]['id'], right=right['atoms'][0]['id'],
                    page=left['page']+1, obstacle_bbox=obstacles[0]['note_bbox'],
                    source_boxes=[left['bbox'],right['bbox']]))
        if n+2 >= len(rows): continue
        image, right = rows[n+1:n+3]
        images = [a for a in image['atoms'] if a['kind']=='image']
        if (len(images) != 1 or not prose(right) or not image['contiguous'] or not image['bbox']
                or left['hi'] != image['lo'] or image['hi'] != right['lo']
                or left['page'] != image['page'] or left['page'] != right['page']
                or any(a['page'] != left['page'] or a['id'] in changed
                       or a.get('source_panel') or a.get('recovery_issue')
                       or (a['kind'] != 'image' and a['role'] != 'image_caption') for a in image['atoms'])
                or not continuation(left,right,image['bbox'])):
            continue
        a,c = left['bbox'], image['bbox']
        extent = max(a[2],c[2])-min(a[0],c[0])
        width = (c[2]-c[0])/extent
        gutter = max(c[0]-a[2],a[0]-c[2])/extent
        overlap = min(a[3],c[3])-max(a[1],c[1])
        targets = [r['anchor'] for r in registry if r['kind']=='figure'
                   and r['lo']==image['lo'] and r['hi']<=image['hi']]
        if (not .2 <= width <= .5 or not .008 <= gutter <= .12
                or abs(a[1]-c[1]) > .015 or max(a[3],c[3])-min(a[1],c[1]) > .25
                or overlap < .65*min(a[3]-a[1],c[3]-c[1])
                or right['hi']-left['lo'] > 1800
                or blocked(left['lo'],right['hi'],targets)
                or any(lo < right['hi'] and left['lo'] < hi for lo,hi in occupied)):
            continue
        figures.append(dict(method='source_figure_prose_continuation', page=left['page']+1,
            lo=left['lo'],hi=right['hi'],main_ranges=[[left['lo'],left['hi']],[right['lo'],right['hi']]],
            image_range=[image['lo'],image['hi']],image=images[0]['id'],targets=targets,
            note_width=width,gutter=gutter,note_left=c[0]<a[0],
            source_boxes=[a,c,right['bbox']]))
        occupied.append((left['lo'],right['hi']))
    return joins, figures
