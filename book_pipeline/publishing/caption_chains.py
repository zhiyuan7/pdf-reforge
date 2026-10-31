"""Keep geometrically continuous caption groups in their figure's float."""
import re

from book_pipeline.structure.headings import valid_box


def continued_figure_captions(index, floats, barriers=(), skipped=(), changed=()):
    result = dict(floats)
    rows = []
    for group, (lo, hi, kind) in floats.items():
        if kind != 'figure':
            continue
        owned = [a for a, _, _ in index.fragments(lo, hi)]
        if not owned or owned[-1]['role'] not in {'image_caption', 'chart_caption'}:
            continue
        role = owned[-1]['role']
        tail = []
        for atom in reversed(owned):
            if atom['role'] != role:
                break
            tail.append(atom)
        groups = list(dict.fromkeys(a['group'] for a in reversed(tail)))
        joined = []
        while hi < index.prefix[-1]:
            first = index.ids[index.address(hi)['atom']]
            following = index.atoms[first]
            positions = index.groups[following['group']]
            if positions != list(range(first, max(positions)+1)):
                break
            atoms = [index.atoms[n] for n in positions]
            old = [a for a in tail if a['group'] == tail[0]['group']]
            def box(items):
                boxes = [a.get('bbox') for a in items]
                if not all(valid_box(b) for b in boxes):
                    return None
                return [min(b[0] for b in boxes), min(b[1] for b in boxes),
                        max(b[2] for b in boxes), max(b[3] for b in boxes)]
            b, c = box(old), box(atoms)
            end = index.prefix[max(positions)+1]
            text = ''.join(a['content'] for a in atoms).lstrip()
            if (any(a['kind'] not in {'text','math'} or a['role'] != role
                    or a['page'] != owned[-1]['page'] or a.get('recovery_issue')
                    or a.get('source_panel') or a['id'] in changed for a in [*tail, *atoms])
                    or not b or not c or not -.002 <= c[1]-b[3] <= .012
                    or abs(c[0]-b[0]) > .012 or min(b[2],c[2])-max(b[0],c[0]) <= .03
                    or re.match(r'(?:Fig(?:ure)?\.?|图|Table|表)\s*\d', text, re.I)
                    or any(lo < p < end for p in barriers)
                    or any(x < end and lo < z for x,z,*_ in skipped)
                    or any(g != group and x < end and hi < z for g,(x,z,_) in floats.items())):
                break
            joined.append((tail[0]['id'], atoms[0]['id']))
            groups.append(following['group'])
            hi = end
            tail = list(reversed(atoms))
        if joined:
            result[group] = (lo, hi, kind)
            rows.append(dict(method='continuous_source_caption_groups', group=group,
                caption_groups=groups, joins=joined, lo=lo, hi=hi,
                page=owned[-1]['page']+1))
    return result, rows
