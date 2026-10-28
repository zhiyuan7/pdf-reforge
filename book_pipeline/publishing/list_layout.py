"""Conservative paragraph styles for explicit source list labels.

Labels stay in the source stream, including their links and coverage. Geometry
only corroborates nesting and indented prose following an item's display math.
"""
from collections import defaultdict
import re


LABEL = re.compile(r'^\s*(?P<label>[•◦▪‣]|(?P<number>\d+)[.)]|\([a-zA-Z0-9]+\)|[a-zA-Z]\))(?=\s)')


def multiline_list_rows(index, barriers=(), skipped=(), changed_atoms=(), corrections=()):
    """Use literal source newlines and consecutive labels inside one OCR block.

    Math is opaque during detection, so array rows and equation numbers cannot
    invent list entries. The renderer keeps every source byte and interval.
    """
    rows = []
    changed_atoms = set(changed_atoms)
    group_names = list(index.groups)
    for group, positions in index.groups.items():
        atoms = [index.atoms[n] for n in positions]
        if (atoms[0]['block_type'] != 'text' or atoms[0]['role'] != 'text'
                or any(a['kind'] not in {'text', 'math'} or a['id'] in changed_atoms for a in atoms)):
            continue
        proxy = ''.join(a['content'] if a['kind']=='text' else 'X'*a['length'] for a in atoms)
        matches = list(re.finditer(r'(?m)^[ \t]*(\d+)[.)](?=[ \t]+\S)', proxy))
        if (len(matches) < 2 or proxy[:matches[0].start()].strip()
                or any(int(b[1]) != int(a[1])+1 for a,b in zip(matches,matches[1:]))):
            continue
        lo, hi = index.prefix[positions[0]], index.prefix[positions[-1]+1]
        # Body-only reviewed edits retain source offsets and row boundaries.
        # A changed label, newline or inter-row gap invalidates the layout.
        body_ranges = [(lo+m.end()+len(re.match(r'[ \t]*', proxy[m.end():])[0]),
                        lo+matches[n+1].start() if n+1<len(matches) else hi)
                       for n,m in enumerate(matches)]
        edits = [c for c in corrections if c['atom'] in {a['id'] for a in atoms}]
        if any('\n' in c['expected'] or '\r' in c['expected']
               or '\n' in c['replacement'] or '\r' in c['replacement']
               or not any(a <= index.prefix[index.ids[c['atom']]]+c['start_offset']
                          < index.prefix[index.ids[c['atom']]]+c['end_offset'] <= z
                          for a,z in body_ranges) for c in edits):
            continue
        if (any(lo < p < hi for p in barriers)
                or any(a < hi and b > lo for a,b,*_ in skipped)):
            continue
        keep_lo = None
        box = atoms[0].get('block_bbox') or atoms[0].get('bbox')
        if len(matches) <= 6 and len(proxy) <= 1200 and box and box[3]-box[1] <= .18:
            keep_lo = lo
            number = group_names.index(group)
            if number:
                previous = index.groups[group_names[number-1]]
                intro = [index.atoms[n] for n in previous]
                text = ''.join(a['content'] for a in intro).strip()
                candidate = index.prefix[previous[0]]
                if (intro[0]['page'] == atoms[0]['page'] and len(text) <= 100
                        and text.endswith((':', '：'))
                        and all(a['kind'] in {'text','math'} and a['role']=='text' for a in intro)
                        and not any(candidate <= p < lo for p in barriers)
                        and not any(a < lo and b > candidate for a,b,*_ in skipped)):
                    keep_lo = candidate
        for n, match in enumerate(matches):
            rows.append(dict(method='explicit_multiline_source_list', group=group,
                             lo=lo+match.start(), hi=lo+matches[n+1].start() if n+1<len(matches) else hi,
                             label=match[0].strip(), page=atoms[0]['page'],
                             keep_lo=keep_lo, list_hi=hi, keep_baselines=2*len(matches)+2))
    return rows


def list_paragraphs(index):
    groups = []
    for group, positions in index.groups.items():
        atoms = [index.atoms[p] for p in positions]
        first = atoms[0]
        box = first.get('block_bbox') or first.get('bbox')
        if not box or len(box) != 4:
            groups.append(dict(group=group, atoms=atoms))
            continue
        text = ''.join(a['content'] for a in atoms)
        match = LABEL.match(text) if first['block_type'] == 'text' and first['role'] == 'text' else None
        groups.append(dict(group=group, atoms=atoms, box=box, page=first['page'],
                           panel=first.get('source_panel'), match=match,
                           lo=index.prefix[positions[0]], hi=index.prefix[positions[-1]+1]))
    # A bare "1." may be prose. Require another aligned ordered item; explicit
    # bullet glyphs are unambiguous and do not need a neighbouring item.
    labels = []
    for n, row in enumerate(groups):
        match = row.get('match')
        if not match or any(a['kind'] not in {'text', 'math'} for a in row['atoms']):
            continue
        label = match['label']
        if label[0].isdigit():
            if not any(other.get('match') and other['match']['number']
                       and other['page'] == row['page'] and other['panel'] == row['panel']
                       and abs(other['box'][0]-row['box'][0]) < .012
                       and abs(int(other['match']['number'])-int(match['number'])) == 1
                       for other in groups[max(0,n-6):n]+groups[n+1:n+7]):
                continue
        labels.append(row)
    by_page = defaultdict(list)
    for row in labels:
        by_page[(row['page'], str(row['panel']))].append(row['box'][0])
    styles, pending = {}, None
    label_groups = {row['group'] for row in labels}
    for row in groups:
        if row['group'] in label_groups:
            # Map distinct source left edges to modest, consistent nesting.
            columns = [[]]
            for x in sorted(by_page[(row['page'], str(row['panel']))]):
                if columns[-1] and x-columns[-1][-1] > .1:
                    columns.append([])
                if not columns[-1] or x-columns[-1][-1] > .012:
                    columns[-1].append(x)
            edges = min(columns, key=lambda column: min(abs(x-row['box'][0]) for x in column))
            level = min(range(len(edges)), key=lambda n: abs(edges[n]-row['box'][0]))
            style = dict(group=row['group'], lo=row['lo'], hi=row['hi'],
                         item_group=row['group'],
                         label=row['match']['label'], indent_em=1+1.2*level,
                         continuation=False, source_bbox=row['box'], page=row['page'])
            styles[row['group']] = style
            pending = row, style
        elif pending:
            item, style = pending
            first = row['atoms'][0]
            box = row.get('box')
            same_flow = (box and row['page'] == item['page'] and row['panel'] == item['panel']
                         and box[1] >= item['box'][1])
            if same_flow and all(a['kind'] == 'display_math' for a in row['atoms']):
                continue
            if (same_flow and first['block_type'] == first['role'] == 'text'
                    and all(a['kind'] in {'text','math'} for a in row['atoms'])
                    and .008 <= box[0]-item['box'][0] <= .065
                    and abs(box[2]-item['box'][2]) < .03):
                styles[row['group']] = dict(style, group=row['group'], lo=row['lo'], hi=row['hi'],
                                             continuation=True, source_bbox=box)
            else:
                pending = None
    return styles
