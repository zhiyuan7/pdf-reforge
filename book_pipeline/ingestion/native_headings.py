"""Split a bold run-in title from prose using unchanged native span ownership."""
from copy import deepcopy
import re


METHOD = 'native_bold_run_in_heading_boundary_v1'


def split_run_in_headings(rows, lookup):
    result = []
    for row in rows:
        parts = row['parts']
        ids = [sid for part in parts for sid in part['spans']]
        boundary = 0
        if row['type'] == 'paragraph_title':
            for sid in ids:
                span = lookup[sid]
                if span['text'].strip() and not re.search(r'Bold|CMBX|CMBXS', span['font'], re.I):
                    break
                boundary += 1
        title = ''.join(lookup[s]['text'] for s in ids[:boundary]).strip()
        tail = ''.join(lookup[s]['text'] for s in ids[boundary:]).strip()
        # A colon-terminated, observed bold prefix is a bounded run-in title.
        # Whole titles, numbered headings and ambiguous style changes stay put.
        if (not 0 < boundary < len(ids) or not title.endswith((':', '：'))
                or not 4 <= len(title) <= 120 or len(tail) < 12
                or re.match(r'^(?:\d|Chapter\b|Part\b)', title, re.I)):
            result.append(deepcopy(row))
            continue
        head_ids = set(ids[:boundary])
        # A source math part is atomic: never split its TeX at a font change.
        if any(p['kind'] != 'text' and bool(head_ids.intersection(p['spans']))
               for p in parts):
            result.append(deepcopy(row))
            continue
        for kind, owned_ids in [('paragraph_title', head_ids), ('text', set(ids[boundary:]))]:
            selected = []
            for part in parts:
                kept = [sid for sid in part['spans'] if sid in owned_ids]
                if kept or (not part['spans'] and kind == 'text'):
                    selected.append({**deepcopy(part), 'spans': kept})
            boxes = [lookup[s]['bbox'] for s in ids if s in owned_ids]
            result.append({**deepcopy(row), 'type': kind, 'parts': selected,
                '_run_in_body': kind == 'text',
                'bbox': [min(b[0] for b in boxes), min(b[1] for b in boxes),
                         max(b[2] for b in boxes), max(b[3] for b in boxes)]})
    merged = []
    for row in result:
        previous = merged[-1] if merged else None
        # A word split between the new body and the next model block must
        # reach the normal linguistic hyphen reviewer as one paragraph. We
        # only reunite span ownership here; no hyphen is removed locally.
        if previous and previous.get('_run_in_body') and row['type'] in {'text', 'paragraph'}:
            left_ids = [s for p in previous['parts'] for s in p['spans']]
            right_ids = [s for p in row['parts'] for s in p['spans']]
            if left_ids and right_ids:
                left, right = lookup[left_ids[-1]], lookup[right_ids[0]]
                a, b = left['bbox'], right['bbox']
                height = max(a[3]-a[1], b[3]-b[1])
                dy = (b[1]+b[3]-a[1]-a[3])/2
                title_box = merged[-2]['bbox'] if len(merged)>1 else previous['bbox']
                if (re.search(r'[A-Za-z][-\u00ad\u2010]\s*$',left['text'])
                        and re.match(r'\s*[a-z]',right['text'])
                        and left['line'] != right['line'] and .3*height < dy < 1.8*height
                        and abs(b[0]-title_box[0]) < .015 and b[0] < a[2]-.05
                        and abs(row['bbox'][2]-previous['bbox'][2]) < .04):
                    previous['parts'].extend(row['parts'])
                    x, y = previous['bbox'], row['bbox']
                    previous['bbox'] = [min(x[0],y[0]),min(x[1],y[1]),max(x[2],y[2]),max(x[3],y[3])]
                    continue
        merged.append(row)
    for row in merged:
        row.pop('_run_in_body',None)
    return merged
