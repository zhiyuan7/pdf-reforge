"""Keep a tightly stacked source figure and its shared caption in one float."""
from book_pipeline.structure.headings import valid_box
from book_pipeline.structure.numbered import CAPTION_ROLES


def stacked_figure_floats(index, floats, barriers=(), skipped=()):
    candidates = []
    for group, (lo, hi, kind) in sorted(floats.items(), key=lambda r: r[1][0]):
        atoms = [a for a, _, _ in index.fragments(lo, hi)]
        images = [a for a in atoms if a['kind'] == 'image']
        captions = [a for a in atoms if a['role'] in CAPTION_ROLES]
        if (kind != 'figure' or len(images) != 1
                or any(a['kind'] != 'image' and a['role'] not in CAPTION_ROLES for a in atoms)
                or any(a.get('extraction_method') != 'native' or a.get('recovery_issue') for a in atoms)
                or len({a['page'] for a in atoms}) != 1
                or not all(valid_box(a.get('bbox')) for a in atoms)):
            continue
        box = images[0]['bbox']
        if captions and any(not -.01 <= a['bbox'][1]-box[3] <= .08
                            or min(a['bbox'][2], box[2]) <= max(a['bbox'][0], box[0])
                            for a in captions):
            continue
        candidates.append(dict(group=group, lo=lo, hi=hi, image=images[0], captions=captions))
    rows, pending = [], []

    def flush():
        if len(pending) >= 2 and pending[-1]['captions']:
            first, last = pending[0], pending[-1]
            bottom = max(a['bbox'][3] for a in last['captions'])
            lo, hi = first['lo'], last['hi']
            if (bottom-first['image']['bbox'][1] <= .74
                    and not any(lo < p < hi for p in barriers)
                    and not any(a < hi and lo < b for a, b, *_ in skipped)):
                rows.append(dict(method='stacked_source_figure_float', lo=lo, hi=hi,
                    page=first['image']['page']+1, groups=[r['group'] for r in pending],
                    images=[r['image']['id'] for r in pending]))
        pending.clear()

    for row in candidates:
        if pending:
            previous = pending[-1]
            a, b = previous['image']['bbox'], row['image']['bbox']
            overlap = min(a[2], b[2])-max(a[0], b[0])
            if (previous['captions'] or previous['hi'] != row['lo']
                    or previous['image']['page'] != row['image']['page']
                    or not 0 <= b[1]-a[3] <= .018
                    or overlap < .9*min(a[2]-a[0], b[2]-b[0])):
                flush()
        pending.append(row)
        if row['captions']:
            flush()
    flush()
    return rows
