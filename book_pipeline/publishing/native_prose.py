"""Repair page-local native prose fragmentation without changing source atoms.

Splits on a live source baseline or at a filled source-column line can seed a
chain. Continuations must resume at its left edge with ordinary line leading.
This is a layout decision, never a repair of a fragment's translated meaning.
"""
from collections import Counter
from pathlib import Path
import re
from statistics import median

from book_pipeline.publishing.native_lateral import native_groups


def fragmented_native_prose(index, registry=(), exclusions=(), changed=(), *, margin_notes=None):
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return []
    groups = native_groups(index)
    if margin_notes is None:
        from book_pipeline.publishing.native_margin_notes import native_margin_evidence
        margin_notes = native_margin_evidence(index, allow_fragmented_owners=True)
    # Only live small-font margin glyphs may be bypassed in source order.
    # A note with several fragmented owners is not attached until those
    # owners have independently proved their paragraph continuity below.
    notes = [g for g in groups if g['group'] in margin_notes
             and not any(a['id'] in changed for a in g['atoms'])
             and not any(r['lo'] < g['hi'] and g['lo'] < r['hi']
                         for r in [*registry, *exclusions])]
    note_groups = {g['group'] for g in notes}
    def adjacent(left, right):
        cursor = left['hi']
        for note in notes:
            if note['lo'] == cursor and note['hi'] <= right['lo']:
                cursor = note['hi']
        return cursor == right['lo']
    eligible = []
    for group in groups:
        if (group['group'] in note_groups or any(a['kind'] not in {'text', 'math'} or a['role'] not in {'text', 'paragraph'}
                or a['block_type'] not in {'text', 'paragraph'} or a['id'] in changed
                or a.get('recovery_issue') for a in group['atoms'])
                or any(r['lo'] < group['hi'] and group['lo'] < r['hi']
                       for r in [*registry, *exclusions])):
            continue
        eligible.append(group)
    if not eligible:
        return []
    import pymupdf
    rows = []
    with pymupdf.open(source) as pdf:
        for number in sorted({g['page'] for g in eligible}):
            if not 0 <= number < len(pdf):
                continue
            page = pdf[number]
            glyphs = [dict(c, size=s['size'], bold=bool(s['flags'] & 16))
                      for b in page.get_text('rawdict')['blocks'] for line in b.get('lines', [])
                      if line.get('dir') == (1., 0.)
                      for s in line['spans'] for c in s['chars'] if c['c'].strip()]
            units = []
            page_groups = [g for g in eligible if g['page'] == number]
            for group in page_groups:
                b = group['bbox']
                box = pymupdf.Rect(b[0]*page.rect.width, b[1]*page.rect.height,
                                   b[2]*page.rect.width, b[3]*page.rect.height)
                expanded = box + (-.6, -.6, .6, .6)
                # A multiline union may enclose the separately serialized tail
                # of its final line. The smaller adjacent box owns that tail.
                tails = []
                for other in page_groups:
                    if not adjacent(group, other):
                        continue
                    c = other['bbox']
                    tail = pymupdf.Rect(c[0]*page.rect.width, c[1]*page.rect.height,
                                        c[2]*page.rect.width, c[3]*page.rect.height)
                    if (expanded.contains(tail) and tail.height < .75*box.height
                            and tail.width < .8*box.width):
                        tails.append(tail + (-.1, -.1, .1, .1))
                owned = [c for c in glyphs if expanded.contains(pymupdf.Point(
                    (c['bbox'][0]+c['bbox'][2])/2, (c['bbox'][1]+c['bbox'][3])/2))
                    and not any(t.contains(pymupdf.Point(
                        (c['bbox'][0]+c['bbox'][2])/2, (c['bbox'][1]+c['bbox'][3])/2)) for t in tails)]
                if not owned:
                    continue
                # The dominant body size excludes sub/superscript baselines.
                size = Counter(round(c['size'], 1) for c in owned).most_common(1)[0][0]
                lines = []
                for c in sorted((c for c in owned if c['size'] >= .85*size),
                                key=lambda c: (c['origin'][1], c['bbox'][0])):
                    if not lines or abs(median(x['origin'][1] for x in lines[-1])-c['origin'][1]) > .3*size:
                        lines.append([])
                    lines[-1].append(c)
                if not lines:
                    continue
                def line_info(chars):
                    chars = sorted(chars, key=lambda c: c['bbox'][0])
                    return dict(x0=min(c['bbox'][0] for c in chars),
                                x1=max(c['bbox'][2] for c in chars),
                                y=median(c['origin'][1] for c in chars),
                                text=''.join(c['c'] for c in chars))
                units.append(dict(group, size=size, first=line_info(lines[0]), last=line_info(lines[-1]),
                                  bold_label=len(lines)==1 and all(c['bold'] for c in owned)))
            chain = None
            for left, right in zip(units, units[1:]):
                a, b = left['last'], right['first']
                size = left['size']
                is_adjacent = (adjacent(left, right) and .9 <= right['size']/size <= 1.1
                            and not left['bold_label'])
                same_line = (is_adjacent and abs(a['y']-b['y']) <= .3*size
                             and -.3 <= b['x0']-a['x1'] <= 1.8*size)
                if not chain or chain['end'] != left['lo']:
                    chain = dict(x0=left['bbox'][0]*page.rect.width,
                                 x1=left['bbox'][2]*page.rect.width, seeded=False, end=left['lo'])
                list_marker = re.match(r'^\s*(?:[\(（]?(?:[A-Za-z]|\d{1,3}|[一二三四五六七八九十]{1,3})[.)）、]|[•·])',b['text'])
                continuation = (is_adjacent and chain['seeded']
                    and .95*size <= b['y']-a['y'] <= 1.5*size
                    and abs(b['x0']-chain['x0']) <= .25*size
                    and abs(a['x1']-chain['x1']) <= .6*size
                    and not re.search(r'[.!?。！？:：;；][\"\'”’）)]*$', a['text'])
                    and bool(re.match(r'[\(（\"\'“‘]*[a-z\u3400-\u9fff]', b['text']))
                    and not list_marker)
                # Some extractors serialize every physical line as a block.
                # A prior same-baseline split then never exists. Require live
                # right-edge fill, the same column and ordinary leading, and
                # a lower-case/parenthetical continuation before joining it.
                full_column_wrap = (is_adjacent and not chain['seeded']
                    and .3*page.rect.width <= (chain['x1']-chain['x0'])
                    and .95*size <= b['y']-a['y'] <= 1.5*size
                    and abs(b['x0']-chain['x0']) <= .25*size
                    and abs(a['x1']-chain['x1']) <= .6*size
                    and abs(right['bbox'][0]*page.rect.width-chain['x0']) <= .5*size
                    # A paragraph's last line can stop before the column edge.
                    # The preceding filled line proves the wrap; the new line
                    # must stay within that column, not fill it again.
                    and right['bbox'][2]*page.rect.width <= chain['x1']+size
                    and not re.search(r'[.!?。！？:：;；][\"\'”’）)]*$',a['text'])
                    and bool(re.match(r'[\(（\"\'“‘]*[a-z\u3400-\u9fff]',b['text']))
                    and not list_marker)
                if not (same_line or continuation or full_column_wrap):
                    chain = None
                    continue
                rows.append(dict(method='native_fragmented_prose_continuation',
                    left=left['atoms'][-1]['id'], right=right['atoms'][0]['id'],
                    page=number+1, boundary=('same_source_baseline' if same_line else
                        'native_full_column_wrap' if full_column_wrap else 'source_column_wrap'),
                    source_boxes=[left['bbox'], right['bbox']]))
                chain.update(seeded=True, end=right['lo'],
                             x0=min(chain['x0'], right['bbox'][0]*page.rect.width),
                             x1=max(chain['x1'], right['bbox'][2]*page.rect.width))
    return rows
