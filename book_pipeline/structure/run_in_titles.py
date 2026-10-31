"""Identify split run-in labels using live PDF glyphs, independent of translation."""
from collections import defaultdict
from pathlib import Path
from statistics import median
import re

from .headings import group_blocks, parse_number, span_for, valid_box


def run_in_title_rows(index, barriers=(), changed=()):
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return []
    candidates = defaultdict(list)
    blocks = group_blocks(index)
    for title, body in zip(blocks, blocks[1:]):
        b, c = title['bbox'], body['bbox']
        if (title['type'] != 'paragraph_title' or body['type'] not in {'text', 'paragraph'}
                or title['page'] != body['page'] or not title['text'].rstrip().endswith((':', '：'))
                or parse_number(title['text'])['kind']
                or re.match(r'\s*(?:Chapter\s+\d|Appendix\s+[A-Z]|\d+(?:[.．]\d+)*)\b', title['text'], re.I)
                or not 2 <= len(title['text'].strip()) <= 80
                or not valid_box(b) or not valid_box(c)
                or b[2]-b[0] > .35 or b[3]-b[1] > .04
                or abs(b[1]-c[1]) > .015):
            continue
        lo, mid = index.span(span_for(index, title))
        body_lo, hi = index.span(span_for(index, body))
        atoms = [index.atoms[n] for n in title['indices']+body['indices']]
        if (mid != body_lo or any(a['id'] in changed or a.get('source_panel')
                or a.get('recovery_issue') or a['kind'] not in {'text', 'math'} for a in atoms)
                or any(r['lo'] < hi and lo < r['hi'] for r in barriers)):
            continue
        candidates[title['page']-1].append((title, body, lo, mid, hi))
    if not candidates:
        return []
    import pymupdf
    rows = []
    with pymupdf.open(source) as pdf:
        for number, pairs in candidates.items():
            if not 0 <= number < len(pdf):
                continue
            page = pdf[number]
            glyphs = [dict(c, size=s['size'], bold=bool(s['flags'] & 16))
                for b in page.get_text('rawdict')['blocks'] for line in b.get('lines', [])
                if line.get('dir') == (1., 0.) for s in line['spans']
                for c in s['chars'] if c['c'].strip()]
            def rect(box):
                return pymupdf.Rect(*(v*(page.rect.width if n%2 == 0 else page.rect.height)
                                      for n,v in enumerate(box)))
            def center(char):
                return pymupdf.Point((char['bbox'][0]+char['bbox'][2])/2,
                                     (char['bbox'][1]+char['bbox'][3])/2)
            for title, body, lo, mid, hi in pairs:
                b, c = rect(title['bbox']), rect(body['bbox'])
                title_box, body_box = b+(-.6,-.6,.6,.6), c+(-.6,-.6,.6,.6)
                label = [g for g in glyphs if title_box.contains(center(g))]
                if not label or not all(g['bold'] for g in label):
                    continue
                size = median(g['size'] for g in label)
                baseline = median(g['origin'][1] for g in label)
                label = sorted(label, key=lambda g:g['bbox'][0])
                if (max(abs(g['origin'][1]-baseline) for g in label) > .2*size
                        or not ''.join(g['c'] for g in label).endswith((':','：'))):
                    continue
                # OCR may serialize the body's whole paragraph rectangle,
                # including the separately owned label on its first line.
                normal = [g for g in glyphs if body_box.contains(center(g))
                    and not title_box.contains(center(g))
                    and abs(g['origin'][1]-baseline) <= .2*size
                    and g['bbox'][0] >= max(x['bbox'][2] for x in label)-.2]
                if (len(normal) < 3 or any(g['bold'] or not .85 <= g['size']/size <= 1.15 for g in normal)
                        or not 0 <= min(g['bbox'][0] for g in normal)-max(g['bbox'][2] for g in label) <= 2*size):
                    continue
                rows.append(dict(method='source_glyph_run_in_title', page=number+1,
                    group=title['group'], lo=lo, mid=mid, hi=hi,
                    left=index.atoms[title['indices'][-1]]['id'],
                    right=index.atoms[body['indices'][0]]['id'],
                    source_boxes=[title['bbox'], body['bbox']]))
    return rows


def mixed_run_in_title_spans(index, barriers=(), changed=()):
    """Keep prose out of headings when native glyphs prove a bold run-in label.

    Translation may share a text atom between label and body. Split only at
    its first colon, never inside inline mathematics or across source groups.
    """
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return []
    candidates = defaultdict(list)
    blocks = group_blocks(index)
    following_blocks = {b['group']:z for b,z in zip(blocks,blocks[1:])}
    for block in blocks:
        box = block['bbox']
        if (block['type'] != 'paragraph_title' or parse_number(block['text'])['kind']
                or not valid_box(box) or not .2 <= box[2]-box[0] <= .85
                or box[3]-box[1] > .18):
            continue
        lo, hi = index.span(span_for(index, block))
        if any(r['lo'] < hi and lo < r['hi'] for r in barriers):
            continue
        mid = None
        for n in block['indices']:
            atom = index.atoms[n]
            if (atom['id'] in changed or atom.get('source_panel') or atom.get('recovery_issue')
                    or atom['kind'] not in {'text', 'math'}):
                mid = None; break
            if mid is None:
                if atom['kind'] != 'text':
                    break
                match = re.search('[:：]', atom['content'])
                if match:
                    mid = index.prefix[n]+match.end()
        if mid is None or not 3 <= mid-lo <= 110 or hi-mid < 12:
            continue
        candidates[block['page']-1].append((block, lo, mid, hi))
    if not candidates:
        return []
    import pymupdf
    rows = []
    with pymupdf.open(source) as pdf:
        for number, blocks in candidates.items():
            if not 0 <= number < len(pdf):
                continue
            page = pdf[number]
            glyphs = [dict(c, size=s['size'], bold=bool(s['flags'] & 16))
                for b in page.get_text('rawdict')['blocks'] for line in b.get('lines', [])
                if line.get('dir') == (1., 0.) for s in line['spans']
                for c in s['chars'] if c['c'].strip()]
            for block, lo, mid, hi in blocks:
                box = pymupdf.Rect(*(v*(page.rect.width if k%2==0 else page.rect.height)
                                     for k,v in enumerate(block['bbox'])))+(-.6,-.6,.6,.6)
                owned = [g for g in glyphs if box.contains(pymupdf.Point(
                    (g['bbox'][0]+g['bbox'][2])/2, (g['bbox'][1]+g['bbox'][3])/2))]
                if not owned:
                    continue
                baseline = min(g['origin'][1] for g in owned)
                first = sorted((g for g in owned if abs(g['origin'][1]-baseline) < 2),
                               key=lambda g:g['bbox'][0])
                prefix = []
                for g in first:
                    if not g['bold']:
                        break
                    prefix.append(g)
                    if g['c'] in ':：':
                        break
                if not prefix or prefix[-1]['c'] not in ':：' or len(prefix) < 3:
                    continue
                size = median(g['size'] for g in prefix)
                tail = [g for g in first if g['bbox'][0] >= prefix[-1]['bbox'][2]-.1
                        and abs(g['origin'][1]-prefix[0]['origin'][1]) <= .2*size]
                if (len(tail) < 4 or sum(not g['bold'] for g in tail) < 4
                        or min(g['bbox'][0] for g in tail)-prefix[-1]['bbox'][2] > 2.5*size):
                    continue
                row = dict(method='native_mixed_run_in_label', group=block['group'],
                    page=number+1, lo=lo, mid=mid, hi=hi,
                    source_label=''.join(g['c'] for g in prefix), source_bbox=block['bbox'])
                following = following_blocks.get(block['group'])
                b, c = block['bbox'], following['bbox'] if following else None
                next_glyphs = []
                if valid_box(c):
                    next_box = pymupdf.Rect(*(v*(page.rect.width if k%2==0 else page.rect.height)
                                             for k,v in enumerate(c)))+(-.6,-.6,.6,.6)
                    next_glyphs = [g for g in glyphs if next_box.contains(pymupdf.Point(
                        (g['bbox'][0]+g['bbox'][2])/2,(g['bbox'][1]+g['bbox'][3])/2))]
                normal = [g for g in owned if g['size'] >= .85*size]
                last_y = max((g['origin'][1] for g in normal),default=0)
                last_line = [g for g in normal if abs(g['origin'][1]-last_y) <= .2*size]
                if (following and following['type'] in {'text','paragraph'}
                        and following['page']==block['page'] and valid_box(c)
                        and len(next_glyphs)>=3 and last_line
                        and max(g['bbox'][2] for g in last_line) >= b[2]*page.rect.width-2*size
                        and not re.search(r'[.!?。！？:：;；]["\'”’）)]*$',block['text'].rstrip())
                        and 0 <= c[1]-b[3] <= .015
                        and abs(c[0]-b[0]) <= .005 and abs(c[2]-b[2]) <= .015
                        and all(index.atoms[n]['kind'] in {'text','math'}
                                and index.atoms[n]['id'] not in changed
                                and not index.atoms[n].get('source_panel')
                                and not index.atoms[n].get('recovery_issue') for n in following['indices'])
                        and not any(r['lo'] < index.prefix[following['indices'][-1]+1]
                                    and lo < r['hi'] for r in barriers)):
                    row.update(left=index.atoms[block['indices'][-1]]['id'],
                               right=index.atoms[following['indices'][0]]['id'])
                rows.append(row)
    return rows
