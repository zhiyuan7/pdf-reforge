"""Prove misclassified running heads from native ink and neighboring pages."""
from collections import defaultdict
from pathlib import Path
from statistics import median

from .headings import group_blocks, span_for, valid_box


def running_header_rows(index):
    """Return whole source groups confined to a repeated ruled header band.

    Neither translated text nor a top-of-page coordinate alone establishes a
    header. Require an isolated page numeral, native text on its baseline, a
    rule directly below it, and the same band on two other nearby PDF pages.
    Split OCR header groups are checked against that complete native band.
    """
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return []
    candidates = defaultdict(list)
    for b in group_blocks(index):
        if (b['type'] in {'doc_title', 'paragraph_title', 'text', 'paragraph'}
                and valid_box(b['bbox']) and b['bbox'][1] < .2
                and all(index.atoms[n]['kind'] == 'text'
                        and not index.atoms[n].get('recovery_issue')
                        and not index.atoms[n].get('source_panel') for n in b['indices'])):
            candidates[b['page']-1].append(b)
    if not candidates:
        return []
    import pymupdf
    rows, cache = [], {}
    with pymupdf.open(source) as pdf:
        def bands(number):
            if number in cache:
                return cache[number]
            if not 0 <= number < len(pdf):
                return []
            page = pdf[number]
            w, h = page.rect.width, page.rect.height
            glyphs = [dict(c, size=s['size'])
                for b in page.get_text('rawdict')['blocks'] for l in b.get('lines', [])
                if l.get('dir') == (1., 0.) for s in l['spans'] for c in s['chars']
                if c['c'].strip() and c['origin'][1] < .22*h]
            lines = []
            for g in sorted(glyphs, key=lambda g:g['origin'][1]):
                if not lines or abs(g['origin'][1]-lines[-1][0]['origin'][1]) > .25*g['size']:
                    lines.append([])
                lines[-1].append(g)
            rules = [d['rect'] for d in page.get_drawings()
                     if d['rect'].height < .5 and d['rect'].width >= .4*w
                     and 0 <= d['rect'].y0 < .22*h
                     and any(item[0] == 'l' for item in d['items'])]
            found = []
            for line in lines:
                line.sort(key=lambda g:g['bbox'][0])
                size = median(g['size'] for g in line)
                baseline = median(g['origin'][1] for g in line)
                chunks = [[]]
                for g in line:
                    if chunks[-1] and g['bbox'][0]-chunks[-1][-1]['bbox'][2] > 3*size:
                        chunks.append([])
                    chunks[-1].append(g)
                if len(chunks) < 2:
                    continue
                numeral = next((c for c in (chunks[0], chunks[-1])
                    if 1 <= len(c) <= 5 and all(g['c'].isascii() and g['c'].isdigit() for g in c)), None)
                if not numeral:
                    continue
                other = [g for c in chunks if c is not numeral for g in c]
                if (sum(g['c'].isalpha() for g in other) < 3
                        or any(not .8 <= g['size']/size <= 1.2 for g in line)):
                    continue
                matching = [r for r in rules if 0 < r.y0-baseline < 1.5*size
                            and r.x0 <= line[0]['bbox'][0]+size
                            and r.x1 >= line[-1]['bbox'][2]-size]
                if not matching:
                    continue
                r = matching[0]
                found.append(dict(baseline=baseline/h, size=size/h,
                    printed_page=int(''.join(g['c'] for g in numeral)),
                    rule=[r.x0/w, r.y0/h, r.x1/w, r.y1/h], glyphs=line))
            cache[number] = found
            return found

        for number, blocks in candidates.items():
            if not 0 <= number < len(pdf):
                continue
            page = pdf[number]
            for band in bands(number):
                witnesses = [n+1 for n in range(number-2, number+3) if n != number
                    and any(abs(b['baseline']-band['baseline']) <= .003
                            and b['printed_page']-band['printed_page'] == n-number
                            and .8 <= b['size']/band['size'] <= 1.2
                            and max(abs(a-z) for a,z in zip(b['rule'], band['rule'])) <= .005
                            for b in bands(n))]
                if len(witnesses) < 2:
                    continue
                for block in blocks:
                    box = pymupdf.Rect(*(v*(page.rect.width if k%2==0 else page.rect.height)
                                         for k,v in enumerate(block['bbox'])))+(-.6,-.6,.6,.6)
                    # Check all ink in the source group, including any body
                    # line that an overly broad OCR rectangle may also cover.
                    owned = [c for b in page.get_text('rawdict')['blocks'] for l in b.get('lines', [])
                             for s in l['spans'] for c in s['chars'] if c['c'].strip()
                             and box.contains(pymupdf.Point((c['bbox'][0]+c['bbox'][2])/2,
                                                           (c['bbox'][1]+c['bbox'][3])/2))]
                    if not owned or any(abs(c['origin'][1]/page.rect.height-band['baseline'])
                                        > .3*band['size'] for c in owned):
                        continue
                    lo, hi = index.span(span_for(index, block))
                    rows.append(dict(method='native_repeated_ruled_running_header',
                        reason='source_marginal', group=block['group'], page=number+1,
                        lo=lo, hi=hi, bbox=block['bbox'], witness_pages=witnesses,
                        source_text=''.join(c['c'] for c in band['glyphs']),
                        native_baseline=band['baseline'], rule=band['rule']))
    return rows
