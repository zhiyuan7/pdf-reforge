"""Recover searchable ruled tables from complete native row groups.

Three live rules, a separated header, repeated live column starts and explicit
cell separators must agree. This only changes layout; no cell is transcribed.
"""
from pathlib import Path
from collections import Counter

from book_pipeline.ingestion.native import spans_for
from book_pipeline.ingestion.panels import intersection_fraction
from book_pipeline.publishing.native_lateral import native_groups


def native_ruled_tables(index, registry=(), exclusions=()):
    import pymupdf
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return []
    groups = native_groups(index)
    result = []
    with pymupdf.open(source) as pdf:
        for number in sorted({g['page'] for g in groups if 0 <= g['page'] < len(pdf)}):
            page = pdf[number]
            rules = sorted([d for d in page.get_drawings()
                if len(d['items']) == 1 and d['items'][0][0] == 'l'
                and d.get('fill') is None and d.get('color')
                and max(d['color']) < .65 and max(d['color'])-min(d['color']) < .03
                and .2 <= d['width'] <= 2 and d['rect'].height < .1
                and d['rect'].width > page.rect.width*.3], key=lambda d:d['rect'].y0)
            spans = spans_for(page)
            for top, middle, bottom in zip(rules, rules[1:], rules[2:]):
                u,m,v = (d['rect'] for d in (top,middle,bottom))
                if (any(abs(r.x0-u.x0) > .5 or abs(r.x1-u.x1) > .5 for r in (m,v))
                        or not .05 < (v.y0-u.y0)/page.rect.height < .6):
                    continue
                box = [u.x0/page.rect.width,u.y0/page.rect.height,u.x1/page.rect.width,v.y0/page.rect.height]
                rows = [g for g in groups if g['page'] == number
                        and box[0] <= g['bbox'][0] < g['bbox'][2] <= box[2]
                        and box[1] < g['bbox'][1] < g['bbox'][3] < box[3]]
                if len(rows) < 3 or not rows[0]['bbox'][3] < m.y0/page.rect.height < rows[1]['bbox'][1]:
                    continue
                lo,hi = rows[0]['lo'],rows[-1]['hi']
                atoms = [a for a,_,_ in index.fragments(lo,hi)]
                owned = {a['id'] for g in rows for a in g['atoms']}
                if (any(a['id'] not in owned or a['kind'] not in {'text','math'}
                        or a['role'] not in {'text','paragraph'} for a in atoms)
                        or any(a['hi'] != b['lo'] or a['bbox'][3] > b['bbox'][1] for a,b in zip(rows,rows[1:]))
                        or any(r['lo'] < hi and lo < r['hi'] for r in [*registry,*exclusions])
                        or any(a['page'] == number and a['id'] not in owned and a.get('bbox')
                            and box[0] < a['bbox'][2] and a['bbox'][0] < box[2]
                            and box[1] < a['bbox'][3] and a['bbox'][1] < box[3] for a in index.atoms)):
                    continue
                header = sorted([s for s in spans if s['text'].strip()
                    and rows[0]['bbox'][1]-.002 <= s['bbox'][1]
                    and s['bbox'][3] <= rows[0]['bbox'][3]+.003
                    and box[0] <= s['bbox'][0] < s['bbox'][2] <= box[2]], key=lambda s:s['bbox'][0])
                clusters = []
                for s in header:
                    if not clusters or s['bbox'][0]-clusters[-1][1] > .025:
                        clusters.append([s['bbox'][0],s['bbox'][2]])
                    else:
                        clusters[-1][1] = max(clusters[-1][1],s['bbox'][2])
                if not 2 <= len(clusters) <= 5:
                    continue
                cells = []
                for g in rows:
                    cuts = [index.prefix[index.ids[a['id']]+1] for a in g['atoms']
                            if a['kind'] == 'text' and a['content'] == '\n']
                    bounds = [g['lo'],*cuts,g['hi']]
                    if len(cuts) != len(clusters)-1 or any(not index.text(a,z).strip() for a,z in zip(bounds,bounds[1:])):
                        break
                    observed = [s for s in spans if s['text'].strip()
                        and g['bbox'][1]-.006 <= s['bbox'][1]
                        and s['bbox'][3] <= g['bbox'][3]+.004
                        and box[0] <= s['bbox'][0] < s['bbox'][2] <= box[2]]
                    starts = [c[0] for c in clusters]
                    assigned = [[s for s in observed if start-.008 <= s['bbox'][0]
                        and s['bbox'][2] <= (starts[k+1]-.008 if k+1 < len(starts) else box[2])]
                        for k,start in enumerate(starts)]
                    if (any(not ss or abs(min(s['bbox'][0] for s in ss)-start) > .006
                            for ss,start in zip(assigned,starts))
                            or sum(map(len,assigned)) != len(observed)):
                        break
                    cells.append(list(map(list,zip(bounds,bounds[1:]))))
                if len(cells) != len(rows):
                    continue
                caption = next((g for g in groups if g['hi'] == lo and g['page'] == number
                    and all(a['role'] == 'table_caption' for a in g['atoms'])
                    and 0 <= box[1]-g['bbox'][3] <= .04),None)
                if caption and any(r['lo'] < caption['hi'] and caption['lo'] < r['hi'] for r in [*registry,*exclusions]):
                    continue
                body_boxes=[g['bbox'] for g in groups if g['page']==number
                    and (g['hi']<=lo or g['lo']>=hi) and g['bbox'][2]-g['bbox'][0]>.6
                    and all(a['role'] in {'text','paragraph'} for a in g['atoms'])]
                body_sizes=Counter();table_sizes=Counter()
                for s in spans:
                    weight=len(s['text'].strip())
                    if intersection_fraction(s['bbox'],box)>.98:table_sizes[s['size']]+=weight
                    elif any(intersection_fraction(s['bbox'],b)>.98 for b in body_boxes):body_sizes[s['size']]+=weight
                ratio=(max(.8,min(1,table_sizes.most_common(1)[0][0]/body_sizes.most_common(1)[0][0]))
                       if table_sizes and body_sizes else 1)
                result.append(dict(method='native_ruled_table',lo=caption['lo'] if caption else lo,hi=hi,
                    table_lo=lo,page=number+1,bbox=box,cells=cells,
                    body_font_ratio=ratio,
                    caption_range=[caption['lo'],caption['hi']] if caption else None,
                    column_starts=[c[0] for c in clusters],
                    column_widths=[(z-a)/(box[2]-clusters[0][0]) for a,z in zip(
                        [c[0] for c in clusters],[*[c[0] for c in clusters[1:]],box[2]])],targets=[]))
    return [r for r in result if not any(q is not r and q['lo'] < r['hi'] and r['lo'] < q['hi'] for q in result)]
