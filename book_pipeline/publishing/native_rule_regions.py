"""Recover paired native rules and their exclusively enclosed prose inset."""
from pathlib import Path

from book_pipeline.publishing.native_lateral import native_groups


def native_rule_regions(index, registry=(), exclusions=(), changed=()):
    import pymupdf
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return []
    groups = native_groups(index)
    if not groups:
        return []
    result = []
    with pymupdf.open(source) as pdf:
        for number in sorted({g['page'] for g in groups if 0 <= g['page'] < len(pdf)}):
            page = pdf[number]
            candidates = []
            for d in page.get_drawings():
                color = d.get('color')
                if (len(d['items']) == 1 and d['items'][0][0] == 'l'
                        and d.get('fill') is None and color and max(color) < .65
                        and max(color)-min(color) < .03 and .2 <= d['width'] <= 4
                        and d['rect'].height < .1 and d['rect'].width > page.rect.width*.35):
                    candidates.append(d)
            candidates.sort(key=lambda d:d['rect'].y0)
            for upper,lower in zip(candidates,candidates[1:]):
                u,v = upper['rect'],lower['rect']
                if (abs(u.x0-v.x0) > .5 or abs(u.x1-v.x1) > .5
                        or abs(upper['width']-lower['width']) > .05
                        or upper['color'] != lower['color']
                        or not .035 < (v.y0-u.y0)/page.rect.height < .25):
                    continue
                box = [u.x0/page.rect.width,u.y0/page.rect.height,
                       u.x1/page.rect.width,v.y0/page.rect.height]
                inside = [g for g in groups if g['page'] == number
                          and box[0]-.002 <= g['bbox'][0] < g['bbox'][2] <= box[2]+.002
                          and box[1] < g['bbox'][1] < g['bbox'][3] < box[3]]
                if len(inside) < 2:
                    continue
                lo,hi = min(g['lo'] for g in inside),max(g['hi'] for g in inside)
                atoms = [a for a,_,_ in index.fragments(lo,hi)]
                owned = {a['id'] for g in inside for a in g['atoms']}
                if (any(a['id'] not in owned or a['id'] in changed
                        or a['kind'] not in {'text','math','display_math'}
                        or a.get('role') in {'header','footer','page_number'} for a in atoms)
                        or any(r['lo'] < hi and lo < r['hi']
                               and r.get('kind') not in {'equation','formula'} for r in [*registry,*exclusions])
                        or any(a['page'] == number and a['id'] not in owned and a.get('bbox')
                               and box[1] < a['bbox'][3] and a['bbox'][1] < box[3]
                               and box[0] < a['bbox'][2] and a['bbox'][0] < box[2] for a in index.atoms)):
                    continue
                context = [g for g in groups if g['page'] == number and g['hi'] <= lo
                           and g['bbox'][2]-g['bbox'][0] > .4]
                if not context:
                    continue
                main = max(context,key=lambda g:g['hi'])['bbox']
                width = main[2]-main[0]
                left,right = (box[0]-main[0])/width,(main[2]-box[2])/width
                if not .01 <= left <= .18 or not -.01 <= right <= .18:
                    continue
                result.append(dict(method='native_paired_rule_inset_v1',lo=lo,hi=hi,
                    page=number+1,bbox=box,left_fraction=left,right_fraction=max(0,right),
                    thickness_pt=upper['width'],color=list(upper['color'])))
    # Adjacent regions may share a rule; overlapping ownership is ambiguous.
    return [r for r in result if not any(q is not r and q['lo'] < r['hi'] and r['lo'] < q['hi'] for q in result)]
