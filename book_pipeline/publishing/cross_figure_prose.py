"""Recover reviewed native sentences interrupted by physical page boundaries.

Move only the proven continuation to the preceding prose paragraph. Figures,
captions and matched footnotes keep their own rendering and source intervals.
"""
from pathlib import Path
import re
from collections import Counter
from statistics import median

from book_pipeline.publishing.native_lateral import native_groups


def cross_figure_prose(index, registry=(), exclusions=(), occupied=(), notes=(), changed=(), *, pages=None):
    if pages is not None and not pages:
        return []
    source = index.source.get('source_pdf')
    if not source or not Path(source).is_file():
        return []
    groups = native_groups(index)
    def prose(g):
        return (all(a['kind'] in {'text', 'math'} and a['role'] in {'text', 'paragraph'}
                    and a['block_type'] in {'text', 'paragraph'} and a['id'] not in changed
                    and not a.get('recovery_issue') for a in g['atoms'])
                and not any(r['lo'] < g['hi'] and g['lo'] < r['hi']
                            for r in [*registry, *exclusions, *occupied])
                # A moved reference would also need its page insertion moved.
                # Keep such paragraphs in ordinary flow until reviewed as a unit.
                and not any(n.get('reference') and n['reference']['lo'] < g['hi']
                            and g['lo'] < n['reference']['hi'] for n in notes))
    mains = [g for g in groups if prose(g)]
    decorative = {'header', 'footer', 'page_number'}
    note_ranges = [(n['lo'], n['hi']) for n in notes if n['status'] == 'matched']
    header_ranges = [(r['lo'], r['hi']) for r in exclusions
                     if r.get('reason') == 'source_marginal'
                     and r.get('method') == 'native_repeated_ruled_running_header']
    def obstacle(a, lo, hi):
        return (a['role'] in decorative or a['kind'] == 'image'
                or a['kind'] in {'text', 'math'} and a['role'] in {'image_caption', 'chart_caption'}
                or any(x <= lo and hi <= z for x, z in header_ranges)
                or any(x <= lo and hi <= z for x, z in note_ranges))
    import pymupdf
    result = []
    with pymupdf.open(source) as pdf:
        profiles = {}
        def profile(g):
            if g['group'] in profiles:
                return profiles[g['group']]
            page = pdf[g['page']]
            b = g['bbox']
            clip = pymupdf.Rect(b[0]*page.rect.width, b[1]*page.rect.height,
                                b[2]*page.rect.width, b[3]*page.rect.height) + (-.6, -.6, .6, .6)
            chars = [dict(c, size=s['size']) for block in page.get_text('rawdict')['blocks']
                     for line in block.get('lines', []) if line.get('dir') == (1., 0.)
                     for s in line['spans'] for c in s['chars'] if c['c'].strip()
                     and clip.contains(pymupdf.Point((c['bbox'][0]+c['bbox'][2])/2,
                                                     (c['bbox'][1]+c['bbox'][3])/2))]
            if not chars:
                profiles[g['group']] = None
                return None
            size = Counter(round(c['size'], 1) for c in chars).most_common(1)[0][0]
            lines = []
            for c in sorted((c for c in chars if c['size'] >= .85*size),
                            key=lambda c: (c['origin'][1], c['bbox'][0])):
                if not lines or abs(median(x['origin'][1] for x in lines[-1])-c['origin'][1]) > .3*size:
                    lines.append([])
                lines[-1].append(c)
            def row(cs):
                return dict(text=''.join(c['c'] for c in sorted(cs, key=lambda c: c['bbox'][0])),
                            x0=min(c['bbox'][0] for c in cs), x1=max(c['bbox'][2] for c in cs))
            value = dict(first=row(lines[0]), last=row(lines[-1]), size=size,
                         x0=b[0]*page.rect.width, x1=b[2]*page.rect.width,
                         width=page.rect.width)
            profiles[g['group']] = value
            return value
        for left, right in zip(mains, mains[1:]):
            if (pages is not None and not set(range(left['page']+1, right['page']+2)) <= set(pages)):
                continue
            if (not 1 <= right['page']-left['page'] <= 2
                    or left['bbox'][3] < .75 or right['bbox'][1] > .65
                    or right['hi']-right['lo'] > 2400):
                continue
            between = [(a, index.prefix[n], index.prefix[n+1])
                       for n in range(index.ids[left['atoms'][-1]['id']]+1,
                                      index.ids[right['atoms'][0]['id']])
                       for a in [index.atoms[n]]]
            figures = [a['id'] for a, _, _ in between if a['kind'] == 'image']
            # With no figure to bypass, only an ordinary consecutive page
            # boundary is eligible. Body text, headings and unmatched notes
            # still block the join, and live glyphs below prove continuity.
            if ((not figures and (right['page'] != left['page']+1 or right['bbox'][1] > .3))
                    or any(not obstacle(a, lo, hi) or a['id'] in changed
                           or a.get('source_panel') or a.get('recovery_issue') for a, lo, hi in between)
                    or not set(range(left['page'], right['page']+1)) <= {a['page'] for a in index.atoms}
                    or any(r['lo'] < right['hi'] and left['lo'] < r['hi']
                           and r.get('kind') != 'figure' for r in registry)
                    or any(r['lo'] < right['hi'] and left['lo'] < r['hi']
                           and r.get('reason') != 'source_marginal' for r in exclusions)):
                continue
            a, b = profile(left), profile(right)
            if (not a or not b or not .9 <= b['size']/a['size'] <= 1.1
                    or a['x1']-a['x0'] < .3*a['width']
                    or not .97 <= (b['x1']-b['x0'])/(a['x1']-a['x0']) <= 1.03
                    or ((right['page']-left['page']) % 2 == 0
                        and abs(b['x0']/b['width']-a['x0']/a['width'])*b['width'] > .5*b['size'])
                    or abs(a['last']['x1']-a['x1']) > 1.2*a['size']
                    or abs(b['first']['x0']-b['x0']) > .4*b['size']
                    or re.search(r'[-\u00ad.!?。！？:：;；][\"\'”’）)]*$', a['last']['text'])
                    or not re.match(r'[a-z]', b['first']['text'])
                    or re.match(r'[a-z][.)]', b['first']['text'])):
                continue
            result.append(dict(method=('native_sentence_across_figures' if figures else
                                       'native_sentence_across_pages'),
                left=left['atoms'][-1]['id'], right=right['atoms'][0]['id'],
                insert_after=left['hi'], moved_range=[right['lo'], right['hi']],
                source_pages=[left['page']+1, right['page']+1],
                source_boxes=[left['bbox'], right['bbox']],
                source_boundary=[a['last']['text'], b['first']['text']],
                figures=figures))
    return result
