"""Recover reviewed example ink and literal labeled rows from native glyphs.

Uniform ink needs no character-offset mapping after translation. Label rows do:
their complete text must still match the live source, ignoring only whitespace.
"""
from collections import Counter
from pathlib import Path
import re

from book_pipeline.publishing.native_lateral import native_groups


INK_METHOD = 'native_uniform_block_ink_v1'


def native_examples(index, registry=(), exclusions=(), changed=(), *, pages=()):
    source = index.source.get('source_pdf')
    if not pages or not source or not Path(source).is_file():
        return {}, {}
    groups = [g for g in native_groups(index) if g['page']+1 in pages
              and all(a['kind'] == 'text' and a['role'] in {'text', 'paragraph'}
                      and not a.get('recovery_issue') and a['id'] not in changed for a in g['atoms'])
              and not any(r['lo'] < g['hi'] and g['lo'] < r['hi'] for r in [*registry, *exclusions])]
    import pymupdf
    ink, labels = {}, {}
    compact = lambda text: re.sub(r'\s+', '', text)
    with pymupdf.open(source) as pdf:
        for number in sorted({g['page'] for g in groups}):
            page = pdf[number]
            glyphs = [dict(c, color=s['color'], bold=bool(s['flags'] & 16), size=s['size'])
                      for b in page.get_text('rawdict')['blocks'] for l in b.get('lines', [])
                      if l.get('dir') == (1., 0.) for s in l['spans'] for c in s['chars']]
            def observed(box):
                clip = pymupdf.Rect(*(v*(page.rect.width if n%2==0 else page.rect.height)
                                       for n,v in enumerate(box)))+(-.6,-.6,.6,.6)
                return [c for c in glyphs if clip.contains(pymupdf.Point(
                    (c['bbox'][0]+c['bbox'][2])/2, (c['bbox'][1]+c['bbox'][3])/2))]
            for group in (g for g in groups if g['page'] == number):
                chars = observed(group['bbox'])
                colors = {c['color'] for c in chars if c['c'].strip()}
                # Never infer the offsets of a mixed-color translated block.
                if len(colors) == 1:
                    color = next(iter(colors))
                    rgb = [(color>>shift & 255)/255 for shift in (16,8,0)]
                    for atom in group['atoms']:
                        if max(rgb)-min(rgb) > .08 and not atom.get('source_text_runs'):
                            ink[atom['id']] = [dict(start=0, end=atom['length'], color=rgb,
                                method=INK_METHOD, page=number+1, bbox=group['bbox'])]
                # Bold field labels are genuine row boundaries, unlike the
                # ordinary physical line wraps inside their example values.
                # Require repeated labels and an exact complete glyph match.
                if len(group['atoms']) != 1:
                    continue
                atom = group['atoms'][0]
                if compact(atom['content']) != compact(''.join(c['c'] for c in chars)):
                    continue
                candidates = []
                for start,c in enumerate(chars):
                    if not c['bold'] or not c['c'].strip():
                        continue
                    previous = chars[start-1] if start else None
                    if previous and (previous['bold'] or abs(previous['origin'][1]-c['origin'][1]) < .5*c['size']):
                        continue
                    end = start
                    while end < len(chars) and chars[end]['bold'] and abs(chars[end]['origin'][1]-c['origin'][1]) < .3*c['size']:
                        end += 1
                    text = ''.join(v['c'] for v in chars[start:end]).strip()
                    if (3 <= len(text) <= 60 and text.endswith((':','：'))
                            and abs(c['bbox'][0]-group['bbox'][0]*page.rect.width) <= c['size']*.6):
                        candidates.append(text)
                counts = Counter(candidates)
                if len(counts) < 2 or sum(counts.values()) < 4:
                    # Individual source blocks can each hold a single example
                    # row. Verify the same labels recur elsewhere on the page.
                    all_bold = ''.join(c['c'] if c['bold'] else '\n' for c in glyphs)
                    if not candidates or any(all_bold.count(t) < 2 for t in candidates):
                        continue
                cursor, runs = 0, []
                for text in candidates:
                    found = re.search(r'\s+'.join(re.escape(w) for w in text.split()), atom['content'][cursor:])
                    if not found:
                        runs = []
                        break
                    start, end = cursor+found.start(), cursor+found.end()
                    runs.append(dict(start=start, end=end, quote=atom['content'][start:end],
                                     method='native_literal_example_label', page=number+1))
                    cursor = end
                if runs and not atom['content'][:runs[0]['start']].strip():
                    labels[atom['id']] = runs
    return ink, labels
