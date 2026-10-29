"""Preserve native lateral captions and small colored marginal references."""
from collections import OrderedDict
import re

from book_pipeline.structure.headings import valid_box


def native_groups(index, allow_ocr=False, split_pages=False):
    groups=OrderedDict()
    for n,atom in enumerate(index.atoms):
        key=(atom['group'],atom['page']) if split_pages else atom['group']
        groups.setdefault(key,[]).append(n)
    result=[]
    for group,positions in groups.items():
        atoms=[index.atoms[n] for n in positions]
        boxes=[a.get('bbox') for a in atoms]
        if (positions!=list(range(min(positions),max(positions)+1))
                or len({a['page'] for a in atoms})!=1
                or any(a.get('extraction_method') not in ({'native','ocr'} if allow_ocr else {'native'}) or a.get('source_panel')
                       or not valid_box(b) for a,b in zip(atoms,boxes))):
            continue
        result.append(dict(group=atoms[0]['group'],atoms=atoms,page=atoms[0]['page'],
            lo=index.prefix[positions[0]],hi=index.prefix[positions[-1]+1],
            bbox=[min(b[0] for b in boxes),min(b[1] for b in boxes),
                  max(b[2] for b in boxes),max(b[3] for b in boxes)]))
    return result


def side_caption_rows(index,registry=(),exclusions=(),changed=(),image_boxes=None):
    groups=native_groups(index)
    images=[g for g in groups if len(g['atoms'])==1 and g['atoms'][0]['kind']=='image'
            and not g['atoms'][0].get('recovery_issue')]
    for image in images:
        box=(image_boxes or {}).get(image['atoms'][0]['id'])
        if valid_box(box):image['bbox']=box
    captions=[g for g in groups if all(a['kind'] in {'text','math'}
              and a['role'] in {'image_caption','chart_caption'} for a in g['atoms'])]
    pairs=[]
    for image in images:
        for caption in captions:
            lo,hi=min(image['lo'],caption['lo']),max(image['hi'],caption['hi'])
            atoms=image['atoms']+caption['atoms'];b,c=image['bbox'],caption['bbox']
            left,right=sorted((b,c),key=lambda b:b[0]);extent=right[2]-left[0]
            overlap=min(b[3],c[3])-max(b[1],c[1])
            if (image['page']!=caption['page']
                    or not (image['hi']==caption['lo'] or caption['hi']==image['lo'])
                    or any(a['id'] in changed for a in atoms)
                    or not .008<=right[0]-left[2]<=.12
                    or min(b[3]-b[1],c[3]-c[1])<=0
                    or overlap<.6*min(b[3]-b[1],c[3]-c[1])
                    or abs(b[1]-c[1])>.05 or max(b[3],c[3])-min(b[1],c[1])>.36
                    or min((b[2]-b[0])/extent,(c[2]-c[0])/extent)<.2
                    or any(r['lo']<hi and lo<r['hi'] for r in exclusions)
                    or any(r['lo']<hi and lo<r['hi'] and r['kind']!='figure' for r in registry)):
                continue
            pairs.append(dict(method='native_lateral_caption',lo=lo,hi=hi,
                image=image['atoms'][0]['id'],caption_range=[caption['lo'],caption['hi']],
                widths=[(b[2]-b[0])/extent,(c[2]-c[0])/extent],image_left=b[0]<c[0],
                image_bbox=b,caption_bbox=c,page=image['page']+1,
                targets=[r['anchor'] for r in registry if lo<=r['lo']<r['hi']<=hi]))
    # Neither reading order nor a nearest-distance guess can choose among
    # several possible owners. Retain the ordinary flow for ambiguous pairs.
    return [p for p in pairs if sum(q['image']==p['image'] for q in pairs)==1
            and sum(q['caption_range']==p['caption_range'] for q in pairs)==1]


def marginal_figures(index,registry=(),exclusions=(),changed=(),image_boxes=None):
    """Keep a bounded native margin diagram and caption beside owned prose.

    Reading order alone cannot establish a side figure: the complete adjacent
    prose must occupy a separate column overlapping the diagram/caption band.
    This also handles a final excerpt paragraph with no full-width continuation.
    """
    groups=native_groups(index);result=[]
    for n,image in enumerate(groups[:-1]):
        caption=groups[n+1]
        if (len(image['atoms'])!=1 or image['atoms'][0]['kind']!='image'
                or image['hi']!=caption['lo'] or image['page']!=caption['page']
                or not all(a['kind'] in {'text','math'} and a['role']=='image_caption'
                           for a in caption['atoms'])):continue
        b=(image_boxes or {}).get(image['atoms'][0]['id'],image['bbox']);c=caption['bbox']
        if (not valid_box(b) or not 0<=c[1]-b[3]<=.04
                or min(b[2],c[2])-max(b[0],c[0])<.6*(b[2]-b[0])):continue
        aside=[min(b[0],c[0]),b[1],max(b[2],c[2]),c[3]]
        if not .08<=aside[2]-aside[0]<=.3 or aside[3]-aside[1]>.26:continue
        main=[]
        for g in reversed(groups[max(0,n-3):n]):
            a=g['bbox']
            if (g['page']!=image['page'] or g['hi']!=(main[0]['lo'] if main else image['lo'])
                    or not all(a['kind'] in {'text','math'} and a['role'] in {'text','paragraph'}
                               for a in g['atoms'])
                    or not .35<=a[2]-a[0]<=.7
                    or not (.008<=aside[0]-a[2]<=.12 or .008<=a[0]-aside[2]<=.12)
                    or not aside[1]-.025<=a[1]<aside[3]
                    or a[3]>aside[3]+.08
                    or (main and (abs(a[0]-main[0]['bbox'][0])>.01
                                  or not 0<=main[0]['bbox'][1]-a[3]<=.025))):break
            main.insert(0,g)
        if not main:continue
        a=[main[0]['bbox'][0],main[0]['bbox'][1],max(g['bbox'][2] for g in main),main[-1]['bbox'][3]]
        overlap=min(a[3],aside[3])-max(a[1],aside[1])
        if abs(a[1]-aside[1])>.025 or overlap<.6*min(a[3]-a[1],aside[3]-aside[1]):continue
        lo,hi=main[0]['lo'],caption['hi']
        if (hi-lo>2200 or any(a['id'] in changed or a.get('recovery_issue')
                for g in [*main,image,caption] for a in g['atoms'])
                or any(r['lo']<hi and lo<r['hi'] for r in exclusions)
                or any(r['lo']<hi and lo<r['hi'] and not
                    (r['kind']=='figure' and image['lo']<=r['lo']<r['hi']<=hi) for r in registry)):
            continue
        extent=max(a[2],aside[2])-min(a[0],aside[0])
        result.append(dict(method='native_marginal_figure',lo=lo,hi=hi,page=image['page']+1,
            main_ranges=[[g['lo'],g['hi']] for g in main],image=image['atoms'][0]['id'],
            caption_range=[caption['lo'],caption['hi']],main_left=a[0]<aside[0],
            image_width_fraction=(b[2]-b[0])/(aside[2]-aside[0]),
            widths=[(a[2]-a[0])/extent,(aside[2]-aside[0])/extent],
            main_bbox=a,image_bbox=b,caption_bbox=c,
            targets=[r['anchor'] for r in registry if image['lo']<=r['lo']<r['hi']<=hi]))
    return [r for r in result if sum(q['lo']<r['hi'] and r['lo']<q['hi'] for q in result)==1]


def marginal_references(index,registry=(),exclusions=(),changed=()):
    groups=native_groups(index);result=[]
    for main,note in zip(groups,groups[1:]):
        b,c=main['bbox'],note['bbox'];atoms=main['atoms']+note['atoms']
        text=index.text(note['lo'],note['hi']).strip()
        colored=any(max(r.get('color',[0,0,0]))-min(r.get('color',[0,0,0]))>.12
                    for a in note['atoms'] for r in a.get('source_text_runs',[]))
        if (main['hi']!=note['lo'] or main['page']!=note['page']
                or not 4<=len(text)<=100 or not colored
                or not .035<=c[2]-c[0]<=.18 or c[3]-c[1]>.08 or b[2]-b[0]<.4
                or not (.008<=b[0]-c[2]<=.12 or .008<=c[0]-b[2]<=.12)
                or not b[1]-.01<=c[1]<=b[3]+.01
                or len(index.text(main['lo'],main['hi']).strip())<60
                or any(a['id'] in changed or a['kind'] not in {'text','math'}
                       or a['role'] not in {'text','paragraph','equation_inline'} for a in atoms)
                or any(r['lo']<note['hi'] and main['lo']<r['hi'] for r in [*registry,*exclusions])):
            continue
        result.append(dict(method='native_colored_marginal_reference',lo=note['lo'],hi=note['hi'],
            attach_after=main['atoms'][-1]['id'],page=note['page']+1,
            main_bbox=b,note_bbox=c))
    return result


def marginal_glossaries(index,registry=(),exclusions=(),changed=()):
    """Pair a paragraph with successive small-font native marginal terms.

    Unlike a short colored reference, a glossary needs its own bounded column.
    Live glyph sizes and complete text ownership are required, not term meaning.
    """
    from pathlib import Path
    import pymupdf
    from book_pipeline.ingestion.native import spans_for
    from book_pipeline.ingestion.panels import intersection_fraction
    source=index.source.get('source_pdf')
    if not source or not Path(source).is_file():return []
    groups=native_groups(index);rows=[]
    with pymupdf.open(source) as pdf:
        pages={}
        for pos,main in enumerate(groups):
            b=main['bbox'];notes=[]
            if (b[2]-b[0]<.45 or b[3]-b[1]>.28
                    or not 100<=len(index.text(main['lo'],main['hi']))<=1600
                    or any(a['kind'] not in {'text','math'} or a['role'] not in {'text','paragraph'}
                           for a in main['atoms']) or not 0<=main['page']<len(pdf)):
                continue
            if main['page'] not in pages:pages[main['page']]=spans_for(pdf[main['page']])
            spans=pages[main['page']]
            body=[s for s in spans if s['text'].strip() and intersection_fraction(s['bbox'],b)>.98]
            if not body:continue
            size=sorted(s['size'] for s in body)[len(body)//2]
            previous=main
            for note in groups[pos+1:]:
                c=note['bbox'];text=index.text(note['lo'],note['hi'])
                if (previous['hi']!=note['lo'] or note['page']!=main['page']
                        or any(a['kind']!='text' or a['role'] not in {'text','paragraph'} for a in note['atoms'])
                        or not 4<=len(text.strip())<=100 or c[3]-c[1]>.08
                        or not b[1]-.01<=c[1]<=c[3]<=b[3]+.04):break
                box=[c[0]-.002,c[1]-.002,c[2]+.002,c[3]+.002]
                observed=[s for s in spans if s['text'].strip() and intersection_fraction(s['bbox'],box)>.98]
                if (not observed or any(s['size']>size*.82 for s in observed)
                        or ''.join(''.join(s['text'].split()) for s in observed)!=''.join(text.split())):break
                actual=[min(s['bbox'][0] for s in observed),min(s['bbox'][1] for s in observed),
                        max(s['bbox'][2] for s in observed),max(s['bbox'][3] for s in observed)]
                if not (.008<=actual[0]-b[2]<=.08 or .008<=b[0]-actual[2]<=.08):break
                colors={ink.get('color',0) for s in observed
                        for ink in s.get('source_ink',dict(runs=[s]))['runs'] if ink['text'].strip()}
                ink=next(iter(colors)) if len(colors)==1 else None
                color=([round(((ink>>shift)&255)/255,4) for shift in (16,8,0)] if ink is not None else None)
                notes.append({**note,'bbox':actual,'uniform_ink':color});previous=note
            if len(notes)<2:continue
            atoms=main['atoms']+[a for n in notes for a in n['atoms']]
            lo,hi=main['lo'],notes[-1]['hi']
            # The main text remains native, including explicit render edits.
            # Never relocate an edited note on its old small-font evidence.
            if (any(a.get('recovery_issue') for a in atoms)
                    or any(a['id'] in changed for n in notes for a in n['atoms'])
                    or any(r['lo']<hi and lo<r['hi'] for r in [*registry,*exclusions])):continue
            c=[min(n['bbox'][0] for n in notes),min(n['bbox'][1] for n in notes),
               max(n['bbox'][2] for n in notes),max(n['bbox'][3] for n in notes)]
            extent=max(b[2],c[2])-min(b[0],c[0])
            widths=[(b[2]-b[0])/extent,(c[2]-c[0])/extent]
            if not .07<=widths[1]<=.22:continue
            rows.append(dict(method='native_marginal_glossary',lo=lo,hi=hi,main_range=[main['lo'],main['hi']],
                note_ranges=[[n['lo'],n['hi']] for n in notes],note_colors=[n['uniform_ink'] for n in notes],
                widths=widths,main_left=b[0]<c[0],
                page=main['page']+1,main_bbox=b,note_bbox=c,targets=[]))
    return rows


def native_side_notes(index, tables=(), registry=(), exclusions=(), footnotes=(), changed=()):
    """Recover a small-font side note even if it was called a footnote.

    A matched callout always wins. Notes need complete native ownership and a
    live smaller font outside their immediately preceding main column.
    """
    from pathlib import Path
    import pymupdf
    from book_pipeline.ingestion.native import spans_for
    from book_pipeline.ingestion.panels import intersection_fraction
    source=index.source.get('source_pdf')
    if not source or not Path(source).is_file():return [],set()
    # Preserve the established ownership of colored references and multi-term
    # glossaries; the new recovery only handles notes those paths leave over.
    supported=marginal_glossaries(index,registry,exclusions,changed)
    supported+=marginal_references(index,registry,[*exclusions,*supported],changed)
    exclusions=[*exclusions,*supported]
    groups=native_groups(index);result=[];relocated=set()
    with pymupdf.open(source) as pdf:
        for main,note in zip(groups,groups[1:]):
            table=next((r for r in tables if r['hi']==main['hi']),None)
            b,c=table['bbox'] if table else main['bbox'],note['bbox']
            # Short paragraphs and inset tables still occupy the established
            # main column, rather than shrinking to their last glyph.
            columns=[g['bbox'] for g in groups if g['page']==main['page']
                and .6<g['bbox'][2]-g['bbox'][0]<.93
                and g['bbox'][0]<=b[0]+.003 and b[2]<=g['bbox'][2]+.003
                and all(a['role'] in {'text','paragraph'} for a in g['atoms'])]
            if columns:
                column=min(columns,key=lambda x:x[2]-x[0])
                b=[column[0],b[1],column[2],b[3]]
            text=index.text(note['lo'],note['hi']).strip()
            claimed=[n for n in footnotes if n['lo']<note['hi'] and note['lo']<n['hi']]
            short_glossary=(not claimed and len(text)<20
                and re.fullmatch(r'[A-Za-z][A-Za-z -]{3,19}',text)
                and len(re.findall(r'(?<![A-Za-z])'+re.escape(text)+r'(?![A-Za-z])',
                                   index.text(main['lo'],main['hi'])))==1)
            if (main['hi']!=note['lo'] or main['page']!=note['page'] or not 0<=note['page']<len(pdf)
                    or not 4<=len(text)<=180 or not .035<=c[2]-c[0]<=.19
                    or c[3]-c[1]>.12 or b[2]-b[0]<.3
                    or not (.008<=b[0]-c[2]<=.2 or .008<=c[0]-b[2]<=.2)
                    or not b[1]-.01<=c[1]<=b[3]+.01
                    or (not claimed and len(text)<20 and not short_glossary)
                    or any(n.get('marker') or n.get('status')=='matched' for n in claimed)
                    or any(a['id'] in changed or a['kind'] not in {'text','math'}
                        or a['role'] not in {'text','paragraph','page_footnote'} for a in note['atoms'])
                    or (not table and any(a['kind'] not in {'text','math'}
                        or a['role'] not in {'text','paragraph'} for a in main['atoms']))):continue
            lo=table['lo'] if table else main['lo'];hi=note['hi']
            if any(r['lo']<hi and lo<r['hi'] for r in [*registry,*exclusions]):continue
            spans=spans_for(pdf[note['page']])
            # A model box may end at the shorter final line. Recover the live
            # line width only if every resulting glyph matches the whole note.
            observed=[s for s in spans if s['text'].strip()
                and intersection_fraction(s['bbox'],[c[0]-.002,c[1]-.002,c[2]+.06,c[3]+.002])>.98]
            body=[s for s in spans if s['text'].strip() and intersection_fraction(s['bbox'],b)>.98]
            if not observed or not body:continue
            size=sorted(s['size'] for s in body)[len(body)//2]
            if any(s['size']>size*(.82 if short_glossary else .93) for s in observed):continue
            pattern=''.join('('+re.escape(''.join(a['content'].split()))+')' if a['kind']=='text' else '.+?'
                            for a in note['atoms'])
            live=''.join(''.join(s['text'].split()) for s in observed)
            first=re.fullmatch(pattern,live);last=re.fullmatch(pattern.replace('.+?','.+'),live)
            if not first or not last or first.regs!=last.regs:continue
            c=[min(s['bbox'][0] for s in observed),min(s['bbox'][1] for s in observed),
               max(s['bbox'][2] for s in observed),max(s['bbox'][3] for s in observed)]
            if (c[2]-c[0]>.19 or not (.008<=b[0]-c[2]<=.2 or .008<=c[0]-b[2]<=.2)
                    or any(a['group']!=note['group'] and a['page']==note['page']
                           and intersection_fraction(a.get('bbox'),c)>.05 for a in index.atoms)):continue
            extent=max(b[2],c[2])-min(b[0],c[0])
            widths=[(b[2]-b[0])/extent,(c[2]-c[0])/extent]
            if not .07<=widths[1]<=.25:continue
            result.append(dict(method='native_small_font_side_note',lo=lo,hi=hi,
                main_range=[main['lo'],main['hi']],table=table,
                relocated_footnotes=[n['id'] for n in claimed],
                note_ranges=[[note['lo'],note['hi']]],note_colors=[None],
                widths=widths,main_left=b[0]<c[0],page=note['page']+1,
                note_top_fraction=max(0,min(.8,(c[1]-b[1])/max(.001,b[3]-b[1]))),
                main_bbox=b,note_bbox=c,targets=[]))
            relocated.update(n['id'] for n in claimed)
    return result,relocated
