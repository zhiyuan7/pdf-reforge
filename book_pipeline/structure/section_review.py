"""Helpers used by the current PDF Reforge workflow."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import re
from book_pipeline.config import fingerprint, save
from book_pipeline.ingestion.math_syntax import formula_tags

METHOD = 'section_text_review_v1'


MODULE_NUMBER = r"\d+(?:\.\d+)*(?:_\d+)?°?[a-z]?(?:['′’]+)?"


def whitespace_occurrences(text,quote):
    """Ignore whitespace only; retain every mathematical delimiter and token."""
    positions=[n for n,c in enumerate(text) if not c.isspace()]
    compact=''.join(text[n] for n in positions);needle=''.join(quote.split())
    if not needle:return []
    return [(positions[m.start()],positions[m.end()-1]+1) for m in re.finditer(re.escape(needle),compact)]


def formal_view(text):
    """Normalize only presentation of labels, retaining exact source positions."""
    chars=[(c,n,n+1) for n,c in enumerate(text)]
    def replace(pattern,kind):
        nonlocal chars
        value=''.join(c[0] for c in chars);out=[];cursor=0
        for m in re.finditer(pattern,value):
            out.extend(chars[cursor:m.start()])
            if kind in {'inner','subscript'}:
                if kind=='subscript':out.append(('_',chars[m.start()][1],chars[m.start()][2]))
                out.extend(chars[m.start(1):m.end(1)])
                if kind=='subscript':out[-1]=(out[-1][0],out[-1][1],chars[m.end()-1][2])
            else:out.append((kind,chars[m.start()][1],chars[m.end()-1][2]))
            cursor=m.end()
        out.extend(chars[cursor:]);chars=out
    replace(r'\\(?:text|mathrm|operatorname)\s*\{\s*([^{}]*?)\s*\}','inner')
    replace(r'\^\s*\{\s*\\circ\s*\}|\^\s*\\circ','°')
    replace(r'_\s*\{\s*(\d+)\s*\}', 'subscript')
    out=[]
    for c,a,b in chars:
        if c.isspace():continue
        if c in '₀₁₂₃₄₅₆₇₈₉':
            out.extend([('_',a,b),('0123456789'['₀₁₂₃₄₅₆₇₈₉'.index(c)],a,b)])
        else:out.append((c.replace('．','.').replace('º','°').replace('′',"'").replace('’',"'"),a,b))
    return ''.join(c[0] for c in out),[(a,b) for _,a,b in out]


def formal_text(text):
    return formal_view(text)[0]


def split_source(index, outline):
    """Cut every group at all heading boundaries; exactly partition all atoms."""
    headings = sorted(outline['headings'], key=lambda h: index.position(h['start']))
    spans = [(index.span(h), h) for h in headings]
    previous = -1
    for (lo, hi), _ in spans:
        if lo < previous:
            raise ValueError('Confirmed headings overlap')
        previous = hi
    cuts = {p for (lo, hi), _ in spans for p in (lo, hi)}
    blocks, formula_ids, split_groups = [], {}, []
    for group, indices in index.groups.items():
        lo, hi = index.prefix[indices[0]], index.prefix[indices[-1]+1]
        boundaries = sorted({lo, hi, *(p for p in cuts if lo < p < hi)})
        fragments = []
        for a, b in zip(boundaries, boundaries[1:]):
            index.position(index.address(a))
            index.position(index.address(b, end=True))
            parts = []
            for atom, start, end in index.fragments(a, b):
                item = {'atom': atom['id'], 'offset': start, 'end_offset': end,
                        'lo': index.prefix[index.ids[atom['id']]]+start,
                        'hi': index.prefix[index.ids[atom['id']]]+end,
                        'kind': atom['kind'], 'text': atom['content'][start:end]}
                if atom['kind'] in {'math', 'display_math'}:
                    fid = formula_ids.setdefault(atom['id'], f'F{len(formula_ids)+1:05d}')
                    item['f'] = fid
                parts.append(item)
            heading = next((h for (x, y), h in spans if x <= a and b <= y), None)
            block = {'b': f'B{len(blocks)+1:05d}', 'group': group,
                     'page': index.atoms[indices[0]]['page']+1, 'lo': a, 'hi': b,
                     'start': index.address(a), 'end': index.address(b, end=True),
                     'parts': parts, 'text': index.text(a, b),
                     'heading': heading['toc_id'] if heading else None,
                     'mineru_type': index.atoms[indices[0]]['block_type'],
                     'source_type': 'heading' if heading else index.atoms[indices[0]]['block_type']}
            blocks.append(block)
            fragments.append(block['b'])
        if len(fragments) > 1:
            split_groups.append({'group': group, 'blocks': fragments,
                                 'headings': [b['heading'] for b in blocks[-len(fragments):] if b['heading']]})
    excluded_pages = {e['toc_page'] for e in outline.get('toc_entries', [])}
    excluded_pages.update(outline.get('metadata_config', {}).get('exclude_pages', []))
    sections = []
    for n, h in enumerate(headings):
        start, body_start = index.span(h)
        following = headings[n+1] if n+1 < len(headings) else None
        end = index.position(following['start']) if following else index.prefix[-1]
        owned = [b for b in blocks if body_start <= b['lo'] < end and b['page'] not in excluded_pages]
        # A parent followed immediately by its child may have no direct content.
        sections.append({'id': h['toc_id'], 'heading': h, 'next_heading': following,
                         'lo': start, 'body_lo': body_start, 'hi': end,
                         'pages': [index.page_for(start)+1,
                                   index.page_for(end)+1 if following else index.source['source_page_count']],
                         'blocks': [b['b'] for b in owned],
                         'text_characters': sum(len(b['text']) for b in owned),
                         'nonempty': any(b['text'].strip() for b in owned),
                         'exercise_context': h.get('exercise_context', False),
                         'exercise_root': h.get('exercise_root')})
    stamp = {k: index.source[k] for k in ('source_pdf', 'input_sha256', 'mineru_sha256', 'source_page_count')}
    return {'method': METHOD, 'source': stamp, 'metadata':deepcopy(outline.get('metadata',{})), 'heading_sha256': fingerprint(headings),
            'excluded_pages': sorted(excluded_pages), 'blocks': blocks,
            'split_groups': split_groups, 'sections': sections,
            'toc_entries': deepcopy(outline.get('toc_entries',[])),
            'exercise_heading_classification': deepcopy(outline.get('exercise_heading_classification',{})),
            'ignored_bold_candidates': len(outline.get('bold_spans', []))}


def ancestors(heading, headings):
    by_id = {h['toc_id']: h for h in headings}
    result, parent = set(), heading.get('toc_parent_id')
    while parent:
        if parent in result or parent not in by_id:
            raise ValueError('Invalid heading ancestry')
        result.add(parent)
        parent = by_id[parent].get('toc_parent_id')
    return result


def materialize(index, section, blocks, edits):
    """A separate processed version with original-address provenance per unit."""
    units = []
    for block in blocks:
        if block['b'] not in section['blocks']:
            continue
        deletions = [d for d in edits['d'] if d['lo'] < block['hi'] and d['hi'] > block['lo']]
        bounds = sorted({block['lo'], block['hi'], *(max(block['lo'], d['lo']) for d in deletions),
                         *(min(block['hi'], d['hi']) for d in deletions)})
        retained = [(lo, hi) for lo, hi in zip(bounds, bounds[1:])
                    if not any(d['lo'] <= lo and hi <= d['hi'] for d in deletions)]
        parts = []
        for lo, hi in retained:
            for atom, start, end in index.fragments(lo, hi):
                base = index.prefix[index.ids[atom['id']]]
                value = atom['content'][start:end]
                for c in sorted((c for c in edits['c'] if base+start <= c['lo'] < c['hi'] <= base+end),
                                key=lambda c: -c['lo']):
                    a, b = c['lo']-base-start, c['hi']-base-start
                    value = value[:a]+c['new']+value[b:]
                parts.append({'kind': atom['kind'], 'text': value, 'source_atom': atom['id'],
                              'start_offset': start, 'end_offset': end})
        if parts:
            units.append({'blocks': [block['b']], 'page': block['page'], 'parts': parts,
                          'content_role': block.get('content_role', 'body')})
    for merge in edits['m']:
        positions = [n for n, unit in enumerate(units) if unit['blocks'][0] in merge['blocks']]
        intervening = [units[n] for n in range(positions[0], positions[-1]+1) if n not in positions] if positions else []
        if not positions or any(u.get('content_role') != 'page_footnote' for u in intervening):
            raise ValueError('Validated merge is not contiguous after deletions')
        chosen = [units[n] for n in positions]
        joined = {'blocks': merge['blocks'], 'page': chosen[0]['page'], 'content_role': 'body',
                  'parts': [p for unit in chosen for p in unit['parts']]}
        units[positions[0]:positions[-1]+1] = [joined, *intervening]
    modules=[]
    for item in edits['s']:
        target=item['body_resolved'] or item
        parts=[]
        lo,hi=target['lo'],target['hi']
        selected=item.get('selected_ranges',[{'lo':lo,'hi':hi}])
        cuts=sorted({lo,hi,*(max(lo,d['lo']) for d in edits['d'] if d['lo']<hi and d['hi']>lo),
                          *(min(hi,d['hi']) for d in edits['d'] if d['lo']<hi and d['hi']>lo),
                          *(max(lo,x['lo']) for x in selected if x['lo']<hi and x['hi']>lo),
                          *(min(hi,x['hi']) for x in selected if x['lo']<hi and x['hi']>lo)})
        for a,b in zip(cuts,cuts[1:]):
            if any(d['lo']<=a and b<=d['hi'] for d in edits['d']):continue
            if not any(x['lo']<=a and b<=x['hi'] for x in selected):continue
            for atom,start,end in index.fragments(a,b):
                base=index.prefix[index.ids[atom['id']]]
                text=atom['content'][start:end]
                for c in sorted((c for c in edits['c'] if base+start<=c['lo']<c['hi']<=base+end),key=lambda c:-c['lo']):
                    x,y=c['lo']-base-start,c['hi']-base-start
                    text=text[:x]+c['new']+text[y:]
                parts.append({'kind':atom['kind'],'text':text,'source_atom':atom['id'],
                              'start_offset':start,'end_offset':end})
        modules.append({**item,'content_parts':parts})
    return {'section': section['id'], 'heading': section['heading'],
            'units': units, 'modules': modules, 'corrections': edits['c']}


def numbered_formula_registry(plan, rows):
    """Index literal tags even inside another module, without semantic guessing."""
    blocks={b['b']:b for b in plan['blocks']};result=[]
    for row in rows:
        for bid in row['section']['blocks']:
            block=blocks[bid]
            if block.get('content_role') == 'page_footnote':continue
            for part in block['parts']:
                if part['kind'] not in {'math','display_math'}:continue
                if any(part['lo']<d['hi'] and part['hi']>d['lo'] for d in row['resolved']['d']):continue
                for tag in formula_tags(part['text']):
                    parents=[s['label'] for s in row['resolved']['s'] if s['type']!='formula'
                             and s['lo']<=part['lo'] and part['hi']<=s['hi']
                             and any(x['lo']<=part['lo'] and part['hi']<=x['hi'] for x in s.get('selected_ranges',[s]))]
                    independent=any(s['type']=='formula' and s['lo']<=part['lo'] and part['hi']<=s['hi'] for s in row['resolved']['s'])
                    result.append({'section':row['section']['id'],'b':bid,'f':part['f'],
                        'number':tag['number'],'source_tag':tag['tex'],'source_atom':part['atom'],
                        'lo':part['lo'],'hi':part['hi'],'parent_labels':parents,
                        'model_separate_formula_module':independent,'method':'literal_latex_tag_index'})
    return result


def render_parts(parts):
    chunks = []
    for p in parts:
        text, kind = p['text'], p['kind']
        if kind == 'math':
            chunks.append('$'+text+'$')
        elif kind == 'display_math':
            chunks.append('\n$$\n'+text+'\n$$\n')
        elif kind in {'image','table'}:
            chunks.append(f'〔保留原{kind}：{p.get("source_atom",p.get("atom"))}〕')
        else:
            chunks.append(text)
    return ''.join(chunks)


def verify_partition(index, plan):
    previous = 0
    for b in plan['blocks']:
        if b['lo'] != previous or index.text(b['lo'], b['hi']) != b['text']:
            raise ValueError('Block splitting lost, duplicated or rewrote source')
        previous = b['hi']
    if previous != index.prefix[-1]:
        raise ValueError('Block splitting does not cover all source atoms')
    return {'source_partition':'pass', 'source_atoms':len(index.atoms), 'split_blocks':len(plan['blocks']),
            'original_groups':len(index.groups), 'split_groups':len(plan['split_groups']),
            'headings':len(plan['sections']), 'nonempty_sections':sum(s['nonempty'] for s in plan['sections'])}


def save_processed_section(directory,section,row,processed):
    sub=Path(directory)/'sections'/section['id']
    save(sub/'result.json',row);save(sub/'processed.json',processed)
    text='# '+section['heading']['title']+'\n\n'
    text+='\n\n'.join('<!-- '+','.join(u['blocks'])+' -->\n'+render_parts(u['parts']) for u in processed['units'])
    (sub/'processed.md').write_text(text+'\n',encoding='utf-8')
    module_text='# '+section['heading']['title']+'：提取的特殊模块\n\n'
    for module in processed['modules']:
        module_text+='## '+module['label']+(' '+module['name'] if module['name'] and module['name'] not in module['label'] else '')+'\n\n'
        module_text+=f"来源：{module['span']['from']['b']} → {module['span']['to']['b']}\n\n"
        module_text+=render_parts(module['content_parts'])+'\n\n'
    if not processed['modules']:module_text+='本部分未提取到特殊模块。\n'
    (sub/'modules.md').write_text(module_text,encoding='utf-8')
