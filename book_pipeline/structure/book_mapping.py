"""Helpers used by the current PDF Reforge workflow."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import re
from book_pipeline.config import fingerprint, read
from book_pipeline.ingestion.math_syntax import formula_tags
from book_pipeline.structure.section_review import render_parts

def occurrences(text,quote):
    return [m.start() for m in re.finditer(re.escape(quote),text)] if quote else []


def raw_tag_ranges(text):
    ranges=[]
    for m in re.finditer(r'\\tag\*?\s*\{',text):
        depth=1;end=m.end()
        while end<len(text) and depth:
            depth+=int(text[end]=='{')-int(text[end]=='}');end+=1
        ranges.append([m.start(),end])
    return ranges


def source_pieces(index,unit,corrections):
    """Produce exact edited text plus a reversible piecewise source map."""
    text=[];pieces=[];offset=0
    for part in unit['parts']:
        atom=index.atoms[index.ids[part['source_atom']]];base=index.prefix[index.ids[atom['id']]]
        lo,hi=base+part['start_offset'],base+part['end_offset']
        edits=sorted((c for c in corrections if lo<=c['lo']<c['hi']<=hi),key=lambda c:c['lo'])
        cursor=lo
        def append(value,a,b,changed=False):
            nonlocal offset
            if not value:return
            text.append(value);pieces.append({'lo':offset,'hi':offset+len(value),'source_lo':a,'source_hi':b,
                'atom':atom['id'],'source_start_offset':a-base,'source_end_offset':b-base,'corrected':changed,
                'page':atom['page']+1})
            offset+=len(value)
        for c in edits:
            append(atom['content'][cursor-base:c['lo']-base],cursor,c['lo'])
            append(c['new'],c['lo'],c['hi'],True);cursor=c['hi']
        append(atom['content'][cursor-base:hi-base],cursor,hi)
        assert ''.join(text)[offset-len(part['text']):offset]==part['text'] if part['text'] else True
    return ''.join(text),pieces


def project_source(pieces,lo,hi):
    ranges=[]
    for p in pieces:
        a,b=max(lo,p['source_lo']),min(hi,p['source_hi'])
        if a>=b:continue
        if p['corrected']:ranges.append([p['lo'],p['hi']])
        else:ranges.append([p['lo']+a-p['source_lo'],p['lo']+b-p['source_lo']])
    return ranges


def source_for_quote(pieces,lo,hi):
    values=[]
    for p in pieces:
        a,b=max(lo,p['lo']),min(hi,p['hi'])
        if a>=b:continue
        if p['corrected']:
            x,y=p['source_start_offset'],p['source_end_offset']
        else:
            x,y=p['source_start_offset']+a-p['lo'],p['source_start_offset']+b-p['lo']
        item={'atom':p['atom'],'start_offset':x,'end_offset':y,'corrected':p['corrected']}
        if p.get('page') is not None:item['page']=p['page']
        values.append(item)
    return values


def build_book(index,plan,report,directory):
    """Registry includes every heading and every retained figure/table body."""
    directory=Path(directory)
    rows=report['sections'];headings=[r['section']['heading'] for r in rows]
    by_heading={h['toc_id']:h for h in headings}
    def context(h):
        chain=[];p=h
        while p:
            chain.append(p);p=by_heading.get(p.get('toc_parent_id'))
        chain.reverse()
        chapter=next((x for x in chain if x.get('toc_category')=='body' and x['heading_level']==1),None)
        section=next((x for x in reversed(chain) if x['kind']=='section'),None)
        return {'chapter':chapter['toc_id'] if chapter else None,'chapter_number':chapter['number'] if chapter else '',
                'section':section['toc_id'] if section else None,'section_number':section['number'] if section else '',
                'heading':h['toc_id'],'heading_title':h['title']}
    atlas=[];units={};registry=[]
    for row in rows:
        sid=row['section']['id'];h=row['section']['heading'];scope=context(h)
        processed=read(directory/'sections'/sid/'processed.json');units[sid]=processed
        for unit in processed['units']:
            text,pieces=source_pieces(index,unit,row['resolved']['c'])
            atlas.append({'b':unit['blocks'][0],'source_blocks':unit['blocks'],'section':sid,'page':unit['page'],
                          'text':text,'pieces':pieces,'parts':unit['parts'],'context':scope,'declarations':[]})
        lo,hi=index.span(h)
        registry.append({'id':'H'+sid.removeprefix('toc-'),'kind':'heading','heading_kind':h['kind'],
                         'number':h['number'],'name':h['title'],'label':index.text(lo,hi),'lo':lo,'hi':hi,
                         'page':index.page_for(lo)+1,'context':scope,'body_preview':'','origin':'matched_toc_heading'})
    target_count=0
    for row in rows:
        sid=row['section']['id'];scope=context(row['section']['heading'])
        for module in units[sid]['modules']:
            if module['type']=='formula':continue # Index every tag below, including nested equations.
            target_count+=1;body=module['body_resolved'] or module
            registry.append({'id':f'O{target_count:05d}','kind':module['type'],'number':module['number'],
                'name':module['name'],'label':module['label'],'lo':module['lo'],'hi':module['hi'],
                'declaration_lo':module['lo'],'declaration_hi':max(module.get('formal_label_end',module['lo']),body['lo']) if module['body_resolved'] else module['hi'],
                **{k:module[k] for k in ('name_span','name_parts','name_source','name_method') if k in module},
                **({'selected_ranges':module['selected_ranges']} if 'selected_ranges' in module else {}),
                'page':index.page_for(module['lo'])+1,'context':scope,
                'body_preview':''.join(p['text'] for p in module['content_parts'])[:320],'origin':'deepseek_section_module'})
    # Already-corrected formulas supply displayed numbers; each source formula is unique.
    equation_names={}
    for processed in units.values():
        for module in processed['modules']:
            if module['type']!='formula' or not module.get('name'):continue
            tags=[(part['source_atom'],tag['number']) for part in module['content_parts']
                  if part['kind'] in {'math','display_math'} for tag in formula_tags(part['text'])]
            if len(tags)==1 and (not module['number'] or module['number']==tags[0][1]):
                equation_names[tags[0]]=module
    formula_count=0;seen=set()
    for block in atlas:
        for part in block['parts']:
            if part['kind'] not in {'math','display_math'}:continue
            atom_id=part['source_atom'];base=index.prefix[index.ids[atom_id]]
            for tag in formula_tags(part['text']):
                key=(atom_id,tag['number'])
                if key in seen:continue
                seen.add(key);formula_count+=1
                named=equation_names.get(key,{})
                registry.append({'id':f'E{formula_count:05d}','kind':'equation','number':tag['number'],'name':named.get('name',''),
                    **{k:named[k] for k in ('name_span','name_parts','name_source','name_method') if k in named},
                    'label':'('+tag['number']+')','lo':base+part['start_offset'],'hi':base+part['end_offset'],
                    'page':block['page'],'context':block['context'],'body_preview':part['text'][:400],
                    'source_atom':atom_id,'source_tag':tag['tex'],'origin':'explicit_latex_tag'})
                # Locate the actual tag in edited text without treating its number as a reference.
                matches=occurrences(block['text'],tag['tex'])
                if len(matches)==1:block['declarations'].append([matches[0],matches[0]+len(tag['tex'])])
    # MinerU sometimes puts a displayed equation's (19) in a text atom rather
    # than a LaTeX tag. Preserve explicitly extracted formula objects as well.
    for sid,processed in units.items():
        for module in processed['modules']:
            if module['type']!='formula':continue
            if any(t['kind']=='equation' and
                   module['lo']<=t['lo']<t['hi']<=module['hi'] for t in registry):continue
            formula_count+=1;scope=context(by_heading[sid])
            original=index.text(module['lo'],module['hi']);positions=occurrences(original,module['label'])
            target={'id':f'E{formula_count:05d}','kind':'equation','number':module['number'],'name':module.get('name',''),
                    **{k:module[k] for k in ('name_span','name_parts','name_source','name_method') if k in module},
                    'label':module['label'],'lo':module['lo'],'hi':module['hi'],
                    'page':index.page_for(module['lo'])+1,'context':scope,
                    'body_preview':''.join(p['text'] for p in module['content_parts'])[:400],
                    'origin':'deepseek_explicit_numbered_formula'}
            if len(positions)==1:
                target['declaration_lo']=module['lo']+positions[0]
                target['declaration_hi']=target['declaration_lo']+len(module['label'])
            registry.append(target)
    # Captions belong to their original MinerU group. Keep anonymous images as targets too.
    assets=defaultdict(list);caption_parts=defaultdict(list);part_blocks={}
    for block in atlas:
        for part in block['parts']:
            a=index.atoms[index.ids[part['source_atom']]];group=a['group']
            part_blocks[part['source_atom']]=block
            if part['kind'] in {'image','table'}:assets[(group,part['kind'])].append(part)
            if 'caption' in a['role']:caption_parts[group].append(part)
    asset_count=Counter()
    for (group,kind),parts in assets.items():
        block=part_blocks[parts[0]['source_atom']];target_kind='figure' if kind=='image' else 'table'
        asset_count[target_kind]+=1;label='图' if kind=='image' else '表'
        caption=''.join(p['text'] for p in caption_parts[group])
        match=re.search(re.escape(label)+r'\s*(\d+(?:\s*[.．]\s*\d+)*)',caption)
        a=index.atoms[index.ids[parts[0]['source_atom']]]
        lo=index.prefix[index.ids[a['id']]];hi=lo+a['length']
        target={'id':('F' if kind=='image' else 'T')+f'{asset_count[target_kind]:05d}','kind':target_kind,
                'number':re.sub(r'\s','',match[1]).replace('．','.') if match else '',
                'name':caption,'label':match[0] if match else '', 'lo':lo,'hi':hi,'page':block['page'],
                'context':block['context'],'body_preview':caption or (parts[0]['text'][:320] if kind=='table' else ''),
                'source_group':group,'source_atom':a['id'],'asset':a.get('asset'),'origin':'mineru_'+target_kind}
        registry.append(target)
        if match:
            for cb in atlas:
                if any(p['source_atom'] in {x['source_atom'] for x in caption_parts[group]} for p in cb['parts']):
                    hits=occurrences(cb['text'],match[0])
                    if len(hits)==1:cb['declarations'].append([hits[0],hits[0]+len(match[0])])
    # A standalone caption can be a separate ordinary text group. Associate
    # only a unique page asset with a unique bare label; never guess adjacency.
    for kind,label in (('figure','图'),('table','表')):
        pages={t['page'] for t in registry if t['kind']==kind and not t['number']}
        for page in pages:
            page_targets=[t for t in registry if t['kind']==kind and t['page']==page]
            if len(page_targets)!=1:continue
            captions=[]
            for cb in atlas:
                if cb['page']!=page:continue
                match=re.fullmatch(r'\s*'+label+r'\s*(\d+(?:\s*[.．]\s*\d+)*)\s*',cb['text'])
                if match:captions.append((cb,match))
            if len(captions)!=1:continue
            cb,match=captions[0];t=page_targets[0]
            t.update(number=re.sub(r'\s','',match[1]).replace('．','.'),name=cb['text'],label=cb['text'].strip(),
                     caption_block=cb['b'],caption_origin='unique_standalone_label_on_same_page')
            cb['declarations'].append([0,len(cb['text'])])
    for target in registry:
        for block in atlas:
            if target['kind']!='heading' and 'block' not in target:
                for p in block['parts']:
                    base=index.prefix[index.ids[p['source_atom']]]
                    if base+p['start_offset']<=target['lo']<base+p['end_offset']:
                        target['block']=block['b'];break
            if 'declaration_lo' in target:
                block['declarations'].extend(project_source(block['pieces'],target['declaration_lo'],target['declaration_hi']))
    for block in atlas:
        block['declarations'].extend(raw_tag_ranges(block['text']))
        block['declarations']=[list(x) for x in sorted(set(tuple(d) for d in block['declarations']))]
    return {'source':report['source'],'heading_sha256':plan['heading_sha256'],
            'processed_sha256':fingerprint([r['resolved'] for r in rows]),'blocks':atlas,'targets':registry,
            'sections':[{'id':r['section']['id'],'heading':r['section']['heading'],'context':context(r['section']['heading'])} for r in rows]}


def write_processed_book(output,book,processing,processing_directory):
    """Export existing edited content and stable anchors without any AI calls."""
    output=Path(output);targets={t['id']:t for t in book['targets']}
    chunks=[]
    anchors=defaultdict(list)
    for t in book['targets']:
        if t.get('block'):anchors[t['block']].append(t['id'])
    for row in processing['sections']:
        sid=row['section']['id'];h=row['section']['heading'];processed=read(Path(processing_directory)/'sections'/sid/'processed.json')
        tid='H'+sid.removeprefix('toc-')
        chunks += [f'<a id="{tid}"></a>', '#'*h['heading_level']+' '+targets[tid]['label'],'']
        for unit in processed['units']:
            bid=unit['blocks'][0]
            chunks += [f'<a id="{bid}"></a>']+[f'<a id="{tid}"></a>' for tid in anchors[bid]]
            chunks += [render_parts(unit['parts']),'']
    (output/'processed_book.md').write_text('\n'.join(chunks),encoding='utf-8')
