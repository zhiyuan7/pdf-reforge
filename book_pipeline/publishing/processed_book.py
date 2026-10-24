"""Render saved stages, with image-based recovery of located formula errors."""
from bisect import bisect_right
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
import re
import shutil

from book_pipeline.config import digest,fingerprint,read,save
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.structure.book_mapping import build_book, project_source
from book_pipeline.structure.headings import source_heading_end
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.formula_repair import compile_with_formula_recovery
from book_pipeline.publishing.environment import doctor
from book_pipeline.publishing.local_qa import LocalPDFReviewer
from book_pipeline.publishing.navigation import NavigationBuilder
from book_pipeline.publishing.validation import audit_pdf

METHOD='saved_three_stage_book_render_v1'


class EditedBook:
    """Materialize saved units; map original intervals into the edited stream."""
    def __init__(self,original,book,processing):
        self.original=original;self.book=book;self.processing=Path(processing)
        self.atoms=[];self.segments=[];self.offset=0;self.block_parts={};self.heading_spans={}
        by_section=defaultdict(list)
        for b in book['blocks']:by_section[b['section']].append(b)
        for section in book['sections']:
            h=section['heading'];lo,hi=original.span(h);start=self.offset
            hi=source_heading_end(original,lo,hi)
            for atom,a,z in original.fragments(lo,hi):
                self.append(atom,a,z,atom['content'][a:z],section['id'],None,None)
            self.heading_spans[section['id']]=(start,self.offset)
            for block in by_section[section['id']]:
                char=0;parts=[]
                # Old saved books may still own the title suffix in a body
                # unit. Move it once, using the existing correction/source map.
                cut=max((z for a,z in project_source(block['pieces'],lo,hi)),default=0)
                for part in block['parts']:
                    atom=original.atoms[original.ids[part['source_atom']]]
                    skip=min(len(part['text']),max(0,cut-char))
                    if skip<len(part['text']) or (skip==0 and atom['kind']!='text'):
                        a=part['start_offset']
                        if skip:a=max(a,hi-original.prefix[original.ids[atom['id']]])
                        new=self.append(atom,a,part['end_offset'],part['text'][skip:],block['b'],block,char+skip)
                        if new:parts.append(dict(atom=new['id'],char_lo=char+skip,char_hi=char+len(part['text'])))
                    char+=len(part['text'])
                assert char==len(block['text'])
                self.block_parts[block['b']]=parts
        self.index=SourceIndex({**book['source'],'atoms':self.atoms})
        self.segments.sort(key=lambda s:(s['source_lo'],s['source_hi']))
        self.source_ends=[s['source_hi'] for s in self.segments]
        self.registry=[];self.bold=[];self.references=[];self.linked_hints=[];self.unlinked=[];self.notes=[]
        for t in book['targets']:
            if t['kind']=='heading':lo,hi=self.heading_spans[t['context']['heading']]
            else:
                source_hi=source_heading_end(original,t['lo'],t['hi']) if t['kind']=='subheading' else t['hi']
                lo,hi=self.project(t['lo'],source_hi)
            kind={'other':'remark'}.get(t['kind'],t['kind'])
            if t['kind']=='heading':
                h=next(s['heading'] for s in book['sections'] if s['id']==t['context']['heading'])
                kind=h['kind']
            item={**t,'kind':kind,'lo':lo,'hi':hi,'anchor':'book-'+t['id'],
                  'start':self.index.address(lo),'end':self.index.address(hi,end=True)}
            for field in ('question_ranges', 'answer_ranges'):
                if field in t:
                    item[field] = [dict(zip(('lo', 'hi'), self.project(r['lo'], r['hi']))) for r in t[field]]
            if t.get('caption_block'):
                parts=self.block_parts[t['caption_block']]
                first,last=self.index.ids[parts[0]['atom']],self.index.ids[parts[-1]['atom']]
                item['caption_span']=[self.index.prefix[first],self.index.prefix[last+1]]
            if t['kind']=='heading':item.update(heading_level=h['heading_level'],toc_category=h['toc_category'],toc_id=h['toc_id'])
            if t['kind'] in {'heading','subheading'}:
                item['label']=self.index.text(lo,hi)
                if t['kind']=='heading':item['name']=item['label']
            if t['kind'] not in {'heading','equation','figure','table','subheading'}:
                title_end=t.get('declaration_hi',t['lo'])
                if t['lo']<title_end<t['hi']:
                    _,end=self.project(t['lo'],title_end)
                    address=self.index.address(end)
                    try:self.index.position(address)
                    except ValueError:
                        self.notes.append(dict(target=t['id'],reason='label_inside_atomic_formula_kept_in_body'))
                    else:item['body_start']=address
            selected=t.get('selected_ranges')
            if selected:
                spans=[]
                for source in selected:
                    a,z=self.project(source['lo'],source['hi'])
                    if spans and a<=spans[-1][1]:spans[-1]=(spans[-1][0],max(z,spans[-1][1]))
                    else:spans.append((a,z))
                for n,(a,z) in enumerate(spans):
                    piece={**item,'lo':a,'hi':z,'start':self.index.address(a),'end':self.index.address(z,True)}
                    if n:
                        piece.update(id=t['id']+f'-segment-{n+1}',anchor=item['anchor']+f'-segment-{n+1}')
                    if n or ('body_start' in piece and not a<self.index.position(piece['body_start'])<z):
                        piece.pop('body_start',None)
                    self.registry.append(piece)
            else:self.registry.append(item)
        # Legacy formula suggestions do not create colored ranges. Keep native
        # math and the independently indexed equation anchors and references.
        self.ignored_formula_modules=[]
        for s in book['sections']:
            processed=read(self.processing/'sections'/s['id']/'processed.json')
            for module in processed['modules']:
                if module['type']!='formula':continue
                self.ignored_formula_modules.append(dict(section=s['id'],number=module['number'],
                    name=module.get('name',''),start=module['start'],end=module['end'],method=module.get('method','legacy')))
        self.registry.sort(key=lambda r:(r['lo'],-r['hi'],r['anchor']))
        self.footnotes = []
        retained_blocks = {b['b'] for b in book['blocks']}
        for note in book.get('footnotes', []):
            if note['block'] not in retained_blocks:
                continue
            lo, hi = self.project(note['lo'], note['hi'])
            item = {**note, 'lo':lo, 'hi':hi, 'anchor':'book-'+note['id'],
                    'reference_anchor':'ref-'+note['id'], 'source_lo':note['lo'], 'source_hi':note['hi']}
            reference = note.get('reference')
            if reference:
                a, z = self.project(reference['lo'], reference['hi'])
                atom = self.index.atoms[self.index.ids[self.index.address(z, True)['atom']]]
                item['insert_after'] = z if atom['kind']=='text' else self.index.prefix[self.index.ids[atom['id']]+1]
                item['reference'] = {**reference, 'edited_lo':a, 'edited_hi':z}
                marker = note['marker']
                item['marker_lo'], item['marker_hi'] = self.project(note['lo']+marker['start'], note['lo']+marker['end'])
            else:
                item['insert_after'] = lo
            self.footnotes.append(item)

    def append(self,atom,a,z,text,group,block,char):
        if not text and atom['kind']=='text':return None
        new={**atom,'id':f'edited-{len(self.atoms)+1:06d}','group':group,'content':text,'length':max(1,len(text)),
             'original_atom':atom['id'],'original_start_offset':a,'original_end_offset':z}
        if block and len(block['source_blocks'])>1 and atom['kind']=='text':new['role']='text'
        start=self.offset;self.offset+=new['length'];self.atoms.append(new)
        base=self.original.prefix[self.original.ids[atom['id']]]
        if block and atom['kind']=='text':
            for p in block['pieces']:
                x,y=max(char,p['lo']),min(char+len(text),p['hi'])
                if x>=y:continue
                if p['corrected']:slo,shi=p['source_lo'],p['source_hi']
                else:slo,shi=p['source_lo']+x-p['lo'],p['source_lo']+y-p['lo']
                self.segments.append(dict(source_lo=slo,source_hi=shi,lo=start+x-char,hi=start+y-char,corrected=p['corrected']))
        else:self.segments.append(dict(source_lo=base+a,source_hi=base+z,lo=start,hi=self.offset,corrected=len(text)!=z-a))
        return new

    def project(self,lo,hi):
        spans=[]
        for s in self.segments[bisect_right(self.source_ends,lo):]:
            if s['source_lo']>=hi:break
            a,z=max(lo,s['source_lo']),min(hi,s['source_hi'])
            if a>=z:continue
            spans.append((s['lo'],s['hi']) if s['corrected'] else
                         (s['lo']+a-s['source_lo'],s['lo']+z-s['source_lo']))
        if not spans:raise ValueError(f'Saved object has no retained content: {lo}:{hi}')
        return min(a for a,z in spans),max(z for a,z in spans)

    def map_hints(self,mapping):
        targets={t['id']:t for t in self.book['targets']}
        unique_by_block=defaultdict(list)
        for row in mapping['sections']:
            for h in row['hints']:
                if h['status']=='unique':unique_by_block[h['block']].append(h)
        seen=set()
        for row in mapping['sections']:
            for hint in row['hints']:
                if hint['status']!='unique':self.unlinked.append(hint);continue
                if any(other['candidate_ids']!=hint['candidate_ids'] and
                       other['start_offset']<hint['end_offset'] and other['end_offset']>hint['start_offset']
                       for other in unique_by_block[hint['block']]):
                    self.unlinked.append({**hint,'render_reason':'overlapping_reference_targets'});continue
                tid=hint['candidate_ids'][0]
                if targets[tid]['kind']=='example':raise ValueError('Example mapping is disabled')
                # Link text fragments of a confirmed phrase; adjoining inline
                # formula fragments remain native math rather than raw TeX text.
                pieces=[]
                for p in self.block_parts[hint['block']]:
                    a,z=max(hint['start_offset'],p['char_lo']),min(hint['end_offset'],p['char_hi'])
                    atom=self.atoms[self.index.ids[p['atom']]]
                    if a<z and atom['kind']=='text':pieces.append((atom,a-p['char_lo'],z-p['char_lo']))
                if not pieces:raise ValueError('A numbered reference has no clickable text')
                for atom,a,z in pieces:
                    signature=(atom['id'],a,z,tid)
                    if signature in seen:continue
                    seen.add(signature)
                    self.references.append(dict(atom=atom['id'],start_offset=a,end_offset=z,
                                                text=atom['content'][a:z],target='book-'+tid))
                self.linked_hints.append(hint)
        return self.references


def merge_footnote_continuations(original, notes):
    """Rejoin adjacent note fragments contained in one original note rectangle."""
    removed=set()
    ordered=sorted(notes,key=lambda n:n['source_lo'])
    for position,note in enumerate(ordered):
        if note['id'] in removed or not note.get('marker'):continue
        first=next(original.fragments(note['source_lo'],note['source_hi']))[0]
        box=first.get('bbox') or first.get('block_bbox')
        if not box:continue
        for child in ordered[position+1:]:
            if (child['page']!=note['page'] or child.get('marker') or
                    child['source_lo']!=note['source_hi'] or child['lo']!=note['hi']):break
            atom=next(original.fragments(child['source_lo'],child['source_hi']))[0]
            b=atom.get('bbox') or atom.get('block_bbox')
            if not b or not (box[0]-.005<=b[0] and b[2]<=box[2]+.005 and
                             box[1]-.005<=b[1] and b[3]<=box[3]+.005):break
            note.update(hi=child['hi'],source_hi=child['source_hi'],end=child['end'],
                        text=note['text']+child['text'])
            note.setdefault('continuation_note_ids',[]).append(child['id']);removed.add(child['id'])
    notes[:]=[n for n in notes if n['id'] not in removed]


def render_saved_book(original,processing,mapping_directory,metadata,config,output,progress=lambda *args:None):
    processing=Path(processing).resolve();mapping_directory=Path(mapping_directory).resolve();output=Path(output).resolve()
    inputs=[processing,mapping_directory,Path(original.source['mineru_dir']).resolve()]
    if any(output==p or output.is_relative_to(p) for p in inputs):raise ValueError('Render into a separate output directory')
    report=read(processing/'section_results.json');plan=read(processing/'split_source.json')
    mapping=read(mapping_directory/'mapping_results.json')
    if mapping.get('status')!='complete' or mapping.get('excluded_target_kinds')!=['example'] or mapping.get('match_policy')!='type_and_number_only':
        raise ValueError('Use the complete local numbered mapping with examples and names disabled')
    book=build_book(original,plan,report,processing)
    if any(book['source'][k]!=original.source[k] for k in ('input_sha256','mineru_sha256')):
        raise ValueError('Processed text source hashes differ from the original PDF/MinerU')
    if book!=read(mapping_directory/'edited_text_index.json') or fingerprint(book['targets'])!=mapping['targets_sha256']:
        raise ValueError('Mapping and current saved processed contents differ')
    before={str(p):digest(p) for p in [processing/'section_results.json',processing/'split_source.json',
                                      mapping_directory/'mapping_results.json',Path(original.source['mineru_dir'])/'middle_json.json']}
    tools=doctor(config,require_key=False)
    progress('读取已保存的三阶段结果',1,1)
    edited=EditedBook(original,book,processing);references=edited.map_hints(mapping)
    decisions=dict(objects=[],exclusions=[],corrections=[],joins=[],issues=[],bold_spans=edited.bold,
                   page_footnotes=edited.footnotes)
    settings={**metadata['heading_layout'],'standalone_levels':[1],'page_break_after_levels':[2]}
    native_config={**config,'float_assets':True,'start_page_levels':[2]}
    project=output/'latex';reports=output/'reports';reports.mkdir(parents=True,exist_ok=True)
    correction_path=output/'text_corrections.json'
    if correction_path.exists():
        correction=read(correction_path)
        if correction.get('source_sha256')!=fingerprint(edited.index.source):
            raise ValueError('Render text corrections do not match the edited source')
        for row in correction['items']:
            atom=edited.index.atoms[edited.index.ids[row['atom']]]
            lo,hi=row['start_offset'],row['end_offset']
            if (atom['kind'] not in {'text','math','display_math'} or not 0<=lo<hi<=atom['length'] or
                    atom['content'][lo:hi]!=row['expected'] or not isinstance(row['replacement'],str)):
                raise ValueError('Render text correction does not match its exact source interval')
            if atom['kind']!='text' and (atom['content'][max(0,lo-6):lo]!=r'\text{' or
                    atom['content'][hi:hi+1]!='}' or re.search(r'[{}\\$]',row['replacement'])):
                raise ValueError('Math text corrections must address a complete plain text label')
            decisions['corrections'].append(row)
        before[str(correction_path)]=digest(correction_path)
    footnote_path=output/'footnote_overrides.json'
    if footnote_path.exists():
        overrides=read(footnote_path)
        if overrides.get('source_sha256')!=fingerprint(edited.index.source):
            raise ValueError('Footnote overrides do not match the edited source')
        for override in overrides['items']:
            note=next(n for n in edited.footnotes if n['id']==override['id'])
            if override.get('marker'):
                marker=override['marker']
                if original.text(note['source_lo']+marker['start'],note['source_lo']+marker['end'])!=marker['text']:
                    raise ValueError('Reviewed footnote marker differs from the source')
                note['marker']=marker
            if override.get('reference'):
                ref=override['reference'];source_atom=original.atoms[original.ids[ref['atom']]]
                a,z=ref['start_offset'],ref['end_offset']
                if (source_atom['kind'] not in {'text','math'} or source_atom['page']+1!=note['page'] or
                        not 0<=a<z<=source_atom['length'] or source_atom['content'][a:z]!=ref['expected']):
                    raise ValueError('Reviewed footnote reference differs from the exact same-page source')
                base=original.prefix[original.ids[ref['atom']]]
                options=[dict(atom=ref['atom'],block=source_atom['group'],lo=base+a,hi=base+z,
                              start_offset=a,end_offset=z,insert_after=base+z,kind=source_atom['kind'],
                              text=ref['expected'],page=note['page'])]
            else:
                options=[c for c in note['candidates'] if c['atom']==override['atom'] and
                         c['start_offset']==override['start_offset']]
            if len(options)!=1 or not note.get('marker'):
                raise ValueError('A footnote override must select one existing source candidate')
            reference=options[0];a,z=edited.project(reference['lo'],reference['hi'])
            atom=edited.index.atoms[edited.index.ids[edited.index.address(z,True)['atom']]]
            note.update(status='matched',reason='reviewed_original_source_marker',
                        reference={**reference,'edited_lo':a,'edited_hi':z},
                        insert_after=z if atom['kind']=='text' else edited.index.prefix[edited.index.ids[atom['id']]+1])
            marker=note['marker']
            note['marker_lo'],note['marker_hi']=edited.project(note['source_lo']+marker['start'],note['source_lo']+marker['end'])
        before[str(footnote_path)]=digest(footnote_path)
    merge_footnote_continuations(original,edited.footnotes)
    save(reports/'input_sources.json',dict(source=book['source'],processing=str(processing),mapping=str(mapping_directory),
         processed_sha256=book['processed_sha256'],targets_sha256=mapping['targets_sha256'],input_files_sha256=before))
    save(reports/'edited_source.json',edited.index.source)
    save(reports/'render_registry.json',edited.registry)
    save(reports/'footnotes.json',edited.footnotes)
    save(reports/'footnote_formula_deletions.json',book.get('footnote_duplicate_formulas', []))
    save(reports/'render_references.json',references)
    save(reports/'unlinked_reference_hints.json',edited.unlinked)
    save(reports/'ignored_formula_modules.json',edited.ignored_formula_modules)
    def generate(replacements):
        decisions['formula_repairs'] = {k:v for k,v in replacements.items() if isinstance(v,str)}
        decisions['table_repairs'] = {k:v for k,v in replacements.items() if isinstance(v,dict)}
        coverage=write_project(edited.index,decisions,edited.registry,references,metadata['book'],native_config,project,heading_layout=settings)
        coverage['saved_module_notes']=edited.notes
        coverage['footnote_formula_deletions']=book.get('footnote_duplicate_formulas', [])
        save(reports/'coverage.json',coverage)
        progress('生成 LaTeX',1,1)
        return coverage
    if (reports/'compilation.json').exists():
        history=reports/'compile_history';history.mkdir(exist_ok=True)
        attempt=len(list(history.glob('attempt-*.json')))+1
        save(history/f'attempt-{attempt:03d}.json',read(reports/'compilation.json'))
        for filename in ('main.log','compile-output.txt'):
            if (project/filename).exists():shutil.copy2(project/filename,history/f'attempt-{attempt:03d}-{filename}')
    coverage,compilation,formula_repairs=compile_with_formula_recovery(
        edited.index,native_config,project,reports,tools,generate,progress)
    save(reports/'compilation.json',compilation)
    save(reports/'formula_repairs.json',formula_repairs)
    result=dict(method=METHOD,processing_method=report['method'],status='compiled' if compilation['success'] else 'compilation_failed',
        output=str(output),tex=str(project/'main.tex'),pdf=str(project/'main.pdf') if compilation['success'] else None,
        source=book['source'],usage=formula_repairs['usage'],
        formula_repairs=dict(status=formula_repairs['status'],repaired_atoms=formula_repairs['repaired_atoms'],
                             report=str(reports/'formula_repairs.json')),
        table_repairs=dict(status=formula_repairs['table_repairs']['status'],repaired_atoms=formula_repairs['table_repairs']['repaired_atoms'],
                           report=str(reports/'table_repair/repairs.json')),
        statistics=dict(headings=len(book['sections']),processed_blocks=len(book['blocks']),
            named_special_blocks=sum(bool(t['name']) for t in book['targets'] if t['kind'] not in {'heading','figure','table'}),
            styled_subheadings=sum(t['kind']=='subheading' for t in edited.registry),formula_modules=sum(t['kind']=='formula' for t in edited.registry),
            ignored_legacy_formula_modules=sum(not m['method'].startswith('page_content_keywords_v') for m in edited.ignored_formula_modules),
            native_formula_special_blocks=sum(m['method'].startswith('page_content_keywords_v') for m in edited.ignored_formula_modules),
            examples=sum(t['kind']=='example' for t in edited.registry),linked_reference_hints=len(edited.linked_hints),
            footnotes=len(edited.footnotes), matched_footnotes=sum(n['status']=='matched' for n in edited.footnotes),
            unmatched_footnotes=sum(n['status']=='unmatched' for n in edited.footnotes),
            duplicate_footnote_formula_blocks=len(book.get('footnote_duplicate_formulas', [])),
            ambiguous_or_unmatched_hints=len(edited.unlinked),asset_float_groups=sum(r['method']=='deferred_asset_float' for r in coverage['source_order_layout']),
            horizontal_image_rows=sum(r['method']=='horizontal_source_image_row' for r in coverage['source_order_layout']),
            horizontal_images=sum(len(r['images']) for r in coverage['source_order_layout'] if r['method']=='horizontal_source_image_row')),
        heading_layout=settings,compilation=compilation,block_colors=config.get('block_colors',{}),
        typography=config.get('typography'))
    if compilation['success']:
        progress('编译完成',1,1)
        audit=audit_pdf(project/'main.pdf',coverage,edited.registry)
        save(reports/'pdf_audit.json',audit)
        numbering = {n['id']:n for n in audit['footnotes']}
        save(reports/'footnotes.json', [{**note, **{k:v for k,v in numbering[note['id']].items()
             if k in {'rendered_page','reference_page','rendered_number','rendered_marker','same_page'}}}
             for note in edited.footnotes])
        save(reports/'footnote_numbering.json', dict(method='matched_footnotes_per_rendered_physical_page_v1',
             pages=audit['footnote_numbering_pages'], items=audit['footnotes']))
        qa=LocalPDFReviewer().review(project/'main.pdf',output/'previews',coverage,progress)
        save(reports/'local_qa.json',qa)
        navigation=NavigationBuilder().finalize(project/'main.pdf',output/'navigation',edited.registry)
        valid = not any(audit[key] for key in ('missing_destinations', 'broken_links', 'outside_page', 'footnote_page_mismatches')) and not qa['issues']
        result.update(status='complete' if valid else 'validation_failed',pages=len(audit['pages']),internal_links=audit['internal_links'],
                      outside_page=audit['outside_page'],local_qa_issues=qa['issues'],navigation_links=navigation['links'],
                      large_bottom_gap_pages=qa['large_bottom_gap_pages'])
    assert before=={p:digest(p) for p in before},'A saved input was modified'
    save(output/'render_result.json',result)
    write_render_report(output,result)
    return result


def write_render_report(output,result):
    s=result['statistics'];c=result['compilation']
    current=result.get('processing_method','').startswith('page_content_keywords_v')
    lines=['# 三阶段结果的 LaTeX 排版报告','',
           '复用保存的标题、类型删除、合并和本地编号映射结果；仅对定位到编译错误的公式发送原图进行 LaTeX 格式恢复。','',
           f"公式编译恢复状态：{result.get('formula_repairs',{}).get('status','not_needed')}；修复公式 {result.get('formula_repairs',{}).get('repaired_atoms',0)} 个。模型请求与重试用量见 reports/formula_repairs.json，原图、回复和编译历史均保留。",'',
           f"表格编译恢复状态：{result.get('table_repairs',{}).get('status','not_needed')}；修复表格 {result.get('table_repairs',{}).get('repaired_atoms',0)} 个。只调整排版模板，原单元格逐字填回；原图、反馈与验证见 reports/table_repair。",'',
           f"状态：{result['status']}。标题 {s['headings']} 个，处理块 {s['processed_blocks']} 个；加粗小标题 {s['styled_subheadings']} 个，公式颜色框 {s['formula_modules']} 个，例题 {s['examples']} 个。",'',
           ('一级标题独占居中标题页；二级标题另起一页；三级及以下不强制换页。特殊块使用用户指定的类型配色，具体设置见 render_result.json 的 block_colors。小标题加粗，独立公式使用原生数学排版。' if result.get('block_colors') else
            '一级标题独占居中标题页；二级标题另起一页；三级及以下不强制换页。定义用绿色块，定理/命题/引理/推论用橙色块，算法用青色块；例题使用黑色分隔线，小标题加粗。独立公式按普通数学公式排版，不添加颜色框。'),'',
           f"原文明示名称 {s.get('named_special_blocks',0)} 个进入对象登记记录；特殊块标题显示原文的类型、编号和名称，保留原有括号、引号等符号。名称不重复出现在框内正文开头；无明确名称时不添加名称。",'',
           (f"第二步提取的 {s.get('native_formula_special_blocks',0)} 个公式特殊块保留识别与来源记录，排版使用原生公式，不添加颜色框。原始识别内容保持原样；编译失败公式的排版替换单独记录，不改变原来源坐标与锚点。编号小标题蓝色加粗，字号大于正文；例题题目用楷体、解答用仿宋。" if current else
            f"旧 DeepSeek 结果中的 {s['ignored_legacy_formula_modules']} 个公式特殊块不再单独着色；公式原文、保存的纠错、编号与引用跳转保留。[原生公式样式记录](reports/ignored_formula_modules.json)"),'',
           f"唯一候选引用提示 {s['linked_reference_hints']} 条生成跳转；多候选、未命中或同一片段目标冲突 {s['ambiguous_or_unmatched_hints']} 条保留原文，不任选目标。名称与例题编号不参与映射。",'',
           '[未生成跳转的引用提示](reports/unlinked_reference_hints.json)','',
           f"脚注 {s.get('footnotes',0)} 个，其中已匹配 {s.get('matched_footnotes',0)} 个按最终物理页从 1 重新编号；正文引用点使用小字号上标，与页底脚注共享编号。未匹配 {s.get('unmatched_footnotes',0)} 个保留原标号。实际页面与编号见 reports/footnote_numbering.json。",'',
           f"图表使用 htp 浮动的分组 {s['asset_float_groups']} 个：当前位置放得下就放，放不下允许正文继续并在后续页顶或浮动页显示。图表与所属图注一起移动，跳转锚点置于实际图表内。位于例题等框内的图表通过分段续框移到框外浮动；长表保留可跨页的原生 longtable。一级标题页和二级分页造成的留白属于指定规则。",'',
           f"同排图片恢复 {s.get('horizontal_image_rows',0)} 组，共 {s.get('horizontal_images',0)} 张，保留原书左右位置、相对尺寸和邻近标签。来源及目标锚点保留在共享图片行内。",'',
           '[LaTeX 主文件](latex/main.tex) · [编译记录](reports/compilation.json) · [内容覆盖记录](reports/coverage.json)','',
           f"编译成功：{c['success']}；用时 {c['seconds']} 秒；编译警告 {len(c['warnings'])} 条。原数据和已保存的三个阶段结果保持原样。",'']
    if c['success']:
        lines += [f"生成 PDF {result['pages']} 页，内部链接 {result['internal_links']} 个；实际链接与目标验证见 reports/pdf_audit.json 和 reports/navigation.json。",'',
                  '[生成的 PDF](latex/main.pdf) · [本地页面检查](reports/local_qa.json)','']
        if result.get('large_bottom_gap_pages'):
            lines += ['自动留白筛查（排除独立标题页、强制分页前页和末页）：'+
                      ', '.join(str(p['page']) for p in result['large_bottom_gap_pages'])+'。这些是待查看页面，不等同于内容丢失。','']
    else:lines += ['编译恢复尚未成功，未标记完成；详情见 reports/formula_repairs.json。源码与 latex/main.log、latex/compile-output.txt 保留。','', '```text',c['log_tail'][-3500:],'```','']
    history=Path(output)/'reports/compile_history'
    if history.exists():
        lines += ['## 编译历史','',
                  '较早编译记录保存在 reports/compile_history。失败公式的原图、模型回复、独立公式语法验证和排版替换保存在 reports/formula_repair。','']
    (Path(output)/'render_report.md').write_text('\n'.join(lines),encoding='utf-8')
