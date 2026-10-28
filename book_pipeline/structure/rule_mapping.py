"""Local type/number lookup in saved edited text; results are reference hints."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
import unicodedata

from book_pipeline.config import fingerprint, read, save
from book_pipeline.structure.book_mapping import build_book, source_for_quote, write_processed_book
from book_pipeline.structure.references import chinese_number
from book_pipeline.structure.section_review import formal_view

METHOD = 'local_numbered_reference_hints_v3'
EXCLUDED_TARGET_KINDS = {'example'}
NUMBER = r"(?:[A-Z]\.)?\d+(?:[._]\d+)*(?:[a-z])?°?'*"
CHINESE = r'[零〇一二三四五六七八九十百]+'
HEADING_NUMBER = rf'(?:{NUMBER}|{CHINESE}|[IVXLCDM]+|[a-z])'
LABELS = {
    '公式': 'equation', '关系式': 'equation', '等式': 'equation', '不等式': 'equation',
    '恒等式': 'equation', '式': 'equation', '定理': 'theorem', '命题': 'proposition',
    '引理': 'lemma', '推论': 'corollary', '定义': 'definition', '算法': 'algorithm',
    '图': 'figure', '表': 'table', '记号': 'notation',
    '公理': '公理', '性质': '性质', '准则': '准则', '附注': '附注',
}
FORMAL_REFERENCE = re.compile(
    r'(?P<label>' + '|'.join(sorted(LABELS, key=len, reverse=True)) + rf')\(?(?P<number>{NUMBER})\)?'
    rf'|第(?P<heading_numbers>{HEADING_NUMBER}(?:[、,，和及与]{HEADING_NUMBER})*)(?P<level>小节|章|节)'
    rf'|§(?P<section_number>{NUMBER})'
    rf'|小节(?P<small_section>[a-z]|[IVXLCDM]+)'
    rf'|附录(?P<appendix>[A-Z]|\d+|{CHINESE})')
PARENTHESIS = re.compile(rf'\((?P<number>{NUMBER})\)')
EQUATION_CUE = re.compile(r'(?:由|根据|利用|结合|代入|参见|见|关系|等式|不等式|公式|表达式|估计|恒等式)$')


def search_view(text):
    value, positions = formal_view(text)
    chars = []
    for char in value:
        if char in '−—–－':char = '-'
        elif '！' <= char <= '～':char = unicodedata.normalize('NFKC', char)
        chars.append(char)
    return ''.join(chars), positions


def key(number):
    number = search_view(str(number))[0].strip('()').removesuffix('°')
    if re.fullmatch(CHINESE, number):return chinese_number(number)
    if re.fullmatch('[IVXLCDM]+', number):return number.upper()
    return number


def heading_number(target):
    if target['number']:return key(target['number'])
    value = search_view(target['label'])[0]
    match = re.match(rf'(?:§|第)?({HEADING_NUMBER})(?:章|节|[.、])', value)
    return key(match[1]) if match else ''


class LocalMapper:
    """Index numbered objects only; names never participate in matching."""
    def __init__(self, book):
        self.book = book
        self.targets = {t['id']: t for t in book['targets']}
        self.by_number = defaultdict(list)
        self.aliases = {}
        headings = {t['context']['heading']: t for t in book['targets'] if t['kind'] == 'heading'}
        for target in book['targets']:
            if target['kind'] in EXCLUDED_TARGET_KINDS:
                self.aliases[target['id']] = {'number_aliases': [], 'searchable': False,
                                               'excluded_reason': 'example_mapping_disabled'}
                continue
            kind = target.get('heading_kind') if target['kind'] == 'heading' else target['kind']
            if kind == 'subheading':kind = 'subsection'
            if kind == 'other':
                label = search_view(target['label'])[0]
                kind = next((LABELS[word] for word in sorted(LABELS, key=len, reverse=True) if label.startswith(word)), kind)
            number = heading_number(target) if target['kind'] == 'heading' else key(target['number'])
            numbers = [number] if number else []
            if kind == 'section' and number and '.' not in number and target['context']['chapter_number']:
                numbers.append(key(target['context']['chapter_number']) + '.' + number)
            if target['kind'] == 'subheading' and re.fullmatch('[a-z]', number):
                parent = headings.get(target['context']['heading'])
                prefix = heading_number(parent) if parent else ''
                if prefix:numbers.append(prefix + number)
            for n in numbers:self.by_number[(kind, n)].append(target)
            prefix = {'chapter': '第{n}章', 'section': '§{n}', 'subsection': '第{n}小节'}.get(kind)
            words = [word for word, family in LABELS.items() if family == kind]
            labels = [prefix.format(n=n) for n in numbers] if prefix else [word+n for word in words for n in numbers]
            self.aliases[target['id']] = {'number_aliases': labels, 'searchable': bool(labels)}
        filename = str(book.get('source', {}).get('source_pdf', ''))
        volume = re.search(rf'第({HEADING_NUMBER})卷', filename)
        self.volume = key(volume[1]) if volume else None

    def scoped(self, hits, context, value, start, kind):
        """Use explicit qualifiers, then heading/section/chapter; never distance."""
        prefix = re.split(r'[。；;！？!?\n]', value[max(0, start-100):start])[-1]
        volume = list(re.finditer(rf'第({HEADING_NUMBER})卷', prefix))
        if volume and self.volume and key(volume[-1][1]) != self.volume:
            return [], 'other_volume'
        chapters = list(re.finditer(rf'第({HEADING_NUMBER})章', prefix))
        sections = list(re.finditer(rf'§({NUMBER})|第({HEADING_NUMBER})节', prefix))
        narrowed = hits
        if kind != 'chapter' and chapters:
            chapter = key(chapters[-1][1])
            narrowed = [t for t in narrowed if key(t['context']['chapter_number']) == chapter]
        if kind not in {'chapter', 'section'} and sections:
            section = key(sections[-1][1] or sections[-1][2])
            if re.fullmatch(r'\d+\.\d+',section):
                chapter,section = section.split('.')
                narrowed = [t for t in narrowed if key(t['context']['chapter_number']) == chapter]
            narrowed = [t for t in narrowed if key(t['context']['section_number']) == section]
        explicit = (kind != 'chapter' and bool(chapters)) or (kind not in {'chapter', 'section'} and bool(sections))
        if explicit:
            # “§2中的定理1” without a chapter qualifier means this chapter first.
            qualified_section = sections and '.' in key(sections[-1][1] or sections[-1][2])
            if not chapters and not qualified_section and context['chapter']:
                local = [t for t in narrowed if t['context']['chapter'] == context['chapter']]
                if local:return local, 'explicit_section_in_current_chapter'
            return narrowed, 'explicit_chapter_or_section'
        if kind == 'chapter':return narrowed, 'chapter_number'
        for field in ('heading', 'section', 'chapter'):
            if kind == 'section' and field in {'heading', 'section'}:continue
            if context[field]:
                local = [t for t in narrowed if t['context'][field] == context[field]]
                if local:return local, 'current_' + field
        return narrowed, 'book_wide_candidates'

    def scan(self, block):
        value, positions = search_view(block['text'])
        text_spans = [];offset = 0
        for part in block['parts']:
            if part['kind'] == 'text':text_spans.append((offset, offset+len(part['text'])))
            offset += len(part['text'])
        rows = [];seen = set();covered = []
        def add(start, end, kind, number, hits, method):
            lo, hi = positions[start][0], positions[end-1][1]
            if not any(a <= lo < b for a,b in text_spans):return
            if any(lo < b and hi > a for a,b in block['declarations']):return
            selected, rule = self.scoped(hits, block['context'], value, start, kind)
            ids = sorted({t['id'] for t in selected})
            signature = (lo, hi, kind, number, tuple(ids))
            if signature in seen:return
            seen.add(signature)
            source = source_for_quote(block['pieces'], lo, hi)
            if not source:raise ValueError('Reference hint has no original source address')
            quote = block['text'][lo:hi]
            occurrence = 1 + len(list(re.finditer(re.escape(quote), block['text'][:lo])))
            rows.append({'section':block['section'], 'block':block['b'], 'text':quote,
                         'normalized_text':value[start:end], 'start_offset':lo, 'end_offset':hi,
                         'physical_page':source[0].get('page',block['page']), 'source':source,
                         'occurrence':occurrence, 'kind':kind, 'number':number, 'match_rule':method,
                         'scope_rule':rule, 'candidate_ids':ids,
                         'status':'unique' if len(ids)==1 else ('ambiguous' if ids else 'missing')})
        for match in FORMAL_REFERENCE.finditer(value):
            if match['label']:
                kind, number = LABELS[match['label']], key(match['number'])
                add(match.start(), match.end(), kind, number, self.by_number[(kind,number)], 'type_and_number')
                # “式(6)、(9)和(10)” retains the original type for each member.
                cursor = match.end()
                while True:
                    member = re.match(rf'[,，、和及与]\(?(?P<number>{NUMBER})\)?', value[cursor:])
                    if not member:break
                    n = key(member['number']);a = cursor + 1;z = cursor + member.end()
                    add(a,z,kind,n,self.by_number[(kind,n)],'numbered_list')
                    covered.append((a,z));cursor = z
            elif match['heading_numbers']:
                kind = {'章':'chapter','节':'section','小节':'subsection'}[match['level']]
                for number in re.split('[、,，和及与]',match['heading_numbers']):
                    n = key(number)
                    add(match.start(),match.end(),kind,n,self.by_number[(kind,n)],'heading_number')
            elif match['section_number']:
                n = key(match['section_number']);add(match.start(),match.end(),'section',n,self.by_number[('section',n)],'section_mark')
            elif match['small_section']:
                n = key(match['small_section']);add(match.start(),match.end(),'subsection',n,self.by_number[('subsection',n)],'subsection_letter')
            else:
                n = key(match['appendix'])
                hits = [t for t in self.book['targets'] if t['kind']=='heading' and search_view(t['label'])[0].startswith('附录'+match['appendix'])]
                add(match.start(),match.end(),'appendix',n,hits,'appendix_label')
            covered.append((match.start(),match.end()))
        for match in PARENTHESIS.finditer(value):
            if any(match.start()<b and match.end()>a for a,b in covered):continue
            if EQUATION_CUE.search(value[max(0,match.start()-16):match.start()]):
                n = key(match['number']);add(match.start(),match.end(),'equation',n,self.by_number[('equation',n)],'equation_reference_cue')
        return sorted(rows, key=lambda r:(r['start_offset'],r['end_offset'],r['kind'],r['number']))


def heading_correspondence(index,book,plan,heading_directory=None):
    catalogue = {}
    if heading_directory:
        toc = read(Path(heading_directory)/'toc_catalog.json')
        if any(toc[k] != book['source'][k] for k in ('input_sha256','mineru_sha256')):
            raise ValueError('TOC catalogue and processed book belong to different original sources')
        saved = read(Path(heading_directory)/'outline.json')
        if fingerprint(saved['headings']) != plan['heading_sha256']:
            raise ValueError('TOC heading matches differ from processed heading boundaries')
        catalogue = {h['id']:h for h in toc['entries']}
    targets = {t['context']['heading']:t for t in book['targets'] if t['kind']=='heading'}
    rows = []
    for section in book['sections']:
        if section['heading'].get('implicit'):
            continue
        h = section['heading'];t = targets[section['id']];toc = catalogue.get(section['id'],{})
        rows.append({'toc_id':section['id'],'target_id':t['id'],'toc_title':toc.get('title',h['title']),
                     'source_title':t['label'],'category':h['toc_category'],'level':h['heading_level'],
                     'toc_page':toc.get('toc_page'),'printed_page':toc.get('printed_page'),
                     'physical_page':t['page'],'source_start':index.address(t['lo']),
                     'source_end':index.address(t['hi'],end=True),'match_origin':'existing_confirmed_heading_match'})
    return rows


def run_rule_mapping(index,processing_directory,output,heading_directory=None,progress=None):
    processing_directory = Path(processing_directory);output = Path(output)
    if output.resolve() in {processing_directory.resolve(),Path(index.source['mineru_dir']).resolve()}:
        raise ValueError('Use a separate output directory for local mapping')
    if (output/'mapping_results.json').exists() and read(output/'mapping_results.json')['method'] not in {METHOD,'local_book_reference_hints_v1','local_numbered_reference_hints_v2'}:
        raise ValueError('Use a new output directory to preserve earlier model mapping results')
    report = read(processing_directory/'section_results.json');plan = read(processing_directory/'split_source.json')
    if report['sampling']['selection'] != 'all_confirmed_heading_scopes':raise ValueError('Local mapping requires the complete saved processed book')
    if any(index.source[k] != report['source'][k] for k in ('input_sha256','mineru_sha256')):
        raise ValueError('Processed book does not match the original PDF/MinerU hashes')
    book = build_book(index,plan,report,processing_directory)
    titles = heading_correspondence(index,book,plan,heading_directory)
    mapper = LocalMapper(book);by_section = defaultdict(list)
    for block in book['blocks']:by_section[block['section']].append(block)
    sections = []
    for n,section in enumerate(book['sections'],1):
        hints = [hint for block in by_section[section['id']] for hint in mapper.scan(block)]
        sections.append({'section':section['id'],'heading_title':section['heading']['title'],'hints':hints})
        if progress:progress('local-reference-hints',n,len(book['sections']))
    counts = Counter(h['status'] for row in sections for h in row['hints'])
    result = {'method':METHOD,'match_policy':'type_and_number_only','role':'reference_hints_only','status':'complete','source':book['source'],
              'excluded_target_kinds':sorted(EXCLUDED_TARGET_KINDS),
              'processed_sha256':book['processed_sha256'],'targets_sha256':fingerprint(book['targets']),
              'sections':sections,'heading_correspondence':titles,
              'usage':{'requests':0,'tokens':0,'estimated_cost':0,'network_required':False},
              'statistics':{'headings':len(sections),'processed_blocks':len(book['blocks']),'targets':len(book['targets']),
                            'target_types':dict(Counter(t['kind'] for t in book['targets'])),
                            'hints':sum(counts.values()),'unique_hints':counts['unique'],
                            'ambiguous_hints':counts['ambiguous'],'unmatched_hints':counts['missing'],
                            'heading_matches':len(titles),'searchable_targets':sum(a['searchable'] for a in mapper.aliases.values()),
                            'excluded_example_targets':sum(t['kind']=='example' for t in book['targets'])}}
    save(output/'edited_text_index.json',book);save(output/'target_registry.json',book['targets'])
    save(output/'search_aliases.json',mapper.aliases);save(output/'mapping_results.json',result)
    if (output/'verification.json').exists():
        audit=read(output/'verification.json')
        if any(audit.get(k)!=result[k] for k in ('method','source','statistics')):
            (output/'verification.json').unlink()
    write_rule_report(output,book,result,mapper.aliases,report,processing_directory)
    return result


def write_csv(path,fields,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def write_rule_report(output,book,result,aliases,processing,processing_directory):
    targets = {t['id']:t for t in book['targets']};stats = result['statistics']
    hints = [h for r in result['sections'] for h in r['hints']]
    fields = ['section','block','source_page','text','match_rule','scope_rule','status','kind','number','candidate_ids','candidate_names','start_offset','end_offset']
    csv_rows = [{**{k:h[k] for k in fields if k not in {'source_page','candidate_names','candidate_ids'}},
                 'source_page':h['physical_page'],'candidate_ids':';'.join(h['candidate_ids']),
                 'candidate_names':';'.join(targets[t]['name'] or targets[t]['label'] for t in h['candidate_ids'])} for h in hints]
    write_csv(output/'reference_mapping.csv',fields,csv_rows)
    write_csv(output/'heading_correspondence.csv',
              ['toc_id','target_id','toc_title','source_title','category','level','toc_page','printed_page','physical_page'],
              [{k:v for k,v in h.items() if k not in {'source_start','source_end','match_origin'}} for h in result['heading_correspondence']])
    coverage = Counter(tid for h in hints for tid in h['candidate_ids'])
    coverage_rows = [{'target_id':t['id'],'kind':t['kind'],'number':t['number'],'name':t['name'],
                      'heading':t['context']['heading'],'physical_page':t['page'],
                      'searchable':aliases[t['id']]['searchable'],'hint_occurrences':coverage[t['id']],
                      'search_aliases':' / '.join(aliases[t['id']]['number_aliases'])} for t in book['targets']]
    write_csv(output/'target_coverage.csv',list(coverage_rows[0]) if coverage_rows else
              ['target_id','kind','number','name','heading','physical_page','searchable','hint_occurrences','search_aliases'],coverage_rows)
    write_processed_book(output,book,processing,processing_directory)
    esc = lambda text:str(text).replace('|','\\|').replace('\n',' ')
    lines = ['# 本地类型与编号映射提示','',
             '**本次映射 DeepSeek 请求 0 次，输入/输出 tokens 0，API 费用 0。** 删除、合并、纠错、特殊块沿用已保存的处理版本，本次没有重新进行模型处理。','',
             f"扫描 {stats['headings']} 个标题范围、{stats['processed_blocks']} 个处理块；对象登记表保留 {stats['targets']} 个对象，其中 {stats['excluded_example_targets']} 个例题已排除出引用匹配。",'',
             f"检出 {stats['hints']} 条提示：唯一候选 {stats['unique_hints']} 条、多候选 {stats['ambiguous_hints']} 条、未找到候选 {stats['unmatched_hints']} 条。目录—正文标题对照 {stats['heading_matches']} 条。",'',
             '[引用提示表](reference_mapping.csv) · [目录—正文标题对照](heading_correspondence.csv) · [全部对象的检索覆盖](target_coverage.csv) · [处理后整书内容](processed_book.md) · [机器结果](mapping_results.json)','',
             '## 规则','',
             '只按原类型与原编号在每个已确认标题下的处理文字里查找，如“第一章”“定理3”“公式3.1”“图5”。支持第几节、§编号、第几小节、定理/命题/引理/定义/图/表编号、公式/式编号及明确引用词后的括号公式编号。空格、全角符号、撇号等做形式归一化，定位仍保存原文字符范围。','',
             '例题不参与引用匹配：“例12”“例题3.1”等引用不建立映射，也不记录为未命中。此前提取的例题内容和编码继续保留在处理版本及对象登记表中。','',
             '明确的章、节限定优先；否则按本标题范围、本节、本章查候选，最后保留全书候选。同一范围仍有多个同号对象时列出全部候选，不调用模型、不按距离选一个。对象声明、公式自身编号、图表自身图注不当作引用。','',
             '不检索对象名称：“基上的极限”“柯西准则”“柯西—康托尔原理”等文字即使带有“参见”“由”等引用词，也不建立映射。名称仅保留在对象登记表中供阅读；无编号对象仍登记，但不参与正文引用匹配。上式、上述命题等语义指代不处理。结果仅作查阅提示，多个候选和未命中也保留在表中。','',
             '## 目录—正文标题','',
             '复用此前确认的标题地址；本次导出目录条目与正文标题之间的对应，不再识别目录图片。','',
             '| 目录编码 | 目录标题 | 正文标题 | 目标 | 正文物理页 |',
             '|---|---|---|---|---:|']
    for h in result['heading_correspondence']:
        lines.append(f"| {h['toc_id']} | {esc(h['toc_title'])} | {esc(h['source_title'])} | [{h['target_id']}](processed_book.md#{h['target_id']}) | {h['physical_page']} |")
    lines += ['','## 引用提示','', '| 标题范围 | 文本块 | 原文片段 | 状态 | 候选对象 | 检索规则 |','|---|---|---|---|---|---|']
    labels = {'unique':'唯一候选','ambiguous':'多个候选','missing':'未找到'}
    for h in hints:
        links = ', '.join(f"[{tid}](processed_book.md#{tid}) {esc(targets[tid]['label'] or targets[tid]['name'])}" for tid in h['candidate_ids']) or '—'
        lines.append(f"| {h['section']} | [{h['block']}](processed_book.md#{h['block']}) | {esc(h['text'])} | {labels[h['status']]} | {links} | {h['match_rule']} / {h['scope_rule']} |")
    lines += ['','## 检索覆盖与验证','',
              f"有可检索类型与编号的对象 {stats['searchable_targets']} 个；不参与编号检索的对象 {stats['targets']-stats['searchable_targets']} 个。各对象是否命中和编号别名见 target_coverage.csv。",'',
              '原 PDF、MinerU、已保存的删除/合并/纠错及目录匹配结果保持原样。未生成最终 LaTeX 或排版 PDF。','']
    if (output/'verification.json').exists():lines += ['[本地运行与来源验证结果](verification.json)','']
    (output/'full_book_report.md').write_text('\n'.join(lines),encoding='utf-8')
