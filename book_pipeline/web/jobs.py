"""Persistent local jobs and a serialized worker for OCR and book rendering."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import subprocess
import threading
import time
import uuid

from book_pipeline.application.pipeline import load_book, match_headings, process_content
from book_pipeline.ai.client import DeepSeekClient
from book_pipeline.config import ROOT, digest, read, save
from book_pipeline.publishing.colors import COLOR_KINDS, DEFAULT_COLORS, validate_colors
from book_pipeline.publishing.block_preview import render_block_preview
from book_pipeline.publishing.processed_book import render_saved_book
from book_pipeline.publishing.typography import validate_typography, heading_pattern, pattern_depth, recommended_size
from book_pipeline.structure.subheading_patterns import pattern_kind, pattern_sort_key
from book_pipeline.structure.page_content import METHOD as CONTENT_METHOD, run_page_content
from book_pipeline.structure.keyword_matching import validate_candidates
from book_pipeline.structure.book_mapping import (
    build_book, special_block_key, special_block_type, removed_special_rules,
    REMOVED_BLOCKS_FILE, SPECIAL_BLOCK_POLICY,
)
from book_pipeline.structure.headings import validate_metadata
from book_pipeline.structure.exercise_headings import classify_saved_plan, classify_exercise_headings
from book_pipeline.structure.toc_matching import write_matching_report
from book_pipeline.structure.rule_mapping import run_rule_mapping
from book_pipeline.structure.semantics import LABELS
from pipeline_progress import MinerUProgress

STAGES = [('mineru', 'MinerU 识别'), ('headings', '标题匹配'), ('content', '正文整理'),
          ('references', '引用映射'), ('render', '整书排版')]
ACTIVE = {'queued', 'processing', 'rendering', 'extracting'}
PHASES = {'footnote-matching': '同页脚注标号匹配', 'subheading-continuity': '小标题编号连续性复核',
          'leading-punctuation-cleanup': '段首标点清理', 'cross-page-pairs': '跨页段落配对',
          'keyword-special-blocks': '特殊块识别', 'local-reference-hints': '编号引用匹配',
          'keyword-discovery': '关键词本地匹配', 'keyword-boundary-review': '特殊块范围复核'}


class JobBusy(ValueError):
    """A running job cannot accept settings for its next recognition pass."""


def now():
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, directory, config):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='book-worker')
        self.jobs = {}
        for path in self.directory.glob('*/job.json'):
            try:
                job = read(path)
                if job.get('preview_status') in ACTIVE:
                    job.update(preview_status='interrupted', preview_error='上次特殊块预览已中断，可重新生成。')
                    self._persist(job)
                if job['status'] in ACTIVE:
                    log_path = path.parent / 'mineru.log'
                    if log_path.is_file():
                        progress = MinerUProgress(job['source_pages'])
                        latest = None
                        with log_path.open(encoding='utf-8', errors='replace') as log:
                            for line in log:
                                event = progress.consume(line)
                                if event:
                                    latest = event
                        stage = next(s for s in job['stages'] if s['id'] == 'mineru')
                        if latest and stage['status'] == 'running':
                            stage.update(latest)
                    for stage in job['stages']:
                        if stage['status'] == 'running':
                            stage['status'] = 'interrupted'
                    job.update(status='interrupted', error='上次处理已中断，上传文件和已保存结果仍保留。')
                    self._persist(job)
                self.jobs[job['id']] = job
            except (OSError, ValueError, KeyError):
                continue
        self._import_completed()

    def _persist(self, job):
        save(self.directory / job['id'] / 'job.json', job)

    def _new(self, metadata, pdf, filename, pages, **paths):
        identifier = uuid.uuid4().hex
        job = dict(id=identifier, metadata=metadata, filename=filename, pdf=str(pdf),
                   source_pages=pages, status='uploaded', created=now(), updated=now(),
                   stages=[dict(id=key, label=label, status='pending', detail='', done=0, total=None)
                           for key, label in STAGES], colors={}, rendered_colors=None,
                   render_pdf=None, render_pages=0, revision=0, error=None, **paths)
        job.update(preview_status='idle', preview_error=None, preview_colors=None,
                   preview_revision=0, preview_samples=[], typography=validate_typography(),
                   keyword_candidates=validate_candidates(self.config.get('keyword_candidates'), self.config.get('numbered_object_labels')))
        self.jobs[identifier] = job
        self._persist(job)
        return identifier

    def create(self, metadata, pdf, filename, pages, keyword_candidates=None):
        selected = validate_candidates(keyword_candidates if keyword_candidates is not None else self.config.get('keyword_candidates'),
                                       self.config.get('numbered_object_labels'))
        with self.lock:
            identifier = self._new(metadata, pdf, filename, pages)
            directory = self.directory / identifier
            uploaded = directory / 'source.pdf'
            Path(pdf).replace(uploaded)
            self.update(identifier, pdf=str(uploaded), headings=str(directory / 'headings'),
                        content=str(directory / 'content'), references=str(directory / 'references'),
                        keyword_candidates=selected)
            return identifier

    def _import_completed(self):
        imported = {j.get('imported_from') for j in self.jobs.values()}
        for path in sorted((ROOT / 'output').rglob('render_result.json')):
            if path.is_relative_to(self.directory) or str(path) in imported:
                continue
            try:
                result = read(path)
                sources = read(path.parent / 'reports' / 'input_sources.json')
                pdf = Path(result['source']['source_pdf'])
                rendered = Path(result['pdf']) if result.get('pdf') else None
                if result['status'] != 'complete' or not pdf.is_file() or not rendered or not rendered.is_file():
                    continue
                content = Path(sources['processing'])
                report = read(content / 'section_results.json')
                metadata = validate_metadata({'book': report['metadata'], 'heading_layout': result['heading_layout']})
                middle = next(Path(p) for p in sources['input_files_sha256'] if p.endswith('/middle_json.json'))
                identifier = self._new(metadata, pdf, pdf.name, result['source']['source_page_count'],
                                       content=str(content), references=sources['mapping'], mineru=str(middle.parent),
                                       imported_from=str(path), headings=None)
                job = self.jobs[identifier]
                job.update(status='complete', render_pdf=str(rendered), render_pages=result['pages'],
                           render_directory=str(path.parent), rendered_colors=result.get('block_colors', {}),
                           colors=result.get('block_colors', {}), revision=1)
                for stage in job['stages']:
                    stage['status'] = 'complete'
                self._cache_blocks(identifier, read(Path(sources['mapping']) / 'edited_text_index.json'), policy_checked=False)
                self._persist(job)
            except (OSError, ValueError, KeyError, StopIteration, TypeError):
                continue

    def get(self, identifier):
        with self.lock:
            if identifier not in self.jobs:
                raise KeyError(identifier)
            return deepcopy(self.jobs[identifier])

    def update(self, identifier, **values):
        with self.lock:
            job = self.jobs[identifier]
            job.update(values, updated=now())
            self._persist(job)

    def snapshot(self, identifier):
        job = self.get(identifier)
        keys = ('id', 'filename', 'source_pages', 'status', 'created', 'updated', 'stages',
                'colors', 'rendered_colors', 'render_pages', 'revision', 'error', 'statistics',
                'preview_status', 'preview_error', 'preview_colors', 'preview_revision', 'preview_samples',
                'typography', 'preview_typography', 'rendered_typography', 'content_method', 'render_content_stale', 'block_revision')
        result = {key: job.get(key) for key in keys}
        result.update(book=job['metadata']['book'], has_pdf=bool(job.get('render_pdf')),
                      has_blocks=(self.directory / identifier / 'blocks.json').is_file())
        result['typography'] = validate_typography(job.get('typography'))
        result['keyword_candidates'] = validate_candidates(job.get('keyword_candidates', self.config.get('keyword_candidates')),
                                                          self.config.get('numbered_object_labels'))
        result['keywords_changed'] = job.get('content_candidates') != result['keyword_candidates']
        return result

    def set_keyword_candidates(self, identifier, candidates):
        selected = validate_candidates(candidates)
        with self.lock:
            job = self.jobs[identifier]
            if job['status'] in ACTIVE or job.get('preview_status') in ACTIVE:
                raise JobBusy('请等待当前处理完成后再修改关键词。')
            self.update(identifier, keyword_candidates=selected)

    def list(self):
        with self.lock:
            ids = sorted(self.jobs, key=lambda key: self.jobs[key]['updated'], reverse=True)
        return [self.snapshot(key) for key in ids]

    def stage(self, identifier, key, status='running', detail='', done=0, total=None, **fields):
        with self.lock:
            job = self.jobs[identifier]
            stage = next(s for s in job['stages'] if s['id'] == key)
            stage.update(status=status, detail=detail, done=done, total=total, **fields)
            job['updated'] = now()
            self._persist(job)

    def progress(self, identifier, key):
        return lambda phase, n, total: self.stage(identifier, key, detail=PHASES.get(phase, phase), done=n, total=total)

    def _cache_blocks(self, identifier, book, *, policy_checked=True):
        blocks = [dict(id=t['id'], kind='remark' if t['kind'] == 'other' else t['kind'], label=t['label'], name=t['name'],
                       page=t['page'], section=t['context']['heading_title'], text=t['body_preview'],
                       removable=t.get('origin') == 'deepseek_section_module',
                       **({'pattern':heading_pattern(t)} if t['kind']=='subheading' else {}))
                  for t in book['targets'] if t['kind'] in COLOR_KINDS | {'other', 'subheading'}]
        save(self.directory / identifier / 'blocks.json', blocks)
        self.update(identifier, block_policy_version=SPECIAL_BLOCK_POLICY if policy_checked else None,
                    block_revision=(self.get(identifier).get('block_revision') or 0) + 1,
                    statistics=dict(headings=len(book['sections']),
                                               blocks=len(book['blocks']), special_blocks=len(blocks)))

    def _ensure_block_policy(self, identifier, *, force=False):
        """Refresh legacy cached annotations locally, keeping OCR and AI decisions."""
        with self.lock:
            job = self.get(identifier)
            if job.get('block_policy_version') == SPECIAL_BLOCK_POLICY:
                return
            if not force and (job['status'] in ACTIVE or job.get('preview_status') in ACTIVE):
                return
            content = Path(job.get('content') or '')
            if not (content / 'section_results.json').is_file() or not (content / 'split_source.json').is_file():
                return
            report = read(content / 'section_results.json')
            if 'sections' not in report:
                return
            index = load_book(job['pdf'], job['mineru'])
            book = build_book(index, read(content / 'split_source.json'), report, content)
            cache = self.directory / identifier / 'blocks.json'
            before = {(b['id'], b['kind'], b.get('pattern')) for b in read(cache)} if cache.is_file() else set()
            after = {(t['id'], 'remark' if t['kind'] == 'other' else t['kind'],
                      heading_pattern(t) if t['kind'] == 'subheading' else None)
                     for t in book['targets'] if t['kind'] in COLOR_KINDS | {'other', 'subheading'}}
            mapping_index = Path(job.get('references') or '') / 'edited_text_index.json'
            mapping_changed = mapping_index.is_file() and read(mapping_index) != book
            if before != after or mapping_changed:
                kinds = {'remark' if t['kind'] == 'other' else t['kind'] for t in book['targets']}
                self.update(identifier, references=str(self.directory / identifier / ('references-' + uuid.uuid4().hex[:8])),
                            colors={k:v for k,v in job['colors'].items() if k in kinds},
                            preview_samples=[], preview_colors=None, preview_typography=None,
                            preview_status=job.get('preview_status') if job.get('preview_status') in ACTIVE else 'idle',
                            render_content_stale=bool(job.get('render_pdf')))
            self._cache_blocks(identifier, book)

    def blocks(self, identifier):
        self._ensure_block_policy(identifier)
        path = self.directory / identifier / 'blocks.json'
        blocks = read(path) if path.is_file() else []
        for block in blocks:
            block.setdefault('removable', block['id'].startswith('O'))
        kinds = sorted({b['kind'] for b in blocks} & COLOR_KINDS)
        patterns = sorted({b['pattern'] for b in blocks if b['kind']=='subheading'}, key=pattern_sort_key)
        typography = validate_typography(self.get(identifier).get('typography'))
        return dict(job=self.snapshot(identifier), blocks=blocks, types=[dict(kind=k, label=LABELS.get(k, {'remark': '注释', 'notation': '记号'}.get(k, k)),
                                             count=sum(b['kind'] == k for b in blocks), default=DEFAULT_COLORS[k])
                                         for k in kinds], subheadings=[dict(pattern=p, kind=pattern_kind(p),
                                             depth=pattern_depth(p),
                                             count=sum(b.get('pattern')==p for b in blocks),
                                             recommended=recommended_size(typography['body_size'],p)) for p in patterns])

    def remove_block(self, identifier, block_id):
        """Cancel one annotation, keeping source text and the previous saved book."""
        with self.lock:
            job = self.get(identifier)
            if job['status'] in ACTIVE or job.get('preview_status') in ACTIVE:
                raise JobBusy('请等待当前处理完成后再取消特殊块。')
            previous = Path(job['content'])
            if not (previous / 'section_results.json').is_file():
                raise KeyError(block_id)
            index = load_book(job['pdf'], job['mineru'])
            plan, report = read(previous / 'split_source.json'), read(previous / 'section_results.json')
            book = build_book(index, plan, report, previous)
            target = next((t for t in book['targets'] if t['id'] == block_id and
                           t.get('origin') == 'deepseek_section_module'), None)
            if not target:
                raise KeyError(block_id)
            section = target['context']['heading']
            rules = removed_special_rules(previous)
            rules.setdefault('blocks', []).append(dict(key=special_block_key(section, target), id=block_id, kind=target['kind'],
                                 label=target['label'], lo=target['lo'], hi=target['hi'], removed_at=now()))
            self._save_special_rules(job, index, previous, plan, report, rules)

    def _save_special_rules(self, job, index, previous, plan, report, rules):
        """Save an annotation edit as a new version and invalidate its render caches."""
        identifier = job['id']
        output = self.directory / identifier / ('content-' + uuid.uuid4().hex[:8])
        shutil.copytree(previous, output, ignore=shutil.ignore_patterns('references'))
        save(output / REMOVED_BLOCKS_FILE, rules)
        book = build_book(index, plan, report, output)
        kinds = {'remark' if t['kind'] == 'other' else t['kind'] for t in book['targets']}
        self.update(identifier, content=str(output), references=str(output / 'references'),
                    colors={k:v for k,v in job['colors'].items() if k in kinds},
                    preview_status='idle', preview_error=None, preview_samples=[],
                    preview_colors=None, preview_typography=None,
                    render_content_stale=bool(job.get('render_pdf')))
        self._cache_blocks(identifier, book)

    def remove_block_type(self, identifier, selector):
        """Disable a whole displayed type, or one exact subheading form."""
        if not isinstance(selector, dict) or set(selector) - {'kind', 'pattern'}:
            raise ValueError('请指定特殊块类型；小标题须指定编号形式。')
        kind, pattern = selector.get('kind'), selector.get('pattern')
        if not isinstance(kind, str) or kind not in COLOR_KINDS | {'subheading'} or (kind == 'subheading' and not pattern_depth(pattern)) or (kind != 'subheading' and pattern is not None):
            raise ValueError('特殊块类型或小标题编号形式无效。')
        if pattern:
            from book_pipeline.structure.subheading_patterns import canonical_pattern
            pattern = canonical_pattern(pattern)
        with self.lock:
            job = self.get(identifier)
            if job['status'] in ACTIVE or job.get('preview_status') in ACTIVE:
                raise JobBusy('请等待当前处理完成后再删除特殊块类型。')
            previous = Path(job.get('content') or '')
            if not (previous / 'section_results.json').is_file():
                raise KeyError(kind)
            index = load_book(job['pdf'], job['mineru'])
            plan, report = read(previous / 'split_source.json'), read(previous / 'section_results.json')
            book = build_book(index, plan, report, previous)
            matches = [t for t in book['targets'] if t.get('origin') == 'deepseek_section_module' and
                       special_block_type({**t, 'type': t['kind']}) == (kind, pattern)]
            if not matches:
                raise KeyError(kind)
            rules = removed_special_rules(previous)
            rules.setdefault('types', []).append(dict(kind=kind, **({'pattern':pattern} if pattern else {}), removed_at=now()))
            self._save_special_rules(job, index, previous, plan, report, rules)

    def _submit(self, identifier, action, **settings):
        with self.lock:
            job = self.jobs[identifier]
            if job['status'] in ACTIVE or job.get('preview_status') in ACTIVE:
                raise ValueError('这个书稿已有任务正在运行。')
            self.update(identifier, status='queued', error=None, **settings)
            self.executor.submit(self._run, identifier, action)

    def process(self, identifier):
        job = self.get(identifier)
        if job.get('imported_from'):
            raise ValueError('已有书稿可直接调整配色并生成预览。')
        self._submit(identifier, 'process')

    def _validated_colors(self, identifier, colors):
        types = {t['kind'] for t in self.blocks(identifier)['types']}
        colors = validate_colors(colors)
        if set(colors) - types:
            raise ValueError('配色包含当前书稿不存在的特殊块类型。')
        if not (self.directory / identifier / 'blocks.json').is_file():
            raise ValueError('请先完成识别、标题匹配和正文整理。')
        return colors

    def render(self, identifier, colors, typography=None):
        self._submit(identifier, 'render', colors=self._validated_colors(identifier, colors),
                     typography=validate_typography(self.get(identifier).get('typography') if typography is None else typography))

    def preview(self, identifier, colors, typography=None):
        colors = self._validated_colors(identifier, colors)
        typography = validate_typography(self.get(identifier).get('typography') if typography is None else typography)
        with self.lock:
            job = self.jobs[identifier]
            if job['status'] in ACTIVE or job.get('preview_status') in ACTIVE:
                raise ValueError('这个书稿已有任务正在运行。')
            self.update(identifier, preview_status='queued', preview_error=None, colors=colors, typography=typography)
            self.executor.submit(self._run, identifier, 'preview')

    def _run(self, identifier, action):
        try:
            if action == 'process':
                self._process(identifier)
            elif action == 'preview':
                self._preview(identifier)
            elif action == 'extract':
                self._extract(identifier)
            else:
                self._render(identifier)
        except (Exception, KeyboardInterrupt) as error:
            interrupted = isinstance(error, (InterruptedError, KeyboardInterrupt))
            if action == 'preview':
                self.update(identifier, preview_status='interrupted' if interrupted else 'failed',
                            preview_error=str(error) or '特殊块预览已中断，可重新生成。')
                return
            with self.lock:
                for stage in self.jobs[identifier]['stages']:
                    if stage['status'] == 'running':
                        stage['status'] = 'interrupted' if interrupted else 'failed'
                self.update(identifier, status='interrupted' if interrupted else 'failed',
                            error=str(error) or '处理已中断，上传文件和已保存结果仍保留。')

    def _process(self, identifier):
        self.update(identifier, status='processing')
        job = self.get(identifier)
        pdf = Path(job['pdf'])
        self.stage(identifier, 'mineru', detail='检查识别结果', done=None, unit='',
                   source_total=job['source_pages'], completed_pages=0, window=None)
        sha = digest(pdf)
        mineru = Path(job.get('mineru') or self.directory / identifier / 'mineru')
        # Only reuse a complete result for the identical uploaded PDF.
        for path in (ROOT / 'output').rglob('manifest.json'):
            try:
                manifest = read(path)
                if manifest.get('input_sha256') == sha and (path.parent / 'middle_json.json').is_file():
                    load_book(pdf, path.parent)
                    mineru = path.parent
                    break
            except (OSError, ValueError, KeyError):
                continue
        self.update(identifier, mineru=str(mineru))
        try:
            load_book(pdf, mineru)
            complete = True
        except (OSError, ValueError, KeyError):
            complete = False
        if not complete:
            python = Path(self.config['mineru_python']).expanduser()
            if not python.is_absolute():
                python = ROOT / python
            command = [str(python), str(ROOT / 'run_mineru.py'), str(pdf), '-o', str(mineru), '--pages', 'all',
                       '--tier', self.config['mineru_tier'], '--ocr-mode', self.config['mineru_ocr_mode'],
                       '--model-source', self.config['mineru_model_source'],
                       '--checkpoint-dir', str(self.directory / identifier / 'mineru_cache')]
            self.stage(identifier, 'mineru', detail='加载模型并识别', done=None, unit='')
            progress = MinerUProgress(job['source_pages'])
            last_update, last_phase = 0, None
            with (self.directory / identifier / 'mineru.log').open('a', encoding='utf-8') as log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      env={**os.environ, 'PDF_PIPELINE_PROGRESS': '1'}) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        event = progress.consume(line)
                        if event:
                            current = time.monotonic()
                            phase = (event['detail'], event['window'], event['total'])
                            if phase != last_phase or current - last_update >= .25 or event['done'] == event['total']:
                                self.stage(identifier, 'mineru', **event)
                                last_update, last_phase = current, phase
                    returncode = process.wait()
                    if returncode in {-2, -15, 130, 143}:
                        raise InterruptedError('MinerU 识别已中断，上传文件和已保存结果仍保留。')
                    if returncode:
                        raise RuntimeError('MinerU 识别失败，运行记录保存在本书稿的 mineru.log 中。')
        index = load_book(pdf, mineru)
        self.stage(identifier, 'mineru', 'complete', '识别完成', job['source_pages'], job['source_pages'],
                   unit='页', completed_pages=job['source_pages'], source_total=job['source_pages'], window=None)
        headings, content = Path(job['headings']), Path(job['content'])
        self.stage(identifier, 'headings', detail='读取目录并匹配标题')
        heading_result = match_headings(index, job['metadata'], self.config, headings, self.progress(identifier, 'headings')) or {}
        detail = '标题匹配完成' + (f" · {heading_result['exercise_headings']} 个习题标题" if 'exercise_headings' in heading_result else '')
        self.stage(identifier, 'headings', 'complete', detail)
        self.stage(identifier, 'content', detail='整理正文与识别特殊块')
        content_config = {**self.config, 'keyword_candidates': validate_candidates(
            job.get('keyword_candidates', self.config.get('keyword_candidates')), self.config.get('numbered_object_labels'))}
        result = process_content(index, headings, content_config, content, 20261008, self.progress(identifier, 'content'))
        if result['status'] != 'complete':
            raise RuntimeError('正文整理存在未解决项，处理报告已保存；解决后可继续处理。')
        book = build_book(index, read(content / 'split_source.json'), read(content / 'section_results.json'), content)
        self._cache_blocks(identifier, book)
        self.stage(identifier, 'content', 'complete', '正文整理完成')
        self.update(identifier, status='ready', content_method=CONTENT_METHOD, content_candidates=content_config['keyword_candidates'])

    def extract(self, identifier):
        job = self.get(identifier)
        if not (Path(job['content']) / 'split_source.json').is_file():
            raise ValueError('请先完成识别与正文整理。')
        self._submit(identifier, 'extract')

    def _extract(self, identifier):
        """Reidentify saved source from scratch, retaining old versions and OCR."""
        self.update(identifier, status='extracting')
        job = self.get(identifier)
        previous = Path(job['content'])
        root = self.directory / identifier
        output = root / ('content-' + uuid.uuid4().hex[:8])
        self.stage(identifier, 'content', detail='识别习题章节与特殊块')
        index = load_book(job['pdf'], job['mineru'])
        config = {**self.config, 'shared_cache_dirs': [str(previous / 'cache'), *self.config.get('shared_cache_dirs', [])],
                  'keyword_candidates': validate_candidates(job.get('keyword_candidates', self.config.get('keyword_candidates')),
                                                            self.config.get('numbered_object_labels'))}
        client = DeepSeekClient(config, output / 'cache')
        plan = read(previous / 'split_source.json')
        heading_directory = Path(job['headings']) if job.get('headings') else None
        if heading_directory and (heading_directory/'outline.json').is_file():
            outline = classify_exercise_headings(read(heading_directory/'outline.json'),client,heading_directory,
                                                self.progress(identifier,'content'))
            write_matching_report(heading_directory,outline,index,read(heading_directory/'heading_matches.json').get('verification'))
            plan['exercise_heading_classification'] = outline['exercise_heading_classification']
            plan['toc_entries'] = outline.get('toc_entries',[])
            self.stage(identifier,'headings','complete',f"标题匹配完成 · {len(outline['exercise_heading_classification']['exercise_heading_ids'])} 个习题标题")
        plan = classify_saved_plan(plan,client,output,
                                   self.progress(identifier,'content'))
        report = run_page_content(index, None, client, output, seed=20261008,
                                  progress=self.progress(identifier, 'content'), saved_plan=plan)
        if report['status'] != 'complete':
            raise RuntimeError('补充识别存在未解决项，报告已保存在 ' + str(output))
        # Reidentification resets all manual annotation cancellations. Keep
        # the old version intact, and switch only after recognition succeeds.
        save(output / REMOVED_BLOCKS_FILE, dict(blocks=[], types=[]))
        book = build_book(index, read(output / 'split_source.json'), report, output)
        self.update(identifier, content=str(output), references=str(output / 'references'), content_method=CONTENT_METHOD,
                    content_candidates=config['keyword_candidates'])
        self._cache_blocks(identifier, book)
        self.stage(identifier, 'content', 'complete', '例题与小标题识别完成')
        self.update(identifier, status='ready', preview_status='idle', preview_error=None,
                    render_content_stale=bool(job.get('render_pdf')),
                    preview_samples=[], preview_colors=None, preview_typography=None)

    def _preview(self, identifier):
        self._ensure_block_policy(identifier, force=True)
        self.update(identifier, preview_status='rendering')
        job = self.get(identifier)
        index = load_book(job['pdf'], job['mineru'])
        revision = (job.get('preview_revision') or 0) + 1
        output = self.directory / identifier / f'block-preview-{revision:04d}'
        result = render_block_preview(index, job['content'], job['metadata'],
                                      {**self.config, 'block_colors': job['colors'], 'typography': job.get('typography')}, output)
        self.update(identifier, preview_status='complete', preview_error=None, preview_revision=revision,
                    preview_directory=str(output), preview_samples=result['samples'], preview_colors=job['colors'],
                    preview_typography=validate_typography(job.get('typography')))

    def sample_path(self, identifier, kind, number):
        job = self.get(identifier)
        sample = next((s for s in job.get('preview_samples', []) if s['kind'] == kind), None)
        if not sample or number not in sample['pages']:
            raise ValueError('特殊块预览不存在。')
        path = Path(job['preview_directory']) / 'samples' / kind / f'{number}.png'
        if not path.is_file():
            raise ValueError('特殊块预览图片不存在，可重新生成。')
        return path

    def _render(self, identifier):
        self._ensure_block_policy(identifier, force=True)
        self.update(identifier, status='rendering')
        job = self.get(identifier)
        index = load_book(job['pdf'], job['mineru'])
        content, mapping = Path(job['content']), Path(job['references'])
        self.stage(identifier, 'references', detail='匹配编号引用')
        if not (mapping / 'mapping_results.json').is_file():
            run_rule_mapping(index, content, mapping, heading_directory=Path(job['headings']) if job['headings'] else None,
                             progress=self.progress(identifier, 'references'))
        self.stage(identifier, 'references', 'complete', '引用映射完成')
        self.stage(identifier, 'render', detail='生成并编译书稿')
        revision = job['revision'] + 1
        output = self.directory / identifier / f'render-{revision:04d}'
        result = render_saved_book(index, content, mapping, job['metadata'],
                                  {**self.config, 'block_colors': job['colors'], 'typography': job.get('typography')}, output,
                                  progress=self.progress(identifier, 'render'))
        if result['status'] != 'complete':
            raise RuntimeError('排版编译失败，LaTeX 源码和编译记录已保存。')
        self.stage(identifier, 'render', 'complete', '整书 PDF 已生成', result['pages'], result['pages'])
        self.update(identifier, status='complete', render_pdf=result['pdf'], render_pages=result['pages'],
                    render_directory=str(output), revision=revision, rendered_colors=job['colors'],
                    rendered_typography=validate_typography(job.get('typography')),render_content_stale=False)

    def pdf_path(self, identifier):
        path = self.get(identifier).get('render_pdf')
        if not path or not Path(path).is_file():
            raise ValueError('书稿还没有生成 PDF。')
        return Path(path)
