#!/usr/bin/env python3
"""Download attributed books, run resumable excerpts, and record real artifacts.

Run inside the CPU application container. The manifest is data, never code.
An artifact success and a quality pass are separate outcomes. No full English
book is translated: the conversion retains English and translation tests use
at most two short source paragraphs per English book.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import sys
import time
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from book_pipeline.config import ROOT, digest, load_config, read, save


def acquire(book, directory):
    import pymupdf
    pdf = directory / (book['id'] + '.pdf')
    if not pdf.exists():
        partial = pdf.with_suffix('.part')
        request = Request(book['url'], headers={'User-Agent': 'PDF-Reforge-Evaluation/1.0'})
        with urlopen(request, timeout=90) as response, partial.open('wb') as stream:
            total = 0
            while data := response.read(1024 * 1024):
                total += len(data)
                if total > 200 * 1024 * 1024:
                    raise ValueError('Source exceeds 200 MB acquisition limit')
                stream.write(data)
        with pymupdf.open(partial) as document:
            if not document.is_pdf or document.needs_pass or not len(document):
                raise ValueError('Invalid/encrypted source')
        partial.replace(pdf)
    with pymupdf.open(pdf) as document:
        if not document.is_pdf or document.needs_pass or not len(document):
            raise ValueError('Cached source is not an unencrypted PDF')
    if book.get('sha256') and digest(pdf) != book['sha256']:
        raise ValueError('Source SHA differs from the attributed corpus record')
    return pdf


def excerpt(book, source, variant, directory):
    import pymupdf
    target = directory / (book['id'] + '-' + variant + '.pdf')
    if target.exists():
        return target
    with pymupdf.open(source) as original, pymupdf.open() as output:
        for number in book['pages']:
            page = original[number - 1]
            if variant == 'rasterized':
                # Deliberately remove the text layer; keep provenance explicit.
                new = output.new_page(width=page.rect.width, height=page.rect.height)
                new.insert_image(new.rect, stream=page.get_pixmap(dpi=150).tobytes('png'))
            elif variant in {'native','scan'}:
                output.insert_pdf(original, from_page=number - 1, to_page=number - 1)
            else:
                raise ValueError('Unsupported benchmark variant: ' + variant)
        output.save(target, garbage=4, deflate=True)
    return target


def translation_items(index):
    """Select at most two whole prose blocks, retaining their inline formulas."""
    from book_pipeline.ingestion.translation import language
    candidates = []
    for group, positions in index.groups.items():
        atoms = [index.atoms[n] for n in positions]
        if (atoms[0].get('block_type') not in {'text', 'paragraph', 'page_footnote'}
                or any(a['kind'] not in {'text', 'math'} or a.get('role') in {'code', 'code_body'} for a in atoms)):
            continue
        text = ''.join('$'+a['content']+'$' if a['kind'] == 'math' else a['content'] for a in atoms).strip()
        if 80 <= len(text) <= 1200 and language(text) == 'en':
            candidates.append(dict(id=group, text=text))
    # Exercise the placeholder contract with real source math when available.
    with_math = [item for item in candidates if '$' in item['text']]
    selected = (with_math or candidates)[:2]
    assert sum(len(item['text']) for item in selected) <= 2400
    return selected


def translation_probe(book, result_dir, config, source_directory=None):
    from book_pipeline.application.pipeline import load_book
    from book_pipeline.ai.client import DeepSeekClient
    from book_pipeline.ingestion.translation import Translator, protect
    source_dir = Path(source_directory) if source_directory else result_dir / 'source'
    if not (source_dir / 'manifest.json').exists():
        return dict(status='not_run', reason='source extraction incomplete')
    manifest = read(source_dir / 'manifest.json')
    index = load_book(manifest['input_pdf'], source_dir)
    items = translation_items(index)
    if not items:
        return dict(status='not_run', reason='no suitable short English prose')
    client = DeepSeekClient({**config, '_budget_job_id': config.get('_budget_job_id', digest(manifest['input_pdf']))}, result_dir / 'translation_probe' / 'cache')
    translator = Translator(client)
    translated = translator.translate(items)
    from collections import Counter
    protected = {item['id']: protect(item['text'])[1] for item in items}
    for item in items:
        if Counter(protect(translated[item['id']])[1]) != Counter(protected[item['id']]):
            raise ValueError('Translated probe changed formula/number placeholders')
    result = dict(status='placeholders_validated_semantics_unreviewed',
                  placeholder_validation='passed', semantic_quality='not_evaluated',
                  paragraphs=len(items), input=items, output=translated,
                  characters=sum(len(item['text']) for item in items), protected_values=protected,
                  glossary=translator.glossary, usage=client.ledger.summary(),
                  scope='at most two complete paragraphs and 2400 characters per benchmark invocation; no full-book translation')
    save(result_dir / 'translation_probe' / 'result.json', result)
    return result


def inspect_artifacts(result_dir, rendered):
    import pymupdf
    pdf, tex = result_dir / 'pdf/latex/main.pdf', result_dir / 'pdf/latex/main.tex'
    result = dict(artifact_success=False, quality_status='not_evaluated', issues=[])
    if not pdf.is_file() or not tex.is_file():
        result['issues'].append('missing_pdf_or_latex')
        return result
    with pymupdf.open(pdf) as document:
        chars = sum(len(p.get_text().strip()) for p in document)
        result.update(artifact_success=len(document) > 0 and chars > 0, pages=len(document),
                      searchable_characters=chars, pdf=str(pdf), latex=str(tex))
        if not chars:
            result['issues'].append('no_searchable_text')
    compilation = result_dir / 'pdf/reports/compilation.json'
    if compilation.exists():
        warnings = read(compilation).get('warnings', [])
        result['compiler_warnings'] = warnings
        if any(re.search(r'Missing character|undefined', w, re.I) for w in warnings):
            result['issues'].append('missing_glyph_or_undefined_reference')
    qa = result_dir / 'pdf/reports/local_qa.json'
    if qa.exists():
        local = read(qa)
        result['issues'].extend(local.get('issues', []))
        result['previews'] = local.get('previews', [])
    coverage = result_dir / 'pdf/reports/coverage.json'
    edited_source = result_dir / 'pdf/reports/edited_source.json'
    if coverage.exists():
        rendering = read(coverage)
        problems = rendering.get('issues',[])
        result['issues'].extend(sorted({i['reason'] for i in problems if i.get('reason')}))
        if edited_source.exists():
            body = [a for a in read(edited_source)['atoms'] if a['kind'] in {'text','math','display_math','table'}
                    and a.get('content','').strip() and a.get('role') not in {'header','footer','page_number'}]
            raster_only = {i['atom'] for i in problems if i.get('reason')=='complex_table_preserved_as_image'}
            result['body_text_atoms'] = len(body)
            result['body_text_atoms_preserved_only_as_images'] = sum(a['id'] in raster_only for a in body)
            if body and all(a['id'] in raster_only for a in body):
                result['artifact_success'] = False
                result['issues'].append('no_searchable_body_text')
    if rendered.get('status') not in {'complete', 'complete_with_unmatched'}:
        result['issues'].append('pipeline_validation_incomplete')
    source_dir = rendered.get('source_directory')
    if source_dir and (Path(source_dir)/'manifest.json').exists():
        recovery = read(Path(source_dir)/'manifest.json').get('table_prose_recovery', {})
        if recovery.get('requires_visual_review'):
            result['issues'].append('ocr_table_prose_recovery_requires_visual_review')
            result['table_prose_recovered_regions'] = recovery['recovered_regions']
    if edited_source.exists():
        result['issues'].extend(sorted({a['recovery_issue'] for a in read(edited_source)['atoms'] if a.get('recovery_issue')}))
    # Geometry and compilation alone do not establish semantic/visual fidelity.
    result['quality_status'] = 'issues_found' if result['issues'] else 'needs_visual_and_semantic_review'
    return result


def write_report(directory, report):
    save(directory / 'results.json', report)
    lines = ['# PDF 重构回归报告', '', '成功表示得到可搜索 PDF 与 LaTeX；质量结论另列。', '',
             '| 样本 | 结果产物 | 质量 | 耗时（秒） |', '| --- | --- | --- | --- |']
    for case in report['cases']:
        lines.append(f"| {case['id']} | {'成功' if case.get('artifact_success') else '失败/未完成'} | {case.get('quality_status', case['status'])} | {case.get('seconds',0)} |")
    lines += ['', '扫描测试件若标为 rasterized，是从原书栅格化的受控测试，不能替代真实扫描书验证。',
              '所有测试均记录实际页码；节选成功不代表整本书成功。英文全文重构保留英文，翻译只抽样。', '']
    (directory / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'benchmarks/corpus.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/benchmark')
    parser.add_argument('--acquire-only', action='store_true')
    parser.add_argument('--case', action='append', help='Limit to selected book-variant case IDs')
    parser.add_argument('--run-id', default='', help='Independent revision outputs; budget remains tied to input SHA')
    parser.add_argument('--reuse-cache-from', type=Path, action='append', default=[],
                        help='Search prior run directories for signature-validated API caches')
    args = parser.parse_args()
    directory = args.output.resolve()
    if args.run_id and (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id) or args.run_id in {'.','..'}):
        parser.error('--run-id must be a single safe directory name')
    report_directory = directory / 'runs' / args.run_id if args.run_id else directory
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit('Benchmark already running; no duplicate paid work started')
    try:
        return run_cases(args, directory, report_directory)
    finally:
        lock.close()


def run_cases(args, directory, report_directory):
    config = load_config()
    config['translation'] = 'off'
    config['publication_scope'] = 'excerpt'
    config['shared_cache_dirs'] = list(dict.fromkeys([
        *config.get('shared_cache_dirs', []),
        *(str(p.resolve()) for root in args.reuse_cache_from for p in root.rglob('cache') if p.is_dir())]))
    sources, samples = directory / 'sources', directory / 'samples'
    sources.mkdir(exist_ok=True); samples.mkdir(exist_ok=True)
    report = dict(started=datetime.now(timezone.utc).isoformat(), scope='excerpt_regression', cases=[])
    translation_probe_case = None
    for book in read(args.manifest)['books']:
        selected = [v for v in book['variants'] if not args.case or book['id']+'-'+v in args.case]
        if not selected:
            continue
        try:
            original = acquire(book, sources)
        except Exception as error:
            report['cases'].append(dict(id=book['id'], status='acquisition_failed', error_type=type(error).__name__))
            write_report(report_directory, report)
            continue
        for variant in selected:
            identifier = book['id'] + '-' + variant
            case = dict(id=identifier, book=book, variant=variant, original_sha256=digest(original),
                        status='running', artifact_success=False, scope='excerpt', source_pages=book['pages'])
            case.update(real_scanned_book=book.get('source_type')=='real_scan',
                        controlled_rasterization=variant=='rasterized')
            report['cases'].append(case)
            started = time.monotonic()
            write_report(report_directory, report)
            try:
                pdf = excerpt(book, original, variant, samples)
                case['sample_pdf'] = str(pdf)
                if args.acquire_only:
                    case['status'] = 'acquired'
                    continue
                from book_pipeline.application.pipeline import run_workflow
                import yaml
                metadata = pdf.with_suffix('.book.yaml')
                metadata.write_text(yaml.safe_dump(dict(book=dict(title=book['title'], author=book['author'])), allow_unicode=True), encoding='utf-8')
                result_dir = report_directory / identifier if args.run_id else directory / 'runs' / identifier
                def progress(stage, n, total):
                    print(f'{identifier}: {stage}: {n}/{total}', flush=True)
                case_config = {**config, '_budget_job_id': digest(original)}
                if book.get('source_type') == 'real_scan':
                    # Re-evaluate real page images even when an archive's PDF
                    # carries an old, incomplete OCR overlay. Keep it intact.
                    case_config['input_mode'] = 'ocr'
                rendered = run_workflow(pdf, None, metadata, case_config, result_dir, 20261025, progress)
                save(result_dir / 'workflow_result.json', rendered)
                case.update(inspect_artifacts(result_dir, rendered), status=rendered['status'])
                if book['language'] == 'en':
                    if translation_probe_case is None:
                        translation_probe_case = identifier
                        case['translation_probe'] = translation_probe(book, result_dir, case_config, rendered.get('source_directory'))
                    else:
                        case['translation_probe'] = dict(status='shared_run_probe', case=translation_probe_case,
                            scope='run-level two-paragraph cap; reconstruction retains English')
            except Exception as error:
                import traceback
                traceback.print_exc()
                case.update(status='failed', error_type=type(error).__name__, error=str(error)[:1000])
            finally:
                case['seconds'] = round(time.monotonic() - started, 2)
                write_report(report_directory, report)
    return 0 if all(c.get('artifact_success') or c['status']=='acquired' for c in report['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
