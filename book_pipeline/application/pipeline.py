"""Shared ingestion, translation and book publishing workflow."""
from pathlib import Path

from book_pipeline.ai.client import DeepSeekClient
from book_pipeline.config import ROOT, digest, read
from book_pipeline.ingestion.source import SourceIndex, load_source
from book_pipeline.structure.headings import load_metadata
from book_pipeline.structure.toc_matching import analyze_toc_headings, verify_matching, write_matching_report
from book_pipeline.structure.page_content import run_page_content
from book_pipeline.structure.exercise_headings import classify_exercise_headings, METHOD as EXERCISE_METHOD
from book_pipeline.structure.rule_mapping import run_rule_mapping
from book_pipeline.publishing.processed_book import render_saved_book


def load_book(pdf, mineru_directory, formula_corrections=None):
    index = SourceIndex(load_source(pdf, mineru_directory))
    if formula_corrections:
        from book_pipeline.ingestion.formula_bands import apply_formula_corrections
        index = apply_formula_corrections(index, read(formula_corrections))
    return index


def match_headings(index, metadata, config, output, progress):
    output = Path(output)
    if output.resolve() == Path(index.source['mineru_dir']).resolve():
        raise ValueError('标题结果目录不能覆盖 MinerU 原始识别目录')
    client = DeepSeekClient(config, output / 'cache')
    outline = analyze_toc_headings(index, metadata, client, config, output, progress=progress)
    verification = verify_matching(outline, index)
    verification['模型用量'] = client.ledger.summary()
    write_matching_report(output, outline, index, verification)
    return {'method': 'toc_headings', 'status': 'complete' if not outline['toc_matching']['counts']['unmatched'] else 'complete_with_unmatched',
            'exercise_headings':len(outline.get('exercise_heading_classification',{}).get('exercise_heading_ids',[])),
            **outline['toc_matching']['counts'], 'usage': client.ledger.summary(), 'output': str(output.resolve())}


def process_content(index, heading_directory, config, output, seed, progress):
    heading_directory, output = Path(heading_directory), Path(output)
    saved = read(heading_directory / 'heading_matches.json')['source']
    if any(saved[k] != index.source[k] for k in ('input_sha256', 'mineru_sha256')):
        raise ValueError('标题结果与当前 PDF / MinerU 来源不匹配')
    if output.resolve() in {heading_directory.resolve(), Path(index.source['mineru_dir']).resolve()}:
        raise ValueError('正文处理需要独立的输出目录')
    outline = read(heading_directory / 'outline.json')
    client = DeepSeekClient(config, output / 'cache')
    if outline.get('exercise_heading_classification', {}).get('method') != EXERCISE_METHOD:
        outline = classify_exercise_headings(outline,client,heading_directory,progress)
        write_matching_report(heading_directory,outline,index,read(heading_directory/'heading_matches.json').get('verification'))
    report = run_page_content(index, outline, client, output, seed=seed, progress=progress)
    return {'method': report['method'], 'status': report['status'], 'sections': len(report['sections']),
            'usage': report['usage'], 'output': str(output.resolve())}


def prepare_book_source(pdf, source_directory, config, output, progress=lambda *args:None):
    """One ingestion/translation entry point for both CLI and web jobs."""
    from book_pipeline.ingestion.native import prepare_source
    from book_pipeline.ingestion.translation import translate_source
    output = Path(output).resolve()
    if source_directory is not None:
        source_path = Path(source_directory).resolve()
        if output == source_path or output.is_relative_to(source_path):
            raise ValueError('输出目录不能覆盖原始提取目录')
    if source_directory is None and not (output / 'source' / 'manifest.json').exists():
        # Retain cross-job reuse of validated legacy OCR for identical uploads.
        # Never select translated/corrected sources under a different policy.
        sha = digest(pdf)
        for path in (ROOT / 'output').rglob('manifest.json'):
            try:
                manifest = read(path)
                if (manifest.get('mineru_version') and manifest.get('input_sha256') == sha
                        and not manifest.get('translation') and not manifest.get('corrected_source_sha256')):
                    load_source(pdf, path.parent)
                    source_directory = path.parent
                    break
            except (OSError, ValueError, KeyError):
                continue
    if source_directory is None:
        # Preserve old resumable MinerU runs when present.
        legacy = output / 'mineru'
        if (legacy / 'manifest.json').is_file():
            load_source(pdf, legacy)
            source_directory = legacy
        else:
            source_directory = prepare_source(pdf, output / 'source', config, progress)
    if output == Path(source_directory).resolve() or output.is_relative_to(Path(source_directory).resolve()):
        raise ValueError('输出目录不能覆盖原始提取目录')
    raw = load_book(pdf, source_directory)
    if raw.source.get('translation'):
        return Path(source_directory)
    # Visual formula recovery reads original English; it must precede translation.
    from book_pipeline.ingestion.formula_bands import correct_formula_bands, apply_formula_corrections
    correction = correct_formula_bands(raw, config, output / 'formula_corrections', progress)
    if correction['records']:
        from book_pipeline.ingestion.source import materialize_source
        source_directory = materialize_source(apply_formula_corrections(raw, correction), output / 'corrected_source')
    return translate_source(pdf, source_directory, output / 'translated_source', config, progress)


def run_workflow(pdf, mineru_directory, metadata_path, config, output, seed, progress):
    pdf, output = Path(pdf).resolve(), Path(output).resolve()
    if output == pdf.parent or pdf.is_relative_to(output):
        raise ValueError('输出目录不能包含输入 PDF')
    metadata = load_metadata(pdf, metadata_path)
    mineru_directory = prepare_book_source(pdf, mineru_directory, config, output, progress)
    index = load_book(pdf, mineru_directory)
    if output == Path(mineru_directory).resolve() or output.is_relative_to(Path(mineru_directory).resolve()):
        raise ValueError('输出目录不能覆盖 MinerU 原始识别目录')
    headings, content, mapping = (output / name for name in ('headings', 'content', 'references'))
    corrections_path = output / 'formula_corrections' / 'formula_corrections.json'
    formula_repairs = read(corrections_path) if corrections_path.is_file() else dict(records=[], usage={})
    first = match_headings(index, metadata, config, headings, progress)
    second = process_content(index, headings, config, content, seed, progress)
    if second['status'] != 'complete':
        return {'status': second['status'], 'headings': first, 'content': second}
    mapped = run_rule_mapping(index, content, mapping, heading_directory=headings, progress=progress)
    rendered = render_saved_book(index, content, mapping, metadata, config, output / 'pdf', progress=progress)
    return {'status': rendered['status'], 'formula_corrections': dict(
                regions=len(formula_repairs['records']), usage=formula_repairs['usage']),
            'headings': first, 'content': second,
            'references': mapped['statistics'], 'render': rendered}
