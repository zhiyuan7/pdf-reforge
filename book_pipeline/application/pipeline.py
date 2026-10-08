"""The current four stages, with optional reuse of complete MinerU output."""
from pathlib import Path
import subprocess

from book_pipeline.ai.client import DeepSeekClient
from book_pipeline.config import ROOT, read
from book_pipeline.ingestion.source import SourceIndex, load_source
from book_pipeline.structure.headings import load_metadata
from book_pipeline.structure.toc_matching import analyze_toc_headings, verify_matching, write_matching_report
from book_pipeline.structure.page_content import run_page_content
from book_pipeline.structure.rule_mapping import run_rule_mapping
from book_pipeline.publishing.processed_book import render_saved_book


def load_book(pdf, mineru_directory):
    return SourceIndex(load_source(pdf, mineru_directory))


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
    report = run_page_content(index, outline, client, output, seed=seed, progress=progress)
    return {'method': report['method'], 'status': report['status'], 'sections': len(report['sections']),
            'usage': report['usage'], 'output': str(output.resolve())}


def run_workflow(pdf, mineru_directory, metadata_path, config, output, seed, progress):
    pdf, output = Path(pdf).resolve(), Path(output).resolve()
    if output == pdf.parent or pdf.is_relative_to(output):
        raise ValueError('输出目录不能包含输入 PDF')
    metadata = load_metadata(pdf, metadata_path)
    if mineru_directory is None:
        mineru_directory = output / 'mineru'
        if not (mineru_directory / 'manifest.json').is_file():
            python = Path(config['mineru_python']).expanduser()
            if not python.is_absolute():
                python = ROOT / python
            subprocess.run([str(python), str(ROOT / 'run_mineru.py'), str(pdf), '-o', str(mineru_directory),
                            '--pages', 'all', '--tier', config['mineru_tier'], '--ocr-mode', config['mineru_ocr_mode'],
                            '--model-source', config['mineru_model_source'], '--checkpoint-dir', str(output / 'mineru_cache')], check=True)
    index = load_book(pdf, mineru_directory)
    if output == Path(mineru_directory).resolve() or output.is_relative_to(Path(mineru_directory).resolve()):
        raise ValueError('输出目录不能覆盖 MinerU 原始识别目录')
    headings, content, mapping = (output / name for name in ('headings', 'content', 'references'))
    first = match_headings(index, metadata, config, headings, progress)
    second = process_content(index, headings, config, content, seed, progress)
    if second['status'] != 'complete':
        return {'status': second['status'], 'headings': first, 'content': second}
    mapped = run_rule_mapping(index, content, mapping, heading_directory=headings, progress=progress)
    rendered = render_saved_book(index, content, mapping, metadata, config, output / 'pdf', progress=progress)
    return {'status': rendered['status'], 'headings': first, 'content': second,
            'references': mapped['statistics'], 'render': rendered}
