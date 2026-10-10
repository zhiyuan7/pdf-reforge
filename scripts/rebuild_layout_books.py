"""Rebuild saved books with full-width formula repair and source image rows.

Usage: .venv/bin/python scripts/rebuild_layout_books.py [--phase recognize|render|all]
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from book_pipeline.ai.client import DeepSeekClient
from book_pipeline.application.pipeline import load_book
from book_pipeline.config import load_config, read, save
from book_pipeline.ingestion.formula_bands import correct_formula_bands, apply_formula_corrections
from book_pipeline.publishing.processed_book import render_saved_book
from book_pipeline.publishing.colors import COLOR_KINDS
from book_pipeline.publishing.typography import heading_pattern
from book_pipeline.structure.book_mapping import SPECIAL_BLOCK_POLICY
from book_pipeline.structure.page_content import run_page_content
from book_pipeline.structure.rule_mapping import run_rule_mapping


def rebuild(job_path, phase, config, update_job=False):
    job = read(job_path)
    root = job_path.parent / 'layout-repair-20261010'
    root.mkdir(exist_ok=True)
    provenance = root / 'previous_job.json'
    if not provenance.exists():
        save(provenance, job)
    job = read(provenance)
    old_content = Path(job['content'])
    old_plan = read(old_content / 'split_source.json')
    original = load_book(job['pdf'], job['mineru'])
    def progress(stage, done, total):
        print(f"{job['metadata']['book']['title']} | {stage}: {done}/{total}", flush=True)
    correction_path = root / 'formulas/formula_corrections.json'
    if phase in {'recognize', 'all'}:
        protected = [section['heading']['group'] for section in old_plan['sections']]
        corrected = correct_formula_bands(original, config, root / 'formulas', progress, protected)
    else:
        corrected = read(correction_path)
    if phase == 'recognize':
        print('公式校正完成：' + str(correction_path), flush=True)
        return
    index = apply_formula_corrections(original, corrected)
    # Heading source IDs survive because their complete groups are protected.
    outline = dict(headings=[deepcopy(s['heading']) for s in old_plan['sections']],
                   metadata=old_plan['metadata'], metadata_config={**job['metadata'],
                       'exclude_pages': old_plan.get('excluded_pages', [])},
                   toc_entries=old_plan.get('toc_entries', []),
                   exercise_heading_classification=old_plan.get('exercise_heading_classification', {}))
    for heading in outline['headings']:
        index.span(heading)
    save(root / 'reused_outline.json', outline)
    content = root / 'content'
    if (content / 'section_results.json').exists() and read(content / 'section_results.json')['source']['mineru_sha256'] != index.source['mineru_sha256']:
        content = root / ('content-' + index.source['mineru_sha256'][:10])
    prior_content = [p for p in root.glob('content*') if p.is_dir() and p != content]
    content_config = {**config, 'keyword_candidates': job.get('keyword_candidates'),
                      'shared_cache_dirs': [str(old_content / 'cache'), *(str(p / 'cache') for p in prior_content),
                                            *config.get('shared_cache_dirs', [])]}
    saved_report = content / 'section_results.json'
    if saved_report.exists() and read(saved_report).get('status') == 'complete':
        report = read(saved_report)
        if report['source']['mineru_sha256'] != index.source['mineru_sha256']:
            raise ValueError('Saved content belongs to other formula corrections')
    else:
        client = DeepSeekClient(content_config, content / 'cache')
        try:
            report = run_page_content(index, outline, client, content, seed=20261008, progress=progress)
        finally:
            if client._api is not None:
                client._api.close()
    if report['status'] != 'complete':
        raise ValueError('Content processing incomplete: ' + report['status'])
    old_rules = old_content / 'removed_special_blocks.json'
    if old_rules.exists():
        # Type cancellations are independent of source positions. Individual
        # cancellations are remapped below through their original addresses.
        rules = read(old_rules)
        from book_pipeline.structure.book_mapping import special_block_key
        key_map = {}
        for section in read(old_content / 'section_results.json')['sections']:
            sid = section['section']['id']
            for module in section['resolved']['s']:
                try:
                    new_lo = index.position(original.address(module['lo']))
                except ValueError:
                    continue
                key_map[special_block_key(sid, module)] = special_block_key(sid, {**module, 'lo': new_lo})
        remapped = []
        for block in rules.get('blocks', []):
            if block['key'] not in key_map:
                raise ValueError('A manually cancelled block cannot be mapped to repaired source')
            remapped.append({**block, 'key': key_map[block['key']]})
        save(content / 'removed_special_blocks.json', {**rules, 'blocks': remapped})
    mapping = root / 'references'
    run_rule_mapping(index, content, mapping, progress=progress)
    render_config = {**config, 'block_colors': job.get('colors', {}), 'typography': job.get('typography')}
    result = render_saved_book(index, content, mapping, job['metadata'], render_config, root / 'pdf', progress)
    result['formula_corrections'] = dict(regions=len(corrected['records']), usage=corrected['usage'],
                                        path=str(correction_path.resolve()))
    result['content_usage'] = report['usage']
    result['previous_content_usage'] = [dict(path=str(p.resolve()), usage=read(p / 'section_results.json')['usage'])
                                       for p in prior_content if (p / 'section_results.json').exists()]
    result['total_new_cost'] = round(corrected['usage']['estimated_cost'] + report['usage']['estimated_cost'] +
                                    sum(p['usage']['estimated_cost'] for p in result['previous_content_usage']), 8)
    save(root / 'result.json', result)
    print('完成：' + str(root / 'result.json'), flush=True)
    if update_job and result['status'] == 'complete':
        current = read(job_path)
        revision = int(current.get('revision', 0)) + 1
        book = read(mapping / 'edited_text_index.json')
        blocks = [dict(id=t['id'], kind='remark' if t['kind'] == 'other' else t['kind'],
                       label=t['label'], name=t['name'], page=t['page'], section=t['context']['heading_title'],
                       text=t['body_preview'], removable=t.get('origin') == 'deepseek_section_module',
                       **({'pattern': heading_pattern(t)} if t['kind'] == 'subheading' else {}))
                  for t in book['targets'] if t['kind'] in COLOR_KINDS | {'other', 'subheading'}]
        save(job_path.parent / 'blocks.json', blocks)
        current.update(mineru=job['mineru'], formula_corrections=str(correction_path.resolve()),
                       formula_correction_statistics=result['formula_corrections'], headings=None,
                       content=str(content.resolve()), references=str(mapping.resolve()),
                       render_directory=result['output'], render_pdf=result['pdf'], render_pages=result['pages'],
                       revision=revision, status='complete', updated=datetime.now().astimezone().isoformat(),
                       render_content_stale=False, preview_status='idle', preview_samples=[],
                       block_policy_version=SPECIAL_BLOCK_POLICY,
                       block_revision=int(current.get('block_revision') or 0) + 1,
                       statistics=dict(headings=len(book['sections']), blocks=len(book['blocks']), special_blocks=len(blocks)),
                       rendered_colors=job.get('colors', {}), rendered_typography=job.get('typography'))
        for stage in current['stages']:
            stage['status'] = 'complete'
            if stage['id'] == 'render':
                stage.update(detail='公式横带已校正，同排图片已恢复横排', done=result['pages'], total=result['pages'])
        save(job_path, current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['recognize', 'render', 'all'], default='all')
    parser.add_argument('--job', type=Path, action='append')
    parser.add_argument('--update-jobs', action='store_true')
    args = parser.parse_args()
    config = load_config()
    config['pricing'] = {**config['pricing'], 'checked_on': '2026-10-10'}
    for job in args.job or sorted((ROOT / 'output/web').glob('*/job.json')):
        rebuild(job, args.phase, config, args.update_jobs)


if __name__ == '__main__':
    main()
