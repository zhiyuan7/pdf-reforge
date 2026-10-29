"""Recover available native pages without credentials, network or reservations."""
from pathlib import Path

from book_pipeline.ai.client import DeepSeekClient, DecisionUncertain
from book_pipeline.config import digest, fingerprint, save
from book_pipeline.ingestion.native import classify_page, native_page, normalize_marginals, spans_for


class CacheMiss(RuntimeError):
    pass


class CacheOnlyClient(DeepSeekClient):
    def _connect(self):
        # ask() checks and validates response caches before connecting or
        # reserving money. Missing/invalid responses must stop at that boundary.
        raise CacheMiss('No valid response for the current request signature')


def recover_native_checkpoint(pdf, cache_directory, output, config, progress=lambda *args: None):
    """Write an explicitly partial source; never mutate the interrupted run.

    Request signatures include the PDF spans, page image, prompt and model.
    A cached layout alone is insufficient if a required hyphen review is absent.
    OCR/fallback pages remain pending rather than being inferred from previews.
    """
    import pymupdf

    pdf, cache_directory, output = (Path(p).resolve() for p in (pdf, cache_directory, output))
    if (output.exists() or pdf.is_relative_to(output)
            or output.is_relative_to(cache_directory) or cache_directory.is_relative_to(output)):
        raise ValueError('Recovery requires a new output separate from the PDF and response cache')
    if not cache_directory.is_dir():
        raise ValueError('Response cache is missing')
    source_sha = digest(pdf)
    config = {**config, '_budget_job_id': config.get('_budget_job_id', source_sha),
              'shared_cache_dirs': [str(cache_directory)]}
    client = CacheOnlyClient(config, output / 'cache')
    pages, records = [], []
    with pymupdf.open(pdf) as document:
        if document.needs_pass:
            raise ValueError('Encrypted PDF')
        count, bookmarks = len(document), document.get_toc()
        for page in document:
            spans = spans_for(page)
            mode, reason = classify_page(page, spans)
            if config.get('input_mode') == 'ocr' and mode != 'blank':
                mode, reason = 'ocr', 'forced_ocr'
            record = dict(page=page.number+1, method=mode, reason=reason, status='pending')
            recovered = None
            if mode == 'blank':
                recovered = dict(page_idx=page.number, blocks=[])
            elif mode == 'native':
                try:
                    recovered = native_page(page, spans, client, output, config['image_dpi'])
                except CacheMiss:
                    record['reason'] = 'required_response_not_cached'
                except DecisionUncertain:
                    record['reason'] = 'cached_layout_requires_ocr'
            if recovered is not None:
                record['status'] = 'recovered'
                pages.append(recovered)
                # Preserve completed work even if later CPU processing fails.
                save(output / 'recovered_pages' / f'{page.number+1:06d}.json', recovered)
            records.append(record)
            progress('离线恢复缓存页', page.number+1, count)
    if digest(pdf) != source_sha:
        raise RuntimeError('PDF changed during recovery')
    normalize_marginals(pages)
    complete = len(pages) == count
    save(output / 'middle_json.json', dict(schema='docvortex.middle', schema_version='2.0',
         is_full_document=complete, pages=pages))
    manifest = dict(method='cache_only_native_recovery_v1', input_pdf=str(pdf), input_sha256=source_sha,
                    canonical_budget_job=config['_budget_job_id'], source_page_count=count,
                    source_page_numbers=[p['page_idx']+1 for p in pages], is_full_document=complete,
                    pages=records, bookmarks=bookmarks, middle_sha256=digest(output / 'middle_json.json'),
                    response_cache=str(cache_directory), usage=client.ledger.summary(),
                    translation_status='pending', quality_status='not_reviewed',
                    formula_correction_status='pending')
    manifest['signature'] = fingerprint(dict(pdf=source_sha, config=config, method=manifest['method']))
    save(output / 'manifest.json', manifest)
    return manifest
