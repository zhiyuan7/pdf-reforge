"""Join signed native recovery and verified CPU exports without paid requests."""
from copy import deepcopy
from pathlib import Path
import shutil

from book_pipeline.config import digest, fingerprint, read, save
from book_pipeline.ingestion.native import annotate_panels, normalize_marginals
from book_pipeline.ingestion.ocr_cache import validate_ocr_cache
from book_pipeline.ingestion.source import load_source, safe_asset

METHOD = 'verified_native_cpu_assembly_v1'


def assemble_source(pdf, native, ocr_directories, output, config, *, allow_partial=False):
    """Keep physical page identities and refuse gaps, overlap or changed inputs.

    Partial output is explicitly declared and never promoted to a full book.
    The original exports remain immutable; copied assets get separate namespaces.
    """
    import pymupdf

    pdf, native, output = (Path(p).resolve() for p in (pdf, native, output))
    sources = [native, *(Path(p).resolve() for p in ocr_directories)]
    if (output.exists() or pdf.is_relative_to(output)
            or any(output.is_relative_to(p) or p.is_relative_to(output) for p in sources)):
        raise ValueError('Assembly requires a new output separate from its inputs')
    sha = digest(pdf)
    with pymupdf.open(pdf) as document:
        count, bookmarks = len(document), document.get_toc()
    pages, records, provenance, assets = {}, {}, [], []
    for serial, directory in enumerate(sources):
        manifest = read(directory/'manifest.json')
        middle = read(directory/'middle_json.json')
        selection = manifest.get('source_page_numbers')
        if (manifest.get('input_sha256') != sha or manifest.get('source_page_count') != count
                or not isinstance(selection, list) or selection != sorted(set(selection))
                or any(type(n) is not int or not 1 <= n <= count for n in selection)
                or middle.get('schema') != 'docvortex.middle' or middle.get('schema_version') != '2.0'
                or [p['page_idx']+1 for p in middle['pages']] != selection):
            raise ValueError('Source PDF, page selection or schema mismatch')
        if serial == 0:
            if (manifest.get('method') != 'cache_only_native_recovery_v1'
                    or manifest.get('middle_sha256') != digest(directory/'middle_json.json')
                    or manifest.get('canonical_budget_job') != config.get('_budget_job_id', sha)):
                raise ValueError('Native recovery signature or budget identity mismatch')
            original_records = {r['page']: r for r in manifest['pages'] if r['status'] == 'recovered'}
            if set(original_records) != set(selection):
                raise ValueError('Native recovery records do not match recovered pages')
        else:
            validate_ocr_cache(pdf, directory, config, selection)
            original_records = {n: dict(page=n, method='ocr', reason='verified_cpu_export') for n in selection}
        if set(selection).intersection(pages):
            raise ValueError('Overlapping source pages; choose disjoint exports explicitly')

        def copy_plan(node):
            if isinstance(node, list):
                for child in node:
                    copy_plan(child)
            elif isinstance(node, dict):
                if node.get('image_path'):
                    asset = Path(safe_asset(directory, node['image_path']))
                    target = Path('assets')/f'source-{serial:03d}'/asset.relative_to(directory)
                    assets.append((asset, target, digest(asset)))
                    node['image_path'] = str(target)
                for child in node.values():
                    if isinstance(child, (list, dict)):
                        copy_plan(child)

        copied = deepcopy(middle['pages'])
        copy_plan(copied)
        pages.update({p['page_idx']+1: p for p in copied})
        records.update(original_records)
        provenance.append(dict(directory=str(directory), manifest_sha256=digest(directory/'manifest.json'),
                               middle_sha256=digest(directory/'middle_json.json'), pages=selection))
    complete = set(pages) == set(range(1, count+1))
    if not complete and not allow_partial:
        raise ValueError('Missing source pages; full-book assembly remains pending')
    if not pages:
        raise ValueError('No recovered pages to assemble')
    with pymupdf.open(pdf) as document:
        for number, page in pages.items():
            annotate_panels(document[number-1], page['blocks'])
    normalize_marginals(list(pages.values()))
    for original, relative, expected_sha in assets:
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if digest(target) != expected_sha:
            raise ValueError('Source asset changed during assembly')
    if digest(pdf) != sha or any(digest(Path(p['directory'])/'manifest.json') != p['manifest_sha256']
            or digest(Path(p['directory'])/'middle_json.json') != p['middle_sha256'] for p in provenance):
        raise ValueError('Source changed during assembly')
    selected = sorted(pages)
    save(output/'middle_json.json', dict(schema='docvortex.middle', schema_version='2.0',
         is_full_document=complete, pages=[pages[n] for n in selected]))
    manifest = dict(method=METHOD, input_pdf=str(pdf), input_sha256=sha, source_page_count=count,
        source_page_numbers=selected, is_full_document=complete, bookmarks=bookmarks,
        canonical_budget_job=config.get('_budget_job_id', sha), pages=[records[n] for n in selected],
        provenance=provenance, middle_sha256=digest(output/'middle_json.json'),
        asset_hashes={str(p): s for _, p, s in assets}, translation_status='pending',
        formula_correction_status='pending', quality_status='not_reviewed')
    manifest['signature'] = fingerprint(manifest)
    save(output/'manifest.json', manifest)
    load_source(pdf, output, 'all' if complete else None)
    return manifest
