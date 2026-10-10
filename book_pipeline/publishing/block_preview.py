"""Render one saved, complete special block per type without mapping or a book PDF."""
from pathlib import Path

from book_pipeline.config import read, save
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.colors import COLOR_KINDS
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.environment import doctor
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.processed_book import EditedBook
from book_pipeline.publishing.typography import heading_pattern
from book_pipeline.structure.subheading_patterns import pattern_kind
from book_pipeline.structure.book_mapping import build_book


def select_samples(edited):
    """Keep the first logical object of each type, including all selected segments.

    The cached body_preview is only a search excerpt. Slice the edited source
    atoms instead so long text, native formulas and assets remain intact.
    """
    chosen = {}
    for item in edited.registry:
        if item['kind'] in COLOR_KINDS or item['kind'] == 'subheading':
            key = pattern_kind(heading_pattern(item)) if item['kind'] == 'subheading' else item['kind']
            chosen.setdefault(key, item)
    # A real ordinary paragraph makes body/heading size comparisons meaningful.
    for block in getattr(edited, 'book', {}).get('blocks', []):
        if block.get('content_role') == 'page_footnote':continue
        if not 30 <= len(block['text'].strip()) <= 400:
            continue
        lo, hi = edited.project(block['pieces'][0]['source_lo'], block['pieces'][-1]['source_hi'])
        if any(r['lo'] < hi and r['hi'] > lo for r in edited.registry):
            continue
        chosen['body'] = dict(id='body-reference', kind='body', lo=lo, hi=hi, label='正文对照',
                              page=block['page'], context=block['context'], anchor='book-body-reference')
        break
    atoms, registry, samples, breaks = [], [], [], []
    offset = 0
    order = lambda key: (0 if key == 'body' else 1 if key.startswith('subheading-') else 2, key)
    for kind in sorted(chosen, key=order):
        identifier = chosen[kind]['id'].split('-segment-')[0]
        pieces = [chosen[kind]] if kind == 'body' else [r for r in edited.registry if r['id'] == identifier or r['id'].startswith(identifier + '-segment-')]
        first = pieces[0]
        samples.append(dict(kind=kind, id=identifier, label=first['label'], name=first.get('name', ''),
                            source_page=first['page'], section=first['context']['heading_title'], anchor=first['anchor'],
                            content_kind=first['kind'], **({'pattern': heading_pattern(first)} if first['kind']=='subheading' else {})))
        breaks.append(first['anchor'])
        for piece in pieces:
            start = offset
            for atom, a, z in edited.index.fragments(piece['lo'], piece['hi']):
                content = atom['content'][a:z]
                new = {**atom, 'id': f'sample-{len(atoms) + 1:06d}',
                       'group': piece['id'] + ':' + atom['group'], 'content': content, 'length': z - a}
                atoms.append(new)
                offset += new['length']
            item = {**piece, 'lo': start, 'hi': offset}
            for field in ('question_ranges', 'answer_ranges'):
                if field in piece:
                    item[field] = [dict(lo=start + max(r['lo'], piece['lo']) - piece['lo'],
                                        hi=start + min(r['hi'], piece['hi']) - piece['lo'])
                                   for r in piece[field] if r['lo'] < piece['hi'] and r['hi'] > piece['lo']]
            item.pop('body_start', None)
            if piece.get('body_start'):
                item['_body_offset'] = start + edited.index.position(piece['body_start']) - piece['lo']
            registry.append(item)
    if not samples:
        raise ValueError('正文中未提取到可预览的特殊块。')
    index = SourceIndex({**edited.index.source, 'atoms': atoms})
    for item in registry:
        item.update(start=index.address(item['lo']), end=index.address(item['hi'], end=True))
        if '_body_offset' in item:
            item['body_start'] = index.address(item.pop('_body_offset'))
    return index, registry, samples, breaks


def render_block_preview(original, processing, metadata, config, output):
    import pymupdf
    from PIL import Image, ImageChops

    processing, output = Path(processing), Path(output)
    book = build_book(original, read(processing / 'split_source.json'),
                      read(processing / 'section_results.json'), processing)
    edited = EditedBook(original, book, processing)
    index, registry, samples, breaks = select_samples(edited)
    project = output / 'latex'
    decisions = dict(corrections=[], joins=[], exclusions=[], bold_spans=[])
    settings = {**config, 'float_assets': False, 'page_break_before_anchors': breaks}
    write_project(index, decisions, registry, [], metadata['book'], settings, project, sample_mode=True)
    save(output / 'reports' / 'samples.json', samples)
    save(output / 'reports' / 'sample_source.json', index.source)
    compilation = compile_project(project, doctor(config, require_key=False), timeout=180)
    save(output / 'reports' / 'compilation.json', compilation)
    if not compilation['success']:
        raise RuntimeError('特殊块预览编译失败，源码和编译记录已保存在 ' + str(output))
    with pymupdf.open(project / 'main.pdf') as document:
        names = document.resolve_names()
        starts = [names[s['anchor']]['page'] for s in samples] + [len(document)]
        for n, sample in enumerate(samples):
            sample['pages'] = list(range(starts[n] + 1, starts[n + 1] + 1))
            if not sample['pages']:
                raise RuntimeError('特殊块预览缺少渲染页面：' + sample['label'])
            for page_number in sample['pages']:
                pixmap = document[page_number - 1].get_pixmap(matrix=pymupdf.Matrix(1.7, 1.7), alpha=False)
                image = Image.frombytes('RGB', (pixmap.width, pixmap.height), pixmap.samples)
                bounds = ImageChops.difference(image, Image.new('RGB', image.size, 'white')).getbbox()
                if not bounds:
                    raise RuntimeError('特殊块预览页面为空：' + sample['label'])
                a, b, c, d = bounds
                # Keep the full book text-column width. Cropping a short title
                # horizontally would enlarge it in the UI and falsify size comparisons.
                margin = round(12.7 * 72 / 25.4 * 1.7)
                a, c = min(a, margin), max(c, image.width - margin)
                image = image.crop((max(0, a - 14), max(0, b - 14), min(image.width, c + 14), min(image.height, d + 14)))
                path = output / 'samples' / sample['kind'] / f'{page_number}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
    result = dict(samples=samples, colors=config.get('block_colors', {}), pdf=str(project / 'main.pdf'))
    save(output / 'preview_result.json', result)
    return result
