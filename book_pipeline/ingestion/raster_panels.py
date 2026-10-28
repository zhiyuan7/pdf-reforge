"""Conservative CPU recovery of flat colored panels in image-only pages.

Color geometry supplies the boundary; independently extracted title and body
blocks must corroborate it. Photos, tables, code and partial/mixed ownership
remain unchanged. Raw OCR and original PDFs are never rewritten.
"""
from copy import deepcopy
from pathlib import Path

from book_pipeline.config import digest, fingerprint, read, save
from book_pipeline.ingestion.panels import intersection_fraction
from book_pipeline.ingestion.scan_panels import raster_page, monochrome_regions


METHOD = 'raster_color_text_panels_v3'
DPI = 110


def color_regions(page):
    import cv2
    import numpy as np
    import pymupdf

    pix = page.get_pixmap(dpi=DPI, alpha=False, colorspace=pymupdf.csRGB)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).astype(np.int16)
    mask = ((rgb.max(2)-rgb.min(2) > 6) & (rgb.max(2) > 120)).astype(np.uint8)
    # Join a colored title and body across a thin *ink* separator. A white
    # gutter between neighboring boxes is never filled by this operation.
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    separator_ink = rgb.max(2) < 225
    connected = mask | (closed & separator_ink.astype(np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(connected, 8)
    regions = []
    for x, y, w, h, count in stats[1:]:
        if not (w/pix.width > .22 and h/pix.height > .018
                and .002 < count/(pix.width*pix.height) < .8 and count/(w*h) > .65):
            continue
        pixels = rgb[y:y+h, x:x+w]
        background = np.median(pixels.reshape(-1, 3), axis=0)
        # A contrasting title band is observed in row medians, independently
        # of the OCR title's wording. Ignore rounded corners at either end.
        rows = np.median(pixels[:, w//10:9*w//10], axis=1)
        differing = np.abs(rows-background).max(1) > 25
        strip_end = 0
        chromatic = (rows.max(1) > 120) & (rows.max(1)-rows.min(1) > 6)
        starts = np.flatnonzero((differing & chromatic)[:max(2, h//10)])
        for start in starts:
            end = int(start)
            while end < h and differing[end]:
                end += 1
            # Ignore a one-pixel antialias halo at the top of the panel.
            if end-start >= max(2, .04*h) and .06*h <= end <= .45*h:
                strip_end = end
                break
        box = [float(x/pix.width), float(y/pix.height),
               float((x+w)/pix.width), float((y+h)/pix.height)]
        title_box = ([box[0], box[1], box[2], float((y+strip_end)/pix.height)]
                     if .06*h <= strip_end <= .45*h else None)
        body_pixels = pixels[strip_end:] if title_box else pixels
        uniform = (np.abs(body_pixels-background).max(2) <= 12).mean()
        if uniform < .8:
            continue
        ink = (np.median(rows[:strip_end], axis=0) if title_box else background)
        regions.append(dict(bbox=box, title_bbox=title_box,
                            color=[round(float(v/255), 4) for v in ink], method=METHOD,
                            evidence=dict(dpi=DPI, color_coverage=round(float(count/(w*h)), 4),
                                          flat_background_coverage=round(float(uniform), 4))))
    return sorted(regions, key=lambda p: (p['bbox'][1], p['bbox'][0]))


def panel_regions(page):
    return color_regions(page) + monochrome_regions(page)


def annotate_raster_panels(page, blocks):
    """Require an entire title/body sequence and reject crossing objects."""
    from book_pipeline.ingestion.source import flatten
    if not raster_page(page):
        return []
    evidence = []
    for n, panel in enumerate(panel_regions(page), 1):
        panel['id'] = f'p{page.number+1:06d}-raster-panel-{n:03d}'
        row = dict(panel=deepcopy(panel), page=page.number+1, status='rejected')
        overlapping = [b for b in blocks if intersection_fraction(b.get('bbox'), panel['bbox']) > .05]
        owned = [b for b in overlapping if intersection_fraction(b.get('bbox'), panel['bbox']) >= .9]
        titles = [b for b in owned if b['type'] in {'paragraph_title', 'doc_title'}]
        body = [b for b in owned if b['type'] in {'text', 'paragraph', 'equation'}]
        if any(b['type'] not in {'text', 'paragraph', 'paragraph_title', 'doc_title', 'equation'}
               for b in overlapping):
            row['reason'] = 'unsupported_content_kind'
        elif len(overlapping) != len(owned):
            row['reason'] = 'partial_source_ownership'
        elif (not body or sum(len(flatten(b).strip()) for b in owned) < 12
              or len(titles) != (1 if panel['title_bbox'] else 0)):
            row['reason'] = 'missing_independent_title_and_body'
        elif panel['title_bbox'] and intersection_fraction(titles[0].get('bbox'), panel['title_bbox']) < .8:
            row['reason'] = 'title_not_in_observed_strip'
        elif any(b.get('source_panel') for b in owned):
            row['reason'] = 'existing_panel_ownership'
        else:
            positions = [blocks.index(b) for b in owned]
            if (positions != list(range(min(positions), max(positions)+1))
                    or (titles and owned[0] is not titles[0])
                    or (not titles and not all(b.get('panel_prose_reviewed') for b in owned))):
                row['reason'] = 'noncontiguous_source_ownership'
            else:
                for block in owned:
                    block.update(source_panel=deepcopy(panel), source_panel_title=bool(titles) and block is titles[0])
                row.update(status='annotated', groups=[b['index'] for b in owned])
        evidence.append(row)
    return evidence


def recover_raster_panels(pdf, origin, directory, config=None, progress=lambda *args: None, client=None):
    """Apply after source selection, so reused OCR receives the same checks."""
    import pymupdf
    from book_pipeline.ingestion.source import SourceIndex, load_source, materialize_source, MATERIALIZATION_METHOD

    pdf, origin, directory = Path(pdf).resolve(), Path(origin).resolve(), Path(directory).resolve()
    if directory == origin or directory.is_relative_to(origin):
        raise ValueError('Raster panel recovery requires a separate output directory')
    manifest = read(origin/'manifest.json')
    middle = read(origin/'middle_json.json')
    source = dict(pdf=digest(pdf), middle=digest(origin/'middle_json.json'))
    if manifest['input_sha256'] != source['pdf']:
        raise ValueError('Raster panel input SHA mismatch')
    from book_pipeline.ingestion.raster_prose import review_algorithm_panels, policy_signature
    from book_pipeline.ingestion.scan_panels import METHOD as MONO_METHOD, PAGE_METHOD, DPI as MONO_DPI
    signature = fingerprint(dict(method=METHOD, dpi=DPI, source=source,
                                 monochrome_method=MONO_METHOD, monochrome_dpi=MONO_DPI, raster_page_method=PAGE_METHOD,
                                 materialization_method=MATERIALIZATION_METHOD,
                                 prose_policy=policy_signature(config)))
    target = directory/'manifest.json'
    if target.exists():
        saved = read(target)
        if (saved.get('raster_panel_recovery', {}).get('signature') != signature
                or saved.get('middle_sha256') != digest(directory/'middle_json.json')):
            raise ValueError('Raster panel cache changed; use an independent output directory')
        return directory
    reviews, prose_reviews = [], []
    index = SourceIndex(load_source(pdf, origin))
    with pymupdf.open(pdf) as document:
        for page in middle['pages']:
            index, page['blocks'], reviewed = review_algorithm_panels(
                document[page['page_idx']], page['blocks'], index, panel_regions(document[page['page_idx']]),
                config, directory/'prose_review', progress, client)
            prose_reviews.extend(reviewed)
            reviews.extend(annotate_raster_panels(document[page['page_idx']], page['blocks']))
    save(directory/'review.json', dict(method=METHOD, signature=signature, source=source,
                                      reviews=reviews, prose_reviews=prose_reviews))
    if not any(row['status'] == 'annotated' for row in reviews):
        return origin
    panels = {(p['page_idx'], b['index']): b for p in middle['pages'] for b in p['blocks'] if b.get('source_panel')}
    for atom in index.atoms:
        block = panels.get((atom['page'], atom.get('raster_block_index',
                              int(atom['group'].split('-b', 1)[1].split('-')[0]))))
        if block:
            atom.update(source_panel=block['source_panel'], source_panel_title=block['source_panel_title'])
    if source != dict(pdf=digest(pdf), middle=digest(origin/'middle_json.json')):
        raise RuntimeError('Source changed during raster panel recovery')
    index.source['mineru_sha256'] = fingerprint(dict(signature=signature, reviews=reviews, prose_reviews=prose_reviews))
    materialize_source(index, directory)
    saved = read(target)
    saved['raster_panel_recovery'] = dict(method=METHOD, signature=signature, source=source,
                                         reviews=reviews, prose_reviews=prose_reviews)
    save(target, saved)
    return directory
