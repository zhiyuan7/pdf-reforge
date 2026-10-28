"""Isolate neighboring leaves only behind independently visible page edges.

The source PDF and OCR stay immutable. This derivative removes wholly external
OCR groups and bounds subsequent recognition crops; it does not rewrite text.
"""
from copy import deepcopy
from pathlib import Path

from book_pipeline.config import fingerprint, read, save
from book_pipeline.ingestion.source import SourceIndex, load_source, materialize_source

METHOD = 'paired_physical_scan_edges_v1'


def detect_frame(gray):
    import cv2
    import numpy as np
    height, width = gray.shape
    edges = cv2.Canny(gray, 10, 30)
    lines = cv2.HoughLinesP(edges, 1, np.pi/1800, 70,
                            minLineLength=round(height*.55), maxLineGap=55)
    if lines is None:
        return None
    sides = [[], []]
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        if abs(y1-y0) < height*.65 or abs(x1-x0) > abs(y1-y0)*.06:
            continue
        slope = (x1-x0)/(y1-y0)
        top = (x0-slope*y0)/width
        bottom = (x0+slope*(height-y0))/width
        if .025 < min(top, bottom) and max(top, bottom) < .2:
            sides[0].append([float(top), float(bottom)])
        elif .8 < min(top, bottom) and max(top, bottom) < .975:
            sides[1].append([float(top), float(bottom)])
    if not all(sides):
        return None
    # Stay inside all witnessed edge segments. A narrow safety inset excludes
    # the edge itself without deleting any content absent separate OCR proof.
    left = [max(e[k] for e in sides[0])+2/width for k in (0, 1)]
    right = [min(e[k] for e in sides[1])-2/width for k in (0, 1)]
    if min(right[k]-left[k] for k in (0, 1)) < .65:
        return None
    return dict(method=METHOD, left=left, right=right,
                witnessed_edges=sides, render_size=[width, height])


def horizontal_bounds(frame, top, bottom):
    def at(edge, y): return edge[0]+(edge[1]-edge[0])*y
    return (max(at(frame['left'], y) for y in (top, bottom)),
            min(at(frame['right'], y) for y in (top, bottom)))


def frame_groups(index, page, frame):
    """Require both physical edges and edge-clipped OCR before narrowing scope."""
    groups = [[index.atoms[n] for n in positions] for positions in index.groups.values()
              if index.atoms[positions[0]]['page'] == page]
    outside, crossing, inside = [], [], []
    for atoms in groups:
        box = atoms[0].get('block_bbox')
        if not box:
            continue
        left, right = horizontal_bounds(frame, box[1], box[3])
        if box[2] <= left or box[0] >= right:
            # Assets and footnotes outside a proposed leaf invalidate it.
            if any(a['kind'] not in {'text', 'math'} or a['role'] != 'text' for a in atoms):
                return None
            outside.append(atoms)
        else:
            inside.append(atoms)
            if box[0] < left or box[2] > right:
                crossing.append(atoms[0]['group'])
    if (not outside or len(inside) < 10
            or not any(a[0]['block_bbox'][0] < .01 or a[0]['block_bbox'][2] > .975 for a in outside)):
        return None
    external_chars = sum(a['length'] for group in outside for a in group)
    total_chars = sum(a['length'] for group in groups for a in group)
    if external_chars > .08*total_chars:
        return None
    return dict(frame=frame, excluded_groups=[a[0]['group'] for a in outside],
                excluded_text=[''.join(a['content'] for a in group) for group in outside],
                crossing_groups=crossing)


def recover_scan_page_frames(pdf, source_directory, directory, config=None, progress=lambda *args: None):
    import pymupdf
    import numpy as np
    origin, directory = Path(source_directory), Path(directory)
    manifest = read(origin/'manifest.json')
    if manifest.get('scan_page_frames'):
        return origin
    index = SourceIndex(load_source(pdf, origin))
    reviews = {}
    with pymupdf.open(pdf) as document:
        for page_info in index.source['pages']:
            number = page_info['page'];page = document[number]
            atoms = [a for a in index.atoms if a['page'] == number]
            # A captured leaf may carry an old OCR overlay. Extraction routing
            # and a full-page raster are authoritative; overlay text alone is
            # not evidence that this is native typography.
            if (not atoms or any(a.get('extraction_method') != 'ocr' for a in atoms)):
                continue
            # A physical page frame belongs to a captured raster leaf.
            if not any(pymupdf.Rect(i['bbox']).get_area() > .9*page.rect.get_area()
                       for i in page.get_image_info()):
                continue
            scale = 1000/page.rect.width
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY)
            frame = detect_frame(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width))
            if frame:
                review = frame_groups(index, number, frame)
                if review:
                    reviews[str(number)] = review
    if not reviews:
        return origin
    result = deepcopy(index.source)
    excluded = {g for review in reviews.values() for g in review['excluded_groups']}
    result['atoms'] = [a for a in result['atoms'] if a['group'] not in excluded]
    for atom in result['atoms']:
        review = reviews.get(str(atom['page']))
        if review and atom['group'] in review['crossing_groups']:
            for key in ('bbox', 'block_bbox'):
                if atom.get(key):
                    box = atom[key];left, right = horizontal_bounds(review['frame'], box[1], box[3])
                    atom[key] = [max(box[0], left), box[1], min(box[2], right), box[3]]
    result['pages'] = [{**p, 'atoms':[a['id'] for a in result['atoms'] if a['page']==p['page']]}
                       for p in result['pages']]
    result['mineru_sha256'] = fingerprint(dict(method=METHOD, source=index.source['mineru_sha256'], reviews=reviews))
    target = materialize_source(SourceIndex(result), directory)
    saved = read(target/'manifest.json')
    save(target/'manifest.json', {**saved, 'scan_page_frames':dict(method=METHOD, pages=reviews)})
    progress('隔离扫描邻页残片', len(reviews), len(reviews))
    return target
