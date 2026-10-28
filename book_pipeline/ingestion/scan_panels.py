"""Pixel evidence for monochrome panels, including scans with hidden OCR."""

METHOD = 'closed_monochrome_scan_panels_v1'
PAGE_METHOD = 'raster_page_hidden_text_only_v1'
DPI = 150


def raster_page(page):
    """A visible text layer stays native; hidden OCR never vetoes a scan."""
    if not page.get_text().strip():
        return True
    import pymupdf
    area = page.rect.get_area()
    coverage = max(((pymupdf.Rect(i['bbox']) & page.rect).get_area()/area
                    for i in page.get_image_info()), default=0)
    if coverage < .8:
        return False
    spans = page.get_texttrace()
    return bool(spans) and all(s['type'] == 3 or s.get('opacity') == 0 for s in spans)


def monochrome_regions(page):
    """Require four connected sides and a flat interior, without inventing titles.

    Text/code ownership and visual classification are checked separately. An
    outlined table may be a geometric candidate but cannot become a prose box.
    """
    import cv2
    import numpy as np
    import pymupdf
    pix = page.get_pixmap(dpi=DPI, alpha=False, colorspace=pymupdf.csRGB)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = (gray < 140).astype(np.uint8)
    connected = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates, grids = [], []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (w > .22*pix.width and .025*pix.height < h < .75*pix.height
                and .006 < w*h/(pix.width*pix.height) < .7):
            continue
        corners = cv2.approxPolyDP(contour, .012*cv2.arcLength(contour, True), True)
        if len(corners) != 4 or cv2.contourArea(contour)/(w*h) < .94:
            continue
        # Every side must be observed along most of its length. Text line
        # clusters, partial frames, photographs and rounded shadows abstain.
        pad = max(3, round(min(w, h)*.025))
        crop = ink[y:y+h, x:x+w]
        sides = [crop[:pad].max(0).mean(), crop[-pad:].max(0).mean(),
                 crop[:, :pad].max(1).mean(), crop[:, -pad:].max(1).mean()]
        if min(sides) < .85:
            continue
        interior = rgb[y+pad:y+h-pad, x+pad:x+w-pad].astype(np.int16)
        background = np.median(interior.reshape(-1, 3), axis=0)
        flat = (np.abs(interior-background).max(2) <= 16).mean()
        if flat < .75 or background.max()-background.min() > 6 or background.min() < 210:
            continue
        # Internal rules would give the box a table/grid interpretation.
        inner = crop[pad:-pad, pad:-pad]
        box = [float(x/pix.width), float(y/pix.height), float((x+w)/pix.width), float((y+h)/pix.height)]
        if (inner.mean(0) > .65).any() or (inner.mean(1) > .65).any():
            grids.append(box)
            continue
        candidates.append(dict(bbox=box, title_bbox=None, color=[.1, .1, .1],
            background_color=[round(float(v/255), 4) for v in background], frame_color=[.1, .1, .1],
            frame_visible=True, method=METHOD, evidence=dict(dpi=DPI,
                side_coverage=[round(float(v), 4) for v in sides], flat_background_coverage=round(float(flat), 4))))
    regions = []
    for panel in sorted(candidates, key=lambda p: -(p['bbox'][2]-p['bbox'][0])*(p['bbox'][3]-p['bbox'][1])):
        from book_pipeline.ingestion.panels import intersection_fraction
        if any(intersection_fraction(panel['bbox'], grid) > .95 for grid in grids):
            continue  # A closed table cell is not an independent prose panel.
        if not any(intersection_fraction(panel['bbox'], p['bbox']) > .95 for p in regions):
            regions.append(panel)
    return sorted(regions, key=lambda p: (p['bbox'][1], p['bbox'][0]))
