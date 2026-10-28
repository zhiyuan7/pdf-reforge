"""Preserve local CJK sans emphasis surrounded by native calligraphic prose.

Some Chinese fonts have neither a bold flag nor 'Bold' in their name. Within
an otherwise calligraphic gray note a local switch to Heiti is emphasis. The
PDF font, exact original text and exclusive geometry must all agree; an
ordinary sans note, a heading or an OCR overlay is never such evidence.
"""
from pathlib import Path
import re

from book_pipeline.ingestion.panels import intersection_fraction
from book_pipeline.publishing.raster_styles import METHOD as STYLE_METHOD, eligible_items

METHOD = 'native_calligraphic_note_cjk_contrast_v1'
CJK = r'[\u3400-\u9fff]'


def font_family(name):
    # Common PostScript family names, including subsetted Chinese foundry
    # abbreviations. 'Sans' alone does not prove either Chinese or emphasis.
    name = re.sub(r'^[A-Z]{6}\+', '', name)
    if re.search(r'Kai(?:Ti)?|Kaiti|FZKT|WenKai', name, re.I):
        return 'calligraphic'
    if re.search(r'Hei(?:Ti)?|FZHT|SourceHanSans|NotoSansCJK', name, re.I):
        return 'sans'
    return None


def note_ranges(spans, items, region):
    from book_pipeline.publishing.raster_body_styles import literal_span
    owned = [s for s in spans if intersection_fraction(s['bbox'], region) >= .95]
    counts = {family: sum(len(re.findall(CJK, s['text'])) for s in owned
                          if font_family(s['font']) == family)
              for family in ('calligraphic', 'sans')}
    if counts['calligraphic'] < max(8, counts['sans']*2) or not counts['sans']:
        return {}
    result = {}
    for span in owned:
        if (font_family(span['font']) != 'sans'
                or not re.fullmatch(CJK + r'{2,20}', span['text'])):
            continue
        matches = [(item, literal_span(item['text'], span['text'])) for item in items
                   if intersection_fraction(span['bbox'], item['bbox']) >= .9]
        matches = [(i, hit) for i, hit in matches if hit is not None]
        if len(matches) != 1:
            continue
        item, (a, z) = matches[0]
        if intersection_fraction(item['bbox'], region) < .9:
            continue
        row = dict(method=STYLE_METHOD, start=a, end=z, quote=item['text'][a:z],
                   italic=False, bold=True, cjk_family='sans', observed_bbox=span['bbox'],
                   source_font=span['font'], evidence=METHOD)
        if row not in result.get(item['atom'], []):
            result.setdefault(item['atom'], []).append(row)
    return result


def recover_native_note_styles(index, original, excluded=()):
    import pymupdf
    from book_pipeline.publishing.raster_decorations import gray_regions, page_pixels
    source = index.source.get('source_pdf')
    candidates = eligible_items(index, original, excluded, extraction_method='native')
    candidates = {page: items for page, items in candidates.items()
                  if any(re.search(CJK, i['text']) for i in items)}
    report = dict(method=METHOD, status='not_needed', records=[])
    styles = {}
    if not candidates or not source or not Path(source).is_file():
        return styles, report
    with pymupdf.open(source) as pdf:
        for number, items in candidates.items():
            if not 0 <= number < len(pdf):
                continue
            page = pdf[number]
            # Only true native pages qualify. Old archive OCR layers cannot
            # supply reliable original font information for scanned printing.
            if any(i.get('extraction_method') == 'ocr' for i in original.atoms if i.get('page') == number):
                continue
            spans = [dict(text=s['text'], font=s['font'],
                          bbox=[v/d for v, d in zip(s['bbox'], [page.rect.width, page.rect.height]*2)])
                     for b in page.get_text('dict')['blocks'] for l in b.get('lines', []) for s in l['spans']]
            for region in gray_regions(page_pixels(page)):
                box = region['bbox']
                batch = [i for i in items if intersection_fraction(i['bbox'], box) >= .9]
                if not batch:
                    continue
                # A main paragraph can have a broad box around a wrapped
                # note. Native font spans still give exact text and glyph
                # geometry. Check all possible owners, requiring the unique
                # selected atom to belong wholly to the note itself.
                ranges = note_ranges(spans, items, box)
                for atom, runs in ranges.items():
                    styles.setdefault(atom, []).extend(r for r in runs if r not in styles.get(atom, []))
                if ranges:
                    report['records'].append(dict(page=number+1, note_bbox=box, ranges=ranges))
    if styles:
        report['status'] = 'requires_visual_review'
    return {a: sorted(runs, key=lambda r: r['start']) for a, runs in styles.items()}, report
