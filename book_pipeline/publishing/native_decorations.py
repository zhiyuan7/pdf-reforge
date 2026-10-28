"""Recover bounded source rules and terminal marks from live PDF geometry.

The PDF is evidence only. Scans and edited/ambiguous text retain normal flow;
neither a declaration's wording nor a file name authorizes a decoration.
"""
from collections import OrderedDict
from pathlib import Path


def native_decorations(index, changed=()):
    import pymupdf

    source = index.source.get('source_pdf')
    if (not source or not Path(source).is_file()
            or not any(a.get('extraction_method') == 'native' for a in index.atoms)):
        return {}, {}
    groups = OrderedDict()
    for atom in index.atoms:
        groups.setdefault(atom['group'], []).append(atom)
    marks, rules = {}, {}
    with pymupdf.open(source) as pdf:
        pages, spans_by_page = {}, {}
        for atoms in groups.values():
            last = atoms[-1]
            if (any(a.get('extraction_method') != 'native' or a['id'] in changed
                    or a.get('source_panel') or a['page'] != last['page']
                    or a.get('block_type') not in {'text', 'paragraph', 'paragraph_title', 'equation'}
                    for a in atoms) or not 0 <= last['page'] < len(pdf)):
                continue
            boxes = [a.get('block_bbox') or a.get('bbox') for a in atoms]
            if not all(b and len(b) == 4 for b in boxes):
                continue
            page = pdf[last['page']]
            box = pymupdf.Rect(min(b[0] for b in boxes)*page.rect.width,
                              min(b[1] for b in boxes)*page.rect.height,
                              max(b[2] for b in boxes)*page.rect.width,
                              max(b[3] for b in boxes)*page.rect.height)
            pages.setdefault(last['page'], []).append((last, box))
            if last['kind'] != 'text' or last['content'].rstrip()[-1:] not in {'■', '□', '▪'}:
                continue
            symbol = last['content'].rstrip()[-1]
            if last['page'] not in spans_by_page:
                spans_by_page[last['page']] = [s for b in page.get_text('dict')['blocks']
                    for line in b.get('lines', []) for s in line['spans']]
            spans = [s for s in spans_by_page[last['page']] if s['text'].strip() == symbol
                     and (box+(-.1,-.1,.1,.1)).contains(pymupdf.Rect(s['bbox']))]
            # Inline explanations of a square and formula symbols fail these
            # right-edge, width and final-line constraints.
            spans = [s for s in spans if s['bbox'][0] > page.rect.width*.8
                     and s['bbox'][2]-s['bbox'][0] <= 2*s['size']
                     and abs(s['bbox'][3]-box.y1) <= s['size']]
            if len(spans) != 1:
                continue
            start = len(last['content'].rstrip())-1
            while start and last['content'][start-1].isspace():
                start -= 1
            marks[last['id']] = dict(method='native_right_edge_terminal_mark',
                atom=last['id'], start=start, end=len(last['content']),
                symbol=symbol, bbox=list(spans[0]['bbox']), page=last['page']+1)
        for number, candidates in pages.items():
            page = pdf[number]
            drawings = page.get_drawings()
            for drawing in drawings:
                if (len(drawing['items']) != 1 or drawing['items'][0][0] != 'l'
                        or drawing.get('fill') is not None or not drawing.get('color')
                        or max(drawing['color']) > .65
                        or max(drawing['color'])-min(drawing['color']) > .02
                        or not .6 <= drawing['width'] <= 4):
                    continue
                rect = drawing['rect']
                if rect.height > .1 or rect.width < page.rect.width*.65:
                    continue
                # Closely repeated full-width rules usually enclose table
                # rows. Their ownership needs table recovery, not a separator.
                if any(other is not drawing and other['rect'].height < .1
                       and abs(other['rect'].x0-rect.x0) < 1
                       and abs(other['rect'].x1-rect.x1) < 1
                       and abs(other['rect'].y0-rect.y0) < page.rect.height*.045
                       for other in drawings):
                    continue
                owners = [(a, b, 'after', rect.y0-b.y1) for a, b in candidates
                          if max(drawing['color']) <= .1
                          and 0 < rect.y0-b.y1 < page.rect.height*.025
                          and rect.x0-.5 <= b.x0 < b.x1 <= rect.x1+.5]
                owners.extend((index.atoms[index.groups[a['group']][0]],b,'before',b.y0-rect.y0)
                              for a,b in candidates if a.get('block_type') == 'paragraph_title'
                              and 0 < b.y0-rect.y0 < page.rect.height*.025
                              and rect.x0-.5 <= b.x0 < b.x1 <= rect.x1+.5)
                if not owners:
                    continue
                atom, box, placement, _ = min(owners, key=lambda row:row[3])
                # Reject a rule that also crosses body content or belongs to
                # a table/frame. Header rules cannot acquire a body owner.
                if ((atom['id'], placement) in rules or any(b.y0 <= rect.y0 <= b.y1
                        and b.x0 < rect.x1 and rect.x0 < b.x1 for _, b in candidates)):
                    continue
                rules[atom['id'],placement] = dict(method='native_horizontal_source_rule',
                    atom=atom['id'], bbox=list(rect), thickness_pt=drawing['width'],
                    color=[round(c,4) for c in drawing['color']], placement=placement,page=number+1)
    return marks, rules
