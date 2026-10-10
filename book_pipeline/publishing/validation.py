"""Helpers used by the current PDF Reforge workflow."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import re


def footnote_number_records(aux):
    """Read the number printed on the same pass that shipped each reference."""
    if not Path(aux).is_file():
        return {}
    records = {}
    for name, properties in re.findall(r'\\zref@newlabel\{([^{}]+)\}\{((?:\\\w+\{[^{}]*\})+)\}',
                                      Path(aux).read_text(encoding='utf-8')):
        fields = dict(re.findall(r'\\(abspage|bookfootnotenumber)\{(\d+)\}', properties))
        if fields.keys() >= {'abspage', 'bookfootnotenumber'}:
            records[name] = dict(page=int(fields['abspage']), number=int(fields['bookfootnotenumber']))
    return records

def audit_pdf(pdf, coverage, registry=()):
    import pymupdf
    pages, outside, links, broken = [], [], 0, []
    with pymupdf.open(pdf) as doc:
        destinations = doc.resolve_names()
        missing = (set(coverage["anchors"]) | set(coverage.get("source_anchors", {}).values())) - set(destinations)
        if missing:
            raise ValueError("PDF missing named destinations: " + ", ".join(sorted(missing)[:5]))
        for n, page in enumerate(doc):
            for block in page.get_text("dict", clip=pymupdf.INFINITE_RECT())["blocks"]:
                rect = pymupdf.Rect(block["bbox"])
                if rect.x0 < -1 or rect.y0 < -1 or rect.x1 > page.rect.width + 1 or rect.y1 > page.rect.height + 1:
                    outside.append({"page": n + 1, "bbox": list(rect)})
            for link in page.get_links():
                if link["kind"] == pymupdf.LINK_GOTO or (link["kind"] == pymupdf.LINK_NAMED and "nameddest" in link):
                    links += 1
                    if not 0 <= link.get("page", -1) < len(doc):
                        broken.append({"page": n + 1, "link": str(link)})
                    if "nameddest" in link and link["nameddest"] not in destinations:
                        broken.append({"page": n + 1, "missing_name": link["nameddest"]})
            pages.append({"page": n + 1, "width": page.rect.width, "height": page.rect.height})
        if broken:
            raise ValueError("PDF contains broken internal jumps")
        expected_headings = [o for o in registry if o["kind"] in ("part", "chapter", "section", "subsection", "subsubsection", "paragraph", "exercise_group")]
        actual_headings = doc.get_toc(simple=False)
        if registry and (len(expected_headings) != len(actual_headings) or any(
                row[3].get("page") != destinations[o["anchor"]]["page"] or
                ("nameddest" in row[3] and row[3]["nameddest"] != o["anchor"])
                for row, o in zip(actual_headings, expected_headings))):
            raise ValueError("PDF bookmark does not point to its source heading")
        footnotes = []
        number_records = footnote_number_records(Path(pdf).with_suffix('.aux'))
        numbering_by_page = defaultdict(list)
        for note in coverage.get('footnotes', []):
            destination = destinations[note['anchor']]
            reference = destinations.get(note['reference_anchor']) if note['status']=='matched' else None
            record = dict(id=note['id'], status=note['status'], source_page=note['page'],
                                  rendered_page=destination['page']+1,
                                  reference_page=reference['page']+1 if reference else None,
                                  same_page=reference['page']==destination['page'] if reference else None)
            if note.get('numbering') == 'rendered_page_from_one':
                printed = number_records.get(note['reference_anchor'])
                if reference is None:
                    raise ValueError('脚注缺少正文引用位置：' + note['id'])
                if not record['same_page']:
                    raise ValueError(f"脚注与正文引用不在同一页：{note['id']}，"
                                     f"正文第 {record['reference_page']} 页，脚注第 {record['rendered_page']} 页。")
                if not printed or printed['page'] != record['reference_page']:
                    raise ValueError('脚注的最终页码或编号未稳定：' + note['id'])
                number = printed['number']
                marker = chr(0x245F+number) if note['marker_style']=='circled' and 1 <= number <= 20 else str(number)
                record.update(rendered_number=number, rendered_marker=marker,
                              original_marker=note.get('original_marker'))
                numbering_by_page[record['rendered_page']].append((note['reference_order'], number, note['id']))
            else:
                record.update(rendered_number=None, rendered_marker=note.get('original_marker'),
                              original_marker=note.get('original_marker'))
            footnotes.append(record)
        for page, rows in numbering_by_page.items():
            if [n for _, n, _ in sorted(rows)] != list(range(1, len(rows)+1)):
                raise ValueError(f'Footnote numbering must start at one on rendered page {page}')
        return {"pages": pages, "internal_links": links, "bookmarks": len(doc.get_toc()), "named_destinations": len(destinations),
                "missing_destinations": [], "broken_links": [], "outside_page": outside,
                'footnotes':footnotes, 'footnote_page_mismatches':[n for n in footnotes if n['same_page'] is False],
                'footnote_numbering_pages':len(numbering_by_page)}
