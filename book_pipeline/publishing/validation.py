"""Helpers used by the current PDF Reforge workflow."""
from __future__ import annotations

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
        for note in coverage.get('footnotes', []):
            destination = destinations[note['anchor']]
            reference = destinations.get(note['reference_anchor']) if note['status']=='matched' else None
            footnotes.append(dict(id=note['id'], status=note['status'], source_page=note['page'],
                                  rendered_page=destination['page']+1,
                                  reference_page=reference['page']+1 if reference else None,
                                  same_page=reference['page']==destination['page'] if reference else None))
        return {"pages": pages, "internal_links": links, "bookmarks": len(doc.get_toc()), "named_destinations": len(destinations),
                "missing_destinations": [], "broken_links": [], "outside_page": outside,
                'footnotes':footnotes, 'footnote_page_mismatches':[n for n in footnotes if n['same_page'] is False]}
