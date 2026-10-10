import tempfile
import unittest
from pathlib import Path

import pymupdf

from book_pipeline.publishing.validation import audit_pdf


class FootnotePaginationTests(unittest.TestCase):
    def make_pdf(self, directory, reference_page, footnote_page):
        path = Path(directory) / 'main.pdf'
        with pymupdf.open() as document:
            document.new_page()
            document.new_page()
            rows = []
            for name, page in [('book-note', footnote_page), ('ref-note', reference_page)]:
                if page is not None:
                    rows.append(f'({name}) [{document.page_xref(page-1)} 0 R /XYZ 30 700 null]')
            names = document.get_new_xref()
            document.update_object(names, '<< /Names [' + ' '.join(rows) + '] >>')
            document.xref_set_key(document.pdf_catalog(), 'Names', f'<< /Dests {names} 0 R >>')
            document.save(path)
        if reference_page is not None:
            path.with_suffix('.aux').write_text(
                r'\zref@newlabel{ref-note}{\abspage{' + str(reference_page)
                + r'}\bookfootnotenumber{1}}', encoding='utf-8')
        coverage = dict(anchors=['book-note'], footnotes=[dict(
            id='note', anchor='book-note', reference_anchor='ref-note', status='matched',
            page=105, numbering='rendered_page_from_one', marker_style='circled',
            original_marker='①', reference_order=1)])
        return path, coverage

    def test_same_page_footnote_retains_physical_page_and_number(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf, coverage = self.make_pdf(directory, 2, 2)
            result = audit_pdf(pdf, coverage)
        self.assertEqual(result['footnote_page_mismatches'], [])
        note = result['footnotes'][0]
        self.assertEqual((note['reference_page'], note['rendered_page'], note['rendered_number']), (2, 2, 1))

    def test_moved_footnote_reports_both_actual_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf, coverage = self.make_pdf(directory, 1, 2)
            with self.assertRaisesRegex(ValueError, '正文第 1 页，脚注第 2 页'):
                audit_pdf(pdf, coverage)

    def test_missing_reference_is_distinct_from_a_page_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf, coverage = self.make_pdf(directory, None, 2)
            with self.assertRaisesRegex(ValueError, '缺少正文引用位置'):
                audit_pdf(pdf, coverage)


if __name__ == '__main__':
    unittest.main()
