"""Running heads need native ink, a rule, and advancing neighboring folios."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.structure.recognized_headings import recognized_heading_registry
from book_pipeline.structure.running_headers import running_header_rows


class RunningHeaderTests(unittest.TestCase):
    def source(self, root, *, rule=True, folios=True, advancing=True, broad=False):
        path = root/'source.pdf'
        with pymupdf.open() as pdf:
            for n in range(5):
                p = pdf.new_page(width=500, height=700)
                p.insert_text((70, 80), '4.2 Running topic', fontsize=10)
                if folios:
                    p.insert_text((430, 80), str(20+n if advancing else 20), fontsize=10)
                if rule:
                    p.draw_line((70, 84), (445, 84))
                p.insert_text((70, 115), 'This is ordinary body text.', fontsize=10)
                p.insert_text((70, 180), '4.2 The actual section', fontsize=12)
            pdf.save(path)
        atoms = []
        for n, (text, box, role) in enumerate([
            ('4.2译后页眉22', [.13,.095,.9,.175 if broad else .12], 'paragraph_title'),
            ('4.2 真正的小节', [.13,.235,.7,.26], 'paragraph_title'),
            ('正文仍须保留。', [.13,.148,.8,.17], 'text')]):
            atoms.append(dict(id=str(n),group=str(n),page=2,kind='text',role=role,
                block_type=role,content=text,length=len(text),bbox=box))
        return SourceIndex(dict(source_pdf=str(path),atoms=atoms))

    def test_translated_header_excluded_but_real_heading_and_body_remain(self):
        with tempfile.TemporaryDirectory() as name:
            index = self.source(Path(name))
            rows = running_header_rows(index)
            self.assertEqual([r['group'] for r in rows], ['0'])
            self.assertEqual(rows[0]['witness_pages'], [1,2,4,5])
            self.assertEqual([r['group'] for r in recognized_heading_registry(index)], ['1'])
            self.assertEqual(index.atoms[0]['content'], '4.2译后页眉22')

    def test_split_header_group_can_omit_folio_in_translated_text(self):
        with tempfile.TemporaryDirectory() as name:
            index = self.source(Path(name))
            index.atoms[0]['bbox'] = [.16,.095,.37,.12]
            index.atoms[0]['content'] = '译后标题'
            rows = running_header_rows(index)
            self.assertEqual([r['group'] for r in rows], ['0'])

    def test_no_exclusion_without_independent_marginal_evidence(self):
        for options in (dict(rule=False), dict(folios=False), dict(advancing=False), dict(broad=True)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as name:
                self.assertEqual(running_header_rows(self.source(Path(name), **options)), [])

    def test_missing_pdf_or_scanned_ink_cannot_prove_header(self):
        with tempfile.TemporaryDirectory() as name:
            index = self.source(Path(name))
            Path(index.source['source_pdf']).unlink()
            self.assertEqual(running_header_rows(index), [])
            with pymupdf.open() as pdf:
                for _ in range(5):
                    pdf.new_page(width=500,height=700)
                pdf.save(index.source['source_pdf'])
            self.assertEqual(running_header_rows(index), [])
