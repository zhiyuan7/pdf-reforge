"""Offline assembly must account for every page without losing source assets."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.config import digest, load_config, read, save
from book_pipeline.ingestion.source import load_source
from book_pipeline.ingestion.source_assembly import assemble_source
from mineru_checkpoint import export_hashes, export_signature


class SourceAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.pdf = self.root/'book.pdf'
        with pymupdf.open() as doc:
            for text in ('Native source prose', 'CPU formula source', 'Last native page'):
                doc.new_page().insert_text((50, 80), text)
            doc.save(self.pdf)
        self.config = load_config()
        self.config['budget']['ledger'] = str(self.root/'never-created.sqlite3')
        self.versions = dict(mineru_version='test-1', docvortex_version='test-2')
        self.stack.enter_context(patch('book_pipeline.ingestion.ocr_cache.runtime_versions', return_value=self.versions))
        for method in ('reserve', 'reserve_amount'):
            self.stack.enter_context(patch('book_pipeline.ai.budget.BudgetGuard.'+method,
                                         side_effect=AssertionError('No paid requests')))
        self.native, self.ocr, self.output = (self.root/n for n in ('native', 'ocr', 'assembled'))
        self.write_source(self.native, [1, 3], native=True)
        self.write_source(self.ocr, [2])

    def write_source(self, directory, numbers, native=False):
        directory.mkdir()
        image = directory/'formula.png'
        with pymupdf.open(self.pdf) as doc:
            doc[0].get_pixmap().save(image)
        pages = [dict(page_idx=n-1, blocks=[dict(index=0, type='equation', bbox=[.1,.2,.8,.3],
            content=r'x^2+1', image_path='formula.png')]) for n in numbers]
        save(directory/'middle_json.json', dict(schema='docvortex.middle', schema_version='2.0',
                                              is_full_document=False, pages=pages))
        manifest = dict(input_sha256=digest(self.pdf), source_page_count=3, source_page_numbers=numbers)
        if native:
            manifest.update(method='cache_only_native_recovery_v1', canonical_budget_job=digest(self.pdf),
                middle_sha256=digest(directory/'middle_json.json'),
                pages=[dict(page=n, method='native', status='recovered') for n in numbers])
        else:
            signature = export_signature(digest(self.pdf), numbers, self.config['mineru_tier'],
                self.config['mineru_ocr_mode'], self.config['mineru_model_source'], self.versions)
            manifest.update({k:v for k,v in signature.items() if k != 'method'})
            manifest.update(export_signature=signature, export_hashes=export_hashes(directory))
        save(directory/'manifest.json', manifest)

    def assemble(self, **kwargs):
        return assemble_source(self.pdf, self.native, [self.ocr], self.output, self.config, **kwargs)

    def test_full_coverage_preserves_formula_assets_and_originals(self):
        originals = {p:p.read_bytes() for d in (self.native, self.ocr) for p in d.iterdir()}
        result = self.assemble()
        self.assertTrue(result['is_full_document'])
        self.assertEqual(result['source_page_numbers'], [1, 2, 3])
        self.assertEqual([p['method'] for p in result['pages']], ['native','ocr','native'])
        self.assertEqual(result['canonical_budget_job'], digest(self.pdf))
        source = load_source(self.pdf, self.output, 'all')
        self.assertEqual([a['content'] for a in source['atoms']], [r'x^2+1']*3)
        self.assertEqual(len(set(a['asset'] for a in source['atoms'])), 2)
        self.assertEqual({p:p.read_bytes() for p in originals}, originals)
        self.assertFalse((self.root/'never-created.sqlite3').exists())
        self.assertEqual(result['quality_status'], 'not_reviewed')
        self.assertEqual(result['formula_correction_status'], 'pending')

    def test_missing_pages_require_explicit_partial_scope(self):
        with self.assertRaisesRegex(ValueError, 'Missing source pages'):
            assemble_source(self.pdf, self.native, [], self.output, self.config)
        self.assertFalse(self.output.exists())
        result = assemble_source(self.pdf, self.native, [], self.output, self.config, allow_partial=True)
        self.assertFalse(result['is_full_document'])
        self.assertEqual(result['source_page_numbers'], [1, 3])

    def test_overlap_is_rejected_before_output(self):
        other = self.root/'overlap'
        self.write_source(other, [1])
        with self.assertRaisesRegex(ValueError, 'Overlapping source pages'):
            assemble_source(self.pdf, self.native, [other], self.output, self.config)
        self.assertFalse(self.output.exists())

    def test_changed_native_bytes_are_rejected(self):
        middle = self.native/'middle_json.json'
        middle.write_text(middle.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'Native recovery signature'):
            self.assemble()

    def test_changed_ocr_asset_or_settings_are_rejected(self):
        self.config['mineru_tier'] = 'advanced'
        with self.assertRaisesRegex(ValueError, 'OCR'):
            self.assemble()
        self.config['mineru_tier'] = 'standard'
        (self.ocr/'formula.png').write_bytes(b'changed image')
        with self.assertRaisesRegex(ValueError, 'OCR'):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_budget_identity_and_output_overlap_are_rejected(self):
        self.config['_budget_job_id'] = 'another-book'
        with self.assertRaisesRegex(ValueError, 'budget identity'):
            self.assemble()
        with self.assertRaisesRegex(ValueError, 'separate from its inputs'):
            assemble_source(self.pdf, self.native, [self.ocr], self.native/'output', self.config)


if __name__ == '__main__':
    unittest.main()
