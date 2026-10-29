"""Offline recovery cannot send requests or equate previews with extraction."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.ai.client import DeepSeekClient
from book_pipeline.config import load_config, read
from book_pipeline.ingestion.native import native_page, spans_for
from book_pipeline.ingestion.native_checkpoint import recover_native_checkpoint
from book_pipeline.ingestion.source import load_source


class NativeCheckpointTests(unittest.TestCase):
    def test_only_valid_cached_pages_and_blanks_are_recovered(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            pdf, original, output = root/'book.pdf', root/'original', root/'recovered'
            with pymupdf.open() as doc:
                doc.new_page().insert_text((60, 80), 'A complete cached paragraph.')
                doc.new_page().insert_text((60, 80), 'An uncached paragraph remains pending.')
                doc.new_page()
                doc.save(pdf)
            config = load_config()
            config['budget']['ledger'] = str(root/'never-created.sqlite3')
            def response(task, messages, settings):
                import json
                return dict(content=json.dumps(dict(blocks=[dict(type='text', bbox=[.05,.05,.95,.2],
                    parts=[dict(kind='text', spans=['s0'])])])), finish_reason='stop', usage={})
            # Populate a real signed request cache without reserving money.
            config_without_budget = {**config, 'budget': None}
            client = DeepSeekClient(config_without_budget, original/'cache', transport=response)
            with pymupdf.open(pdf) as doc:
                native_page(doc[0], spans_for(doc[0]), client, original, config['image_dpi'])
            before = {p.name:p.read_bytes() for p in (original/'cache').iterdir()}
            with patch('book_pipeline.ai.budget.BudgetGuard.reserve', side_effect=AssertionError('reservation')):
                result = recover_native_checkpoint(pdf, original/'cache', output, config)
            self.assertEqual(result['source_page_numbers'], [1, 3])
            self.assertFalse(result['is_full_document'])
            self.assertEqual(result['pages'][1]['reason'], 'required_response_not_cached')
            self.assertFalse((root/'never-created.sqlite3').exists())
            self.assertEqual(before, {p.name:p.read_bytes() for p in (original/'cache').iterdir()})
            source = load_source(pdf, output)
            self.assertEqual([p['page'] for p in source['pages']], [0, 2])
            self.assertIn('complete cached', source['atoms'][0]['content'])
            self.assertEqual(read(output/'manifest.json')['quality_status'], 'not_reviewed')
            # Changing a request setting invalidates the old cached response.
            changed = recover_native_checkpoint(pdf, original/'cache', root/'changed',
                                                {**config, 'model':config['model']+'-changed'})
            self.assertEqual(changed['source_page_numbers'], [3])

    def test_refuses_to_overwrite_cache_or_existing_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cache = root/'cache'; cache.mkdir()
            for target in [root, cache, cache/'child']:
                with self.subTest(target=target), self.assertRaises(ValueError):
                    recover_native_checkpoint(root/'book.pdf', cache, target, load_config())

    def test_cached_layout_without_required_hyphen_review_stays_pending(self):
        import json
        from book_pipeline.ingestion.native_hyphens import candidates_for
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            pdf, original = root/'book.pdf', root/'original'
            with pymupdf.open() as doc:
                doc.new_page().insert_text((60, 80), 'The source paragraph describes the trans-\nformation.')
                doc.save(pdf)
            config = load_config()
            layout = dict(blocks=[dict(type='text', bbox=[.05,.05,.95,.3],
                                      parts=[dict(kind='text', spans=['s0','s1'])])])
            def response(*args):
                return dict(content=json.dumps(layout), finish_reason='stop', usage={})
            seed = DeepSeekClient({**config, 'budget':None}, original/'cache', transport=response)
            with pymupdf.open(pdf) as doc:
                spans = spans_for(doc[0])
                self.assertEqual(len(candidates_for(layout['blocks'], {s['id']:s for s in spans})), 1)
                # Simulate interruption after storing layout but before review.
                with patch('book_pipeline.ingestion.native.review_hyphens',
                           return_value=({s['id']:s for s in spans}, set(), {})):
                    native_page(doc[0], spans, seed, original, config['image_dpi'])
            with patch('book_pipeline.ai.budget.BudgetGuard.reserve', side_effect=AssertionError('reservation')):
                result = recover_native_checkpoint(pdf, original/'cache', root/'recovered', config)
            self.assertEqual(result['source_page_numbers'], [])
            self.assertEqual(result['pages'][0]['reason'], 'required_response_not_cached')


if __name__ == '__main__':
    unittest.main()
