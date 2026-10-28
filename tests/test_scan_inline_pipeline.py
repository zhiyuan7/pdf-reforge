"""A CPU-owned inline statement must reach the blind letter consensus stage."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.ingestion.formula_bands import correct_formula_bands
from book_pipeline.ingestion.scan_formula_anchors import restore_left_labels
from book_pipeline.ingestion.source import SourceIndex


class ScanInlinePipelineTests(unittest.TestCase):
    def test_cpu_statement_ownership_precedes_consensus_and_policy_changes_invalidate_cache(self):
        paragraph = 'The printed statement preserves every ordinary word in this paragraph.'
        base = dict(items=[dict(kind='formula', number='12',
            latex=r'\text{If } x/d=2,\ x/dk=2 \text{ if } dk \text{ is consistent.}'),
            dict(kind='text', text=paragraph)])
        first = dict(items=[dict(kind='text',
            text='(12) If $x/h=2$. $x/dk=2$ if $dk$ is consistent.'), dict(kind='text',text=paragraph)])
        second = dict(items=[dict(kind='text',
            text='(12) If $x/h=2$, $x/dk=2$ if $dk$ is consistent.'), dict(kind='text',text=paragraph)])
        calls = []

        class Client:
            _api = None
            def __init__(self, *args, **kwargs):
                self.ledger = self
                self.reads = 0
            def summary(self):
                return {}
            def ask(self, task, prompt, payload, **kwargs):
                calls.append(task)
                if task.startswith('literal_scan_band'):
                    answer = dict(edits=[],unresolved=[])
                elif task == 'independent_scan_prose_reading_v1':
                    answer = second
                else:
                    self.reads += 1
                    answer = base if self.reads == 1 else first
                kwargs['validator'](answer)
                return answer

        rows = [dict(id=0,pixel_bbox=[0,0,1500,50],words=[
            dict(text='(12)',score=.99,bbox=[100,0,40,30]),
            dict(text='If',score=.99,bbox=[160,0,30,30]),
            dict(text='is',score=.99),dict(text='consistent.',score=.99)])]

        def anchors(answer, *args):
            output, changes = restore_left_labels(answer,rows)
            return output, dict(labels=changes)

        with tempfile.TemporaryDirectory() as name, pymupdf.open() as pdf:
            path = Path(name)/'source.pdf'
            pdf.new_page(width=400,height=300)
            pdf.save(path)
            atom = dict(id='a',group='a',page=0,kind='display_math',content='x/d=2',length=5,
                        extraction_method='ocr')
            index = SourceIndex(dict(atoms=[atom],input_sha256='p',mineru_sha256='m',source_pdf=str(path)))
            band = dict(id='b',page=0,bbox=[0,.2,1,.5],groups=['a'],preserve=[])
            config = dict(model='fake',base_url='https://example.invalid',thinking='disabled',concurrency=1)
            module = 'book_pipeline.ingestion.formula_bands.'
            with patch(module+'detect_bands',side_effect=lambda *args:[dict(band)]), \
                 patch(module+'DeepSeekClient',Client), \
                 patch('book_pipeline.ingestion.scan_formula_anchors.policy_signature',return_value={}), \
                 patch('book_pipeline.ingestion.scan_formula_anchors.recover_anchors',side_effect=anchors) as cpu:
                directory = Path(name)/'corrections'
                report = correct_formula_bands(index,config,directory)
                record = report['records'][0]
                self.assertEqual(record['answer']['items'][0]['text'],
                    r'(12) If $x/h=2,\ x/dk=2$ if $dk$ is consistent.')
                self.assertEqual(record['answer']['items'][1],base['items'][1])
                self.assertEqual(len(record['inline_consensus']['accepted']),1)
                self.assertFalse(record['prose_consensus']['proposed'])
                self.assertEqual(len(record['scan_anchors']['labels']),1)
                self.assertEqual(cpu.call_count,1)
                self.assertEqual(len(calls),4)
                correct_formula_bands(index,config,directory)
                self.assertEqual(len(calls),4)
                with patch('book_pipeline.ingestion.scan_inline_consensus.METHOD','changed-policy'):
                    correct_formula_bands(index,config,directory)
                self.assertEqual(len(calls),8)


if __name__ == '__main__':
    unittest.main()
