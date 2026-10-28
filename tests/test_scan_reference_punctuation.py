"""Reference punctuation needs row ownership, unchanged digits and visible ink."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
from PIL import Image

from book_pipeline.config import digest
from book_pipeline.ingestion.scan_reference_punctuation import (
    dot_ink, observe_references, restore_reference_dots)


class ScanReferencePunctuationTests(unittest.TestCase):
    def setUp(self):
        self.answer = dict(items=[dict(kind='formula', latex=r'\text{Since } x/y=1,\ z/y=0 \text{by (23),}', number='')])
        self.rows = [dict(id=7, pixel_bbox=[0,0,1400,50],
                          words=[dict(text='Since', score=.96), dict(text='by', score=.90)])]
        self.observation = dict(row=7, reference='2.3', reader_signature='cpu-model-hash',
            readings=[dict(reference='2.3', score=.98)]*2, ink=dict(dot_bbox=[30,30,34,34]))

    def test_only_inserts_dot_with_a_unique_independent_plain_word_anchor(self):
        before = deepcopy(self.answer)
        output, changes = restore_reference_dots(self.answer, self.rows, [self.observation])
        self.assertEqual(output['items'][0]['latex'], before['items'][0]['latex'].replace('(23)', '(2.3)'))
        self.assertEqual(self.answer, before)
        self.assertEqual(changes[0]['anchors'], ['since'])
        self.assertEqual(output['items'][0]['number'], '')

    def test_digit_replacement_low_confidence_missing_ink_and_ambiguity_abstain(self):
        for change in (dict(reference='2.4'), dict(reference='23'), dict(ink=None),
                       dict(reader_signature=''), dict(row=8),
                       dict(readings=[dict(reference='2.3', score=.95)]*2),
                       dict(readings=[dict(reference='2.3', score=.98)])):
            with self.subTest(change=change):
                self.assertEqual(restore_reference_dots(self.answer, self.rows,
                    [{**self.observation, **change}]), (self.answer, []))
        self.assertEqual(restore_reference_dots(self.answer,self.rows,[self.observation]*2),(self.answer,[]))
        for rows in ([*self.rows,dict(id=8,words=self.rows[0]['words'])],
                     [dict(id=7,words=[dict(text='Hence',score=.99)])]):
            self.assertEqual(restore_reference_dots(self.answer,rows,[self.observation]),(self.answer,[]))
        doubled = dict(items=self.answer['items']*2)
        self.assertEqual(restore_reference_dots(doubled,self.rows,[self.observation]),(doubled,[]))
        # No changes to references outside flat prose, or inside nested TeX.
        for latex in (r'\text{Since } x=1\quad (23)', r'\text{Since \textbf{by (23)}}'):
            answer = dict(items=[{**self.answer['items'][0], 'latex':latex}])
            self.assertEqual(restore_reference_dots(answer,self.rows,[self.observation]),(answer,[]))

    def test_dot_must_lie_between_separate_digit_glyphs_at_the_baseline(self):
        gray = np.full((50,120),255,np.uint8)
        for x in (10,30,70,90):
            gray[10:36,x:x+5] = 0
        gray[32:36,52:56] = 0
        witness = dot_ink(gray,'2.3')
        self.assertEqual(witness['dot_bbox'],[52,32,56,36])
        self.assertIsNone(dot_ink(gray,'23.4'))
        gray[32:36,52:56] = 255
        self.assertIsNone(dot_ink(gray,'2.3'))
        gray[15:19,52:56] = 0
        self.assertIsNone(dot_ink(gray,'2.3'))
        gray[15:19,52:56] = 255
        gray[32:36,52:56] = 0
        gray[32:36,61:65] = 0
        self.assertIsNone(dot_ink(gray,'2.3'))

    def test_unlabelled_scan_formulas_still_recover_dots_and_invalidate_old_policy(self):
        from book_pipeline.ingestion.scan_formula_anchors import recover_anchors, policy_signature
        from book_pipeline.config import fingerprint
        with patch('book_pipeline.ingestion.scan_formula_anchors.physical_rows',return_value=self.rows), patch(
                'book_pipeline.ingestion.scan_reference_punctuation.observe_references',return_value=[self.observation]):
            output, evidence = recover_anchors(self.answer,'source.png','rows')
        self.assertIn('(2.3)',output['items'][0]['latex'])
        self.assertEqual(len(evidence['reference_dots']),1)
        self.assertEqual(evidence['labels'],[])
        with patch('book_pipeline.publishing.latin_word_ocr.runtime_signature',return_value='runtime'), patch(
                'book_pipeline.ingestion.formula_text_anchors.policy_signature',return_value={'reader':'hash'}):
            before = fingerprint(policy_signature())
            with patch('book_pipeline.ingestion.scan_reference_punctuation.METHOD','new-dot-policy'):
                self.assertNotEqual(before,fingerprint(policy_signature()))

    def test_blind_two_view_reader_requires_same_reference_and_revalidates_pixels(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'row.png'
            Image.new('RGB',(1400,50),'white').save(path)
            row = dict(id=3,image=str(path),image_sha256=digest(path),pixel_bbox=[0,0,1400,50],
                words=[dict(text='Since',score=.99,bbox=[200,4,80,30]),
                       dict(text='by',score=.90,bbox=[1100,4,40,35]),
                       dict(text='(23),',score=.1,bbox=[1170,4,80,35])])
            reader = Mock()
            reader.ocr.return_value = [[(None,('by (2.3),',.98))]]
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(reader,'cpu-hash')), patch(
                    'book_pipeline.ingestion.scan_reference_punctuation.dot_ink',return_value={'dot_bbox':[1,2,3,4]}):
                results = observe_references([row],Path(name)/'observations')
                self.assertEqual(len(results),1)
                self.assertEqual(results[0]['reference'],'2.3')
                self.assertEqual(reader.ocr.call_count,2)
                for call in reader.ocr.call_args_list:
                    self.assertEqual(len(call.args),1)
                    self.assertEqual(call.kwargs,{})
                reader.ocr.side_effect = [[[(None,('by (2.3),',.98))]], [[(None,('by (2.4),',.98))]]]
                self.assertEqual(observe_references([row],Path(name)/'disagree'),[])
                reader.ocr.reset_mock()
                Image.new('RGB',(1400,50),'black').save(path)
                self.assertEqual(observe_references([row],Path(name)/'tampered'),[])
                reader.ocr.assert_not_called()


if __name__ == '__main__':
    unittest.main()
