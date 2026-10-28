"""A unique image-only long successor can witness position, never edit text."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np

from book_pipeline.ingestion.formula_text_anchors import following_anchor, observe_prose
from book_pipeline.ingestion.formula_crops import exclusive_bounds


class FormulaTextAnchorTests(unittest.TestCase):
    def setUp(self):
        self.text = '故其乘數可以此式求之'
        self.observation = dict(text=self.text, score=.999, bbox=[170, 300, 210, 700],
                                reader_signature='independent-cpu-runtime')
        self.bounds = [120, 257, 260, 297]
        self.original = [200, 500, 290, 570]
        self.following = self.text + '其他後文'

    def witness(self, **kwargs):
        args = dict(bounds=self.bounds, original=self.original, following=self.following,
                    observations=[self.observation], source_texts=[self.following, '不相干正文'])
        return following_anchor(**{**args, **kwargs})

    def test_unique_long_successor_recovers_position_and_preserves_inputs(self):
        before = deepcopy(self.observation)
        result = self.witness()
        self.assertEqual(result['observation'], before)
        self.assertEqual(self.observation, before)
        self.assertEqual(result['candidate_bbox'], self.bounds)
        self.assertIsNone(self.witness(following='先前文字' + self.following))

    def test_short_low_confidence_duplicate_and_missing_runtime_abstain(self):
        for change in (dict(score=.98), dict(text='又'), dict(reader_signature=''),
                       dict(text='故其乘數可以彼式求之')):
            with self.subTest(change=change):
                self.assertIsNone(self.witness(observations=[{**self.observation, **change}]))
        self.assertIsNone(self.witness(observations=[self.observation]*2))
        self.assertIsNone(self.witness(source_texts=[self.following]*2))
        self.assertIsNone(self.witness(following=''))

    def test_far_nonadjacent_other_column_and_tall_formula_abstain(self):
        for bounds in ([120, 120, 260, 200], [120, 300, 260, 380],
                       [300, 257, 420, 297], [120, 50, 260, 297]):
            self.assertIsNone(self.witness(bounds=bounds))
        self.assertIsNone(self.witness(original=[350, 500, 440, 570]))
        self.assertIsNone(self.witness(observations=[{**self.observation, 'bbox':[170,300,570,340]}]))

    def test_reader_has_only_pixels_and_filters_math_and_horizontal_lines(self):
        model = Mock()
        valid = [[170,300],[210,300],[210,700],[170,700]]
        model.ocr.return_value = [[(valid,(self.text,.999)),(valid,('甲乙天',1.)),
            (valid,(self.text,.98)), ([[20,20],[420,20],[420,60],[20,60]],(self.text,.999))]]
        rgb = np.full((800,600,3),230,dtype=np.uint8)
        with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(model,'cpu-signature')):
            rows = observe_prose(rgb)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['text'],self.text)
        self.assertEqual(rows[0]['reader_signature'],'cpu-signature')
        model.ocr.assert_called_once()
        args, kwargs = model.ocr.call_args
        self.assertEqual(len(args),1)
        self.assertFalse(kwargs)
        np.testing.assert_array_equal(args[0],rgb[:,:,::-1])

    def test_text_witness_keeps_prose_and_other_owner_vetoes(self):
        rgb = np.full((800,600,3),255,dtype=np.uint8)
        boxes = [[140,260,240,295], [180,310,220,600]]
        choices = [dict(block=7,regions=[0],uncertain=False)]
        args = dict(anchors={7:self.original},text_context={7:self.following},
                    observations=[self.observation],source_texts=[self.following])
        evidence = {}
        with patch('book_pipeline.ingestion.formula_crops.ink_bounds',return_value=self.bounds):
            self.assertEqual(exclusive_bounds(rgb,boxes,choices,[0],anchor_evidence=evidence,**args)[7],self.bounds)
        self.assertIn(7,evidence)
        # Prose inside the bounding rectangle remains a hard veto.
        boxes.append([170,267,210,285])
        with patch('book_pipeline.ingestion.formula_crops.ink_bounds',return_value=self.bounds):
            self.assertIsNone(exclusive_bounds(rgb,boxes,choices,[0],**args)[7])
        # Another formula's ownership remains a hard veto as well.
        choices.append(dict(block=8,regions=[2],uncertain=True))
        with patch('book_pipeline.ingestion.formula_crops.ink_bounds',return_value=self.bounds):
            self.assertIsNone(exclusive_bounds(rgb,boxes,choices,[0,2],**args)[7])


if __name__ == '__main__':
    unittest.main()
