"""A font metric needs observed blank borders and a fresh word reading."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from book_pipeline.publishing.raster_body_styles import (
    analyze_page, complete_vertical_box, completed_word_observation)
from book_pipeline.publishing.scan_font_calibration import complete_word_ink


class ScanWordBorderTests(unittest.TestCase):
    def fixture(self):
        rgb = np.full((70, 140, 3), 255, np.uint8)
        rgb[22:40, 25:45] = 0
        word = dict(text='mapping', score=.999, bbox=[20, 20, 80, 40], scores={})
        return rgb, word

    def test_complete_edge_glyph_gets_blank_pixels_without_changing_legacy_strip(self):
        rgb, word = self.fixture()
        self.assertEqual(complete_vertical_box(rgb, word['bbox']), word['bbox'])
        reader = Mock()
        reader.ocr.return_value = [[('mapping', .999)]]
        fixed = completed_word_observation(rgb, reader, word)
        x, y, z, t = fixed['bbox']
        self.assertGreater(t, word['bbox'][3])
        self.assertFalse(complete_word_ink(255-rgb[20:40, 20:80, 0]))
        self.assertTrue(complete_word_ink(255-rgb[y:t, x:z, 0]))
        self.assertEqual(word['bbox'], [20, 20, 80, 40])
        reader.ocr.assert_called_once()
        self.assertEqual(reader.ocr.call_args.kwargs, dict(det=False))
        self.assertEqual(reader.ocr.call_args.args[0].shape, (t-y, z-x, 3))

    def test_unrelated_ink_or_search_edge_cannot_authorize_padding(self):
        for defect in ('neighbor', 'continues'):
            rgb, word = self.fixture()
            if defect == 'neighbor':
                rgb[41:44, 60:64] = 0
            else:
                rgb[30:55, 25:45] = 0
            reader = Mock()
            self.assertIsNone(completed_word_observation(rgb, reader, word), defect)
            reader.ocr.assert_not_called()

    def test_rereading_cannot_change_word_or_raise_confidence(self):
        rgb, word = self.fixture()
        for rows in ([('mopping', .999)], [('mapping', .98)],
                     [('mapping', float('nan'))], [('mapping', 1.01)],
                     [('map', .999), ('ping', .999)], []):
            reader = Mock()
            reader.ocr.return_value = [rows]
            self.assertIsNone(completed_word_observation(rgb, reader, word))
        reader.ocr.return_value = [[('mapping', .995)]]
        self.assertEqual(completed_word_observation(rgb, reader, word)['score'], .995)

    def test_calibration_retains_original_and_reread_observations(self):
        rgb, word = self.fixture()
        text = 'A mapping defines this distribution'
        poly = [[20, 20], [80, 20], [80, 40], [20, 40]]
        class Reader:
            def ocr(self, image, det=True):
                return [[(poly, (text, .999))]] if det else [[('mapping', .999)]]
        module = 'book_pipeline.publishing.raster_body_styles.'
        with patch(module+'word_boxes', return_value=[(0, 64)]), \
             patch(module+'complete_vertical_box', return_value=word['bbox']), \
             patch(module+'complete_line_box', return_value=word['bbox']), \
             patch(module+'completed_word_observation', return_value=dict(word, bbox=[20, 20, 80, 44])):
            observed = analyze_page(rgb, Reader(), calibration=True)
            self.assertTrue(any(w['bbox'][3] == 44 for row in observed for w in row['words']))
            self.assertTrue(any(w['bbox'][3] != 44 for row in observed for w in row['words']))


if __name__ == '__main__':
    unittest.main()
