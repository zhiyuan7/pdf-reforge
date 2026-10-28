"""A clear image-only label must survive an uncertain neighboring formula."""
import unittest
from unittest.mock import patch
import numpy as np
from book_pipeline.publishing.raster_body_styles import (
    analyze_page, label_windows, numbered_prose_contexts, owned_ranges)


class ScanLabelWindowTests(unittest.TestCase):
    def collect(self, reading='Property 7:', confidence=.999, *, calibration=False,
                row='Property 7: X', word='Property'):
        rgb = np.full((30, 135, 3), 255, np.uint8)
        rgb[5:25, 5:55] = 0
        class Model:
            def ocr(self, pixels, det=True):
                if det:
                    return [[([[2, 2], [133, 2], [133, 28], [2, 28]],
                              (row, .97))]]
                width = pixels.shape[1]
                value, score = {60: (word, .999), 30: ('7:', .999),
                    45: ('X', .999), 90: (reading, confidence)}.get(width, ('noise', .97))
                return [[(value, score)]]
        module = 'book_pipeline.publishing.raster_body_styles.'
        bold = dict(tiro=.3, tibo=.92, tiit=.3, tibi=.4)
        with patch(module+'word_boxes', return_value=[(0, 60), (60, 90), (90, 135)]), \
             patch(module+'complete_vertical_box', return_value=[0, 0, 135, 30]), \
             patch(module+'complete_line_box', return_value=[0, 0, 135, 30]), \
             patch(module+'completed_word_observation', return_value=None), \
             patch(module+'context_windows', return_value=[]), \
             patch(module+'scored_scan_word', side_effect=lambda m, g, w: (w, bold)):
            return analyze_page(rgb, Model(), calibration=calibration)

    def test_short_label_retains_independent_word_ink_and_exact_owner(self):
        observed = self.collect()
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]['text'], 'Property 7:')
        self.assertEqual(observed[0]['bbox'], [0, 0, 90, 30])
        self.assertEqual([w['text'] for w in observed[0]['words']], ['Property'])
        contexts = numbered_prose_contexts(observed[0], [
            dict(atom='a', text='Property 7: '), dict(math=True)])
        self.assertEqual(len(contexts), 1)
        items = [dict(atom='a', text='Property 7: ', bbox=[0, 0, 1, 1])]
        result = owned_ranges(observed, items, [135, 30]*2)
        self.assertEqual([(r['quote'], r['bold']) for r in result['a']], [('Property', True)])
        # No proposed word/font score is sent to the image-only collector.
        raw = self.collect(calibration=True)
        self.assertEqual(raw[0]['words'][0]['scores'], {})

    def test_number_first_example_uses_its_own_independent_word(self):
        observed = self.collect('2.4 Example.', row='2.4 Example. X', word='Example.')
        item = dict(atom='a', text='2.4 Example. Let ', bbox=[0, 0, 1, 1])
        result = owned_ranges(observed, [item], [135, 30]*2)
        self.assertEqual(result['a'][0]['quote'], 'Example.')
        self.assertEqual(result['a'][0]['start'], 4)

    def test_uncertain_incomplete_or_formula_bearing_short_context_abstains(self):
        for text, score in [('Property 7:', .98), ('Property 7:', float('nan')),
                            ('Property 7:', 1.01), ('Property 7', .999),
                            ('Property 7: X', .999), ('Property', .999)]:
            self.assertEqual(self.collect(text, score), [], (text, score))

    def test_independent_number_mismatch_cannot_authorize_source_label(self):
        line = self.collect('Property 8:')[0]
        group = [dict(atom='a', text='Property 7: '), dict(math=True)]
        self.assertEqual(numbered_prose_contexts(line, group), [])
        self.assertEqual(owned_ranges([line], [dict(atom='a', text='Property 7: ',
            bbox=[0, 0, 1, 1])], [135, 30]*2), {})

    def test_windows_are_bounded_and_never_cut_connected_ink(self):
        mask = np.zeros((20, 500), np.uint8)
        for a, b in [(5, 45), (55, 75), (85, 125), (135, 170), (180, 220)]:
            mask[3:17, a:b] = 255
        windows = label_windows(mask)
        self.assertLessEqual(len(windows), 3)
        self.assertTrue(windows)
        self.assertTrue(all(not mask[:, end].any() and end-start <= 320
                            for start, end in windows))
        self.assertEqual(label_windows(np.zeros((20, 0), np.uint8)), [])


if __name__ == '__main__':
    unittest.main()
