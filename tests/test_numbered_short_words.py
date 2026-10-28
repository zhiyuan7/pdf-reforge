"""Short formula-neighbor prose requires its own reading and ink evidence."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from book_pipeline.publishing.raster_body_styles import (
    analyze_page, numbered_prose_contexts, numbered_short_span, owned_ranges,
    supported_words)


class NumberedShortWordTests(unittest.TestCase):
    def fixture(self, *, word='In'):
        bold = dict(tiro=.3, tibo=.92, tiit=.3, tibi=.4)
        italic = dict(tiro=.3, tibo=.3, tiit=.91, tibi=.4)
        words = [dict(text='Property', score=.999, bbox=[5, 10, 45, 30], scores=bold),
                 dict(text=word, score=.999, bbox=[60, 10, 75, 30], scores=italic)]
        line = dict(text=f'Property 7: {word}', score=.999,
                    bbox=[5, 10, 75, 30], words=words)
        group = [dict(atom='a', text=f'Property 7: {word} '), dict(math=True)]
        return line, group

    def test_short_prefix_word_has_its_own_style_and_source_offsets(self):
        for word in ('In', 'If', 'on', 'An'):
            line, group = self.fixture(word=word)
            before = deepcopy(line)
            contexts = numbered_prose_contexts(line, group)
            self.assertEqual(contexts[0]['source'], [0, len(line['text'])])
            item = dict(atom='a', text=group[0]['text'], bbox=[0, 0, 1, 1],
                        line_contexts=contexts)
            recovered = owned_ranges([line], [item], [100]*4)
            self.assertEqual([r['quote'] for r in recovered['a']], ['Property', word])
            self.assertTrue(recovered['a'][0]['bold'])
            self.assertTrue(recovered['a'][1]['italic'])
            self.assertFalse(recovered['a'][1]['bold'])
            self.assertEqual(recovered['a'][1]['start'], len('Property 7: '))
            self.assertEqual(line, before)

    def test_label_does_not_supply_short_word_reading_or_style(self):
        for defect in ('missing', 'uncertain', 'nan', 'overconfident', 'upright',
                       'weak', 'wrong_word', 'unread_label', 'weak_label'):
            line, _ = self.fixture()
            if defect == 'missing': line['words'].pop()
            if defect == 'uncertain': line['words'][1]['score'] = .98
            if defect == 'nan': line['words'][1]['score'] = float('nan')
            if defect == 'overconfident': line['words'][1]['score'] = 1.01
            if defect == 'upright': line['words'][1]['scores']['tiro'] = .99
            if defect == 'weak': line['words'][1]['scores']['tiit'] = .5
            if defect == 'wrong_word': line['words'][1]['text'] = 'Or'
            if defect == 'unread_label': line['words'][0]['score'] = .98
            if defect == 'weak_label': line['words'][0]['scores']['tiro'] = .99
            self.assertNotIn('In', [w['text'] for w in supported_words(line['words'], line['text'])], defect)

    def test_partial_repeated_unlabeled_and_single_letter_words_abstain(self):
        line, _ = self.fixture()
        for text in ('In Property 7: In', 'Property 7: Inside', 'Property 7: Inner',
                     'A Property 7: In', 'Property: In', 'Property 7; In',
                     'Property 7:In', 'Property 7: A In', 'Property 7: In In'):
            self.assertIsNone(numbered_short_span(text, line['words'][1]), text)
        for word in ('A', 'X', 'for'):
            line, _ = self.fixture(word=word)
            self.assertIsNone(numbered_short_span(line['text'], line['words'][1]))

    def test_source_must_own_the_whole_short_word_before_formula(self):
        for source in ('Property 7: ', 'Property 7: Inside ', 'Property 7: Or '):
            line, group = self.fixture()
            group[0]['text'] = source
            context = numbered_prose_contexts(line, group)[0]
            self.assertEqual(context['source'], [0, len('Property 7:')])
            item = dict(atom='a', text=source, bbox=[0, 0, 1, 1], line_contexts=[context])
            result = owned_ranges([line], [item], [100]*4)
            self.assertEqual([r['quote'] for r in result['a']], ['Property'])

    def test_duplicate_prose_cannot_own_short_word(self):
        line, group = self.fixture()
        contexts = numbered_prose_contexts(line, group)
        item = dict(atom='a', text=group[0]['text'], bbox=[0, 0, 1, 1],
                    line_contexts=contexts)
        self.assertEqual(owned_ranges([line], [item, dict(item, atom='duplicate')], [100]*4), {})

    def test_number_first_prefix_supports_only_its_matching_short_word(self):
        line, group = self.fixture()
        line['text'] = '2.7 Example. In'
        line['words'][0]['text'] = 'Example.'
        group[0]['text'] = line['text']+' '
        self.assertEqual(numbered_prose_contexts(line, group)[0]['source'], [0, len(line['text'])])
        group[0]['text'] = '2.8 Example. In '
        self.assertEqual(numbered_prose_contexts(line, group), [])

    def test_uncertain_formula_row_uses_independent_bounded_prefix(self):
        rgb = np.full((30, 135, 3), 255, np.uint8)
        bold = dict(tiro=.3, tibo=.92, tiit=.3, tibi=.4)
        italic = dict(tiro=.3, tibo=.3, tiit=.91, tibi=.4)
        class Reader:
            def ocr(self, pixels, det=True):
                if det:
                    return [[([[0, 0], [135, 0], [135, 30], [0, 30]],
                              ('Property 7: In X', .97))]]
                reading = {60: ('Property', .999), 30: ('7:', .999),
                           45: ('In', .999), 90: ('Property 7:', .999),
                           135: ('Property 7: In', .999)}[pixels.shape[1]]
                return [[reading]]
        module = 'book_pipeline.publishing.raster_body_styles.'
        for calibration in (False, True):
            with patch(module+'word_boxes', return_value=[(0, 60), (60, 90), (90, 135)]), \
                 patch(module+'complete_vertical_box', return_value=[0, 0, 135, 30]), \
                 patch(module+'complete_line_box', return_value=[0, 0, 135, 30]), \
                 patch(module+'completed_word_observation', return_value=None), \
                 patch(module+'context_windows', return_value=[]), \
                 patch(module+'scored_scan_word', side_effect=lambda m, g, w: (w, italic if w == 'In' else bold)):
                observed = analyze_page(rgb, Reader(), calibration=calibration)
            prefix = [r for r in observed if r['text'] == 'Property 7: In']
            self.assertEqual(len(prefix), 1)
            self.assertEqual([w['text'] for w in prefix[0]['words']], ['Property', 'In'])
            if calibration:
                self.assertEqual(prefix[0]['words'][1]['scores'], {})


if __name__ == '__main__':
    unittest.main()
