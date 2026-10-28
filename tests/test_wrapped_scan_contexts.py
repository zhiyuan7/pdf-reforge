"""Cross-line prose ownership must retain complete-group and OCR guards."""
from copy import deepcopy
import unittest

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.raster_body_styles import (
    anchored_mixed_prose, mixed_fragments, owned_ranges, wrapped_line_contexts)


class WrappedScanContextTests(unittest.TestCase):
    def fixture(self):
        contents = [('text', 'A mapping '), ('math', 'X'),
                    ('text', ' assigns a real number '), ('math', 'Y'),
                    ('text', ' to each event '), ('math', 'Z'),
                    ('text', ' is a probability distribution')]
        base = dict(page=0, group='G', bbox=[.05, .15, .9, .5],
                    role='paragraph', block_type='paragraph', extraction_method='ocr')
        raw = [dict(base, id=f'r{n}', kind=k, content=v, length=len(v))
               for n, (k, v) in enumerate(contents)]
        edited = [dict(a, id=f'e{n}', original_atom=a['id'], original_start_offset=0,
                       original_end_offset=len(a['content'])) for n, a in enumerate(raw)]
        word = dict(text='event', score=.999, bbox=[10, 33, 25, 42],
                    scores=dict(tiro=.3, tibo=.3, tiit=.9, tibi=.9))
        observed = [dict(text='A mapping X assigns a real number Y to each',
                         score=.999, bbox=[10, 20, 80, 30], words=[]),
                    dict(text='event Z is a probability distribution',
                         score=.999, bbox=[10, 32, 80, 44], words=[word])]
        return SourceIndex(dict(atoms=raw)), SourceIndex(dict(atoms=edited)), observed

    def recover(self, raw, index, observed, excluded=()):
        items = [dict(atom=a['id'], text=a['content'], bbox=a['bbox'])
                 for a in index.atoms if a['kind'] == 'text']
        mixed = mixed_fragments(index, raw, items, excluded)
        prepared = anchored_mixed_prose(observed, index, raw, items, mixed)
        return owned_ranges(observed, prepared, [100]*4, mixed=mixed)

    def test_wrapped_word_uses_two_sided_formula_context_and_keeps_source_offsets(self):
        raw, index, observed = self.fixture()
        result = self.recover(raw, index, observed)
        self.assertEqual([(r['quote'], r['start'], r['end']) for r in result['e4']],
                         [('event', 9, 14)])
        self.assertTrue(result['e4'][0]['italic'])
        self.assertEqual(index.atoms[4]['content'], ' to each event ')

    def test_neighbor_geometry_confidence_and_transcript_are_required(self):
        for defect in ('column', 'gap', 'overlap', 'intervening', 'confidence',
                       'changed_text', 'word_confidence', 'upright', 'short_line'):
            raw, index, observed = self.fixture()
            if defect == 'column': observed[0]['bbox'] = [1, 20, 9, 30]
            if defect == 'gap': observed[1]['bbox'][1:] = [45, 80, 57]
            if defect == 'overlap': observed[1]['bbox'][1] = 29
            if defect == 'intervening':
                observed.append(dict(text='interruption', score=.8, bbox=[10, 30, 80, 32], words=[]))
            if defect == 'confidence': observed[0]['score'] = .98
            if defect == 'changed_text': observed[0]['text'] = observed[0]['text'].replace('number', 'label')
            if defect == 'word_confidence': observed[1]['words'][0]['score'] = .98
            if defect == 'upright': observed[1]['words'][0]['scores']['tiro'] = .95
            if defect == 'short_line': observed[1]['text'] = 'event Z'
            self.assertEqual(self.recover(raw, index, observed), {}, defect)

    def test_changed_removed_or_excluded_sibling_cannot_supply_context(self):
        for defect in ('removed', 'rewritten', 'role', 'excluded', 'duplicate'):
            raw, index, observed = self.fixture()
            changed = deepcopy(index.source)
            excluded = ()
            if defect == 'removed': changed['atoms'].pop(1)
            if defect == 'rewritten': changed['atoms'][1]['content'] = 'W'
            if defect == 'role': changed['atoms'][1]['role'] = 'heading'
            if defect == 'excluded': excluded = ('e1',)
            if defect == 'duplicate':
                changed['atoms'].append(dict(changed['atoms'][4], id='copy', group='other'))
            self.assertEqual(self.recover(raw, SourceIndex(changed), observed, excluded), {}, defect)

    def test_formula_bytes_never_supply_styled_prose(self):
        raw, index, observed = self.fixture()
        raw.source['atoms'][5]['content'] = 'event'
        index.source['atoms'][5].update(content='event', original_end_offset=5)
        raw, index = SourceIndex(raw.source), SourceIndex(index.source)
        self.assertEqual(self.recover(raw, index, observed), {})
        # A weak single-sided formula anchor cannot be repaired by wrapping.
        group = [dict(atom='a', text='A mapping '), dict(math=True),
                 dict(atom='b', text='event is a probability distribution')]
        self.assertEqual(wrapped_line_contexts(observed, group), [])


if __name__ == '__main__':
    unittest.main()
