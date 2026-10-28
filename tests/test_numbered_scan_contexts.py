"""Short scan prose needs a literal numbered label and independent ink."""
from copy import deepcopy
import unittest

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.raster_body_styles import (
    anchored_mixed_prose, mixed_fragments, numbered_prose_contexts, owned_ranges)


class NumberedScanContextTests(unittest.TestCase):
    def fixture(self):
        contents = [('text', 'Property 7: If '), ('math', 'X, Y'),
                    ('text', ' are disjoint then\n')]
        base = dict(page=0, group='G', bbox=[.05, .15, .9, .4],
                    role='paragraph', block_type='paragraph', extraction_method='ocr')
        raw = [dict(base, id=f'r{n}', kind=k, content=v, length=len(v))
               for n, (k, v) in enumerate(contents)]
        edited = [dict(a, id=f'e{n}', original_atom=a['id'], original_start_offset=0,
                       original_end_offset=len(a['content'])) for n, a in enumerate(raw)]
        bold = dict(tiro=.3, tibo=.92, tiit=.3, tibi=.4)
        italic = dict(tiro=.3, tibo=.3, tiit=.91, tibi=.91)
        words = [dict(text=s, score=.999, bbox=[x, 20, z, 30], scores=style)
                 for s, x, z, style in [('Property', 10, 30, bold),
                    ('are', 45, 53, italic), ('disjoint', 53, 66, italic),
                    ('then', 66, 80, italic)]]
        observed = [dict(text='Property 7: If X, Y are disjoint then',
                         score=.999, bbox=[10, 20, 80, 30], words=words)]
        return SourceIndex(dict(atoms=raw)), SourceIndex(dict(atoms=edited)), observed

    def recover(self, raw, index, observed, excluded=()):
        items = [dict(atom=a['id'], text=a['content'], bbox=a['bbox'])
                 for a in index.atoms if a['kind'] == 'text']
        mixed = mixed_fragments(index, raw, items, excluded)
        prepared = anchored_mixed_prose(observed, index, raw, items, mixed)
        return owned_ranges(observed, prepared, [100]*4, mixed=mixed)

    def test_exact_label_and_terminal_prose_keep_source_offsets(self):
        raw, index, observed = self.fixture()
        before = deepcopy(index.source)
        result = self.recover(raw, index, observed)
        self.assertEqual([r['quote'] for r in result['e0']], ['Property'])
        self.assertEqual([r['quote'] for r in result['e2']], ['are', 'disjoint', 'then'])
        self.assertTrue(result['e0'][0]['bold'])
        self.assertTrue(all(r['italic'] for r in result['e2']))
        self.assertEqual(result['e2'][0]['start'], 1)
        self.assertEqual(index.source, before)

    def test_complete_two_word_fragment_before_terminal_math_is_owned(self):
        raw, index, observed = self.fixture()
        for data in (raw.source, index.source):
            data['atoms'][2].update(content=' for every ', length=11)
            data['atoms'].append(dict(data['atoms'][1], id='r3' if data is raw.source else 'e3',
                                      content='Z', length=1))
        index.source['atoms'][2]['original_end_offset'] = 11
        index.source['atoms'][3].update(original_atom='r3', original_end_offset=1)
        row = observed[0]
        row['text'] = 'Property 7: If X, Y for every Z'
        row['words'] = [row['words'][0], dict(row['words'][1], text='for'),
                        dict(row['words'][2], text='every')]
        result = self.recover(SourceIndex(raw.source), SourceIndex(index.source), observed)
        self.assertEqual([r['quote'] for r in result['e2']], ['for', 'every'])
        # A matched substring inside a longer token cannot own a fragment.
        observed[0]['text'] = row['text'].replace('every Z', 'everywhere Z')
        self.assertNotIn('e2', self.recover(SourceIndex(raw.source), SourceIndex(index.source), observed))

    def test_number_and_full_prefix_must_match_independent_row(self):
        for bad in ('Property 8: If X, Y are disjoint then',
                    'Property 7; If X, Y are disjoint then',
                    'noise Property 7: If X, Y are disjoint then',
                    'Property 7:Property 7: If X, Y are disjoint then',
                    'Property 7:If X, Y are disjoint then'):
            raw, index, observed = self.fixture()
            observed[0]['text'] = bad
            group = [dict(math=True) if a['kind'] == 'math' else dict(atom=a['id'], text=a['content']) for a in index.atoms]
            self.assertEqual(numbered_prose_contexts(observed[0], group), [], bad)
        for score in (.98, float('nan'), 1.01):
            raw, index, observed = self.fixture()
            observed[0]['score'] = score
            group = [dict(math=True) if a['kind'] == 'math' else dict(atom=a['id'], text=a['content'])
                     for a in index.atoms]
            self.assertEqual(numbered_prose_contexts(observed[0], group), [])

    def test_label_cannot_borrow_style_or_ownership_from_formula(self):
        for defect in ('low_word_score', 'upright', 'missing_word', 'formula_token', 'outside'):
            raw, index, observed = self.fixture()
            if defect == 'low_word_score': observed[0]['words'][0]['score'] = .98
            if defect == 'upright': observed[0]['words'][0]['scores']['tiro'] = .99
            if defect == 'missing_word': observed[0]['words'].pop(0)
            if defect == 'formula_token':
                raw.source['atoms'][1]['content'] = 'Property'
                index.source['atoms'][1].update(content='Property', original_end_offset=8)
                raw, index = SourceIndex(raw.source), SourceIndex(index.source)
            if defect == 'outside': observed[0]['bbox'] = [10, 50, 80, 60]
            recovered = self.recover(raw, index, observed)
            self.assertNotIn('e0', recovered, defect)
            if defect not in ('formula_token', 'outside'):
                group = [dict(math=True) if a['kind'] == 'math' else dict(atom=a['id'], text=a['content']) for a in index.atoms]
                self.assertEqual(numbered_prose_contexts(observed[0], group), [], defect)

    def test_tail_needs_complete_unique_literal_reading_and_individual_style(self):
        for defect in ('changed', 'extra', 'missing', 'weak', 'mixed_style', 'gap', 'other_line', 'repeated'):
            raw, index, observed = self.fixture()
            if defect == 'changed': observed[0]['text'] = observed[0]['text'].replace('disjoint', 'separate')
            if defect == 'extra': observed[0]['text'] += ' later'
            if defect == 'missing': observed[0]['words'].pop(2)
            if defect == 'weak': observed[0]['words'][2]['scores']['tiro'] = .99
            if defect == 'mixed_style': observed[0]['words'][2]['scores'] = dict(tiro=.3, tibo=.92, tiit=.3, tibi=.4)
            if defect == 'gap': observed[0]['words'][2]['bbox'][0] += 5
            if defect == 'other_line': observed[0]['words'][2]['bbox'][1] += 1
            if defect == 'repeated': observed[0]['text'] += ' are disjoint then'
            result = self.recover(raw, index, observed)
            self.assertEqual([r['quote'] for r in result['e0']], ['Property'], defect)
            group = [dict(math=True) if a['kind'] == 'math' else dict(atom=a['id'], text=a['content']) for a in index.atoms]
            self.assertEqual([c['atom'] for c in numbered_prose_contexts(observed[0], group)], ['e0'], defect)

    def test_group_edits_exclusions_or_duplicate_owners_cannot_supply_context(self):
        for defect in ('removed', 'rewritten', 'role', 'excluded', 'duplicate'):
            raw, index, observed = self.fixture()
            changed = deepcopy(index.source)
            excluded = ()
            if defect == 'removed': changed['atoms'].pop(1)
            if defect == 'rewritten': changed['atoms'][1]['content'] = 'Z'
            if defect == 'role': changed['atoms'][1]['role'] = 'heading'
            if defect == 'excluded': excluded = ('e1',)
            if defect == 'duplicate':
                changed['atoms'].extend(dict(changed['atoms'][n], id=f'copy{n}', group='other') for n in (0, 2))
            self.assertEqual(self.recover(raw, SourceIndex(changed), observed, excluded), {}, defect)

    def test_number_first_label_and_existing_long_context_do_not_duplicate_owners(self):
        raw, index, observed = self.fixture()
        for data in (raw.source, index.source):
            data['atoms'][0].update(content='2.4 Example. If ', length=16)
        index.source['atoms'][0]['original_end_offset'] = len(index.source['atoms'][0]['content'])
        observed[0]['text'] = '2.4 Example. If X, Y are disjoint then'
        observed[0]['words'][0]['text'] = 'Example'
        result = self.recover(SourceIndex(raw.source), SourceIndex(index.source), observed)
        self.assertEqual([r['quote'] for r in result['e0']], ['Example'])
        # This whole group also meets the original formula-bracketing rule;
        # the additional numeric prefix must not create competing owners.
        self.assertEqual(len(result['e2']), 3)

    def test_short_unlabeled_or_single_word_tail_is_ineligible(self):
        line = dict(text='Lemma 2: X disjoint', score=.999, words=[])
        group = [dict(atom='a', text='Lemma 2: '), dict(math=True), dict(atom='b', text=' disjoint')]
        self.assertEqual(numbered_prose_contexts(line, group), [])
        line['words'] = [dict(text='Lemma', score=.999, bbox=[0, 0, 15, 10],
                              scores=dict(tiro=.3, tibo=.92, tiit=.3, tibi=.4))]
        contexts = numbered_prose_contexts(line, group)
        self.assertEqual([c['atom'] for c in contexts], ['a'])
        group[0]['text'] = 'Lemma: '
        self.assertEqual(numbered_prose_contexts(line, group), [])


if __name__ == '__main__':
    unittest.main()
