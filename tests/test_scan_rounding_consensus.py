import unittest
from copy import deepcopy

from book_pipeline.ingestion.scan_rounding_consensus import candidates, reading, restore_rounding


def formula(text, number=''):
    return dict(kind='formula', latex=text, number=number)


class RoundingConsensusTests(unittest.TestCase):
    def setUp(self):
        self.text = r'T_p\leq\sum_{\tau=1}^{T_\infty}\left[\frac{n_\tau}{p}\right]<T_1/p+T_\infty,'
        self.answer = dict(items=[formula(self.text)])
        self.rounded = self.text.replace('[', r'\lceil').replace(']', r'\rceil')
        self.sources = [dict(group='display', latex=self.rounded)]

    def test_complete_cpu_and_blind_reading_repair_only_paired_delimiters(self):
        original = deepcopy(self.answer)
        witness = dict(items=[formula(self.rounded.replace(r'\leq', r'\le').replace('T_p', 'T_{p}'))])
        output, changes = restore_rounding(self.answer, witness, self.sources)
        self.assertEqual(output['items'][0]['latex'], self.rounded)
        self.assertEqual(self.answer, original)
        self.assertEqual(changes[0]['source_group'], 'display')
        self.assertEqual(len(changes[0]['edits']), 2)

    def test_disagreement_or_any_other_character_change_abstains(self):
        for text in (self.text, self.rounded.replace('n_', 'm_'), self.rounded.replace('<', '>'),
                     self.rounded.replace('T_1', 'T_2'), self.rounded[:-1],
                     self.rounded.replace(r'\lceil', r'\lfloor').replace(r'\rceil', r'\rfloor')):
            with self.subTest(text=text):
                self.assertFalse(restore_rounding(self.answer, dict(items=[formula(text)]), self.sources)[1])
        self.assertFalse(restore_rounding(self.answer, dict(items=[formula(self.rounded, '7')]), self.sources)[1])

    def test_duplicate_or_partial_owners_abstain(self):
        witness = dict(items=[formula(self.rounded)])
        for answer, other, sources in (
                (dict(items=self.answer['items']*2), witness, self.sources),
                (self.answer, dict(items=witness['items']*2), self.sources),
                (self.answer, witness, self.sources*2),
                (self.answer, dict(items=[formula(r'\lceil n_\tau/p\rceil')]), self.sources),
                (self.answer, witness, [dict(group='partial',latex=r'\lceil n_\tau/p\rceil')])):
            self.assertFalse(restore_rounding(answer, other, sources)[1])

    def test_floor_and_nested_pairs_keep_tex_command_boundaries(self):
        answer = dict(items=[formula('[x]+[y]z')])
        source = r'\lceil x\rceil+\lfloor y\rfloor z'
        output, changes = restore_rounding(answer, dict(items=[formula(source)]), [dict(group='g',latex=source)])
        self.assertEqual(output['items'][0]['latex'], source)
        self.assertEqual(len(changes[0]['edits']),4)
        nested = r'\lceil\lfloor x\rfloor\rceil'
        output, changes = restore_rounding(dict(items=[formula('[[x]]')]), dict(items=[formula(nested)]),
                                           [dict(group='g',latex=nested)])
        self.assertEqual(reading(output['items'][0]['latex'])['identity'], reading(nested)['identity'])
        self.assertEqual(len(changes[0]['edits']),4)

    def test_ordinary_intervals_compound_groups_and_unsupported_text_abstain(self):
        self.assertIsNone(reading('[a,b)'))
        self.assertIsNone(reading(r'[x]\text{is not}'))
        self.assertIsNone(reading(r'[x]_{a'))
        self.assertIsNone(reading(r'[x]\rceil'))
        answer = dict(items=[formula('[x_{ab}]')])
        self.assertFalse(candidates(answer, [dict(group='g',latex=r'\lceil x_ab\rceil')]))
        # Existing ceilings/floors are never changed into another delimiter.
        answer = dict(items=[formula(r'\lceil x\rceil')])
        self.assertFalse(candidates(answer,[dict(group='g',latex=r'\lfloor x\rfloor')]))


if __name__ == '__main__':
    unittest.main()
