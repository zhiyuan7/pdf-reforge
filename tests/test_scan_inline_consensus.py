from copy import deepcopy
import unittest

from book_pipeline.ingestion.scan_inline_consensus import agreed_inline_edits, propose_inline_edits, reading


class ScanInlineConsensusTests(unittest.TestCase):
    def setUp(self):
        self.text = r'(12) If $x/d=2,\ x/dk=2$ if $dk$ is consistent.'
        self.answer = dict(items=[dict(kind='text', text=self.text),
                                 dict(kind='formula', latex='x/d=2', number='')])
        self.first = self.witness(r'(12) If $x/h=2$. $x/dk=2$ if $dk$ is consistent.')
        self.second = self.witness(r'(12) If $x/h = 2$, $x/dk = 2$ if $dk$ is consistent.')
        self.band = dict(preserve=[])

    def witness(self, text):
        return dict(items=[dict(kind='text', text=text)])

    def test_complete_readings_repair_one_letter_and_preserve_punctuation_grouping_and_other_items(self):
        before = deepcopy(self.answer)
        output, accepted, rejected = agreed_inline_edits(self.answer, self.first, self.second, self.band)
        self.assertEqual(output['items'][0]['text'], self.text.replace('x/d=2,', 'x/h=2,'))
        self.assertEqual(output['items'][1:], before['items'][1:])
        self.assertEqual(self.answer, before)
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)

    def test_partial_disagreeing_duplicate_and_unchanged_readings_abstain(self):
        for second in (self.witness('If $x/h=2$.'), self.witness(self.text),
                       self.witness(self.second['items'][0]['text'].replace('x/h', 'x/z')),
                       dict(items=self.second['items']*2),
                       dict(items=self.second['items']+self.answer['items'][:1])):
            with self.subTest(second=second):
                output, accepted, rejected = agreed_inline_edits(self.answer,self.first,second,self.band)
                self.assertEqual(output,self.answer)
                self.assertFalse(accepted)
                self.assertEqual(len(rejected),1)
        duplicated = dict(items=self.answer['items']+self.answer['items'][:1])
        self.assertFalse(propose_inline_edits(duplicated,self.first))
        ambiguous = dict(items=self.answer['items']+[dict(kind='text',text=self.text.replace('x/d=','x/z='))])
        self.assertFalse(propose_inline_edits(ambiguous,self.first))

    def test_numbers_operators_numerators_prose_case_and_multi_letter_changes_are_fixed(self):
        for changed in (self.text.replace('(12)','(13)'), self.text.replace('=2','=3'),
                        self.text.replace('=2','>2'), self.text.replace('x/d=','y/d='),
                        self.text.replace('consistent','inconsistent'), self.text.replace('x/d=','x/D='),
                        self.text.replace('x/d=','x/h=').replace('x/dk=','x/dh='),
                        self.text.replace('x/d=','x/h=').replace('.', ''),
                        self.text.replace('x/d=','x/h=').replace('is consistent','isc onsistent'),
                        self.text.replace('x/d=','x/h=').replace('If','if')):
            with self.subTest(changed=changed):
                self.assertFalse(propose_inline_edits(self.answer,self.witness(changed)))
        # Decimal identity remains significant even though prose comma/period
        # separator positions can vary between the independent readers.
        answer = self.witness('If $x/d=1.2$, then the statement follows.')
        other = self.witness('If $x/h=1,2$, then the statement follows.')
        self.assertFalse(propose_inline_edits(answer,other))

    def test_unsupported_math_commands_and_unbalanced_delimiters_abstain(self):
        for text in ('If $x/d=2', 'If $x/d=2$ and $', r'If $\frac{x}{d}=2$, then it follows.',
                     r'If $x/d=2$ \text{then} it follows.', '若 $x/d=2$ 则结论成立。'):
            self.assertIsNone(reading(text))
        for text in ('If $x/d = 2$, then it follows.', r'If $x/d=2\quad x/k=2$, then it follows.'):
            self.assertIsNotNone(reading(text))


if __name__ == '__main__':
    unittest.main()
