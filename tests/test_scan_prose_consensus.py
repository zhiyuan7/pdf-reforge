"""Blind agreement can restore letters without authorizing prose rewrites."""
from copy import deepcopy
import unittest

from book_pipeline.ingestion.scan_prose_consensus import (agreed_prose_edits,
    plain_prose, propose_prose_edits, repeated_letter_variant)


class ScanProseConsensusTests(unittest.TestCase):
    def setUp(self):
        self.text = 'The printed address appears in the program, and another address follows the program.'
        self.reading = self.text.replace('program', 'programm')
        self.answer = dict(items=[dict(kind='text', text=self.text),
                                 dict(kind='formula', latex='a/h=1', number='9')])
        self.witness = dict(items=[dict(kind='text', text=self.reading)])
        self.band = dict(preserve=[])

    def test_two_complete_readings_restore_repeated_words_without_changing_other_fields(self):
        original = deepcopy(self.answer)
        second = dict(items=[dict(kind='text', text=self.reading.replace(', and', ',\n and'))])
        corrected, accepted, rejected = agreed_prose_edits(self.answer, self.witness, second, self.band)
        self.assertEqual(self.answer, original)
        self.assertEqual(corrected['items'][0]['text'], self.reading)
        self.assertEqual(corrected['items'][1:], original['items'][1:])
        self.assertEqual(len(accepted), 2)
        self.assertFalse(rejected)
        self.assertTrue(all(len(e['before']) <= 64 and len(e['after']) <= 64 for e in accepted))

    def test_a_substring_or_disagreeing_or_duplicate_paragraph_cannot_confirm(self):
        for texts in ([self.reading[:40]], [self.text], [self.reading, self.reading],
                      [self.reading, self.text], [self.reading.replace('follows', 'followss')]):
            with self.subTest(texts=texts):
                second = dict(items=[dict(kind='text', text=t) for t in texts])
                output, accepted, rejected = agreed_prose_edits(self.answer, self.witness, second, self.band)
                self.assertEqual(output, self.answer)
                self.assertFalse(accepted)
                self.assertEqual(len(rejected), 2)
        for texts in ([self.reading, self.reading], [self.reading, self.text]):
            self.assertFalse(propose_prose_edits(self.answer, dict(items=[dict(kind='text',text=t) for t in texts])))
        duplicated = dict(items=self.answer['items']+[self.answer['items'][0]])
        self.assertFalse(propose_prose_edits(duplicated, self.witness))
        ambiguous = dict(items=self.answer['items']+[dict(kind='text',text=self.reading.replace('programm.', 'program.'))])
        self.assertFalse(propose_prose_edits(ambiguous, self.witness))

    def test_math_numbering_punctuation_case_and_other_word_changes_are_protected(self):
        for after in (self.reading.replace('The', 'the'), self.reading.replace(',', ';'),
                      self.reading.replace('another', 'the other'), self.reading.replace('follows', 'arrives'),
                      self.reading+' It does.', self.reading.replace('address', 'addresz')):
            self.assertFalse(propose_prose_edits(self.answer, dict(items=[dict(kind='text',text=after)])), after)
        for text in (self.text+' $a/h=1$', '(9) '+self.text, self.text+' \\text{label}', 'Too short.',
                     '这是中文，不能由英文单词修复器修改。'):
            self.assertFalse(plain_prose(text))
        for before,after in [('program','programme'), ('proof','proff'), ('address','Address'), ('a','aa')]:
            self.assertFalse(repeated_letter_variant(before,after))

    def test_overlapping_contexts_abstain_as_one_atomic_paragraph(self):
        answer = dict(items=[dict(kind='text',text='The program program is the program program in this printed text.')])
        witness = dict(items=[dict(kind='text',text=answer['items'][0]['text'].replace('program','programm'))])
        self.assertEqual(propose_prose_edits(answer,witness), [])


if __name__ == '__main__':
    unittest.main()
