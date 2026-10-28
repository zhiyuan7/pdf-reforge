import unittest
from book_pipeline.structure.footnotes import extract_marker,marker_occurrences

class FootnoteMarkerTests(unittest.TestCase):
    def test_prose_numerals_do_not_masquerade_as_missing_callouts(self):
        marker=extract_marker('^{1} A source note.')
        for text in ('Axiom 1: positive','Step 1 and 2','There is 1 event.','Definition 1.2'):
            self.assertEqual(list(marker_occurrences(text,marker,'text')),[],text)
        for text in ('After a sentence.¹ Next','After a sentence.^{1} Next',r'After.\textsuperscript{1} Next'):
            self.assertEqual(len(list(marker_occurrences(text,marker,'text'))),1,text)
        self.assertEqual(list(marker_occurrences('x^{1}',marker,'math')),[])
        self.assertEqual(len(list(marker_occurrences('^{1}',marker,'math'))),1)
        self.assertEqual(len(list(marker_occurrences('After① next',extract_marker('① note'),'text'))),1)
