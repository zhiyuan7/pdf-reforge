"""Only verified prose line ends can change; linguistic uncertainty is retained."""
from copy import deepcopy
import unittest

from book_pipeline.ingestion.native import native_content
from book_pipeline.ingestion.native_hyphens import candidates_for, review_hyphens, validate_decisions


def source(left='The geo-', right='metric meaning.', separate_parts=False):
    spans = [dict(id='a', line=0, text=left, size=10, bbox=[.1,.1,.5,.12]),
             dict(id='b', line=1, text=right, size=10, bbox=[.1,.125,.45,.145])]
    parts = ([dict(kind='text', spans=['a']), dict(kind='text', spans=['b'])] if separate_parts
             else [dict(kind='text', spans=['a','b'])])
    return [dict(type='text', bbox=[.1,.1,.5,.145], parts=parts)], {s['id']:s for s in spans}


class ReviewClient:
    def __init__(self, action):
        self.action, self.calls = action, 0

    def ask(self, task, prompt, payload, validator, **kwargs):
        self.calls += 1
        value = dict(decisions=[dict(id=c['id'], action=self.action, evidence='linguistic test evidence')
                                for c in payload['candidates']])
        validator(value)
        return value


class NativeHyphenTests(unittest.TestCase):
    def render(self, rows, lookup, client):
        adjusted, boundaries, records = review_hyphens(rows, lookup, client)
        return ''.join(n['content'] for n in native_content(rows[0], adjusted, 600, boundaries)), records

    def test_discretionary_word_and_proper_name_across_parts(self):
        for left,right,expected in [('The geo-', 'metric meaning.', 'The geometric meaning.'),
                                    ('Edwin Ab- ', ' bott.', 'Edwin Abbott.')]:
            for separate in [False,True]:
                rows, lookup = source(left,right,separate)
                original = deepcopy(lookup)
                text, records = self.render(rows, lookup, ReviewClient('join'))
                self.assertEqual(text, expected)
                self.assertEqual(lookup, original)
                self.assertEqual(records[0][0]['source_lines'], [left,right])

    def test_real_and_suspended_compounds_keep_hyphen_without_space(self):
        for left,right,expected in [('Use high-', 'dimensional spaces.', 'Use high-dimensional spaces.'),
                                    ('Run the Matrix-', 'Plot program.', 'Run the Matrix-Plot program.'),
                                    ('Short- and long-', 'term effects.', 'Short- and long-term effects.')]:
            rows, lookup = source(left,right,True)
            text, _ = self.render(rows, lookup, ReviewClient('retain'))
            self.assertEqual(text,expected)

    def test_uncertain_word_keeps_source_break_and_evidence(self):
        rows, lookup = source()
        text, records = self.render(rows, lookup, ReviewClient('uncertain'))
        self.assertEqual(text, 'The geo-\nmetric meaning.')
        self.assertEqual(records[0][0]['action'], 'uncertain')

    def test_math_same_line_columns_and_unowned_line_end_are_not_candidates(self):
        for variant in ['math','same_line','column','other_line_ink','vertical_gap','paragraph_end']:
            rows, lookup = source()
            if variant == 'math': rows[0]['parts'][0]['kind'] = 'math'
            if variant == 'same_line': lookup['b']['line'] = 0
            if variant == 'column': lookup['b']['bbox'] = [.6,.125,.95,.145]
            if variant == 'other_line_ink':
                lookup['c'] = dict(id='c', line=0, text='x', bbox=[.52,.1,.55,.12])
            if variant == 'vertical_gap': lookup['b']['bbox'] = [.1,.3,.45,.32]
            if variant == 'paragraph_end': lookup['b']['bbox'] = [.18,.125,.45,.145]
            client = ReviewClient('join')
            self.assertEqual(candidates_for(rows,lookup), [],variant)
            review_hyphens(rows,lookup,client)
            self.assertEqual(client.calls,0)

    def test_ordinary_hyphens_and_math_spaces_survive(self):
        rows, lookup = source('Use two-dimensional spaces and', 'x as a vector.')
        rows[0]['parts'] = [dict(kind='text',spans=['a']),dict(kind='math',spans=['b'],latex='x')]
        self.assertEqual(candidates_for(rows,lookup),[])
        self.assertEqual(self.render(rows,lookup,ReviewClient('join'))[0],
                         'Use two-dimensional spaces and\nx')

    def test_validator_rejects_unbounded_or_duplicate_decisions(self):
        rows, lookup = source()
        candidates = candidates_for(rows,lookup)
        valid = dict(id='h0',action='join',evidence='one word')
        validate_decisions(dict(decisions=[valid]),candidates)
        for invalid in [[],[valid,valid],[dict(valid,id='invented')],
                        [dict(valid,action='rewrite')],[dict(valid,evidence='')]]:
            with self.assertRaises(ValueError): validate_decisions(dict(decisions=invalid),candidates)
