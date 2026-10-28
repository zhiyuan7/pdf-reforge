"""Visible PDF word boundaries survive splitting prose from inline formulas."""
import unittest

from book_pipeline.ingestion.native import native_content, join_spans


class NativeSpacingTests(unittest.TestCase):
    def content(self, spans, parts):
        lookup = {s['id']: s for s in spans}
        nodes = native_content(dict(type='text', bbox=[0, 0, 1, 1], parts=parts), lookup, 600)
        return ''.join(n['content'] for n in nodes)

    @staticmethod
    def span(sid, text, x0, x1, line=0):
        return dict(id=sid, text=text, bbox=[x0, .1, x1, .12], line=line, size=10)

    def test_math_owned_spaces_are_kept_on_both_sides(self):
        spans = [self.span('a', 'think of', .1, .2), self.span('b', ' 𝑣 ', .2, .24),
                 self.span('c', 'as a vector', .24, .4)]
        parts = [dict(kind='text', spans=['a']), dict(kind='math', spans=['b'], latex='v'),
                 dict(kind='text', spans=['c'])]
        self.assertEqual(self.content(spans, parts), 'think of v as a vector')

    def test_geometric_gap_after_math_recovers_a_missing_word_space(self):
        spans = [self.span('a', ' 𝑣', .2, .24), self.span('b', 'not as a point', .249, .4)]
        self.assertEqual(self.content(spans, [dict(kind='math', spans=['a'], latex='v'),
                                            dict(kind='text', spans=['b'])]), ' v not as a point')

    def test_math_at_end_of_line_does_not_glue_the_next_line(self):
        spans = [self.span('a', '𝑣', .8, .85), self.span('b', 'has length one', .1, .4, line=1)]
        self.assertEqual(self.content(spans, [dict(kind='math', spans=['a'], latex='v'),
                                            dict(kind='text', spans=['b'])]), 'v\nhas length one')

    def test_chinese_and_punctuation_do_not_gain_word_spaces(self):
        spans = [self.span('a', '向量', .1, .2), self.span('b', '𝑣', .2, .22),
                 self.span('c', '，长度为一。', .225, .4)]
        self.assertEqual(self.content(spans, [dict(kind='text', spans=['a']),
                                            dict(kind='math', spans=['b'], latex='v'),
                                            dict(kind='text', spans=['c'])]), '向量v，长度为一。')

    def test_unencoded_gap_between_text_font_runs_is_preserved(self):
        spans = [self.span('a', 'a', .1, .11), self.span('b', 'vector', .116, .2)]
        self.assertEqual(join_spans(['a', 'b'], {s['id']: s for s in spans}, 600), 'a vector')
