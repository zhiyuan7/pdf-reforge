"""Unicode math in OCR prose retains variables instead of missing glyphs."""
import unittest
from book_pipeline.publishing.latex import escape, math_tex


class MathUnicodeTests(unittest.TestCase):
    def test_unicode_greek_variants_retain_their_distinct_math_shapes(self):
        self.assertEqual(math_tex('εϵφϕ'),r'\varepsilon\epsilon\varphi\phi')

    def test_set_operators_outside_equation_markup_use_math_fonts(self):
        self.assertEqual(escape('A∪B'),r'A\ensuremath{\cup}B')
        self.assertEqual(escape('A∩B'),r'A\ensuremath{\cap}B')
        self.assertEqual(math_tex('A∪B'),r'A\cup B')
    def test_proof_connectives_have_glyphs_in_prose_and_math(self):
        self.assertEqual(escape('∴'),r'\ensuremath{\therefore}')
        self.assertEqual(math_tex('∵'),r'\because')

    def test_double_arrows_and_similarity_keep_math_glyphs_in_table_cells(self):
        from book_pipeline.publishing.latex import table_tex
        for symbol,command in [('⇒',r'\Rightarrow'),('⇐',r'\Leftarrow'),
                               ('⇔',r'\Leftrightarrow'),('∼',r'\sim')]:
            self.assertEqual(math_tex(symbol+'Q'),command+' Q')
            self.assertIn(r'\ensuremath{'+command+'}',table_tex(
                '<table><tr><td>P'+symbol+'Q</td></tr></table>'))
    def test_italic_variables_in_prose_use_math_fonts(self):
        result = escape('当 𝑛 为正整数时，变量 𝑥 可以相加。')
        self.assertIn(r'\ensuremath{n}', result)
        self.assertIn(r'\ensuremath{x}', result)
        self.assertNotIn('𝑛', result)

    def test_styled_numbers_and_greek_remain_semantically_math(self):
        self.assertIn(r'\mathbb{A}', math_tex('𝔸'))
        self.assertIn(r'\mathfrak{R}', math_tex('𝕽'))
        self.assertIn(r'\alpha', math_tex('𝛼'))
        self.assertIn(r'\boldsymbol', math_tex('𝐀'))
        self.assertEqual(escape('中文说明'), '中文说明')
        self.assertIn(r'\ensuremath{\heartsuit}', escape('♡'))
        self.assertIn(r'\ensuremath{\checkmark}', escape('✓'))
        self.assertIn(r'\ensuremath{\cdots}', escape('⋯'))
