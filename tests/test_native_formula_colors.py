import unittest

from book_pipeline.ingestion.native_formula_colors import validate_colors,without_color


class NativeFormulaColorTests(unittest.TestCase):
    def test_color_review_preserves_vectors_macros_numbers_and_each_observed_color(self):
        source=r'\frac{1}{p(\boldsymbol{x}_n)}\frac{\partial p(\boldsymbol{x}_n)}{\partial \theta}'
        contract=dict(number='3.4',latex=source,colors=[[1,.5,0],[0,0,1]])
        colored=(r'\textcolor[rgb]{1,0.5,0}{\frac{1}{p(\boldsymbol{x}_n)}}'
                 r'\textcolor[rgb]{0,0,1}{\frac{\partial p(\boldsymbol{x}_n)}{\partial \theta}}')
        answer=lambda s,n='3.4':dict(items=[dict(kind='formula',latex=s,number=n)])
        validate_colors(answer(colored),[contract])
        self.assertEqual(without_color(colored)[0],source)
        for wrong in (colored.replace(r'\boldsymbol{x}_n','x_n'),colored.replace('0,0,1','1,0.5,0'),
                      colored.replace('0,0,1','0,1,0'),colored.replace('x}_n','x}_{n+1}'),colored[:-1]):
            with self.assertRaises(ValueError):validate_colors(answer(wrong),[contract])
        with self.assertRaises(ValueError):validate_colors(answer(colored,'3.5'),[contract])

    def test_nested_colors_keep_complete_formula_and_reject_unsupported_color_macros(self):
        plain,colors=without_color(r'\textcolor[rgb]{1,0.5,0}{a+\textcolor[rgb]{0,0,1}{b}}')
        self.assertEqual(plain,'a+b');self.assertEqual(len(colors),2)
        for wrong in (r'\color{blue}x',r'\textcolor{red}{x}',r'\textcolor[rgb]{nan,0,1}{x}'):
            with self.assertRaises(ValueError):without_color(wrong)


if __name__=='__main__':unittest.main()
