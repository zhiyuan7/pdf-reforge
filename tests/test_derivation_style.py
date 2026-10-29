"""Displayed derivations retain readable fractions without changing matrices."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.derivation_style import display_derivation_arrays
from book_pipeline.publishing.latex import write_project


class DerivationStyleTests(unittest.TestCase):
    def test_only_style_declarations_added_with_delimiters_and_row_bytes_preserved(self):
        for column in ('l', 'c', 'r'):
            tex = (r'\left|\begin{array} { '+column+r' } \frac{1}{2}\sum_{i=1}^n x_i '
                   r'\\ = \frac{n}{2} \\ \leq n \end{array}\right|')
            styled, count = display_derivation_arrays(tex)
            self.assertEqual(count, 1)
            self.assertEqual(styled.count(r'\displaystyle '), 3)
            self.assertEqual(styled.replace(r'\displaystyle ', ''), tex)
            self.assertEqual(display_derivation_arrays(styled), (styled, 0))

    def test_matrices_cases_nested_explicit_styles_and_uncertain_rows_abstain(self):
        for body, column in [
                (r'1\\2', 'l'), (r'x&=\frac{1}{2}\\y&=2', 'll'),
                (r'x&\frac{1}{2}\\=y&2', 'l'),
                (r'\frac{1}{2}\\x=2', 'c'),
                (r'\textstyle\frac{1}{2}\\=2', 'l'),
                (r'\frac{1}{2}\\[2pt]=2', 'l'),
                (r'\frac{1}{2}\\*=2', 'l'),
                (r'\frac{1}{2}\\', 'l'),
                (r'\frac{1}{2}\\=\begin{matrix}1\\2\end{matrix}', 'l'),
                (r'\frac{1}{2} % comment'+'\n'+r'\\=2', 'l'),
                (r'\frac{1}{2}\\=2}', 'l')]:
            tex = r'\begin{array}{'+column+'}'+body+r'\end{array}'
            with self.subTest(tex=tex):
                self.assertEqual(display_derivation_arrays(tex), (tex, 0))
        tex = r'\begin{cases}\frac{1}{2}\\=2\end{cases}'
        self.assertEqual(display_derivation_arrays(tex), (tex, 0))
        inner = r'\begin{array}{l}\frac{1}{2}\\=2\end{array}'
        for tex in [r'x^{'+inner+'}', r'\textstyle '+inner,
                    r'\begin{pmatrix}'+inner+r'\end{pmatrix}',
                    r'\begin{cases}'+inner+r'\end{cases}', '{'+inner+'}',
                    inner+'+'+inner]:
            self.assertEqual(display_derivation_arrays(tex), (tex, 0))

    def test_compiled_array_fraction_matches_standalone_display_size_and_source_coverage(self):
        formulas = [r'\begin{array}{l}\frac{17}{19}\\=\frac{23}{29}\end{array}\tag{4.2}',
                    r'\frac{31}{37}', r'\begin{pmatrix}41\\43\end{pmatrix}']
        atoms = [dict(id=str(n), group=str(n), kind='display_math', role='equation',
                      block_type='equation', page=0, content=t, length=len(t),
                      bbox=[.1, .1+n*.15, .8, .2+n*.15]) for n, t in enumerate(formulas)]
        source = SourceIndex(dict(atoms=atoms))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            audit = write_project(source, dict(corrections=[], joins=[], exclusions=[], bold_spans=[]),
                [], [], dict(title='Derivation style', author=''), load_config(), root, sample_mode=True)
            self.assertTrue(audit['coverage_complete'])
            self.assertEqual(audit['accounted_units'], source.prefix[-1])
            result = compile_project(root, {t: executable(t) for t in ('xelatex', 'latexmk')})
            self.assertTrue(result['success'], result['log_tail'])
            self.assertEqual(result['warnings'], [])
            with pymupdf.open(root/'main.pdf') as pdf:
                spans = [s for p in pdf for b in p.get_text('dict')['blocks']
                         for line in b.get('lines', []) for s in line['spans']]
                def size(word):
                    found = [s['size'] for s in spans if s['text'] == word]
                    self.assertEqual(len(found), 1, (word, spans))
                    return found[0]
                for number in ('17', '19', '23', '29'):
                    self.assertAlmostEqual(size(number), size('31'), delta=.02)
                self.assertAlmostEqual(size('41'), size('43'), delta=.02)
                names = pdf.resolve_names()
                self.assertTrue(all('source-'+a['id'] in names for a in atoms))
                self.assertEqual(sum(len(p.search_for('(4.2)')) for p in pdf), 1)
            self.assertEqual([a['content'] for a in source.atoms], formulas)


if __name__ == '__main__':
    unittest.main()
