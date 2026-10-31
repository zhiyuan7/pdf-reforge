from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.native_math_rows import native_math_rows


class NativeMathRowTests(unittest.TestCase):
    def source(self, root, single_line=False, count=3):
        path = root/'source.pdf'
        with pymupdf.open() as pdf:
            p = pdf.new_page(width=500, height=700)
            for i in range(count):
                p.insert_text((80+i*80 if single_line else 80, 100 if single_line else 100+i*20),
                              f'{chr(120+i)} = {i+1}', fontsize=11)
            pdf.save(path)
        text = 'x = 1\ny = 2\nz = 3, \\tag{7.2}'
        atom = dict(id='eq', group='eq', page=0, kind='display_math', role='equation',
                    block_type='equation', content=text, length=len(text),
                    extraction_method='native', bbox=[.1,.1,.8,.3])
        return path, atom

    def test_explicit_rows_and_tag_keep_tokens_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as name:
            path, atom = self.source(Path(name)); before = deepcopy(atom)
            rendered = native_math_rows(atom, atom['content'], path)
            self.assertEqual(rendered, r'\begin{aligned}x &= 1\\y &= 2\\z &= 3,\end{aligned}\tag{7.2}')
            self.assertEqual(atom, before)

    def test_source_baselines_and_equal_sign_count_are_required(self):
        for options in (dict(single_line=True), dict(count=2), dict(count=4)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as name:
                path, atom = self.source(Path(name), **options)
                self.assertIsNone(native_math_rows(atom, atom['content'], path))

    def test_spacing_becomes_rows_only_with_independent_aligned_native_relations(self):
        for single_line in (False, True):
            with self.subTest(single_line=single_line), tempfile.TemporaryDirectory() as name:
                path, atom = self.source(Path(name), single_line=single_line)
                atom['content'] = r'x=1\qquad y=2\qquad z=3, \tag{7.2}'
                before = deepcopy(atom)
                result = native_math_rows(atom, atom['content'], path)
                if single_line:
                    self.assertIsNone(result)
                else:
                    self.assertEqual(result, r'\begin{aligned}x&=1\\ y&=2\\ z&=3,\end{aligned}\tag{7.2}')
                self.assertEqual(atom, before)

    def test_nested_spacing_missing_relation_and_staggered_rows_abstain(self):
        for content in (r'x={1\qquad y=2}\qquad z=3',
                        r'x=1\qquad y\qquad z=3'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as name:
                path, atom = self.source(Path(name))
                atom['content'] = content
                self.assertIsNone(native_math_rows(atom, content, path))
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'source.pdf'
            with pymupdf.open() as pdf:
                page = pdf.new_page(width=500,height=700)
                for i in range(3):
                    page.insert_text((80+i*40,100+i*20),f'{chr(120+i)}={i+1}',fontsize=11)
                pdf.save(path)
            content = r'x=1\qquad y=2\qquad z=3'
            atom = dict(kind='display_math',extraction_method='native',page=0,
                        bbox=[.1,.1,.8,.3],content=content)
            self.assertIsNone(native_math_rows(atom,content,path))

    def test_small_equals_in_summation_limits_do_not_count_as_equation_rows(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path, atom = self.source(root)
            extra = root/'with-limits.pdf'
            with pymupdf.open(path) as pdf:
                for i in range(3):
                    pdf[0].insert_text((125, 107+i*20), 'i=1', fontsize=7)
                pdf.save(extra)
            atom['content'] = 'x=\\sum_{i=1}^n i\ny=2\nz=3'
            self.assertEqual(native_math_rows(atom, atom['content'], extra),
                r'\begin{aligned}x&=\sum_{i=1}^n i\\y&=2\\z&=3\end{aligned}')

    def test_edited_scan_nested_and_ambiguous_inputs_abstain(self):
        for defect in ('edited', 'scan', 'missing', 'recovery', 'unbalanced', 'prose',
                       'environment', 'rowbreak', 'comment', 'ampersand', 'multiple_tags',
                       'tag_in_middle', 'relation', 'empty_rhs'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                path, atom = self.source(Path(name)); content = atom['content']
                if defect == 'edited': content += ' '
                if defect == 'scan': atom['extraction_method'] = 'ocr'
                if defect == 'missing': path.unlink()
                if defect == 'recovery': atom['recovery_issue'] = 'unreviewed'
                changes = dict(unbalanced='x = {1\ny = 2\nz = 3', prose='x = 1\nordinary prose\nz = 3',
                    environment=r'\begin{aligned}x &=1\\y &=2\end{aligned}', rowbreak='x=1\\\\\ny=2\nz=3',
                    comment='x=1%\ny=2\nz=3', ampersand='x&=1\ny=2\nz=3',
                    multiple_tags='x=1\\tag{a}\ny=2\nz=3\\tag{b}',
                    tag_in_middle='x=1\\tag{a}\ny=2\nz=3', relation='x<=1\ny=2\nz=3',
                    empty_rhs='x=\ny=2\nz=3')
                if defect in changes: atom['content'] = content = changes[defect]
                self.assertIsNone(native_math_rows(atom, content, path))

    def test_operator_led_continuations_keep_every_relation(self):
        with tempfile.TemporaryDirectory() as name:
            path, atom = self.source(Path(name))
            atom['content'] = 'x=1\n=2\ny=3'
            self.assertEqual(native_math_rows(atom, atom['content'], path),
                             r'\begin{aligned}x&=1\\&=2\\y&=3\end{aligned}')

    def test_explicit_tex_row_breaks_need_the_same_native_baseline_witness(self):
        for single_line in (False, True):
            with self.subTest(single_line=single_line), tempfile.TemporaryDirectory() as name:
                path, atom = self.source(Path(name), single_line=single_line)
                atom['content'] = r'x=1 \\ y=2 \\ z=3\tag{7.2}'
                rendered = native_math_rows(atom, atom['content'], path)
                if single_line:
                    self.assertIsNone(rendered)
                else:
                    self.assertEqual(rendered, r'\begin{aligned}x&=1 \\ y&=2 \\ z&=3\end{aligned}\tag{7.2}')

    def test_compiled_formula_has_separate_readable_rows_and_one_number(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path, atom = self.source(root)
            index = SourceIndex(dict(source_pdf=str(path), atoms=[atom], source_page_count=1))
            audit = write_project(index, dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [], [], dict(title='Equations',author=''), load_config(), root/'latex', sample_mode=True)
            result = compile_project(root/'latex', {t: executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'], result['log_tail'])
            self.assertEqual(result['warnings'], [])
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                self.assertEqual(len(pdf), 1)
                p = pdf[0]
                # Math variables may use Unicode mathematical italic glyphs.
                bounds = [p.search_for(v)[0] for v in ('1', '2', '3')]
                self.assertTrue(bounds[0].y0 < bounds[1].y0 < bounds[2].y0)
                self.assertTrue(all(b.height >= 8 for b in bounds))
                self.assertEqual(p.get_text().count('(7.2)'), 1)
                self.assertIn('source-eq', pdf.resolve_names())
