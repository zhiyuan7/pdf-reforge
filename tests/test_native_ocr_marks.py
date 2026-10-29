"""Live glyphs may restore bounded OCR punctuation, never hidden scan text."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.native_ocr_marks import group_ending, native_ocr_marks


class NativeOCRMarkTests(unittest.TestCase):
    def test_reread_paragraph_retains_native_period_and_square_ownership(self):
        from book_pipeline.ingestion.formula_bands import apply_formula_corrections,METHOD
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=self.source(root)
            source=SourceIndex({**source.source,'input_sha256':'p','mineru_sha256':'m','pages':[dict(page=0)]})
            band=dict(id='band',page=0,bbox=[0,.1,1,.2],groups=['g'],preserve=[],
                answer=dict(items=[dict(kind='text',text=source.atoms[0]['content']+'$x$')]))
            output=apply_formula_corrections(source,dict(status='complete',method=METHOD,
                source=dict(input_sha256='p',mineru_sha256='m'),records=[band],usage={}))
            self.assertEqual(output.atoms[0]['bbox'],source.atoms[0]['bbox'])
            self.assertEqual(output.atoms[0]['extraction_method'],'ocr')
            punctuation,marks,_=native_ocr_marks(output,root/'render')
            self.assertEqual(list(punctuation),[output.atoms[-1]['id']])
            self.assertEqual(list(marks),[output.atoms[-1]['id']])

    def source(self, root, *, square=True, hidden=False, tail='x.', gray=False, outlined=False):
        prose = 'The numerical solution converges rapidly to '
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=500, height=700)
            page.insert_text((50, 100), prose+tail, fontsize=11, render_mode=3 if hidden else 0)
            if square:
                if outlined:
                    page.draw_rect((440, 94, 446, 100), color=(0, 0, 0))
                else:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 1, 1))
                    pix.clear_with(180 if gray else 0)
                    page.insert_image((440, 94, 446, 100), stream=pix.tobytes('png'))
            pdf.save(root/'source.pdf')
        box = [.095, .12, .895, .15]
        atoms = [dict(id=str(n), group='g', page=0, kind=k, role='text', block_type='text',
                      content=t, length=len(t), bbox=box, block_bbox=box, extraction_method='ocr')
                 for n, (k, t) in enumerate([('text', prose), ('math', 'x')])]
        return SourceIndex(dict(source_pdf=str(root/'source.pdf'), atoms=atoms))

    def test_live_complete_paragraph_restores_period_and_isolated_bitmap_square(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); index = self.source(root)
            original = deepcopy(index.source)
            p, m, evidence = native_ocr_marks(index, root/'project')
            self.assertEqual(set(p), {'1'})
            self.assertEqual(p['1']['text'], '.')
            self.assertEqual(set(m), {'1'})
            self.assertTrue((root/'project'/m['1']['image']).is_file())
            self.assertEqual(len(evidence), 2)
            self.assertEqual(index.source, original)
            audit = write_project(index, dict(corrections=[], joins=[], exclusions=[], bold_spans=[]),
                [], [], dict(title='OCR fallback marks', author=''), load_config(), root/'project', sample_mode=True)
            self.assertTrue(audit['coverage_complete'])
            recovered = [r for r in audit['source_order_layout'] if r.get('method') == p['1']['method']]
            self.assertEqual(len(recovered), 2)
            result = compile_project(root/'project', {t: executable(t) for t in ('xelatex', 'latexmk')})
            self.assertTrue(result['success'], result['log_tail'])
            self.assertEqual(result['warnings'], [])
            with pymupdf.open(root/'project'/'main.pdf') as pdf:
                text = ''.join(p.get_text() for p in pdf)
                self.assertRegex(text, r'[x𝑥]\.\s*$')
                squares = [i for p in pdf for i in p.get_image_info()]
                self.assertEqual(len(squares), 1)
                b = squares[0]['bbox']
                self.assertGreater(b[0], pdf[0].rect.width*.8)

    def test_previously_restored_math_period_is_not_added_twice(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); index = self.source(root)
            p, m, evidence = native_ocr_marks(index, root/'project', rendered_math={'1': 'x.'})
            self.assertEqual(p, {})
            self.assertEqual(set(m), {'1'})
            self.assertEqual([r['kind'] for r in evidence], ['mark'])
            audit = write_project(index, dict(corrections=[], joins=[], exclusions=[], bold_spans=[],
                native_math_styles={'1': dict(latex='x.')}), [], [], dict(title='Marks', author=''),
                load_config(), root/'project', sample_mode=True)
            self.assertEqual([r['kind'] for r in audit['source_order_layout']
                              if r.get('method') == evidence[0]['method']], ['mark'])
            tex = ''.join(p.read_text() for p in (root/'project'/'chapters').glob('*.tex'))
            self.assertIn('$x.$', tex)
            self.assertNotIn('end-source-atom: 1\n.', tex)

    def test_native_and_edited_groups_hidden_ocr_and_unowned_glyphs_abstain(self):
        for scenario in ('native', 'hidden', 'edited', 'exclusion', 'rewritten', 'extra', 'missing_dot', 'panel'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                index = self.source(root, hidden=scenario=='hidden',
                                    tail='x. extra' if scenario=='extra' else 'x' if scenario=='missing_dot' else 'x.')
                if scenario=='native':
                    for a in index.atoms: a['extraction_method'] = 'native'
                if scenario=='rewritten': index.atoms[0]['content'] = 'A different numerical paragraph '
                if scenario=='panel': index.atoms[0]['source_panel'] = {'id': 'panel'}
                p, m, _ = native_ocr_marks(index, root/'project',
                    [dict(lo=0, hi=1)] if scenario=='exclusion' else (), {'0'} if scenario=='edited' else ())
                self.assertEqual((p, m), ({}, {}))

    def test_existing_period_is_not_duplicated_and_square_requires_shape_color_ownership(self):
        for scenario in ('existing_math_period', 'existing_text_period', 'none', 'gray', 'outline', 'competing'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                index = self.source(root, square=scenario!='none', gray=scenario=='gray', outlined=scenario=='outline')
                if scenario=='existing_math_period': index.atoms[-1].update(content='x.', length=2)
                if scenario=='existing_text_period':
                    index = SourceIndex({**index.source, 'atoms': [*index.atoms,
                        {**index.atoms[-1], 'id': 'period', 'kind': 'text', 'content': '.', 'length': 1}]})
                if scenario=='competing':
                    index = SourceIndex({**index.source, 'atoms': [*index.atoms,
                        {**index.atoms[-1], 'id': 'other', 'group': 'other', 'kind': 'image',
                         'bbox': [.88, .13, .895, .15]}]})
                p, m, _ = native_ocr_marks(index, root/'project')
                self.assertEqual(bool(p), scenario not in {'existing_math_period', 'existing_text_period'})
                self.assertEqual(bool(m), scenario=='existing_text_period')

    def test_ambiguous_formula_ownership_and_punctuated_wrappers_abstain(self):
        atoms = [dict(kind=k, content=t) for k, t in [('text', 'A complete mathematical sentence containing '),
            ('math', 'x'), ('text', ' same '), ('math', 'y')]]
        glyphs = [dict(c=c) for c in 'A complete mathematical sentence containing x same y same z.']
        self.assertIsNone(group_ending(atoms, glyphs))
        for tail in [r'\mathrm{x.}', r'x\ldots', 'x,', r'x\text{.}', r'x\blacksquare', 'x■']:
            atoms[-1]['content'] = tail
            self.assertIsNone(group_ending(atoms, glyphs))


if __name__ == '__main__':
    unittest.main()
