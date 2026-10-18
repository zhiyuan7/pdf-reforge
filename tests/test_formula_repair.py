"""Formula compatibility and image-repair regressions; no live API in tests."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from book_pipeline.ai.client import DeepSeekClient
from book_pipeline.config import executable, read
from book_pipeline.ingestion.math_syntax import compatible_equation_tag
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.formula_repair import FormulaRepairer, compile_with_formula_recovery, locate_formula_errors
from book_pipeline.publishing.latex import Renderer, table_tex


CONFIG = dict(model='test-model', base_url='https://example.invalid', api_key_env='TEST_KEY',
              attempts=1, max_tokens=1024, thinking='disabled', shared_cache_dirs=[], pricing={})


def index(content=r'\logit(p)', kind='display_math', asset=None):
    atom = dict(id='edited-000001', group='B00001', kind=kind, role='equation', block_type='equation', content=content,
                length=len(content), page=0, bbox=[.1, .1, .9, .3], original_atom='p000001-b000001-s0000')
    if asset:
        atom['asset'] = str(asset)
    return SourceIndex(dict(source_pdf='unused.pdf', input_sha256='source-sha', mineru_sha256='ocr-sha', atoms=[atom]))


class FormulaCompatibilityTests(unittest.TestCase):
    def test_math_number_is_wrapped_and_existing_forms_are_unchanged(self):
        tag = r'\tag{*_{1}}'
        fixed = r'\tag{\ensuremath{*_{1}}}'
        self.assertEqual(compatible_equation_tag(tag), fixed)
        self.assertEqual(compatible_equation_tag(fixed), fixed)
        self.assertEqual(compatible_equation_tag(r'\tag*{3^2}'), r'\tag*{\ensuremath{3^2}}')
        for unchanged in (r'\tag{3.2}', r'\tag{a\_b}', r'\tag{$*_1$}', r'\tag{\(*_1\)}'):
            self.assertEqual(compatible_equation_tag(unchanged), unchanged)

    def test_renderer_keeps_native_math_number_without_requiring_an_image(self):
        source = index(r'P=1\tag{*_1}')
        before = deepcopy(source.source)
        decisions = dict(objects=[], exclusions=[], corrections=[], joins=[], bold_spans=[], page_footnotes=[])
        with tempfile.TemporaryDirectory() as folder:
            renderer = Renderer(source, decisions, [], [], Path(folder))
            atom = source.atoms[0]
            output = renderer.content_fragment(atom, 0, atom['length'], False)
        self.assertIn(r'\tag{\ensuremath{*_1}}', output)
        self.assertNotIn('BookImage', output)
        self.assertEqual(source.source, before)

    def test_render_only_replacement_keeps_source_offsets_and_content(self):
        source = index()
        before = deepcopy(source.source)
        decisions = dict(objects=[], exclusions=[], corrections=[], joins=[], bold_spans=[], page_footnotes=[],
                         formula_repairs={'edited-000001': r'\operatorname{logit}(p)'})
        with tempfile.TemporaryDirectory() as folder:
            renderer = Renderer(source, decisions, [], [], Path(folder))
            output = renderer.content_fragment(source.atoms[0], 0, source.atoms[0]['length'], False)
        self.assertIn(r'\operatorname{logit}', output)
        self.assertEqual(source.source, before)
        self.assertEqual(source.prefix[-1], len(r'\logit(p)'))

    def test_error_mapping_does_not_guess_from_adjacent_prose_or_outside_paths(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root/'chapters').mkdir()
            (root/'chapters/generated-001.tex').write_text(
                r'\hypertarget{source-edited-000001}{}% source-atom: edited-000001'+'\n$bad$\n% end-source-atom: edited-000001\nordinary prose\n')
            errors = {'error_details': [dict(file='chapters/generated-001.tex', line=2, message='Undefined control sequence.'),
                                       dict(file='chapters/generated-001.tex', line=4, message='Other error'),
                                       dict(file='../outside.tex', line=2, message='Other error')]}
            matches = locate_formula_errors(root, errors, index().atoms)
            self.assertEqual([a['id'] for a,e in matches], ['edited-000001'])
            errors['error_details'] = errors['error_details'][1:]
            self.assertEqual(locate_formula_errors(root, errors, index().atoms), [])

    def test_malformed_math_cell_preserves_table_image_instead_of_broken_tex(self):
        valid='<table><tr><td><eq>x_i</eq></td><td>10</td></tr></table>'
        self.assertIsNotNone(table_tex(valid))
        for broken in ['<eq>$x_i</eq>',r'<eq>\(x_i</eq>',r'<eq>x_i}</eq>']:
            self.assertIsNone(table_tex('<table><tr><td>'+broken+'</td><td>10</td></tr></table>'))
        with tempfile.TemporaryDirectory() as folder:
            picture=Path(folder)/'table.png';Image.new('RGB',(100,50),'white').save(picture)
            html='<table><tr><td><eq>$x_i</eq></td><td>10</td></tr></table>'
            source=index(html,kind='table',asset=picture);before=deepcopy(source.source)
            decisions=dict(objects=[],exclusions=[],corrections=[],joins=[],bold_spans=[],page_footnotes=[])
            (Path(folder)/'render').mkdir()
            renderer=Renderer(source,decisions,[],[],Path(folder)/'render')
            output=renderer.content_fragment(source.atoms[0],0,source.atoms[0]['length'],False)
            self.assertIn('BookImage',output);self.assertEqual(source.source,before)

    def test_model_cannot_change_number_or_supply_document_commands(self):
        source = index(r'x=1\tag{*_1}')
        with tempfile.TemporaryDirectory() as name:
            repairer = FormulaRepairer(source, CONFIG, Path(name), {})
            for answer in [dict(latex='x=1',number='2'),dict(latex=r'\input{secret}',number='*_1'),
                           dict(latex='$x=1$',number='*_1'),dict(latex=r'x=1\tag{2}',number='*_1')]:
                with self.assertRaises(ValueError):repairer.content(answer,source.atoms[0])
            self.assertEqual(repairer.content(dict(latex='x=1',number='*_1'),source.atoms[0]), r'x=1\tag{*_1}')


class RecoveryFlowTests(unittest.TestCase):
    def run_flow(self, directory, successful=True, located=True):
        picture = directory/'original.png';Image.new('RGB',(80,30),'white').save(picture)
        source = index(asset=picture);original = deepcopy(source.source)
        reports=directory/'reports';reports.mkdir();project=directory/'latex';project.mkdir();(project/'chapters').mkdir()
        generated=[];requests=[]
        def generate(replacements):
            generated.append(dict(replacements))
            (project/'main.tex').write_text(r'\documentclass{article}\begin{document}\end{document}')
            (project/'chapters/generated-001.tex').write_text(
                '% source-atom: edited-000001\n$'+replacements.get('edited-000001',r'\logit(p)')+'$\n% end-source-atom: edited-000001\nprose\n')
            return dict(source_units=source.prefix[-1],coverage_complete=True)
        def transport(task,messages,config):
            requests.append(messages)
            return dict(content=json.dumps(dict(latex=r'\operatorname{logit}(p)',number='')),
                        finish_reason='stop',usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15))
        failure=dict(success=False,exit_code=1,error_details=[dict(file='chapters/generated-001.tex',line=2 if located else 4,message='Undefined control sequence.')])
        success=dict(success=True,exit_code=0,error_details=[])
        factory=lambda config,cache,**kw:DeepSeekClient(config,cache,transport=transport,sleep=lambda _:None,**kw)
        with patch('book_pipeline.publishing.formula_repair.DeepSeekClient',side_effect=factory), \
             patch.object(FormulaRepairer,'check',return_value=None), \
             patch('book_pipeline.publishing.formula_repair.compile_project',side_effect=[deepcopy(failure),success if successful else deepcopy(failure)]):
            coverage,compilation,recovery=compile_with_formula_recovery(source,CONFIG,project,reports,{},generate,lambda *a:None,max_rounds=1)
        self.assertEqual(source.source,original)
        return source,project,reports,generated,requests,compilation,recovery

    def test_error_sends_image_recompiles_and_persists_resumable_replacement(self):
        with tempfile.TemporaryDirectory() as name:
            source,project,reports,generated,requests,compilation,recovery=self.run_flow(Path(name))
            self.assertTrue(compilation['success']);self.assertEqual(recovery['status'],'complete')
            self.assertEqual(generated,[{}, {'edited-000001':r'\operatorname{logit}(p)'}])
            content=requests[0][1]['content']
            self.assertTrue(any(p.get('image_url',{}).get('url','').startswith('data:image/png;base64,') for p in content))
            resumed=FormulaRepairer(source,CONFIG,reports/'formula_repair',{})
            self.assertEqual(resumed.replacements,generated[-1])
            self.assertEqual(len(list((reports/'compile_history').glob('attempt-*.json'))),2)

    def test_unlocated_errors_do_not_send_unrelated_images(self):
        with tempfile.TemporaryDirectory() as name:
            *_,requests,compilation,recovery=self.run_flow(Path(name),located=False)
            self.assertEqual(requests,[]);self.assertFalse(compilation['success'])
            self.assertEqual(compilation['repair_failure'],'compiler_error_not_in_a_supported_source_block')

    def test_recovery_limit_never_reports_success(self):
        with tempfile.TemporaryDirectory() as name:
            *_,requests,compilation,recovery=self.run_flow(Path(name),successful=False)
            self.assertEqual(len(requests),1);self.assertFalse(compilation['success'])
            self.assertEqual(recovery['status'],'unresolved')
            self.assertEqual(compilation['repair_failure'],'source_repair_round_limit')


@unittest.skipUnless(os.environ.get('PDF_REPAIR_REAL_TEX')=='1','Enable production XeLaTeX syntax checks explicitly')
class RealFormulaCompilationTests(unittest.TestCase):
    def test_real_compiler_accepts_number_and_repaired_operators_and_cases(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name);project=directory/'latex';project.mkdir()
            (project/'main.tex').write_text(r'\documentclass{article}\usepackage{amsmath,amssymb}\begin{document}\end{document}')
            source=index();repairer=FormulaRepairer(source,CONFIG,directory/'repairs',{'xelatex':executable('xelatex')})
            atom=source.atoms[0]
            repairer.check(project,atom,r'P=1\tag{*_1}')
            with self.assertRaises(ValueError):repairer.check(project,atom,r'\logit(p)')
            repairer.check(project,atom,r'\operatorname{logit}(p)')
            with self.assertRaises(ValueError):repairer.check(project,atom,r'\begin{cases}x&x>0&n>0\\0&x=0\end{cases}')
            repairer.check(project,atom,r'\left\{\begin{array}{lll}x&x>0&n>0\\0&x=0&\end{array}\right.')


if __name__=='__main__':unittest.main()
