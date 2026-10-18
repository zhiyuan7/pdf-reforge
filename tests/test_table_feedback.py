"""Actual table failures join image feedback without changing source cells."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from book_pipeline.ai.client import DeepSeekClient
from book_pipeline.config import executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.formula_repair import compile_with_formula_recovery, locate_formula_errors, locate_source_errors
from book_pipeline.publishing.latex import Renderer
from book_pipeline.publishing.table_repair import TableRepairer, expand_template, table_template

CONFIG = dict(model='test',base_url='https://example.invalid',api_key_env='TEST_KEY',attempts=3,
              max_tokens=1000,thinking='disabled',shared_cache_dirs=[],pricing={},concurrency=1)


def source(folder, rows=14):
    html = '<table>'+''.join('<tr><td>'+('[General Information]' if n==0 else 'value '+str(n))+'</td></tr>' for n in range(rows))+'</table>'
    picture = Path(folder)/'original.png'
    Image.new('RGB',(160,90),'white').save(picture)
    atom = dict(id='table-1',group='g',kind='table',role='table_body',block_type='table',
                content=html,length=len(html),page=0,bbox=[.1,.1,.9,.4],asset=str(picture),original_atom='p000001-b000000-s0000')
    return SourceIndex(dict(atoms=[atom],source_pdf='unused.pdf',input_sha256='source',mineru_sha256='ocr'))


def render(index, directory, replacement=None, in_box=False):
    decisions = dict(objects=[],exclusions=[],corrections=[],joins=[],bold_spans=[],page_footnotes=[],
                     table_repairs={} if replacement is None else {'table-1':replacement})
    renderer = Renderer(index,decisions,[],[],Path(directory))
    atom = index.atoms[0]
    return renderer.content_fragment(atom,0,atom['length'],in_box)


def project(folder, fragment):
    folder = Path(folder)
    (folder/'chapters').mkdir(parents=True,exist_ok=True)
    (folder/'main.tex').write_text(r'\documentclass{article}\usepackage{booktabs,longtable,adjustbox}'+'\n'+r'\begin{document}\input{chapters/generated-001.tex}\end{document}')
    (folder/'chapters/generated-001.tex').write_text(fragment)
    line = next(n for n,t in enumerate(fragment.splitlines(),1) if t.startswith('[General Information]'))
    return dict(file='chapters/generated-001.tex',line=line,message='Missing number, treated as zero.')


class TableFeedbackTests(unittest.TestCase):
    def test_only_the_marked_table_is_sent_to_the_table_handler(self):
        with tempfile.TemporaryDirectory() as folder:
            index = source(folder)
            fragment = render(index,folder)
            error = project(Path(folder)/'latex',fragment)
            errors = dict(error_details=[error,dict(file='../outside.tex',line=1),dict(file=error['file'],line=1)])
            self.assertEqual(locate_formula_errors(Path(folder)/'latex',errors,index.atoms),[])
            self.assertEqual([a['id'] for a,e in locate_source_errors(Path(folder)/'latex',errors,index.atoms,{'table'})],['table-1'])

    def test_model_layout_cannot_drop_reorder_rewrite_or_hide_source_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            index = source(folder)
            atom = index.atoms[0]
            data = table_template(atom['content'])
            valid = dict(kind='table_template',template=data['template'].replace('@@CELL0001@@','{@@CELL0001@@}'),in_box=False)
            rendered = expand_template(atom['content'],valid)
            for cell in data['cells']:
                self.assertIn(cell['latex'],rendered)
            bad = [data['template'].replace('@@CELL0001@@',''),
                   data['template'].replace('@@CELL0001@@','@@CELL0002@@'),
                   data['template']+'\n\\input{secret}',
                   data['template']+'\nNEWWORDS',
                   data['template'].replace('@@CELL0001@@','%@@CELL0001@@'),
                   data['template'].replace('longtable','adjustbox'),
                   data['template']+'\n\\begin{document}']
            for template in bad:
                with self.subTest(template=template),self.assertRaises(ValueError):
                    expand_template(atom['content'],{**valid,'template':template})

    def test_render_only_replacement_keeps_original_html_offsets_and_cells(self):
        with tempfile.TemporaryDirectory() as folder:
            index = source(folder)
            before = deepcopy(index.source)
            data = table_template(index.atoms[0]['content'])
            replacement = dict(kind='table_template',template=data['template'].replace('@@CELL0001@@','{@@CELL0001@@}'),in_box=False)
            rendered = render(index,folder,replacement)
            self.assertIn('{[General Information]}',rendered)
            self.assertIn('% source-atom: table-1',rendered)
            self.assertEqual(index.source,before)
            self.assertEqual(index.prefix[-1],len(index.atoms[0]['content']))

    def test_whole_book_flow_sends_image_code_error_and_restores_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            index = source(root)
            before = deepcopy(index.source)
            requests = []
            generated = []
            def generate(replacements):
                generated.append(deepcopy(replacements))
                fragment = render(index,root,replacements.get('table-1'))
                if len(generated)==1:
                    error = project(root/'latex',fragment)
                    failure['error_details']=[error]
                else:
                    (root/'latex/chapters/generated-001.tex').write_text(fragment)
                return dict(coverage_complete=True)
            def transport(task,messages,config):
                requests.append(messages)
                payload=json.loads(messages[1]['content'][0]['text'])
                return dict(content=json.dumps(dict(latex=payload['table_template'].replace('@@CELL0001@@','{@@CELL0001@@}'))),finish_reason='stop',usage={})
            failure = dict(success=False,log_tail='Missing number, treated as zero.')
            factory=lambda config,cache,**kw:DeepSeekClient(config,cache,transport=transport,sleep=lambda _:None,**kw)
            with patch('book_pipeline.publishing.table_repair.DeepSeekClient',side_effect=factory),patch.object(TableRepairer,'check',return_value=None),patch('book_pipeline.publishing.formula_repair.compile_project',side_effect=[failure,dict(success=True)]):
                _,compilation,summary=compile_with_formula_recovery(index,CONFIG,root/'latex',root/'reports',{},generate,lambda *args:None)
            self.assertTrue(compilation['success'])
            self.assertEqual(summary['table_repairs']['repaired_atoms'],1)
            self.assertEqual(summary['repaired_atoms'],0)
            payload=json.loads(requests[0][1]['content'][0]['text'])
            self.assertIn('[General Information]',payload['current_table_latex'])
            self.assertIn('Missing number',payload['compiler_error']['message'])
            self.assertIn('% source-table-in-box: 0',payload['generated_source_fragment'])
            self.assertTrue(any(p.get('type')=='image_url' for p in requests[0][1]['content']))
            resumed=TableRepairer(index,CONFIG,root/'reports/table_repair',{})
            self.assertEqual(resumed.replacements,generated[-1])
            self.assertEqual(index.source,before)

    def test_boxed_table_uses_its_actual_rendering_context(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);index=source(root,rows=2)
            error=project(root/'latex',render(index,root,in_box=True))
            repairer=TableRepairer(index,CONFIG,root/'repair',{})
            def transport(task,messages,config):
                payload=json.loads(messages[1]['content'][0]['text'])
                return dict(content=json.dumps(dict(latex=payload['table_template'].replace('@@CELL0001@@','{@@CELL0001@@}'))),finish_reason='stop')
            repairer.client=DeepSeekClient(CONFIG,root/'cache',transport=transport,sleep=lambda _:None)
            with patch.object(TableRepairer,'check',return_value=None):repairer.repair(root/'latex',index.atoms[0],error)
            replacement=repairer.replacements['table-1']
            self.assertTrue(replacement['in_box'])
            self.assertIn('{[General Information]}',render(index,root,replacement,in_box=True))
            with self.assertRaises(ValueError):render(index,root,replacement,in_box=False)


@unittest.skipUnless(os.environ.get('PDF_REPAIR_REAL_TEX')=='1','Enable real XeLaTeX table feedback checks')
class RealTableFeedbackTests(unittest.TestCase):
    def test_original_bracket_failure_and_more_than_three_feedback_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);index=source(root);before=deepcopy(index.source)
            error=project(root/'latex',render(index,root))
            repairer=TableRepairer(index,CONFIG,root/'repair',dict(xelatex=executable('xelatex')))
            data=table_template(index.atoms[0]['content'])
            original=dict(kind='table_template',template=data['template'],in_box=False)
            with self.assertRaisesRegex(ValueError,'Missing number'):repairer.check(root/'latex',index.atoms[0],original)
            requests=[]
            def transport(task,messages,config):
                requests.append(deepcopy(messages))
                template=data['template'] if len(requests)<5 else data['template'].replace('@@CELL0001@@','{@@CELL0001@@}')
                return dict(content=json.dumps(dict(latex=template)),finish_reason='stop',usage={})
            repairer.client=DeepSeekClient(CONFIG,root/'cache',transport=transport,sleep=lambda _:None)
            repairer.repair(root/'latex',index.atoms[0],error)
            self.assertEqual(len(requests),5)
            self.assertEqual(repairer.records[0]['syntax_validation'],'passed')
            self.assertIn('Missing number',requests[-1][-1]['content'])
            self.assertTrue(all(any(p.get('type')=='image_url' for p in m[1]['content']) for m in requests))
            self.assertEqual(index.source,before)


if __name__=='__main__':unittest.main()
