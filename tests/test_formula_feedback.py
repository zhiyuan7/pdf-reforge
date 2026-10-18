"""Formula feedback keeps the image, candidate and real compiler error together."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from book_pipeline.ai.client import APIFailure, DeepSeekClient
from book_pipeline.publishing.formula_repair import FormulaRepairer, compile_with_formula_recovery


CONFIG=dict(model='test-model',base_url='https://example.invalid',api_key_env='TEST_KEY',
            attempts=3,max_tokens=1024,thinking='disabled',shared_cache_dirs=[],pricing={})


class FormulaFeedbackTests(unittest.TestCase):
    def test_reasoning_budget_exhaustion_switches_to_direct_code_with_same_image(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);picture=directory/'original.png';Image.new('RGB',(50,20),'white').save(picture)
            requests=[]
            def transport(task,messages,config):
                requests.append((deepcopy(messages),config['thinking']))
                if len(requests)==1:return dict(content='',finish_reason='length')
                return dict(content='{"latex":"x=1"}',finish_reason='stop')
            client=DeepSeekClient({**CONFIG,'thinking':'enabled'},directory/'cache',transport=transport,sleep=lambda _:None)
            answer=client.ask('repair','system',{},images=[picture],retry_until_valid=True)
            self.assertEqual(answer['latex'],'x=1')
            self.assertEqual([thinking for _,thinking in requests],['enabled','disabled'])
            self.assertEqual(requests[0][0][1],requests[1][0][1])
            self.assertIn('finish_reason=length',requests[1][0][-1]['content'])

    def test_full_book_feedback_continues_beyond_the_previous_32_round_limit(self):
        from book_pipeline.ingestion.source import SourceIndex
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);project=directory/'latex';(project/'chapters').mkdir(parents=True)
            atom=dict(id='a',group='g',page=0,kind='display_math',content='x',length=1)
            source=SourceIndex(dict(atoms=[atom]))
            def generate(replacements):
                (project/'main.tex').write_text(r'\documentclass{article}\begin{document}\end{document}')
                (project/'chapters/generated-001.tex').write_text('% source-atom: a\nx\n% end-source-atom: a\n')
                return {}
            failure=dict(success=False,error_details=[dict(file='chapters/generated-001.tex',line=2,message='error')])
            def repair(repairer,*args):repairer.replacements['a']='x'
            with patch('book_pipeline.publishing.formula_repair.compile_project',side_effect=[failure]*35+[dict(success=True)]),patch.object(FormulaRepairer,'repair',repair):
                _,compilation,summary=compile_with_formula_recovery(source,CONFIG,project,directory/'reports',{},generate,lambda *args:None)
            self.assertTrue(compilation['success']);self.assertEqual(summary['status'],'complete')
            self.assertEqual(len(list((directory/'reports/compile_history').glob('attempt-*.json'))),36)

    def test_more_than_three_failed_candidates_retain_image_code_and_precise_error(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);picture=directory/'formula.png';Image.new('RGB',(100,40),'white').save(picture)
            requests=[]
            def transport(task,messages,config):
                requests.append(deepcopy(messages));n=len(requests)
                return dict(content=json.dumps(dict(latex='x='+str(n))),finish_reason='stop')
            def validate(answer):
                if answer['latex']!='x=5':raise ValueError('formula.tex:30: Missing control sequence inserted.\n'+answer['latex'])
            client=DeepSeekClient(CONFIG,directory/'cache',transport=transport,sleep=lambda _:None)
            answer=client.ask('repair','Read the original formula',{},images=[picture],validator=validate,retry_until_valid=True)
            self.assertEqual(answer['latex'],'x=5');self.assertEqual(len(requests),5)
            for messages in requests:
                self.assertTrue(any(p.get('type')=='image_url' for p in messages[1]['content']))
            self.assertIn('x=4',requests[-1][-2]['content'])
            self.assertIn('Missing control sequence inserted.',requests[-1][-1]['content'])
            self.assertLessEqual(len(requests[-1]),8)
            self.assertEqual(len(list((directory/'cache/invalid').glob('*.json'))),4)

    def test_cancelled_feedback_resumes_with_previous_candidate_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            def transport(*args):return dict(content='{"latex":"bad"}',finish_reason='stop')
            client=DeepSeekClient(CONFIG,directory,transport=transport,sleep=lambda _:client.cancel.set())
            with self.assertRaises(APIFailure):
                client.ask('repair','system',{},validator=lambda _:(_ for _ in ()).throw(ValueError('actual error')),retry_until_valid=True)
            requests=[]
            def corrected(task,messages,config):
                requests.append(messages);return dict(content='{"latex":"good"}',finish_reason='stop')
            resumed=DeepSeekClient(CONFIG,directory,transport=corrected,sleep=lambda _:None)
            resumed.ask('repair','system',{},retry_until_valid=True)
            self.assertIn('bad',requests[0][-2]['content']);self.assertIn('actual error',requests[0][-1]['content'])
            self.assertEqual(len(list((directory/'invalid').glob('*.json'))),1)

    def test_api_outage_stops_with_failure_instead_of_looping_forever(self):
        with tempfile.TemporaryDirectory() as folder:
            calls=[]
            def transport(*args):calls.append(1);raise OSError('service unavailable')
            client=DeepSeekClient(CONFIG,folder,transport=transport,sleep=lambda _:None)
            with self.assertRaises(APIFailure):client.ask('repair','system',{},retry_until_valid=True)
            self.assertEqual(len(calls),3)

    def test_other_tasks_still_use_the_configured_attempt_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            calls=[]
            def transport(*args):calls.append(1);return dict(content='',finish_reason='stop')
            client=DeepSeekClient(CONFIG,folder,transport=transport,sleep=lambda _:None)
            with self.assertRaises(RuntimeError):client.ask('ordinary','system',{})
            self.assertEqual(len(calls),3)

    def test_local_compiler_feedback_ignores_exclamation_marks_in_package_loading(self):
        from book_pipeline.ingestion.source import SourceIndex
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder);project=directory/'latex';project.mkdir()
            (project/'main.tex').write_text(r'\documentclass{article}\begin{document}\end{document}')
            atom=dict(id='a',group='g',page=0,kind='display_math',content='x',length=1)
            repairer=FormulaRepairer(SourceIndex(dict(atoms=[atom])),CONFIG,directory/'repair',dict(xelatex='/bin/false'))
            output='font loaded!\n(/tex/mathtools.sty)\n/tmp/formula.tex:30: Missing control sequence inserted.\n<inserted text>\n\\inaccessible\nl.30 \\begin{pmatrix}\n'
            with patch('book_pipeline.publishing.formula_repair.subprocess.run',return_value=type('Result',(),dict(returncode=1,stdout=output,stderr=''))()):
                with self.assertRaises(ValueError) as caught:repairer.check(project,atom,r'\begin{pmatrix}A\\B\end{pmatrix}')
            self.assertIn('Missing control sequence inserted.',str(caught.exception))
            self.assertIn('Rendered candidate:',str(caught.exception))
            self.assertNotIn('font loaded!',str(caught.exception))


if __name__=='__main__':unittest.main()
