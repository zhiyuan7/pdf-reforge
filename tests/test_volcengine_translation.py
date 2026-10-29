"""Dedicated MT cannot change formulas, bypass limits, or spend twice on resume."""
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from book_pipeline.ai.budget import BudgetExceeded
from book_pipeline.ai.client import APIFailure, DecisionUncertain, DeepSeekClient
from book_pipeline.ai.volcengine import VolcengineTranslationClient, chunks, signed_headers, normalize_written_numbers, normalize_prose_wraps
from book_pipeline.config import digest, load_config, read
from book_pipeline.ingestion.translation import Translator, translate_source, translate_catalog, protect
from tests.test_auto_ingestion import middle_source


class VolcengineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(os.environ, VOLCENGINE_ACCESS_KEY_ID='test-access-key',
                         VOLCENGINE_SECRET_ACCESS_KEY='test-secret-key')
        env.start(); self.addCleanup(env.stop)
        self.config = load_config()
        self.config['budget'] = {**self.config['budget'], 'ledger':str(self.root/'budget.db')}
        self.config['_budget_job_id'] = 'same-original-book'
        self.config['translation_service'] = dict(provider='volcengine', glossary={'deep learning':'深度学习'},
            free_quota=dict(month=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m'),
                            characters=2_000_000, evidence='User verified unused allowance'))
        self.calls = []

    def transport(self, body):
        self.calls.append(body)
        return dict(TranslationList=[dict(Translation='中文 '+text.replace('If ', '若 ').replace(' then ', ' 则 '))
                                     for text in body['TextList']])

    def client(self, cache='cache', transport=None):
        return VolcengineTranslationClient(self.config, self.root/cache, transport or self.transport)

    def test_math_digits_and_glossary_survive_and_resume_uses_cache(self):
        items = [dict(id='a',text='Deep learning: if $x<2$, then $y=3$; Figure 1.2.')]
        translator = Translator(self.client(), {'deep learning':'深度学习'})
        result = translator.translate(items)['a']
        self.assertIn('深度学习（deep learning）',result)
        for value in ('$x<2$','$y=3$','1.2'): self.assertIn(value,result)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(Translator(self.client(),{'deep learning':'深度学习'}).translate(items)['a'],result)
        self.assertEqual(len(self.calls),1)
        with sqlite3.connect(self.root/'budget.db') as db:
            self.assertEqual(db.execute('SELECT state,cny FROM charges').fetchall(), [('settled',0.0)])
            self.assertEqual(db.execute('SELECT COUNT(*) FROM translation_characters').fetchone()[0],1)

    def test_missing_formula_is_rejected_then_original_prose_is_retranslated_separately(self):
        calls=[]
        def broken(body):
            calls.extend(body['TextList'])
            return dict(TranslationList=[dict(Translation='中文') for text in body['TextList']])
        client=self.client(transport=broken)
        self.assertEqual(Translator(client).translate([dict(id='a',text='Compute $x+1$.')])['a'],'中文 $x+1$.')
        self.assertEqual(calls,['Compute ZXQREF000000QXZ.','Compute'])
        fallback=next((client.cache/'integrity_fallbacks').glob('*.json'))
        self.assertTrue(read(fallback)['semantic_review_required'])

    def test_spaced_tokens_preserve_exact_identity_count_and_diagnostics(self):
        for output,passes in [('计算 ZX Q REF 000000 QX Z。',True),
                              ('计算 ZX Q REF 000001 QX Z。',False),
                              ('计算 ZX Q REF 000000 QX Z 和 ZXQREF000000QXZ。',False)]:
            with self.subTest(output=output):
                def response(body):return dict(TranslationList=[dict(Translation=output)])
                client=self.client('spaces-'+str(len(self.calls))+str(passes)+str(len(output)),transport=response)
                translator=Translator(client)
                if passes:
                    self.assertEqual(translator.translate([dict(id='a',text='Compute $x+1$.')])['a'],'计算 $x+1$。')
                    self.assertEqual(len(list((client.cache/'normalizations').glob('*.json'))),1)
                else:
                    with self.assertRaises(DecisionUncertain):translator.translate([dict(id='a',text='Compute $x+1$.')])

    def test_readable_terms_in_short_titles_are_still_counted_and_validated(self):
        seen=[]
        def response(body):
            seen.extend(body['TextList'])
            return dict(TranslationList=[dict(Translation=text.replace(' for inputs','用于输入')) for text in body['TextList']])
        client=self.client('visible-terms',transport=response)
        translator=Translator(client,{'convolutional networks':'卷积网络'})
        self.assertEqual(translator.translate([dict(id='a',text='Convolutional networks for inputs')])['a'],
                         '卷积网络（convolutional networks）用于输入')
        self.assertEqual(seen,['卷积网络 for inputs'])
        for result in ['卷积网络用于卷积网络输入','卷积网用于输入 42']:
            def changed(body):return dict(TranslationList=[dict(Translation=result)])
            translator=Translator(self.client('changed-'+str(len(result)),transport=changed),{'convolutional networks':'卷积网络'})
            with self.assertRaises(DecisionUncertain):translator.translate([dict(id='a',text='Convolutional networks for inputs')])

    def test_damaged_numeric_tokens_never_enter_fallback_prose_requests(self):
        for prose,passes in [('标题',True),('16 标题 276',False)]:
            calls=[]
            def response(body):
                calls.extend(body['TextList'])
                text=body['TextList'][0]
                result='ZXQREF0000QXZ 标题 ZXQREF000001QXZ' if 'ZXQREF' in text else prose
                return dict(TranslationList=[dict(Translation=result)])
            translator=Translator(self.client('digits-'+str(passes),transport=response))
            if passes:
                self.assertEqual(translator.translate([dict(id='a',text='15 Title 276')])['a'],'15 标题 276')
            else:
                with self.assertRaises(DecisionUncertain):translator.translate([dict(id='a',text='15 Title 276')])
            self.assertEqual(calls,['ZXQREF000000QXZ Title ZXQREF000001QXZ','Title'])

    def test_mixed_script_term_fallback_translates_only_original_surrounding_prose(self):
        calls=[]
        def response(body):
            calls.extend(body['TextList'])
            return dict(TranslationList=[dict(Translation=text.replace('Transformer','变压器').replace('Use','使用'))
                                         for text in body['TextList']])
        client=self.client('alias',transport=response)
        translator=Translator(client,{'transformers':'Transformer模型'})
        self.assertEqual(translator.translate([dict(id='a',text='Use Transformers')])['a'],
                         '使用 Transformer模型（transformers）')
        self.assertEqual(calls,['Use Transformer模型','Use'])
        self.assertEqual(len(list((client.cache/'integrity_fallbacks').glob('*.json'))),1)

    def test_entire_known_term_title_needs_no_machine_translation(self):
        def forbidden(body):raise AssertionError('known title should be local')
        translator=Translator(self.client('local-title',transport=forbidden),{'transformers':'Transformer模型'})
        self.assertEqual(translator.translate([dict(id='a',text='12 Transformers 207')])['a'],
                         '12 Transformer模型（transformers） 207')

    def test_math_heavy_unchanged_prose_is_retranslated_without_sending_formulas(self):
        calls=[]
        def response(body):
            calls.extend(body['TextList'])
            return dict(TranslationList=[dict(Translation=text if 'ZXQREF' in text else '这里说明矩阵乘法。')
                                         for text in body['TextList']])
        client=self.client('unchanged-math',transport=response)
        result=Translator(client).translate([dict(id='a',text='The matrix $W$ multiplies the preceding layer activations.')])['a']
        self.assertIn('$W$',result)
        self.assertIn('说明',result)
        self.assertTrue(all('ZXQREF' not in text for text in calls[1:]))

    def test_figure_letter_suffixes_and_dimension_identifiers_are_protected_whole(self):
        text='Figure 1.4a and Figure 1.4b have 2D and 3D inputs; 1.2 stays numeric.'
        masked,values=protect(text)
        self.assertEqual(values,['1.4a','1.4b','2D','3D','1.2'])
        self.assertNotIn('.4a',masked)
        self.assertNotIn('.4b',masked)
        translator=Translator(self.client('suffixes'))
        result=translator.translate([dict(id='a',text=text)])['a']
        for value in values:self.assertIn(value,result)
        masked,values=protect('1More properly, this is a footnote. x2 2t item_12 H2O')
        self.assertEqual(values,['1'])
        self.assertTrue(masked.startswith('⟪KEEP_0⟫More'))
        self.assertEqual(protect('the 1980s and 2020s; x2 2t H2O')[1],['1980s','2020s'])
        self.assertEqual(protect('Smith (2020a), 2nd Edition; x2 2t H2O')[1],['2020a','2nd'])
        self.assertEqual(protect('1ensures positive weights.',role='page_footnote')[1],['1'])
        self.assertEqual(protect('1BERT predicts labels.',role='page_footnote')[1],['1'])
        def note(body):return dict(TranslationList=[dict(Translation='ZXQREF000000QXZ确保正权重。')])
        result=Translator(self.client('note-marker',transport=note)).translate([
            dict(id='a',text='1ensures positive weights.',role='page_footnote')])['a']
        self.assertEqual(protect(result)[1],['1'])

    def test_written_number_normalization_requires_exact_source_value_and_count(self):
        self.assertEqual(normalize_written_numbers('twenty-five years','工作25年。'),'工作二十五年。')
        self.assertEqual(normalize_written_numbers('one hundred and two examples','102个例子。'),'一百零二个例子。')
        self.assertEqual(normalize_written_numbers('one million cases','1,000,000个案例。'),'一百万个案例。')
        self.assertEqual(normalize_written_numbers('twenty-five years','25年和42年。'),'二十五年和42年。')
        self.assertEqual(normalize_written_numbers('twenty-five years','25年和25年。'),'二十五年和25年。')
        self.assertEqual(normalize_written_numbers('twenty-five years','ZXQREF000025QXZ ⟪KEEP_25⟫ 2.5'),'ZXQREF000025QXZ ⟪KEEP_25⟫ 2.5')
        self.assertEqual(normalize_written_numbers('the eighties and early nineties','80年代和90年代初'),'八十年代和九十年代初')
        self.assertEqual(normalize_written_numbers('the nineties','80年代和90年代'),'80年代和九十年代')
        self.assertEqual(normalize_written_numbers('between zero and one','在0和1之间'),'在零和一之间')
        self.assertEqual(normalize_written_numbers('twenty-five and six','25和6'),'二十五和六')
        self.assertEqual(normalize_written_numbers('one hundred and two and three','102和3'),'一百零二和三')

    def test_terminal_repeat_count_suffix_does_not_become_an_unrelated_number(self):
        calls=[]
        def response(body):
            calls.extend(body['TextList'])
            return dict(TranslationList=[dict(Translation=text if 'ZXQREF' in text else '重复')
                                         for text in body['TextList']])
        client=self.client('repeat-suffix',transport=response)
        result=Translator(client).translate([dict(id='a',text='How often do we repeat $K$ times?')])['a']
        self.assertIn('$K$ 次？',result)
        self.assertNotIn('times?',calls[1:])

    def test_pdf_wraps_join_before_translation_but_blank_paragraphs_survive(self):
        text='More regions than\nthere are atoms.\n\nA high-\ndimensional input.'
        expected='More regions than there are atoms.\n\nA high-dimensional input.'
        self.assertEqual(normalize_prose_wraps(text),expected)
        self.assertEqual(normalize_prose_wraps('A\r\nB\r\n \r\nC'),'A B\n \nC')
        calls=[]
        def response(body):
            calls.extend(body['TextList'])
            return dict(TranslationList=[dict(Translation='区域比原子更多。')])
        Translator(self.client('wraps',transport=response)).translate([dict(id='a',text='More regions than\nthere are atoms.')])
        self.assertEqual(calls,['More regions than there are atoms.'])
        calls.clear()
        translator=Translator(self.client('term-wraps',transport=self.transport),{'deep learning':'深度学习'})
        self.assertEqual(translator.translate([dict(id='a',text='Deep\nlearning')])['a'],
                         '深度学习（deep learning）')
        self.assertEqual(calls,[])

    def test_source_written_number_can_be_translated_without_allowing_invented_digits(self):
        for translated,passes in [('工作25年。',True),('工作42年。',False)]:
            def response(body):return dict(TranslationList=[dict(Translation=translated)])
            translator=Translator(self.client('number-'+str(passes),transport=response))
            if passes:self.assertEqual(translator.translate([dict(id='a',text='Working twenty-five years.')])['a'],'工作二十五年。')
            else:
                with self.assertRaises(DecisionUncertain):translator.translate([dict(id='a',text='Working twenty-five years.')])

    def test_month_names_allow_only_source_attested_numeric_months(self):
        self.assertEqual(normalize_written_numbers('December release','12月发布。'),'十二月发布。')
        self.assertEqual(normalize_written_numbers('February release','2月发布，3月修订。'),'二月发布，3月修订。')
        self.assertEqual(normalize_written_numbers('May release','5月和5月。'),'五月和5月。')
        self.assertEqual(normalize_written_numbers('It may change.','5月更改。'),'5月更改。')
        self.assertEqual(normalize_written_numbers('December release','1.12月发布。'),'1.12月发布。')
        for translated,passes in [('12月发布。',True),('11月发布。',False)]:
            def response(body):return dict(TranslationList=[dict(Translation=translated)])
            translator=Translator(self.client('month-'+str(passes),transport=response))
            if passes:self.assertEqual(translator.translate([dict(id='a',text='December release.')])['a'],'十二月发布。')
            else:
                with self.assertRaises(DecisionUncertain):translator.translate([dict(id='a',text='December release.')])

    def test_request_groups_respect_sixteen_items_and_five_thousand_characters(self):
        items = [dict(id=str(i),text='A long paragraph about learning. '*90) for i in range(20)]
        # Inspect adapter batching without a prose validator unrelated to this test.
        result = self.client().ask('translate-book-zh','',dict(items=items,glossary={}))
        self.assertEqual(len(result['items']),20)
        self.assertTrue(all(len(b['TextList'])<=16 and sum(map(len,b['TextList']))<=5000 for b in self.calls))
        self.calls.clear()
        self.client('small').ask('translate-book-zh','',dict(items=[dict(id=str(i),text='word') for i in range(20)],glossary={}))
        self.assertEqual([len(b['TextList']) for b in self.calls],[16,4])

    def test_long_prose_splits_without_losing_any_original_characters(self):
        text='Original sentence. '*1000
        self.assertEqual(''.join(chunks(text)),text)
        self.assertTrue(all(len(c)<=4500 for c in chunks(text)))
        with self.assertRaises(DecisionUncertain): chunks('x'*5001)

    def test_unknown_request_retains_allowance_and_charge(self):
        self.config['translation_service'].pop('free_quota')
        def lost(body): raise TimeoutError('sensitive service exception must not reach output')
        with self.assertRaises(APIFailure) as failure:
            self.client(transport=lost)._request(['Hello'])
        self.assertNotIn('sensitive',str(failure.exception))
        with sqlite3.connect(self.root/'budget.db') as db:
            self.assertEqual(db.execute('SELECT state FROM charges').fetchone()[0],'unknown')
            self.assertEqual(db.execute('SELECT SUM(characters) FROM translation_characters').fetchone()[0],5)

    def test_free_quota_is_shared_and_never_reinitialized_on_resume(self):
        self.config['translation_service']['free_quota']['characters']=8
        self.client('one')._request(['12345'])
        self.client('two')._request(['67890'])
        with sqlite3.connect(self.root/'budget.db') as db:
            self.assertEqual(db.execute('SELECT SUM(free_characters) FROM translation_characters').fetchone()[0],8)
            self.assertAlmostEqual(db.execute('SELECT SUM(cny) FROM charges').fetchone()[0],2*49/1_000_000)

    def test_previous_month_attestation_does_not_grant_current_allowance(self):
        self.config['translation_service']['free_quota']['month']='2000-01'
        self.client()._request(['hello'])
        with sqlite3.connect(self.root/'budget.db') as db:
            self.assertEqual(db.execute('SELECT SUM(free_characters) FROM translation_characters').fetchone()[0],0)
            self.assertAlmostEqual(db.execute('SELECT SUM(cny) FROM charges').fetchone()[0],5*49/1_000_000)

    def test_budget_rejection_precedes_transport_and_releases_unsubmitted_quota(self):
        self.config['translation_service'].pop('free_quota')
        self.config['budget']['per_call_cny']=.000001
        with self.assertRaises(BudgetExceeded): self.client()._request(['Hello'])
        self.assertEqual(self.calls,[])
        with sqlite3.connect(self.root/'budget.db') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM translation_characters').fetchone()[0],0)

    def test_full_source_translation_preserves_assets_and_records_provider(self):
        pdf,source=middle_source(self.root,[('paragraph_title','Deep learning'),
            ('text',[dict(type='text',content='If '),dict(type='equation_inline',content='x^2')]),
            ('equation','y=f[x]')],image=True)
        before=digest(source/'middle_json.json')
        out=translate_source(pdf,source,self.root/'translated',self.config,client=self.client())
        self.assertEqual(digest(source/'middle_json.json'),before)
        manifest=read(out/'manifest.json')
        self.assertEqual(manifest['translation']['provider'],'volcengine')
        self.assertIn('x^2',(out/'middle_json.json').read_text())
        self.assertEqual(translate_source(pdf,source,out,self.config,client=self.client()),out)
        other={**self.config,'translation_service':{'provider':'deepseek'}}
        with self.assertRaises(ValueError):translate_source(pdf,source,out,other,client=self.client())

    def test_layout_tasks_never_reach_translation_service(self):
        with self.assertRaises(ValueError):self.client().ask('layout','',{})
        self.assertEqual(self.calls,[])

    def test_pending_catalog_titles_use_volcengine_instead_of_the_layout_client(self):
        layout = DeepSeekClient(self.config,self.root/'layout-cache')
        settings = dict(provider='volcengine',service=self.config['translation_service'],
                        glossary={'deep learning':'深度学习'},titles=[])
        with patch('book_pipeline.ingestion.translation.translation_client',return_value=self.client('catalog')), \
             patch.object(layout,'ask',side_effect=AssertionError('Translation reached layout provider')):
            result = translate_catalog([dict(id='a',title='1 Introduction to deep learning')],dict(translation=settings),layout)
        self.assertIn('深度学习',result[0]['title'])
        self.assertIn('1',result[0]['title'])
        self.assertEqual(result[0]['original_title'],'1 Introduction to deep learning')
        self.assertEqual(len(self.calls),1)

    def test_signature_changes_with_body_and_is_deterministic_at_fixed_time(self):
        date='20261029T110000Z'
        first=signed_headers(b'{}','dummy-ak','dummy-sk',date)
        self.assertEqual(first,signed_headers(b'{}','dummy-ak','dummy-sk',date))
        self.assertNotEqual(first['Authorization'],signed_headers(b'{"TextList":[]}', 'dummy-ak','dummy-sk',date)['Authorization'])
        self.assertNotIn('dummy-sk',json.dumps(first))


if __name__=='__main__': unittest.main()
