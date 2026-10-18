"""Answer sections inherit exercise-style suppression without losing text."""
from copy import deepcopy
from pathlib import Path
import json,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
from book_pipeline.structure.exercise_headings import (METHOD,apply_exercise_context,classify_exercise_headings,classify_saved_plan,heading_signature)
from book_pipeline.structure.page_content import run_page_content
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.ai.client import DeepSeekClient

CONFIG=dict(model='test',base_url='https://example.invalid',api_key_env='TEST_KEY',attempts=1,max_tokens=1000,thinking='disabled',shared_cache_dirs=[],pricing={},concurrency=1)
def heading(i,title,parent=None):
 return dict(toc_id=i,title=title,number='',toc_parent_id=parent,heading_level=2 if parent else 1)
class Client:
 def __init__(self,answer):self.answer=answer;self.calls=[]
 def ask(self,task,prompt,payload,validator,**kwargs):
  self.calls.append((task,prompt,payload));validator(self.answer);return self.answer

class AnswerHeadingTests(unittest.TestCase):
 def test_inherited_scope_keeps_original_headings_and_normal_sibling(self):
  h=[heading('a','参考答案'),heading('child','第一章','a'),heading('ordinary','解析几何')];before=deepcopy(h)
  apply_exercise_context(h,{'a'})
  self.assertEqual([x['exercise_context'] for x in h],[True,True,False])
  self.assertEqual([x['is_exercise_heading'] for x in h],[True,False,False])
  for x,y in zip(h,before):self.assertEqual({k:x[k] for k in y},y)
 def test_inheritance_crosses_unmatched_toc_parent(self):
  h=[heading('a','课后解析'),heading('child','详细步骤','unmatched')]
  apply_exercise_context(h,{'a'},[dict(id='unmatched',parent_id='a')])
  self.assertTrue(h[1]['exercise_context']);self.assertEqual(h[1]['exercise_root'],'a')
 def test_v1_cache_is_reclassified_and_saved_plan_receives_new_scope(self):
  h=[heading('a','答案'),heading('b','矩阵','a'),heading('c','普通教学')]
  old=dict(method='exercise_headings_v1',heading_signature=heading_signature(h),exercise_heading_ids=[])
  plan=dict(sections=[dict(id=x['toc_id'],heading=x,exercise_context=False) for x in h],exercise_heading_classification=old)
  with tempfile.TemporaryDirectory() as p:
   c=Client({'e':['a']});result=classify_saved_plan(plan,c,p)
   self.assertEqual(len(c.calls),1);self.assertEqual(result['exercise_heading_classification']['method'],METHOD)
   self.assertEqual([s['exercise_context'] for s in result['sections']],[True,True,False]);self.assertEqual(plan['exercise_heading_classification'],old)
   self.assertEqual(c.calls[0][2]['headings'][1]['parents'],['答案'])
 def test_current_classification_can_resume_without_another_model_call(self):
  h=[heading('a','课后答案')];c=Client({'e':['a']})
  with tempfile.TemporaryDirectory() as p:
   outline=classify_exercise_headings(dict(headings=h),c,p)
   again=classify_exercise_headings(outline,Client(None),p)
   self.assertTrue(again['headings'][0]['exercise_context']);self.assertEqual(len(c.calls),1)
 def test_model_ids_remain_strictly_validated(self):
  for value in [{'e':['unknown']},{'e':['a','a']},{'e':'a'},{'e':[],'extra':1}]:
   with tempfile.TemporaryDirectory() as p,self.assertRaises(ValueError):classify_exercise_headings(dict(headings=[heading('a','解析')]),Client(value),p)
 def test_cli_content_stage_upgrades_existing_v1_heading_classification(self):
  from book_pipeline.application.pipeline import process_content
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);source=dict(input_sha256='source',mineru_sha256='ocr',mineru_dir=str(root/'ocr'))
   outline=dict(exercise_heading_classification=dict(method='exercise_headings_v1'))
   updated=dict(exercise_heading_classification=dict(method=METHOD))
   report=dict(method='content',status='complete',sections=[],usage={})
   with patch('book_pipeline.application.pipeline.read',side_effect=[dict(source=source),outline,dict(verification={})]),patch('book_pipeline.application.pipeline.DeepSeekClient'),patch('book_pipeline.application.pipeline.classify_exercise_headings',return_value=updated) as classify,patch('book_pipeline.application.pipeline.write_matching_report'),patch('book_pipeline.application.pipeline.run_page_content',return_value=report) as run:
    process_content(SimpleNamespace(source=source),root/'headings',{},root/'content',0,lambda *args:None)
   classify.assert_called_once();self.assertIs(run.call_args.args[1],updated)
 def test_second_stage_keeps_answer_numbers_as_body_and_teaching_numbers_as_subheadings(self):
  atoms=[];headings=[]
  for sid,title in [('ordinary','普通教学'),('answers','参考答案')]:
   aid=f'{sid}-heading';atoms.append(dict(id=aid,group=aid,kind='text',role='text',block_type='text',content=title,length=len(title),page=0,bbox=[.1,.2,.9,.3]))
   h=heading(sid,title);h.update(kind='chapter',toc_category='body',start=dict(atom=aid,offset=0),end=dict(atom=aid,offset=len(title)));headings.append(h)
   for n in range(1,13):
    text=f'{n}. 方法{n}。';aid=f'{sid}-{n}';atoms.append(dict(id=aid,group=aid,kind='text',role='text',block_type='text',content=text,length=len(text),page=0,bbox=[.1,.2,.9,.3]))
  source=SourceIndex(dict(source_pdf='unused.pdf',input_sha256='source',mineru_sha256='ocr',source_page_count=1,atoms=atoms));before=deepcopy(source.source)
  apply_exercise_context(headings,{'answers'})
  def transport(task,messages,config):
   if task.endswith('-subheading'):value={'y':1}
   elif task.endswith('-pair'):value={'m':0}
   else:raise AssertionError(task)
   return dict(content=json.dumps(value),finish_reason='stop',usage={})
  with tempfile.TemporaryDirectory() as p:
   c=DeepSeekClient(CONFIG,Path(p)/'cache',transport=transport)
   result=run_page_content(source,dict(headings=headings,metadata=dict(title='测试书',author='测试')),c,Path(p)/'content')
   by_id={r['section']['id']:r for r in result['sections']}
   self.assertEqual(by_id['answers']['resolved']['s'],[]);self.assertEqual(len(by_id['ordinary']['resolved']['s']),12)
   self.assertEqual(result['statistics']['skipped_exercise_subheadings'],12)
   processed=json.loads((Path(p)/'content/sections/answers/processed.json').read_text())
   self.assertEqual(''.join(part['text'] for u in processed['units'] for part in u['parts']),''.join(f'{n}. 方法{n}。' for n in range(1,13)))
   self.assertEqual(source.source,before)

if __name__=='__main__':unittest.main()
