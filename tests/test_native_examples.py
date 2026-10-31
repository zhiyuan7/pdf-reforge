from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.native_examples import native_examples


class NativeExampleTests(unittest.TestCase):
    def source(self, root, *, mixed=False):
        source=root/'examples.pdf'; rows=[]
        labels=['Input: ', 'Output: ', 'Input: ', 'Output: ']
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=500,height=700)
            for n,label in enumerate(labels):
                y=100+14*n
                page.insert_text((40,y),label,fontname='hebo',fontsize=10,color=(.6,.3,.1))
                x=40+pymupdf.get_text_length(label,fontname='hebo',fontsize=10)
                value=f'example value {n}.'
                page.insert_text((x,y),value,fontsize=10,color=(0,.4,.6) if mixed and n==3 else (.6,.3,.1))
                rows.append(label+value)
            pdf.save(source)
        text=' '.join(rows)
        atom=dict(id='a',group='a',page=0,kind='text',role='text',block_type='text',
            extraction_method='native',content=text,length=len(text),bbox=[.08,.12,.8,.21])
        return SourceIndex(dict(source_pdf=str(source),source_page_count=1,atoms=[atom]))

    def test_uniform_translated_ink_is_not_an_offset_guess_and_inputs_are_immutable(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name));index.atoms[0].update(content='完整译文',length=4)
            before=deepcopy(index.source)
            ink,labels=native_examples(index,pages=[1])
            self.assertEqual((ink['a'][0]['start'],ink['a'][0]['end']),(0,4))
            self.assertEqual(labels,{})
            self.assertEqual(index.source,before)
            self.assertEqual(native_examples(index),({},{}))

    def test_complete_native_label_rows_keep_values_and_do_not_require_pdf_linebreaks(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name));ink,labels=native_examples(index,pages=[1])
            self.assertEqual(len(labels['a']),4)
            self.assertEqual([r['quote'] for r in labels['a']],['Input:','Output:']*2)
            self.assertIn('a',ink)
            index.atoms[0]['content']=index.atoms[0]['content'].replace('value','changed',1)
            self.assertEqual(native_examples(index,pages=[1])[1],{})

    def test_mixed_ink_keeps_existing_offsets_and_never_colors_entire_translation(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name),mixed=True)
            ink,labels=native_examples(index,pages=[1]);self.assertEqual(ink,{})
            self.assertEqual(len(labels['a']),4)
            index.atoms[0]['source_text_runs']=[dict(start=0,end=6,color=[1,0,0],method='native_span_ink_v1')]
            self.assertEqual(native_examples(index,pages=[1])[0],{})

    def test_scans_panels_edits_semantics_and_exclusions_are_barriers(self):
        for defect in ('scan','panel','edited','recovery','heading','excluded','missing'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as name:
                index=self.source(Path(name));registry=[];excluded=[];changed=[]
                if defect=='scan':index.atoms[0]['extraction_method']='ocr'
                if defect=='panel':index.atoms[0]['source_panel']={'id':'code'}
                if defect=='edited':changed=['a']
                if defect=='recovery':index.atoms[0]['recovery_issue']='unknown'
                if defect=='heading':registry=[dict(lo=0,hi=index.prefix[-1])]
                if defect=='excluded':excluded=[dict(lo=0,hi=index.prefix[-1])]
                if defect=='missing':Path(index.source['source_pdf']).unlink()
                self.assertEqual(native_examples(index,registry,excluded,changed,pages=[1]),({},{}))

    def test_compiled_example_values_share_their_label_rows_and_remain_searchable(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);project=root/'latex'
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[],reviewed_ink_pages=[1]),
                [],[],dict(title='Examples',author=''),load_config(),project,sample_mode=True)
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(project/'main.pdf') as pdf:
                text=pdf[0].get_text()
                for n in range(4):self.assertEqual(text.count(f'example value {n}.'),1)
                rows=[pdf[0].search_for(f'example value {n}.')[0] for n in range(4)]
                self.assertEqual(len({round(r.y0) for r in rows}),4)
                spans=[s for b in pdf[0].get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if 'example value' in s['text']]
                self.assertEqual(len({s['color'] for s in spans}),1)
                self.assertNotEqual(spans[0]['color'],0)
                self.assertTrue(set(audit['source_anchors'].values())<=pdf.resolve_names().keys())

    def test_compiled_list_numbers_keep_the_same_native_ink_as_the_items(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'list.pdf';atoms=[]
            with pymupdf.open() as pdf:
                page=pdf.new_page(width=500,height=700)
                for n in range(2):
                    text=f'{n+1}. Example item';y=100+20*n
                    page.insert_text((40,y),text,fontsize=10,color=(0,.4,.6))
                    atoms.append(dict(id=str(n),group=str(n),kind='text',role='text',block_type='text',
                        page=0,extraction_method='native',content=text,length=len(text),
                        bbox=[.08,(y-11)/700,.6,(y+3)/700]))
                pdf.save(source)
            index=SourceIndex(dict(source_pdf=str(source),source_page_count=1,atoms=atoms));project=root/'latex'
            write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[],reviewed_ink_pages=[1]),
                [],[],dict(title='List',author=''),load_config(),project,sample_mode=True)
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            with pymupdf.open(project/'main.pdf') as pdf:
                spans=[s for b in pdf[0].get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['text'].strip()]
                numbers=[s for s in spans if s['text'].strip() in {'1.','2.'}]
                self.assertEqual(len(numbers),2)
                self.assertEqual({s['color'] for s in spans},{numbers[0]['color']})
                self.assertNotEqual(numbers[0]['color'],0)


if __name__=='__main__':unittest.main()
