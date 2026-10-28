"""Source-colored prose survives merging, serialization, edits and PDF output."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config, executable
from book_pipeline.ingestion.native import native_content, spans_for
from book_pipeline.ingestion.source import SourceIndex, corrected_atoms
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import Renderer, write_project
from book_pipeline.publishing.processed_book import EditedBook


class NativeInkTests(unittest.TestCase):
    def nodes(self):
        # Identical fonts, adjacent native runs, with a deliberate label gap.
        spans = [dict(id='a',text='证明',bbox=[.1,.1,.14,.12],line=0,size=10,color=0xff8618),
                 dict(id='b',text='假设条件成立。',bbox=[.153,.1,.3,.12],line=0,size=10,color=0)]
        return native_content(dict(type='text',bbox=[.1,.1,.3,.12],
            parts=[dict(kind='text',spans=['a','b'])]),{s['id']:s for s in spans},600)

    def source(self):
        node = self.nodes()[0]
        atom = dict(id='a',group='a',page=0,kind='text',role='text',block_type='text',
            content=node['content'],length=len(node['content']),bbox=[.1,.1,.8,.13],
            source_text_runs=node['source_text_runs'])
        return SourceIndex(dict(atoms=[atom],source_page_count=1))

    def test_chinese_label_keeps_exact_text_color_range_and_observed_gap(self):
        nodes = self.nodes()
        self.assertEqual(len(nodes),1)
        self.assertEqual(nodes[0]['content'],'证明假设条件成立。')
        run = nodes[0]['source_text_runs'][0]
        self.assertEqual((run['start'],run['end']),(0,2))
        self.assertAlmostEqual(run['color'][1],134/255,places=4)
        self.assertEqual(run['gap_after_em'],.78)

    def test_label_gap_also_survives_separate_model_text_parts(self):
        spans = [dict(id='a',text='证明',bbox=[.1,.1,.14,.12],line=0,size=10,color=0xff8618),
                 dict(id='b',text='假设',bbox=[.153,.1,.3,.12],line=0,size=10,color=0)]
        nodes = native_content(dict(type='text',bbox=[.1,.1,.3,.12],parts=[
            dict(kind='text',spans=['a']),dict(kind='text',spans=['b'])]),{s['id']:s for s in spans},600)
        self.assertEqual(nodes[0]['source_text_runs'][0]['gap_after_em'],.78)
        self.assertEqual(''.join(n['content'] for n in nodes),'证明假设')

    def test_same_font_color_boundaries_preserve_legacy_layout_ids_and_native_ink(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((60,80),'Proof',color=(1,.5,0))
            page.insert_text((90,80),'Suppose',color=(0,0,0))
            spans = spans_for(page)
            self.assertEqual(len(spans),1)
            self.assertEqual(spans[0]['id'],'s0')
            self.assertEqual(spans[0]['text'],'Proof Suppose')
            runs=spans[0]['source_ink']['runs']
            self.assertNotEqual(runs[0]['color'],runs[1]['color'])
            node=native_content(dict(type='text',bbox=[0,0,1,1],parts=[
                dict(kind='text',spans=['s0'])]),{'s0':spans[0]},page.rect.width)[0]
            self.assertEqual(node['source_text_runs'][0]['end'],5)

    def test_cropped_style_is_rebased_and_rewritten_prose_drops_stale_offsets(self):
        index = self.source();atom=index.atoms[0]
        for a,z,text,expected in [(1,4,atom['content'][1:4],[(0,1)]),
                                  (0,1,atom['content'][:1],[(0,1)]),
                                  (0,4,'更改过的文字',[])]:
            with self.subTest(a=a,z=z,text=text):
                edited=EditedBook.__new__(EditedBook)
                edited.original=index;edited.atoms=[];edited.offset=0;edited.segments=[]
                result=edited.append(atom,a,z,text,'g',None,None)
                self.assertEqual([(r['start'],r['end']) for r in result.get('source_text_runs',[])],expected)
                if z==1:self.assertNotIn('gap_after_em',result['source_text_runs'][0])

    def test_correction_does_not_color_new_text_or_duplicate_source_bytes(self):
        index=self.source()
        with tempfile.TemporaryDirectory() as name:
            renderer=Renderer(index,dict(corrections=[dict(atom='a',start_offset=2,end_offset=4,replacement='考虑')],
                joins=[],exclusions=[],bold_spans=[]),[],[],Path(name))
            tex=''.join(renderer.render())
            self.assertIn('证明',tex);self.assertIn('考虑',tex);self.assertNotIn('假设',tex)
            self.assertEqual(tex.count(r'\hspace{0.78em}'),1)
            self.assertEqual(sum(r['end']-r['start'] for r in renderer.ledger),index.prefix[-1])

    def test_materialized_rewrite_cannot_reuse_stale_native_ink_offsets(self):
        index=self.source()
        output=corrected_atoms(index.source,[dict(atom='a',start_offset=0,end_offset=2,replacement='论证标签')])
        self.assertNotIn('source_text_runs',output['atoms'][0])
        self.assertIn('source_text_runs',index.atoms[0])

    def test_actual_pdf_retains_orange_label_black_body_gap_and_destination(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source()
            reference=dict(id='ref',group='ref',page=0,kind='text',role='text',block_type='text',
                content='See Chapter 6.',length=14,bbox=[.1,.2,.8,.23],source_text_runs=[
                    dict(start=11,end=13,color=[0,0,1],method='native_span_ink_v1')])
            index=SourceIndex({**index.source,'atoms':[*index.atoms,reference]})
            write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Source ink',author=''),load_config(),root,sample_mode=True)
            result=compile_project(root,{tool:executable(tool) for tool in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                page=pdf[0];label=page.search_for('证明')[0];body=page.search_for('假设')[0]
                self.assertGreater(body.x0-label.x1,7)
                colors = {s['text'].strip():s['color'] for b in page.get_text('dict')['blocks']
                          for line in b.get('lines',[]) for s in line['spans']}
                self.assertEqual(colors['证明'],0xff8618)
                self.assertEqual(colors['假设条件成立。'],0)
                self.assertIn('See Chapter 6.',page.get_text())
                self.assertEqual(colors['6'],0x0000ff)
                target=pdf.resolve_names()['source-a']
                self.assertEqual(target['page'],0);self.assertLess(abs(target['to'][0]-label.x0),2)
