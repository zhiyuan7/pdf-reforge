"""Scan font recovery preserves exact prose, math ownership and saved inputs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.ai.client import DeepSeekClient
from book_pipeline.config import digest, executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import Renderer, write_project
from book_pipeline.publishing.raster_styles import METHOD, eligible_items, recover_raster_styles, style_ranges


class RasterStyleTests(unittest.TestCase):
    def answer(self, **changes):
        return dict(runs=[dict(atom='edited', quote='classification', occurrence=1,
                              italic=True, bold=False, **changes)], uncertain_atoms=[])

    def sources(self, pdf=None):
        atom = dict(id='raw', group='raw', kind='text', page=0, role='text', block_type='text',
            extraction_method='ocr', bbox=[.1,.1,.5,.2], content='classification then classification.',
            length=35)
        raw = SourceIndex(dict(atoms=[atom], source_pdf=str(pdf) if pdf else None))
        edited = SourceIndex(dict(atoms=[dict(atom, id='edited', original_atom='raw',
            original_start_offset=0, original_end_offset=len(atom['content']))], source_pdf=raw.source['source_pdf']))
        return raw, edited

    def test_repeated_quote_selects_only_the_requested_occurrence(self):
        raw, edited = self.sources()
        items = eligible_items(edited, raw)[0]
        runs = style_ranges(self.answer(), items)['edited']
        self.assertEqual([(r['start'],r['end']) for r in runs], [(20,34)])
        self.assertEqual(edited.atoms[0]['content'][20:34],runs[0]['quote'])

    def test_invalid_ownership_rewrite_and_conflicting_styles_fail_closed(self):
        raw, edited = self.sources(); items=eligible_items(edited,raw)[0]
        candidates=[]
        for field,value in [('atom','formula'),('quote','Classification'),('occurrence',True),
                            ('occurrence',2),('italic',1),('quote',' classification ')]:
            answer=self.answer();answer['runs'][0][field]=value;candidates.append(answer)
        duplicate=self.answer();duplicate['runs']*=2;candidates.append(duplicate)
        uncertain=self.answer();uncertain['uncertain_atoms']=['edited'];candidates.append(uncertain)
        for answer in candidates:
            with self.subTest(answer=answer), self.assertRaises(ValueError):style_ranges(answer,items)
        with self.assertRaises(ValueError):
            style_ranges(dict(runs=[dict(atom='cn',quote='中文楷体',occurrence=0,italic=True,bold=False)],
                              uncertain_atoms=[]),[dict(atom='cn',text='中文楷体')])

    def test_normal_annotations_whitespace_and_nested_emphasis_preserve_exact_ranges(self):
        text='Plain. The important term remains.'
        answer=dict(runs=[dict(atom='a',quote='Plain.',occurrence=0,italic=False,bold=False),
            dict(atom='a',quote=' The important term remains.',occurrence=0,italic=True,bold=False),
            dict(atom='a',quote='important term',occurrence=0,italic=False,bold=True)],uncertain_atoms=[])
        runs=style_ranges(answer,[dict(atom='a',text=text)])['a']
        self.assertEqual([(r['quote'],r['italic'],r['bold']) for r in runs],
            [('The ',True,False),('important term',True,True),(' remains.',True,False)])
        self.assertEqual(''.join(r['quote'] for r in runs),text[7:])

    def test_native_math_vertical_translated_and_rewritten_prose_are_ineligible(self):
        raw, edited=self.sources()
        for change in [dict(extraction_method='native'),dict(kind='math'),dict(writing_direction='vertical_rl'),
                       dict(content='A rewrite.'),dict(role='code_body'),dict(recovery_issue='uncertain')]:
            with self.subTest(change=change):
                altered=SourceIndex({**edited.source,'atoms':[{**edited.atoms[0],**change}]})
                self.assertEqual(eligible_items(altered,raw),{})
        self.assertEqual(eligible_items(edited,raw,['edited']),{})
        self.assertEqual(eligible_items(SourceIndex({**edited.source,'translation':dict(method='test')}),raw),{})
        # A cropped unchanged fragment is still eligible, with local offsets.
        cropped=SourceIndex({**edited.source,'atoms':[dict(edited.atoms[0],content='classification',
            original_start_offset=20, original_end_offset=34)]})
        self.assertEqual(eligible_items(cropped,raw)[0][0]['text'],'classification')

    def test_requests_cache_exact_pixels_text_policy_and_model_without_touching_source(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf=root/'scan.pdf'
            with pymupdf.open() as p:
                page=p.new_page()
                page.draw_rect(pymupdf.Rect(page.rect.width*.1,page.rect.height*.1,
                    page.rect.width*.5,page.rect.height*.2),fill=(.9,.9,.9),color=None)
                page.insert_text((70,110),'classification then classification.')
                p.save(pdf)
            raw,edited=self.sources(pdf);before=deepcopy(edited.source);sha=digest(pdf);calls=[]
            config=dict(model='test',base_url='https://example.invalid',attempts=1,max_tokens=512,
                thinking='disabled',shared_cache_dirs=[],pricing={})
            def transport(task,messages,settings):
                calls.append(messages);return dict(content=json.dumps(self.answer()),finish_reason='stop')
            def client(cfg,path):return DeepSeekClient(cfg,path,transport=transport,sleep=lambda _:None)
            with patch('book_pipeline.publishing.raster_styles.DeepSeekClient',side_effect=client):
                a,report=recover_raster_styles(edited,raw,config,root/'styles')
                b,report2=recover_raster_styles(edited,raw,config,root/'styles')
                recover_raster_styles(edited,raw,{**config,'model':'changed'},root/'styles')
                changed=SourceIndex({**edited.source,'atoms':[{**edited.atoms[0],'bbox':[.1,.11,.5,.2]}]})
                recover_raster_styles(changed,raw,config,root/'styles')
            self.assertEqual(a,b);self.assertEqual(len(calls),3)
            self.assertEqual(report2['usage']['current_run']['cache_hits'],1)
            self.assertEqual(report['records'][0]['status'],'requires_visual_review')
            self.assertTrue(any(c['type']=='image_url' for c in calls[0][1]['content']))
            self.assertEqual(edited.source,before);self.assertEqual(digest(pdf),sha)

    def test_plain_body_and_mixed_note_boxes_never_trigger_style_calls(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf=root/'plain.pdf'
            with pymupdf.open() as p:
                p.new_page().insert_text((70,110),'Plain body classification.');p.save(pdf)
            raw,edited=self.sources(pdf)
            with patch('book_pipeline.publishing.raster_styles.DeepSeekClient') as client:
                styles,report=recover_raster_styles(edited,raw,{},root/'plain')
                self.assertEqual(styles,{});self.assertEqual(report['status'],'not_needed');client.assert_not_called()
            # One complete group inside a note cannot authorize a second OCR
            # group crossing the main column and note.
            raw2=deepcopy(raw.source);edited2=deepcopy(edited.source)
            raw2['atoms'].append({**raw.atoms[0],'id':'mixed','bbox':[.05,.1,.65,.2]})
            edited2['atoms'].append({**edited.atoms[0],'id':'mixed-edited','original_atom':'mixed',
                'bbox':[.05,.1,.65,.2]})
            with patch('book_pipeline.publishing.raster_styles.gray_regions',return_value=[dict(bbox=[.1,.1,.5,.2])]), \
                 patch('book_pipeline.publishing.raster_styles.DeepSeekClient') as client:
                styles,_=recover_raster_styles(SourceIndex(edited2),SourceIndex(raw2),{},root/'mixed')
                self.assertEqual(styles,{});client.assert_not_called()

    def test_styles_compose_with_color_links_splits_and_preserve_searchable_pdf(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            text='A vector and a scalar. See Chapter 6. 中文的随机变量。'
            atom=dict(id='a',group='g',page=0,kind='text',role='text',block_type='text',
                content=text,length=len(text),bbox=[.1,.2,.8,.25],
                source_text_runs=[dict(start=2,end=8,color=[0,.5,0],method='native_span_ink_v1')])
            index=SourceIndex(dict(atoms=[atom],source_page_count=1))
            items=[dict(atom='a',text=text)]
            styles=style_ranges(dict(runs=[dict(atom='a',quote='vector',occurrence=0,italic=True,bold=False),
                dict(atom='a',quote='scalar',occurrence=0,italic=False,bold=True),
                dict(atom='a',quote='Chapter 6',occurrence=0,italic=True,bold=False)],uncertain_atoms=[]),items)
            start=text.index('中文')
            styles['a'].append(dict(method=METHOD,start=start,end=start+2,quote='中文',
                                    italic=False,bold=False,cjk_family='calligraphic'))
            start=text.index('随机变量')
            styles['a'].append(dict(method=METHOD,start=start,end=start+4,quote='随机变量',
                                    italic=False,bold=True,cjk_family='sans'))
            decisions=dict(corrections=[],joins=[],exclusions=[],bold_spans=[],source_styles=styles)
            renderer=Renderer(index,decisions,[],[],root)
            self.assertIn(r'\itshape',renderer.source_text_fragment(atom,4,8))
            self.assertEqual(renderer.source_text_fragment(atom,0,2),'A ')
            write_project(index,decisions,[],[],dict(title='Styles',author=''),load_config(),root,sample_mode=True)
            result=compile_project(root,{tool:executable(tool) for tool in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                spans=[s for p in pdf for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]
                vector=next(s for s in spans if 'vector' in s['text'])
                scalar=next(s for s in spans if 'scalar' in s['text'])
                self.assertTrue(vector['flags'] & 2,vector)
                self.assertEqual(vector['color'],0x008000)
                self.assertTrue(scalar['flags'] & 16,scalar)
                cjk=next(s for s in spans if '随机变量' in s['text'])
                self.assertIn('NotoSansCJK',cjk['font'])
                kai=next(s for s in spans if '中文' in s['text'])
                self.assertIn('FandolKai',kai['font'])
                self.assertFalse(kai['flags'] & 2,kai)
                self.assertIn('Chapter 6',''.join(p.get_text() for p in pdf))
                self.assertEqual(len(pdf),1)
            start=text.index('Chapter 6')
            refs=[dict(atom='a',start_offset=start,end_offset=start+9,text='Chapter 6',target='target')]
            renderer=Renderer(index,decisions,[],refs,root)
            self.assertIn(r'\BookLink{target}{{\itshape Chapter 6}}',renderer.text_fragment(atom,0,len(text)))
            bad=deepcopy(styles);bad['a'][0]['quote']='wrong'
            renderer=Renderer(index,{**decisions,'source_styles':bad},[],[],root)
            with self.assertRaises(ValueError):renderer.source_text_fragment(atom,0,len(text))
            for change in (dict(cjk_family='serif'),dict(italic=True),dict(bold=False),dict(quote='other')):
                bad=deepcopy(styles);bad['a'][-1].update(change)
                renderer=Renderer(index,{**decisions,'source_styles':bad},[],[],root)
                with self.assertRaises(ValueError):renderer.source_text_fragment(atom,0,len(text))
