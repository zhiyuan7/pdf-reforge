"""Pixel-backed marks must keep their owner without guessing missing text."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable,load_config
from book_pipeline.ingestion.panels import panel_environments
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.raster_marks import raster_marks,METHOD


class RasterMarkTests(unittest.TestCase):
    def source(self,root,live=False):
        path=root/'source.pdf'
        panel=dict(id='observed-panel',bbox=[.08,.1,.94,.3],color=[0,.65,.32],
                   background_color=[.95,.98,.96],frame_color=[0,.65,.32],method='test')
        with pymupdf.open() as original,pymupdf.open() as raster:
            p=original.new_page(width=500,height=700)
            p.draw_rect((40,70,470,210),color=(0,.65,.32),fill=(.95,.98,.96),width=.5)
            p.insert_text((50,100),'A source definition.',fontsize=11)
            p.insert_text((200,160),'x + y = y + x',fontsize=11)
            # An observed diamond (not a symbol inferred from box color).
            p.draw_polyline([(457,194),(461,198),(457,202),(453,198),(457,194)],
                            color=(0,.65,.32),fill=(0,.65,.32))
            p.insert_text((40,260),'Proof',fontsize=11,color=(1,.52,.08))
            p.insert_text((80,260),'Take a mathematical object.',fontsize=11)
            p.insert_text((40,310),'This completes the argument.',fontsize=11)
            p.draw_rect((455,304,462,311),color=None,fill=(1,.52,.08))
            p.insert_text((40,350),'An inline square is explained here.',fontsize=11)
            p.draw_rect((230,343,237,350),color=None,fill=(1,.52,.08))
            if live: original.save(path)
            else:
                rp=raster.new_page(width=500,height=700)
                rp.insert_image(rp.rect,stream=p.get_pixmap(dpi=150).tobytes('png'))
                raster.save(path)
        specs=[('body','text',[.10,.12,.45,.15],'A source definition.',panel),
               ('formula','display_math',[0,.20,1,.32],'x+y=y+x',panel),
               ('intro','text',[.08,.35,.65,.375],'Proof Take a mathematical object.',None),
               ('end','text',[.08,.425,.93,.45],'This completes the argument.',None),
               ('explanation','text',[.08,.475,.75,.51],'An inline square is explained here.',None)]
        atoms=[]
        for name,kind,box,text,frame in specs:
            a=dict(id=name,group=name,page=0,kind=kind,role='text',content=text,length=len(text),
                   bbox=box,block_bbox=box,block_type='equation' if kind=='display_math' else 'text',
                   extraction_method='native' if live else 'ocr')
            if frame:a['source_panel']=frame
            atoms.append(a)
        return SourceIndex(dict(source_pdf=str(path),source_page_count=1,atoms=atoms))

    def test_shapes_and_label_color_are_owned_without_changing_source(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);before=deepcopy(index.source)
            panels,_=panel_environments(index,[])
            marks,labels,_=raster_marks(index,panels,root/'latex')
            self.assertEqual(index.source,before)
            self.assertEqual(set(marks),{'formula','end'})
            self.assertEqual(set(labels),{'intro'})
            self.assertEqual(labels['intro']['text_end'],5)
            self.assertEqual(labels['intro']['end'],6)
            self.assertGreater(labels['intro']['gap_after_bp'],5)
            self.assertIsNone(marks['end']['panel'])
            for mark in marks.values():
                self.assertTrue((root/'latex'/mark['image']).is_file())
                self.assertGreater(mark['height_pt'],6)

    def test_edited_excluded_ambiguous_and_recognized_marks_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);base=self.source(root)
            for problem in ('edited','excluded','overlap','recognized','edited_panel'):
                with self.subTest(problem=problem):
                    index=SourceIndex(deepcopy(base.source));changed=set();exclusions=[]
                    if problem=='edited':changed={'end','intro'}
                    elif problem=='excluded':exclusions=[dict(lo=index.prefix[2],hi=index.prefix[4])]
                    elif problem=='overlap':
                        index.atoms[4]['bbox']=index.atoms[4]['block_bbox']=[.88,.43,.96,.46]
                    elif problem=='recognized':index.atoms[3]['content']+=' ■'
                    elif problem=='edited_panel':changed={'body'}
                    panels,_=panel_environments(index,[])
                    marks,labels,_=raster_marks(index,panels,root/'latex',exclusions,changed)
                    if problem=='edited_panel':self.assertNotIn('formula',marks)
                    else:self.assertNotIn('end',marks)
                    if problem in {'edited','excluded'}:self.assertNotIn('intro',labels)

    def test_live_text_and_semantic_panel_cannot_acquire_bitmap_marks(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root,True)
            panels,_=panel_environments(index,[])
            self.assertEqual(raster_marks(index,panels,root/'latex'),({},{},[]))
            index=self.source(root)
            marks,_,_=raster_marks(index,[],root/'latex')
            self.assertNotIn('formula',marks)

    def test_shipped_pdf_has_exactly_two_source_marks_and_searchable_proof(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);project=root/'latex'
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Source furniture',author=''),load_config(),project,sample_mode=True)
            self.assertTrue(audit['coverage_complete'])
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            furniture=[r for r in audit['source_order_layout'] if r['method']==METHOD]
            self.assertEqual(len(furniture),3)
            with pymupdf.open(project/'main.pdf') as pdf:
                self.assertEqual(len(pdf),1);page=pdf[0]
                self.assertTrue(page.search_for('Proof Take a mathematical object.'))
                images=page.get_image_info();self.assertEqual(len(images),2)
                for mark in [r for r in furniture if r['kind']=='mark']:
                    image=min(images,key=lambda im:abs(pymupdf.Rect(im['bbox']).height-mark['height_pt']))
                    self.assertAlmostEqual(pymupdf.Rect(image['bbox']).height,mark['height_pt'],delta=.02)
                label=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[])
                       for s in l['spans'] if s['text'].strip()=='Proof']
                self.assertEqual(len(label),1);self.assertNotEqual(label[0]['color'],0)
                end=page.search_for('This completes the argument.')[0]
                terminal=min(images,key=lambda im:abs(pymupdf.Rect(im['bbox']).y1-end.y1))
                self.assertGreater(pymupdf.Rect(terminal['bbox']).x0,end.x1)
                self.assertTrue({'source-'+a['id'] for a in index.atoms} <= set(pdf.resolve_names()))
