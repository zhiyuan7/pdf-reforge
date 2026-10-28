"""Raster note/rule ownership and the PDF that is actually shipped."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.raster_decorations import raster_decorations


class RasterDecorationTests(unittest.TestCase):
    def source(self, root, live=False, table_rules=False, fraction=False):
        path = root/'source.pdf'
        with pymupdf.open() as source, pymupdf.open() as raster:
            page = source.new_page(width=500, height=700)
            page.draw_line((35,40),(465,40),color=(.5,)*3,width=1)
            page.insert_text((40,55),'An observed source heading',fontsize=10)
            page.insert_text((180,100),'x + 0 = x',fontsize=11)
            page.draw_line((35,115),(465,115),color=(0,)*3,width=2)
            if table_rules:
                page.draw_line((35,135),(465,135),color=(0,)*3,width=2)
            if fraction:
                page.insert_text((180,130),'another denominator',fontsize=10)
            page.draw_rect((270,200,465,300),fill=(.922,.925,.929),color=None)
            for y in (217,233,249,265,281):
                page.insert_text((280,y),'Source note gives more context.',fontsize=9)
                page.insert_text((40,y),'Main mathematical argument.',fontsize=10)
            page.insert_text((40,350),'Following full width paragraph.',fontsize=10)
            if live:
                source.save(path)
            else:
                out = raster.new_page(width=500,height=700)
                out.insert_image(out.rect,stream=page.get_pixmap(dpi=150).tobytes('png'))
                raster.save(path)
        specs = [
            ('heading','text','paragraph_title',[.08,.063,.6,.08],'An observed source heading'),
            ('equation','display_math','equation',[0,.115,1,.196],'x+0=x'),
            ('main','text','text',[.08,.286,.49,.42],'Main mathematical argument. '*9),
            ('note','text','text',[.56,.29,.92,.41],'Source note gives more context. '*5),
            ('after','text','text',[.08,.48,.94,.51],'Following full width paragraph.')]
        atoms = [dict(id=n,group=n,page=0,kind=kind,role='equation' if kind=='display_math' else 'text',
                      block_type=block,bbox=box,block_bbox=box,content=text,length=len(text),
                      extraction_method='native' if live else 'ocr') for n,kind,block,box,text in specs]
        return SourceIndex(dict(source_pdf=str(path),source_page_count=1,atoms=atoms))

    def test_flat_note_and_padded_formula_rule_have_independent_owners(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name));before=deepcopy(index.source)
            panels,rows,rules,evidence=raster_decorations(index,[])
            self.assertEqual(index.source,before)
            self.assertEqual(len(panels),1)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['main_bbox'],index.atoms[2]['bbox'])
            self.assertEqual(set(rules),{('heading','before'),('equation','after')})
            self.assertAlmostEqual(rules['equation','after']['thickness_pt'],2,delta=.5)
            self.assertEqual(evidence[0]['status'],'rendered')

    def test_mixed_table_image_code_edited_and_semantic_notes_keep_original_flow(self):
        with tempfile.TemporaryDirectory() as name:
            base=self.source(Path(name))
            for problem in ('mixed','table','image','code','edited','semantic','footnote','changed_main','distant'):
                with self.subTest(problem=problem):
                    index=SourceIndex(deepcopy(base.source));a=index.atoms[3]
                    if problem=='mixed':a['block_bbox']=[.08,.29,.92,.41]
                    elif problem=='table':a['kind']='table'
                    elif problem=='image':a['kind']='image'
                    elif problem=='code':a['block_type']='code'
                    elif problem=='distant':index.atoms[2]['block_bbox']=[.08,.6,.49,.7]
                    registry=[dict(kind='remark',lo=index.prefix[3],hi=index.prefix[4])] if problem=='semantic' else []
                    exclusions=[dict(lo=index.prefix[3],hi=index.prefix[4])] if problem=='footnote' else []
                    changed={'note'} if problem=='edited' else {'main'} if problem=='changed_main' else set()
                    panels,rows,_,_=raster_decorations(index,registry,exclusions,changed)
                    if problem=='changed_main':
                        # Existing links/corrections do not alter the bounded
                        # geometry of a complete neighboring main paragraph.
                        self.assertEqual(len(rows),1)
                    else:self.assertEqual((panels,rows),([],[]))

    def test_repeated_table_rules_and_fraction_bars_cannot_become_separators(self):
        with tempfile.TemporaryDirectory() as name:
            for table,fraction in ((True,False),(False,True)):
                with self.subTest(table=table):
                    index=self.source(Path(name),table_rules=table,fraction=fraction)
                    self.assertNotIn(('equation','after'),raster_decorations(index,[])[2])

    def test_native_text_and_unavailable_source_do_not_enter_raster_detection(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name),live=True)
            self.assertEqual(raster_decorations(index,[]),([],[],{},[]))
            index.source['source_pdf']=str(Path(name)/'missing.pdf')
            self.assertEqual(raster_decorations(index,[]),([],[],{},[]))

    def test_compiled_round_note_padding_rules_destinations_and_following_text(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);before=deepcopy(index.source)
            project=root/'latex'
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                                [],[],dict(title='Raster furniture',author=''),load_config(),project,sample_mode=True)
            self.assertEqual(index.source,before)
            self.assertTrue(audit['coverage_complete'])
            result=compile_project(project,{tool:executable(tool) for tool in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(project/'main.pdf') as pdf:
                self.assertEqual(len(pdf),1);page=pdf[0]
                main=page.search_for('Main mathematical')[0];note=page.search_for('Source note')[0]
                self.assertGreater(note.x0,main.x1)
                self.assertLess(abs(main.y0-note.y0),15)
                self.assertGreater(page.search_for('Following full')[0].y0,
                                   max(b.y1 for b in page.search_for('context.')+page.search_for('argument.')))
                fills=[d for d in page.get_drawings() if d.get('fill') and
                       abs(d['fill'][0]-.922)<.01]
                frame=max(fills,key=lambda d:d['rect'].get_area())
                self.assertTrue(any(i[0]=='c' for i in frame['items']))
                self.assertGreater(note.x0-frame['rect'].x0,5)
                self.assertEqual({a['id'] for a in index.atoms},
                    {a['id'] for a in index.atoms if 'source-'+a['id'] in pdf.resolve_names()})
                lines=[d for d in page.get_drawings() if d['rect'].width>300 and d['rect'].height<.1]
                self.assertTrue(any(d['width']>1.5 and d.get('color')==(0,0,0) for d in lines))
