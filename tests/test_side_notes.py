"""Source geometry must govern lateral note ownership and compiled placement."""
from copy import deepcopy
from pathlib import Path
import re
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config, executable
from book_pipeline.ingestion.panels import panel_environments, text_panels
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.side_notes import side_note_rows, explicit_side_note_rows


class SideNoteTests(unittest.TestCase):
    def explicit_source(self, left=False):
        body='A complete paragraph explains the probability model and its mathematical consequences. '*3
        specs=[('body',body,'text',[.25,.2,.8,.3] if left else [.1,.2,.65,.3]),
               ('following','The following paragraph retains the ordinary reading order.','text',[.1,.35,.8,.4]),
               ('aside','Appendix C.5.1: KL divergence','aside_text',
                [.08,.24,.22,.28] if left else [.69,.24,.83,.28])]
        atoms=[dict(id=i,group=i,page=0,kind='text',role=r,block_type=r,content=t,
                    length=len(t),bbox=b,extraction_method='ocr') for i,t,r,b in specs]
        return SourceIndex(dict(atoms=atoms,source_page_count=1))

    def test_explicit_ocr_aside_needs_unique_disjoint_owner_and_preserves_source(self):
        for defect in ('none','ambiguous','inside','different_page','distant','heading','unclassified','edited','excluded','semantic'):
            with self.subTest(defect=defect):
                index=self.explicit_source();changed=[];exclusions=[];registry=[]
                if defect=='ambiguous':
                    index=SourceIndex(dict(atoms=[dict(index.atoms[0],id='other',group='other'),*index.atoms]))
                note=index.atoms[index.ids['aside']]
                if defect=='inside':note['bbox']=[.3,.24,.44,.28]
                if defect=='different_page':note['page']=1
                if defect=='distant':note['bbox']=[.69,.6,.83,.64]
                if defect=='heading':index.atoms[index.ids['body']]['role']='paragraph_title'
                if defect=='unclassified':note['role']='text'
                if defect=='edited':changed=['aside']
                if defect=='excluded':exclusions=[dict(lo=0,hi=1)]
                if defect=='semantic':registry=[dict(lo=0,hi=1,kind='theorem')]
                before=deepcopy(index.source)
                self.assertEqual(len(explicit_side_note_rows(index,registry,exclusions,changed)),int(defect=='none'))
                self.assertEqual(index.source,before)

    def test_compiled_nonadjacent_aside_keeps_owner_and_destinations_on_both_sides(self):
        for left in (False,True):
            with self.subTest(left=left),tempfile.TemporaryDirectory() as name:
                index=self.explicit_source(left);root=Path(name)
                audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                    [],[],dict(title='Explicit side note',author=''),load_config(),root,sample_mode=True)
                self.assertTrue(audit['coverage_complete'])
                result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
                with pymupdf.open(root/'main.pdf') as pdf:
                    names=pdf.resolve_names();self.assertEqual(names['source-aside']['page'],names['source-body']['page'])
                    page=pdf[names['source-body']['page']]
                    main=page.search_for('A complete paragraph')[0];note=page.search_for('Appendix')[0]
                    self.assertEqual(main.x0>note.x0,left)
                    self.assertLess(abs(main.y0-note.y0),12)
                    self.assertEqual(page.get_text().count('Appendix'),1)
                    self.assertGreater(page.search_for('The following')[0].y0,
                                       max(r.y1 for r in page.search_for('consequences.')))

    def source(self, note_first=False, note_left=False):
        main_box, note_box = ([.54,.20,.94,.36], [.06,.20,.49,.34]) if note_left else ([.06,.20,.49,.36], [.54,.20,.94,.34])
        panel = dict(id='p1-panel-1',bbox=note_box,color=[.25]*3,
                     background_color=[.925]*3,title_color=None)
        atoms = [dict(id='main',group='main',page=0,kind='text',role='text',block_type='text',
                      content='Main paragraph has several lines explaining a mathematical idea. '*4,
                      bbox=main_box,block_bbox=main_box),
                 dict(id='note',group='note',page=0,kind='text',role='text',block_type='text',
                      content='Side note adds context without interrupting the argument. '*3,
                      bbox=note_box,block_bbox=note_box,source_panel=panel)]
        if note_first: atoms.reverse()
        for atom in atoms:atom['length']=len(atom['content'])
        return SourceIndex(dict(atoms=atoms,source_page_count=1))

    def test_geometry_accepts_both_orders_and_sides(self):
        for first in (False,True):
            for left in (False,True):
                with self.subTest(first=first,left=left):
                    index=self.source(first,left);panels,_=panel_environments(index,[])
                    rows=side_note_rows(index,panels,[])
                    self.assertEqual(len(rows),1)
                    self.assertEqual(rows[0]['first_left'],first==left)
                    self.assertEqual((rows[0]['lo'],rows[0]['hi']),(0,index.prefix[-1]))
                    self.assertTrue(panels[0]['lateral_note'])

    def test_ambiguous_geometry_and_semantics_keep_original_flow(self):
        for change in ('same_column','different_page','distant','heading','figure','long','footnote'):
            with self.subTest(change=change):
                index=self.source();atom=index.atoms[0]
                if change=='same_column':atom['block_bbox']=[.56,.2,.92,.35]
                if change=='different_page':atom['page']=1
                if change=='distant':atom['block_bbox']=[.06,.6,.49,.7]
                if change=='heading':atom['block_type']='paragraph_title'
                if change=='figure':atom['kind']='image'
                if change=='long':atom.update(content='x'*1800,length=1800)
                index=SourceIndex(index.source);panels,_=panel_environments(index,[])
                exclusions=[dict(lo=0,hi=1)] if change=='footnote' else []
                self.assertEqual(side_note_rows(index,panels,[],exclusions),[])
                self.assertEqual(side_note_rows(index,panels,[dict(lo=0,hi=index.prefix[-1])]),[])

    def test_compiled_pair_keeps_destinations_color_padding_and_next_paragraph(self):
        for first,left in ((False,False),(True,True),(True,False)):
            with self.subTest(first=first,left=left), tempfile.TemporaryDirectory() as name:
                index=self.source(first,left)
                after=deepcopy(index.atoms[0]);after.update(id='after',group='after',source_panel=None,
                    content='Following paragraph starts below both columns.',length=46,
                    bbox=[.06,.5,.94,.54],block_bbox=[.06,.5,.94,.54])
                index=SourceIndex({**index.source,'atoms':[*index.atoms,after]})
                root=Path(name)
                audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                    [],[],dict(title='Side notes',author=''),load_config(),root,sample_mode=True)
                self.assertTrue(audit['coverage_complete'])
                self.assertEqual(len([r for r in audit['source_order_layout']
                    if r['method']=='bounded_adjacent_source_side_note']),1)
                result=compile_project(root,{tool:executable(tool) for tool in ('xelatex','latexmk')})
                self.assertTrue(result['success'],result['log_tail'])
                self.assertEqual(result['warnings'],[])
                with pymupdf.open(root/'main.pdf') as pdf:
                    self.assertEqual(len(pdf),1)
                    page=pdf[0];main=page.search_for('Main paragraph')[0];note=page.search_for('Side note')[0]
                    self.assertEqual(main.x0>note.x0,left)
                    self.assertLess(abs(main.y0-note.y0),12)
                    self.assertGreater(page.search_for('Following paragraph')[0].y0,
                        max(r.y1 for r in page.search_for('argument.')+page.search_for('idea.'))+4)
                    names=pdf.resolve_names()
                    for atom,rect in [('main',main),('note',note)]:
                        self.assertEqual(names['source-'+atom]['page'],0)
                        self.assertLess(abs(names['source-'+atom]['to'][0]-rect.x0),2)
                    fills=[d for d in page.get_drawings() if d.get('fill') and
                           all(abs(c-.925)<.002 for c in d['fill'])]
                    self.assertTrue(fills)
                    frame=fills[-1]['rect']
                    self.assertLess(frame.width,page.rect.width*.5)
                    self.assertGreater(note.x0-frame.x0,5)

    def wrapped_source(self, root, note_first=False, note_left=False):
        index = self.source(note_first, note_left)
        main = next(a for a in index.atoms if a['id'] == 'main')
        note = next(a for a in index.atoms if a['id'] == 'note')
        frame = [.08,.20,.46,.27] if note_left else [.54,.20,.92,.27]
        main.update(block_bbox=[.08,.20,.92,.37], bbox=[.08,.20,.92,.37],
                    extraction_method='native', content=('The opening explains the mathematical idea and its context. '*9))
        note.update(block_bbox=frame, bbox=frame, extraction_method='native',
                    content='A short note gives context.')
        note['source_panel']['bbox'] = frame
        for a in index.atoms: a['length'] = len(a['content'])
        path = root/'source.pdf'
        with pymupdf.open() as pdf:
            p = pdf.new_page(width=500, height=700)
            x = 270 if note_left else 40
            for y in (151,167,183):
                p.insert_text((x,y), 'The opening explains the main idea.', fontsize=11)
            p.insert_text((40,211), 'The following full line continues across the entire paragraph width with further details.', fontsize=11)
            p.insert_text((40,235), 'The final line completes the explanation.', fontsize=11)
            p.insert_text((frame[0]*500+5,151),'A short note gives context.',fontsize=10)
            pdf.save(path)
        return SourceIndex({**index.source, 'source_pdf':str(path)})

    def test_live_source_lines_restore_width_in_both_orders_and_sides(self):
        for first in (False, True):
            for left in (False, True):
                with self.subTest(first=first,left=left), tempfile.TemporaryDirectory() as name:
                    index = self.wrapped_source(Path(name),first,left)
                    panels,_ = panel_environments(index,[])
                    rows = side_note_rows(index,panels,[])
                    self.assertEqual(len(rows),1)
                    self.assertEqual(rows[0]['method'],'native_wrapped_source_side_note')
                    self.assertEqual(rows[0]['note_left'],left)
                    self.assertEqual(rows[0]['main_first'],not first)
                    self.assertEqual(rows[0]['source_narrow_lines'],3)
                    self.assertEqual(rows[0]['source_full_lines'],2)
                    self.assertEqual(side_note_rows(index,panels,[],changed=['main']),[])
                    index.atoms[index.ids['main']]['extraction_method']='ocr'
                    self.assertEqual(side_note_rows(index,panels,[]),[])

    def test_compiled_wrapping_rejoins_main_paragraph_and_owns_destinations(self):
        for first,left in ((True,False),(False,True)):
            with self.subTest(first=first,left=left), tempfile.TemporaryDirectory() as name:
                root=Path(name);index=self.wrapped_source(root,first,left)
                after=dict(id='after',group='after',page=0,kind='text',role='text',block_type='text',
                           content='Following paragraph starts below the complete wrapped pair.',
                           bbox=[.08,.4,.92,.44],block_bbox=[.08,.4,.92,.44])
                after['length']=len(after['content'])
                prefix=[]
                if not first:
                    before={**after,'id':'prefix','group':'prefix',
                            'content':'Context before the mathematical argument. '*95}
                    before['length']=len(before['content']);prefix=[before]
                index=SourceIndex({**index.source,'atoms':[*prefix,*index.atoms,after]})
                target=root/'render'
                audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                    [],[],dict(title='Wrapping',author=''),load_config(),target,sample_mode=True)
                self.assertTrue(audit['coverage_complete'])
                result=compile_project(target,{tool:executable(tool) for tool in ('xelatex','latexmk')})
                self.assertTrue(result['success'],result['log_tail'])
                self.assertEqual(result['warnings'],[])
                with pymupdf.open(target/'main.pdf') as pdf:
                    names=pdf.resolve_names();page_number=names['source-note']['page']
                    self.assertEqual(page_number,names['source-main']['page'])
                    self.assertEqual(page_number>0,not first)
                    page=pdf[page_number];note=page.search_for('A short note')[0]
                    frames=[d['rect'] for d in page.get_drawings() if d.get('fill') and
                            d['rect'].contains(note) and .92<min(d['fill'])<.93]
                    self.assertTrue(frames);frame=min(frames,key=lambda r:r.get_area())
                    main=page.search_for('The opening')
                    # search_for returns multiple rectangles when a phrase wraps.
                    self.assertEqual(len(re.findall(r'The\s+opening',page.get_text())),9)
                    self.assertLess(abs(main[0].y0-note.y0),12)
                    lines=[l for b in page.get_text('dict')['blocks'] for l in b.get('lines',[])
                           if any('opening' in s['text'] or 'context.' in s['text'] for s in l['spans'])]
                    self.assertTrue(any(l['bbox'][2]-l['bbox'][0]>page.rect.width*.6
                                        and l['bbox'][1]>frame.y1 for l in lines))
                    for word in page.get_text('words'):
                        rect=pymupdf.Rect(word[:4])
                        if word[4] in ('opening','mathematical','idea'):
                            self.assertTrue((rect & frame).is_empty)
                    self.assertGreater(page.search_for('Following paragraph')[0].y0,
                                       max(frame.y1,max(r.y1 for r in main))+4)
                    names=pdf.resolve_names()
                    for atom,rect in [('main',main[0]),('note',note)]:
                        self.assertEqual(names['source-'+atom]['page'],page_number)
                        self.assertLess(abs(names['source-'+atom]['to'][0]-rect.x0),2)

    def test_nested_paint_order_and_opacity_determine_interior_color(self):
        with pymupdf.open() as pdf:
            p=pdf.new_page(width=500,height=700)
            p.draw_rect((40,120,460,220),fill=(.25,.25,.25))
            # The body starts below the title rather than covering the whole frame.
            p.draw_rect((41,131,459,219),fill=(.9,.94,.98),fill_opacity=.8)
            p.draw_rect((50,110,450,130),fill=(0,.65,.32))
            p.insert_text((60,124),'Definition 1.1')
            p.insert_text((60,160),'The sum is defined coordinatewise.')
            panels=text_panels(p)
            self.assertEqual(len(panels),1)
            self.assertEqual(panels[0]['color'],[0,.65,.32])
            self.assertEqual(panels[0]['title_color'],[0,.65,.32])
            self.assertEqual(panels[0]['title_text_color'],[0,0,0])
            for c,expected in zip(panels[0]['background_color'],[.77,.802,.834]):
                self.assertAlmostEqual(c,expected,places=3)

    def test_pair_near_page_end_moves_intact_without_false_continuation(self):
        index=self.source()
        prefix=deepcopy(index.atoms[0]);prefix.update(id='prefix',group='prefix',
            content='Context before the mathematical argument. '*95,
            bbox=[.06,.01,.94,.18],block_bbox=[.06,.01,.94,.18])
        prefix['length']=len(prefix['content'])
        index=SourceIndex({**index.source,'atoms':[prefix,*index.atoms]})
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Pagination',author=''),load_config(),root,sample_mode=True)
            result=compile_project(root,{tool:executable(tool) for tool in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                names=pdf.resolve_names()
                page=names['source-note']['page']
                self.assertEqual(page,names['source-main']['page'])
                self.assertGreater(page,0)
                self.assertNotIn('（续）',''.join(p.get_text() for p in pdf))
                self.assertEqual(len(re.findall(r'Side\s+note',pdf[page].get_text())),3)
