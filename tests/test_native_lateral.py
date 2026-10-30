from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config,executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.native_lateral import side_caption_rows,marginal_references,marginal_glossaries,marginal_figures,native_side_notes
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.figure_recovery import recover_native_vector_labels


class NativeLateralTests(unittest.TestCase):
    def test_single_short_glossary_requires_unique_main_term_and_live_smaller_font(self):
        for defect in ('none','synthetic_band','large','repeated','absent','shifted','rewritten'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as name:
                path=Path(name)/'source.pdf'
                text='The responsibility describes each mixture component.'
                if defect=='repeated':text='The responsibility and responsibility describe components.'
                if defect=='absent':text='The weight describes each mixture component.'
                with pymupdf.open() as pdf:
                    page=pdf.new_page(width=500,height=700)
                    page.insert_text((150,200),text,fontsize=10)
                    page.insert_text((30,260 if defect=='shifted' else 200),'responsibility',
                                     fontsize=10 if defect=='large' else 7)
                    blocks=[line for block in page.get_text('dict')['blocks'] for line in block.get('lines',[])];pdf.save(path)
                atoms=[]
                for n,(value,block) in enumerate(zip([text,'responsibility'],blocks)):
                    atoms.append(dict(id=str(n),group=str(n),page=0,kind='text',role='text',block_type='text',
                        content=value,length=len(value),extraction_method='native',
                        bbox=[v/(500 if k%2==0 else 700) for k,v in enumerate(block['bbox'])]))
                if defect=='rewritten':atoms[-1]['content']='probabilities'
                if defect=='synthetic_band':
                    atoms.append(dict(atoms[0],id='band',group='band',bbox=[0,.4,1,.45],
                                      content='Wide repaired band',length=18))
                rows,moved=native_side_notes(SourceIndex(dict(source_pdf=str(path),atoms=atoms)))
                self.assertEqual(len(rows),int(defect in {'none','synthetic_band'}),defect)
                self.assertEqual(moved,set())
                if rows:self.assertFalse(rows[0]['main_left'])

    def margin_figure(self,root):
        picture=root/'margin.png'
        pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,80),False)
        pix.clear_with(180);pix.save(picture)
        text='A paragraph explains a graph construction with vertices and edges in detail. '*3
        specs=[('a','text','text',text,[.1,.4,.65,.47]),
               ('b','text','text',text,[.1,.48,.65,.6]),
               ('i','image','image_body','',[.74,.4,.88,.47]),
               ('c','text','image_caption','Figure 3.2. The graph and its vertices.',[.7,.48,.92,.56])]
        atoms=[dict(id=i,group=i,page=0,kind=k,role=r,block_type=r,content=t,length=max(1,len(t)),
                    bbox=b,extraction_method='native') for i,k,r,t,b in specs]
        atoms[2]['asset']=str(picture)
        return SourceIndex(dict(atoms=atoms))

    def test_margin_figure_requires_complete_adjacent_native_columns(self):
        with tempfile.TemporaryDirectory() as name:
            for defect in ('none','ocr','edited','inside','far','caption_above','semantic','excluded','foreign'):
                index=self.margin_figure(Path(name));changed=[];registry=[];excluded=[]
                if defect=='ocr':index.atoms[2]['extraction_method']='ocr'
                if defect=='edited':changed=['b']
                if defect=='inside':index.atoms[2]['bbox']=[.4,.4,.6,.47]
                if defect=='far':index.atoms[2]['bbox'][1]=.2
                if defect=='caption_above':index.atoms[3]['bbox']=[.7,.3,.92,.37]
                if defect=='semantic':registry=[dict(kind='theorem',lo=0,hi=index.prefix[-1])]
                if defect=='excluded':excluded=[dict(lo=0,hi=index.prefix[-1])]
                if defect=='foreign':index.atoms[1]['role']='paragraph_title'
                before=deepcopy(index.source)
                rows=marginal_figures(index,registry,excluded,changed)
                self.assertEqual(len(rows),int(defect=='none'),defect)
                self.assertEqual(index.source,before)

    def test_compiled_margin_figure_keeps_caption_and_prose_in_separate_columns(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.margin_figure(root)
            config={**load_config(),'float_assets':True}
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Margin graph',author=''),config,root/'latex',sample_mode=True)
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                body=pdf[0].search_for('A paragraph explains')[0]
                caption=pdf[0].search_for('Figure 3.2.')[0]
                self.assertGreater(caption.x0,body.x1)
                self.assertEqual(pdf[0].get_text().count('Figure 3.2.'),1)
                names=pdf.resolve_names()
                self.assertTrue(all('source-'+a['id'] in names for a in index.atoms))
                image=pymupdf.Rect(pdf[0].get_image_info()[0]['bbox'])
                self.assertLess(image.width,100)  # Diagram is narrower than its caption column.

    def glossary(self,root,size=7):
        source=root/'source.pdf'
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=500,height=700)
            text='A mathematical paragraph introduces several terms with precise meanings. '*3
            page.insert_textbox((40,80,350,180),text,fontname='tiro',fontsize=10)
            page.insert_text((360,100),'independent',fontname='tiro',fontsize=size,color=(.35,.37,.44))
            page.insert_text((360,120),'response',fontname='tiro',fontsize=size,color=(.35,.37,.44))
            pdf.save(source)
        specs=[('body',text,[.08,.114,.7,.257]),('first','independent',[.72,.132,.88,.146]),
               ('second','response',[.72,.16,.88,.175])]
        atoms=[dict(id=i,group=i,page=0,kind='text',role='text',block_type='text',content=t,
                    length=len(t),bbox=b,extraction_method='native') for i,t,b in specs]
        return SourceIndex(dict(source_pdf=str(source),atoms=atoms))

    def test_native_glossary_needs_multiple_small_complete_unedited_notes(self):
        for defect in ('none','body_edit','large','scan','note_edit','excluded','semantic','wrong_text','inside'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as name:
                index=self.glossary(Path(name),size=10 if defect=='large' else 7)
                changed={'body'} if defect=='body_edit' else {'first'} if defect=='note_edit' else ()
                if defect=='scan':index.atoms[1]['extraction_method']='ocr'
                if defect=='wrong_text':index.atoms[1]['content']='unowned'
                if defect=='inside':index.atoms[1]['bbox']=[.5,.132,.65,.146]
                rows=marginal_glossaries(index,
                    [dict(lo=0,hi=index.prefix[-1],kind='example')] if defect=='semantic' else (),
                    [dict(lo=0,hi=index.prefix[-1])] if defect=='excluded' else (),changed)
                self.assertEqual(len(rows),int(defect in {'none','body_edit'}),defect)

    def test_compiled_glossary_preserves_side_order_text_and_destinations(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.glossary(root)
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Margin terms',author=''),load_config(),root/'latex',sample_mode=True)
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                body=pdf[0].search_for('A mathematical paragraph')[0]
                term=pdf[0].search_for('independent')[0]
                response=pdf[0].search_for('response')[0]
                self.assertGreater(term.x0,body.x1);self.assertGreater(response.y0,term.y0)
                span=next(s for b in pdf[0].get_text('dict')['blocks'] for l in b.get('lines',[])
                          for s in l['spans'] if 'independent' in s['text'])
                self.assertNotEqual(span['color'],0)
                self.assertEqual(set(pdf.resolve_names()) & {'source-body','source-first','source-second'},
                                 {'source-body','source-first','source-second'})

    def test_outlined_labels_expand_only_into_unowned_native_figure_space(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'source.pdf'
            with pymupdf.open() as doc:
                page=doc.new_page(width=200,height=200)
                page.draw_rect(pymupdf.Rect(60,60,120,120))
                page.draw_rect(pymupdf.Rect(54,80,58,90),fill=(0,0,0))
                page.draw_rect(pymupdf.Rect(90,124,96,129),fill=(0,0,0))
                doc.save(source)
            image=dict(id='plot',group='plot',page=0,kind='image',role='image_body',
                block_type='image',content='',length=1,extraction_method='native',bbox=[.3,.3,.6,.6])
            for defect in ('none','ocr','edited','owned_text','owned_formula'):
                atoms=[deepcopy(image)];changed=[]
                if defect=='ocr':atoms[0]['extraction_method']='ocr'
                if defect=='edited':changed=['plot']
                if defect.startswith('owned'):
                    atoms.append(dict(image,id='caption',group='caption',kind='text' if defect=='owned_text' else 'math',
                        role='image_caption',content='unrelated',length=9,bbox=[.25,.38,.62,.68]))
                index=SourceIndex(dict(source_pdf=str(source),atoms=atoms))
                result=recover_native_vector_labels(index,root/defect,changed)
                self.assertEqual(bool(result),defect=='none',defect)
                if result:
                    b=result['plot']['bbox'];self.assertLess(b[0],.28);self.assertGreater(b[3],.64)

    def source(self,root,image_left=False):
        image=root/'diagram.png'
        pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,150,100),False)
        pix.clear_with(120);pix.save(image)
        left,right=[.1,.2,.45,.4],[.55,.2,.9,.4]
        cap,img=(right,left) if image_left else (left,right)
        specs=[('caption','C','text','image_caption','Figure 2.1. A diagram with ',cap),
               ('formula','C','math','image_caption','x',cap),
               ('tail','C','text','image_caption',' and a complete caption.',cap),
               ('image','I','image','image_body','',img),
               ('body','P','text','text','This paragraph explains the construction and its mathematical implications in detail.',[.25,.5,.9,.6]),
               ('note','N','text','text','Notebook 2.1',[.08,.58,.18,.6])]
        atoms=[dict(id=i,group=g,kind=k,role=r,content=t,length=max(1,len(t)),
                    page=0,bbox=b,block_type='image' if k=='image' else r,extraction_method='native')
               for i,g,k,r,t,b in specs]
        atoms[3]['asset']=str(image)
        atoms[-1]['source_text_runs']=[dict(start=0,end=12,color=[.2,.5,.6],method='native_span_ink_v1')]
        return SourceIndex(dict(atoms=atoms,source_page_count=1))

    def test_caption_geometry_and_complete_native_ownership_are_required(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for defect in ('none','ocr','edited','page','column','below','semantic','excluded','foreign'):
                index=self.source(root);changed=[];registry=[];excluded=[]
                if defect=='ocr':index.atoms[0]['extraction_method']='ocr'
                if defect=='edited':changed=['formula']
                if defect=='page':index.atoms[3]['page']=1
                if defect=='column':index.atoms[3]['bbox']=[.3,.2,.8,.4]
                if defect=='below':index.atoms[3]['bbox']=[.1,.45,.45,.6]
                if defect=='semantic':registry=[dict(kind='theorem',lo=1,hi=10)]
                if defect=='excluded':excluded=[dict(lo=1,hi=10)]
                if defect=='foreign':
                    other=dict(index.atoms[-2],id='foreign',group='foreign')
                    index=SourceIndex(dict(atoms=[*index.atoms[:3],other,*index.atoms[3:]]))
                before=deepcopy(index.source)
                self.assertEqual(len(side_caption_rows(index,registry,excluded,changed)),int(defect=='none'),defect)
                self.assertEqual(index.source,before)

    def test_only_short_colored_native_margin_text_follows_its_adjacent_prose(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for defect in ('none','black','ocr','edited','page','inside','distant','long','semantic'):
                index=self.source(root);changed=[];registry=[]
                if defect=='black':index.atoms[-1]['source_text_runs']=[]
                if defect=='ocr':index.atoms[-1]['extraction_method']='ocr'
                if defect=='edited':changed=['body']
                if defect=='page':index.atoms[-1]['page']=1
                if defect=='inside':index.atoms[-1]['bbox']=[.3,.58,.4,.6]
                if defect=='distant':index.atoms[-1]['bbox']=[.08,.8,.18,.82]
                if defect=='long':index.atoms[-1]['content']*=20;index.atoms[-1]['length']*=20
                if defect=='semantic':registry=[dict(kind='definition',lo=index.prefix[-2],hi=index.prefix[-1])]
                index=SourceIndex(index.source)
                self.assertEqual(len(marginal_references(index,registry,changed=changed)),int(defect=='none'),defect)

    def test_compiled_pair_keeps_both_sides_math_and_all_destinations(self):
        for image_left in (False,True):
            with self.subTest(image_left=image_left),tempfile.TemporaryDirectory() as name:
                root=Path(name);index=self.source(root,image_left)
                audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                    [],[],dict(title='Native sides',author=''),load_config(),root/'latex',sample_mode=True)
                result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
                self.assertTrue(audit['coverage_complete'])
                with pymupdf.open(root/'latex/main.pdf') as doc:
                    names=doc.resolve_names()
                    self.assertEqual({names['source-'+a['id']]['page'] for a in index.atoms},{0})
                    image=pymupdf.Rect(doc[0].get_image_info()[0]['bbox'])
                    caption=doc[0].search_for('Figure 2.1.')[0]
                    self.assertEqual(image.x0<caption.x0,image_left)
                    self.assertEqual(''.join(p.get_text() for p in doc).count('Notebook'),1)

    def test_lateral_figure_keeps_source_order_after_pending_tall_floats(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root)
            earlier=[dict(index.atoms[3],id='earlier'+str(n),group='earlier'+str(n),
                bbox=[.1,.02+n*.08,.9,.06+n*.08]) for n in range(2)]
            index=SourceIndex({**index.source,'atoms':[*earlier,*index.atoms]})
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Figure order',author=''),{**load_config(),'float_assets':True},
                root/'latex',sample_mode=True)
            self.assertTrue(audit['coverage_complete'])
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'latex/main.pdf') as doc:
                names=doc.resolve_names()
                prior=[names['source-'+a['id']]['page'] for a in earlier]
                pair=names['source-image']['page']
                self.assertGreater(pair,max(prior))
                self.assertEqual(names['source-caption']['page'],pair)
                self.assertEqual(names['source-tail']['page'],pair)
                caption=doc[pair].search_for('and a complete caption.')[0]
                self.assertLess(caption.y1,doc[pair].rect.height-50)


if __name__=='__main__':unittest.main()
