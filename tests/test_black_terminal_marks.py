from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import pymupdf
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.raster_marks import raster_marks
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.config import load_config,executable


class BlackTerminalMarkTests(unittest.TestCase):
    def source(self,root,hidden=False,following=False):
        path=root/'source.pdf';text='The argument ends.'
        with pymupdf.open() as live,pymupdf.open() as scan:
            p=live.new_page(width=500,height=700)
            p.insert_text((50,200),text,fontname='tiro',fontsize=12)
            end=50+pymupdf.Font('tiro').text_length(text,fontsize=12)
            p.draw_rect((end+5,195,end+10,200),color=None,fill=(0,0,0))
            if following:p.insert_text((end+20,200),'More words.',fontname='tiro',fontsize=12)
            rp=scan.new_page(width=500,height=700)
            rp.insert_image(rp.rect,stream=p.get_pixmap(dpi=200).tobytes('png'))
            if hidden:rp.insert_text((50,200),'An unreliable hidden OCR layer.',render_mode=3)
            scan.save(path)
        atom=dict(id='last',group='g',page=0,kind='text',role='text',block_type='text',
                  extraction_method='ocr',content=text,length=len(text),bbox=[.1,.26,.9,.30])
        return SourceIndex(dict(atoms=[atom],source_pdf=str(path),source_page_count=1))

    def test_image_only_and_hidden_ocr_scans_recover_inline_owned_square(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for hidden in (False,True):
                index=self.source(root,hidden=hidden);before=deepcopy(index.source)
                marks,labels,_=raster_marks(index,[],root/'render')
                self.assertEqual(list(marks),['last'])
                self.assertTrue(marks['last']['inline'])
                self.assertTrue(2<marks['last']['gap_bp']<7)
                self.assertEqual(labels,{})
                self.assertEqual(index.source,before)

    def test_following_text_changes_duplicates_and_missing_period_abstain(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);base=self.source(root)
            for defect in ('following','changed','excluded','duplicate','recognized','period'):
                index=SourceIndex(deepcopy(base.source));changed=();excluded=[]
                if defect=='following':index=self.source(root,following=True)
                if defect=='changed':changed=('last',)
                if defect=='excluded':excluded=[dict(lo=0,hi=index.prefix[-1])]
                if defect=='duplicate':
                    index=SourceIndex(dict(base.source,atoms=[*index.atoms,dict(index.atoms[0],id='other',group='other')]))
                if defect in ('recognized','period'):
                    a=index.atoms[0];a['content']=a['content']+' ■' if defect=='recognized' else a['content'][:-1]
                    a['length']=len(a['content']);index=SourceIndex(index.source)
                with self.subTest(defect=defect):
                    self.assertEqual(raster_marks(index,[],root/'render',excluded,changed)[0],{})

    def test_compilation_keeps_end_mark_beside_last_word_and_text_searchable(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root,hidden=True);project=root/'latex'
            result=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],
                                 dict(title='End mark',author=''),load_config(),project,sample_mode=True)
            self.assertTrue(any(r['method']=='raster_period_owned_black_terminal_square_v1'
                                for r in result['source_order_layout']))
            compiled=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(compiled['success'],compiled['log_tail'])
            with pymupdf.open(project/'main.pdf') as pdf:
                self.assertEqual(len(pdf),1);p=pdf[0]
                text=p.search_for('The argument ends.')[0];images=p.get_image_info()
                self.assertEqual(len(images),1);mark=pymupdf.Rect(images[0]['bbox'])
                self.assertTrue(1<mark.x0-text.x1<8)
                self.assertLess(abs(mark.y1-text.y1),5)
