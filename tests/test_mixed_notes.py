"""Independently read note/main lines, conservative failures and shipped PDF."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.config import executable,load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.mixed_notes import mixed_note_rows


class MixedNoteTests(unittest.TestCase):
    def source(self,root):
        path=root/'source.pdf'
        with pymupdf.open() as pdf,pymupdf.open() as raster:
            page=pdf.new_page(width=500,height=700)
            page.draw_rect((270,200,465,222),fill=(.922,)*3,color=None)
            page.insert_text((280,216),'A symbol ends a proof.',fontsize=9)
            page.insert_text((40,216),'The argument continues and',fontsize=9)
            page.insert_text((40,237),'then completes the demonstration.',fontsize=9)
            out=raster.new_page(width=500,height=700)
            out.insert_image(out.rect,stream=page.get_pixmap(dpi=150).tobytes('png'))
            raster.save(path)
        specs=[('start','mixed','text','The argument continues with '),
               ('math','mixed','math','x'),('end','mixed','text',' and A symbol ■ ends a proof.'),
               ('tail','tail','text','then completes the demonstration. '*8),
               ('after','after','text','Following independent paragraph.')]
        atoms=[]
        for n,g,k,t in specs:
            box=([.08,200/700,.93,222/700] if g=='mixed' else
                 [.08,225/700,.93,.40] if g=='tail' else [.08,.43,.93,.45])
            atoms.append(dict(id=n,group=g,kind=k,role='text' if k=='text' else 'equation_inline',
                block_type='text',page=0,bbox=box,block_bbox=box,content=t,length=len(t),extraction_method='ocr'))
        return SourceIndex(dict(source_pdf=str(path),atoms=atoms))

    def observed(self):
        return [[dict(text='A symbol ends a proof.',score=.999,bbox=[5,0,150,25])],
                [dict(text='The argument continues with x and',score=.999,bbox=[0,0,425,25])]]

    def test_exact_note_preserves_symbols_offsets_and_source(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);before=deepcopy(index.source)
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'mock-v1')), \
                 patch('book_pipeline.publishing.mixed_notes.observed_lines',side_effect=self.observed()):
                rows=mixed_note_rows(index,[],[],set(),root/'render')
            self.assertEqual(len(rows),1);r=rows[0]
            self.assertEqual(index.text(*r['note_range']),'A symbol ■ ends a proof.')
            main=''.join(index.text(a,z) for a,z in r['main_ranges'])
            self.assertIn(' and then completes',main)
            self.assertEqual(index.source,before)

    def test_ambiguous_or_changed_content_keeps_source_flow(self):
        for problem in ('mismatch','low_confidence','two_lines','changed','semantic','footnote','native','no_models'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as name:
                root=Path(name);index=self.source(root);obs=self.observed();registry=[];exclusions=[];changed=set()
                if problem=='mismatch':obs[0][0]['text']='A symbol starts a proof.'
                if problem=='low_confidence':obs[0][0]['score']=.98
                if problem=='two_lines':obs[0]*=2
                if problem=='changed':changed={'math'}
                if problem=='semantic':registry=[dict(lo=0,hi=index.prefix[4])]
                if problem=='footnote':exclusions=[dict(lo=index.prefix[2],hi=index.prefix[3])]
                if problem=='native':index.atoms[0]['extraction_method']='native'
                runtime=None if problem=='no_models' else (object(),'mock-v1')
                with patch('book_pipeline.publishing.image_search.local_ocr',return_value=runtime), \
                     patch('book_pipeline.publishing.mixed_notes.observed_lines',side_effect=obs):
                    self.assertEqual(mixed_note_rows(index,registry,exclusions,changed,root/'render'),[])

    def test_compiled_note_is_rounded_separate_searchable_and_keeps_destinations(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);before=deepcopy(index.source);project=root/'latex'
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'mock-v1')), \
                 patch('book_pipeline.publishing.mixed_notes.observed_lines',side_effect=self.observed()):
                audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],
                    dict(title='Mixed note',author=''),load_config(),project,sample_mode=True)
            self.assertTrue(audit['coverage_complete']);self.assertEqual(index.source,before)
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(project/'main.pdf') as pdf:
                self.assertEqual(len(pdf),1);page=pdf[0]
                main=page.search_for('The argument')[0];note=page.search_for('A symbol')[0]
                self.assertGreater(note.x0,main.x1);self.assertLess(abs(main.y0-note.y0),15)
                frames=[d for d in page.get_drawings() if d.get('fill') and abs(d['fill'][0]-.922)<.01]
                self.assertTrue(frames);frame=max(frames,key=lambda d:d['rect'].get_area())
                self.assertTrue(any(item[0]=='c' for item in frame['items']))
                self.assertGreater(note.x0-frame['rect'].x0,5)
                self.assertGreater(page.search_for('Following independent')[0].y0,frame['rect'].y1)
                self.assertEqual(page.get_text().count('A symbol'),1)
                self.assertTrue(all('source-'+a['id'] in pdf.resolve_names() for a in index.atoms))
