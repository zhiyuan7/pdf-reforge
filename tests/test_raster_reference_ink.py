"""Colored references require independent token, context and source ownership."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.raster_reference_ink import raster_reference_ink, reference_range


class RasterReferenceInkTests(unittest.TestCase):
    def row(self, text, score=.999, bbox=None):
        return dict(text=text, score=score, bbox=bbox or [0,0,100,20])

    def test_exact_context_and_unique_terminal_token(self):
        text='A topic. More details in Chapter 12.'
        line=self.row('More details in Chapter 12.')
        self.assertEqual(reference_range(text,line,self.row('12')),(33,35))
        for bad_line,bad_token in [(self.row('More details in chapter 12.'),self.row('12')),
                (line,self.row('2')), (line,self.row('12',.98)),
                (self.row('More details in Chapter 12.' , .98),self.row('12')),
                (self.row('Chapter 12 and Section 12.'),self.row('12')),
                (self.row('Chapter 12 begins.'),self.row('12')),
                (self.row(''),self.row('12'))]:
            self.assertIsNone(reference_range(text,bad_line,bad_token))
        self.assertIsNone(reference_range(text+text,line,self.row('12')))

    def source(self, root, native=False):
        path=root/'scan.pdf'
        with pymupdf.open() as original, pymupdf.open() as scan:
            p=original.new_page(width=500,height=700)
            p.draw_rect((260,190,470,250),fill=(.922,)*3,color=None)
            p.insert_text((270,215),'Details in Chapter ',fontsize=11)
            p.insert_text((362,215),'6',fontsize=11,color=(0,0,1))
            p.insert_text((369,215),'.',fontsize=11)
            if native: original.save(path)
            else:
                q=scan.new_page(width=500,height=700)
                q.insert_image(q.rect,stream=p.get_pixmap(dpi=150).tobytes('png'))
                scan.save(path)
        text='More context. Details in Chapter 6.'
        return SourceIndex(dict(source_pdf=str(path),atoms=[dict(id='a',group='a',page=0,
            kind='text',role='text',block_type='text',content=text,length=len(text),
            bbox=[.53,.28,.93,.35],extraction_method='native' if native else 'ocr')]))

    def observations(self):
        return [[self.row('Details in Chapter 6.',bbox=[40,0,460,40])],[self.row('6')]]

    def test_requires_exact_ocr_and_caches_by_pixels_text_and_model(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);before=deepcopy(index.source)
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'v1')), \
                 patch('book_pipeline.publishing.raster_reference_ink.observed_lines',side_effect=self.observations()):
                ink=raster_reference_ink(index,root/'render')
            self.assertEqual(index.source,before);self.assertEqual(set(ink),{'a'})
            run=ink['a'][0];self.assertEqual(run['quote'],'6')
            self.assertGreater(run['color'][2],.95);self.assertLess(run['color'][0],.3)
            self.assertEqual(index.atoms[0]['content'][run['start']:run['end']],'6')
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'v1')), \
                 patch('book_pipeline.publishing.raster_reference_ink.observed_lines',side_effect=AssertionError('cache miss')):
                self.assertEqual(raster_reference_ink(index,root/'render'),ink)
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'v2')), \
                 patch('book_pipeline.publishing.raster_reference_ink.observed_lines',side_effect=self.observations()) as ocr:
                self.assertTrue(raster_reference_ink(index,root/'render'));self.assertEqual(ocr.call_count,2)

    def test_rewrites_overlap_exclusions_native_and_uncertain_ocr_fail_closed(self):
        for issue in ('changed','excluded','native','overlap','rewrite','low_score','ambiguous_line','math','mixed_box'):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as name:
                root=Path(name);index=self.source(root,native=issue=='native');observed=self.observations()
                if issue=='overlap':index=SourceIndex({**index.source,'atoms':[*index.atoms,{**index.atoms[0],'id':'b','group':'b'}]})
                if issue=='rewrite':index.atoms[0]['content']='Details in Chapter 7.'
                if issue=='low_score':observed[1][0]['score']=.98
                if issue=='ambiguous_line':observed[0]*=2
                if issue=='math':index.atoms[0]['kind']='math'
                if issue=='mixed_box':index.atoms[0]['bbox']=[.1,.28,.93,.35]
                with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'v1')), \
                     patch('book_pipeline.publishing.raster_reference_ink.observed_lines',side_effect=observed):
                    self.assertEqual(raster_reference_ink(index,root/'render',
                        [dict(lo=0,hi=index.prefix[-1])] if issue=='excluded' else [],
                        {'a'} if issue=='changed' else []),{})
