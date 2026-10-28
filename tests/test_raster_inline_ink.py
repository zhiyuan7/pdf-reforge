"""Existing inline squares retain searchable glyphs and verified scan color."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.raster_inline_ink import METHOD, raster_inline_ink


class RasterInlineInkTests(unittest.TestCase):
    def source(self, root, live=False, second_square=False, hollow=False):
        path = root/'source.pdf'
        with pymupdf.open() as original, pymupdf.open() as raster:
            p = original.new_page(width=500, height=700)
            p.draw_rect((270,200,465,222), fill=(.922,)*3, color=None)
            p.insert_text((280,216), 'Symbol', fontsize=9)
            p.draw_rect((313,209,319,215), color=(1,.52,.08),
                        fill=None if hollow else (1,.52,.08), width=.3)
            p.insert_text((325,216), 'means proof finished.', fontsize=9)
            if second_square:
                p.draw_rect((450,209,456,215), color=None, fill=(1,.52,.08))
            if live: original.save(path)
            else:
                q = raster.new_page(width=500,height=700)
                q.insert_image(q.rect,stream=p.get_pixmap(dpi=150).tobytes('png'))
                raster.save(path)
        text = 'Symbol ■ means proof finished.'
        return SourceIndex(dict(source_pdf=str(path), atoms=[dict(id='note', group='note',
            page=0, kind='text', role='text', block_type='text', content=text, length=len(text),
            bbox=[.56,.29,.92,.31], block_bbox=[.56,.29,.92,.31],
            extraction_method='native' if live else 'ocr')]))

    def observed(self):
        return [[dict(text='Symbol',score=.999,bbox=[0,0,35,18])],
                [dict(text='means proof finished.',score=.999,bbox=[0,0,120,18])]]

    def test_color_needs_exact_both_neighbors_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);before=deepcopy(index.source)
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'test-v1')), \
                 patch('book_pipeline.publishing.raster_inline_ink.observed_lines',side_effect=self.observed()):
                ink=raster_inline_ink(index,root/'render')
            self.assertEqual(index.source,before)
            self.assertEqual(set(ink),{'note'})
            r=ink['note'][0]
            self.assertEqual(index.atoms[0]['content'][r['start']:r['end']],'■')
            self.assertEqual(r['method'],METHOD)
            self.assertGreater(r['color'][0],.95);self.assertLess(r['color'][2],.15)
            # Same signature reuses evidence; a changed OCR model cannot.
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'test-v1')), \
                 patch('book_pipeline.publishing.raster_inline_ink.observed_lines',side_effect=AssertionError('cache miss')):
                self.assertEqual(raster_inline_ink(index,root/'render'),ink)
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'test-v2')), \
                 patch('book_pipeline.publishing.raster_inline_ink.observed_lines',side_effect=self.observed()) as ocr:
                self.assertTrue(raster_inline_ink(index,root/'render'))
                self.assertEqual(ocr.call_count,2)

    def test_ambiguous_geometry_text_and_edited_content_are_rejected(self):
        for problem in ('left','right','low_confidence','multiple_lines','two_glyphs','two_shapes',
                        'hollow','overlap','changed','excluded','native','math','multiline','missing_models'):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as name:
                root=Path(name);index=self.source(root,live=problem=='native',
                    second_square=problem=='two_shapes',hollow=problem=='hollow')
                obs=self.observed();changed=set();exclusions=[]
                if problem=='left':obs[0][0]['text']='Another symbol'
                elif problem=='right':obs[1][0]['text']='means proof started.'
                elif problem=='low_confidence':obs[0][0]['score']=.98
                elif problem=='multiple_lines':obs[1]*=2
                elif problem=='two_glyphs':index.atoms[0]['content']+=' ■'
                elif problem=='overlap':
                    index=SourceIndex({**index.source,'atoms':[*index.atoms,
                        {**index.atoms[0],'id':'other','group':'other','content':'Overlapping owner'}]})
                elif problem=='changed':changed={'note'}
                elif problem=='excluded':exclusions=[dict(lo=0,hi=index.prefix[-1])]
                elif problem=='math':index.atoms[0]['kind']='display_math'
                elif problem=='multiline':index.atoms[0]['block_bbox']=[.56,.29,.92,.4]
                runtime=None if problem=='missing_models' else (object(),'test')
                with patch('book_pipeline.publishing.image_search.local_ocr',return_value=runtime), \
                     patch('book_pipeline.publishing.raster_inline_ink.observed_lines',side_effect=obs):
                    self.assertEqual(raster_inline_ink(index,root/'render',exclusions,changed),{})

    def test_compiled_square_color_search_text_and_layout(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);before=deepcopy(index.source)
            paths=[]
            for colored in (False,True):
                project=root/('colored' if colored else 'baseline')
                with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'test')), \
                     patch('book_pipeline.publishing.raster_inline_ink.observed_lines',side_effect=self.observed()), \
                     patch('book_pipeline.publishing.raster_inline_ink.raster_inline_ink',
                           wraps=raster_inline_ink if colored else lambda *a,**k:{}):
                    audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                        [],[],dict(title='Inline ink',author=''),load_config(),project,sample_mode=True)
                self.assertTrue(audit['coverage_complete']);self.assertEqual(index.source,before)
                result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
                paths.append(project/'main.pdf')
            with pymupdf.open(paths[0]) as baseline,pymupdf.open(paths[1]) as colored:
                self.assertEqual(len(baseline),len(colored))
                self.assertEqual(baseline.resolve_names(),colored.resolve_names())
                for p,q in zip(baseline,colored):
                    self.assertEqual(p.get_text(),q.get_text())
                    self.assertEqual(p.get_text('words'),q.get_text('words'))
                glyphs=[(c,s['color']) for b in colored[0].get_text('rawdict')['blocks']
                    for l in b.get('lines',[]) for s in l['spans'] for c in s['chars'] if c['c']=='■']
                self.assertEqual(len(glyphs),1)
                self.assertGreater(glyphs[0][1]>>16,240)
                self.assertLess(glyphs[0][1]&255,40)
                self.assertEqual(sum(len(p.get_images()) for p in colored),0)
