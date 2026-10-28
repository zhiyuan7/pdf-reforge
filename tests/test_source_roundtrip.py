"""Rich caption and algorithm math retains its semantic owner after saving."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.config import digest, read, save
from book_pipeline.ingestion.source import SourceIndex, load_source, materialize_source


class SourceRoundtripTests(unittest.TestCase):
    def test_automatic_whole_book_reuse_cannot_select_a_partial_preflight(self):
        from book_pipeline.application.pipeline import prepare_book_source
        from book_pipeline.config import load_config
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);origin=root/'output'/'preflight';origin.mkdir(parents=True);pdf=root/'input.pdf'
            with pymupdf.open() as document:
                document.new_page();document.new_page();document.save(pdf)
            save(origin/'middle_json.json',dict(schema='docvortex.middle',schema_version='2.0',
                                              pages=[dict(page_idx=1,blocks=[])]))
            save(origin/'manifest.json',dict(input_sha256=digest(pdf),mineru_version='test',source_page_numbers=[2]))
            with patch('book_pipeline.application.pipeline.ROOT',root), patch(
                    'book_pipeline.ingestion.native.prepare_source',side_effect=RuntimeError('fresh extraction required')) as fresh:
                with self.assertRaisesRegex(RuntimeError,'fresh extraction required'):
                    prepare_book_source(pdf,None,load_config(),root/'output'/'new')
                fresh.assert_called_once()

    def test_declared_excerpt_keeps_original_indices_and_scope_after_roundtrip(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);origin=root/'ocr';origin.mkdir();pdf=root/'input.pdf'
            with pymupdf.open() as document:
                for _ in range(3): document.new_page()
                document.save(pdf)
            save(origin/'middle_json.json',dict(schema='docvortex.middle',schema_version='2.0',
                pages=[dict(page_idx=1,blocks=[dict(index=0,type='text',bbox=[.1,.1,.8,.2],
                                                  content='Selected page')])]))
            manifest=dict(input_pdf=str(pdf),input_sha256=digest(pdf),mineru_version='test',
                          source_page_numbers=[2],is_full_document=False)
            save(origin/'manifest.json',manifest)
            source=load_source(pdf,origin)
            self.assertEqual(source['selected_pages'],[1])
            self.assertEqual(source['atoms'][0]['page'],1)
            target=materialize_source(SourceIndex(source),root/'saved')
            self.assertEqual(load_source(pdf,target)['selected_pages'],[1])
            self.assertFalse(read(target/'middle_json.json')['is_full_document'])
            self.assertFalse(read(target/'manifest.json')['is_full_document'])
            with self.assertRaisesRegex(ValueError,'missing selected'):
                load_source(pdf,origin,'all')
            for invalid in ([2,2],[True],[0],[4],[],[3,2]):
                save(origin/'manifest.json',{**manifest,'source_page_numbers':invalid})
                with self.assertRaisesRegex(ValueError,'Invalid manifest'):
                    load_source(pdf,origin)
            manifest.pop('source_page_numbers');save(origin/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'missing selected'):
                load_source(pdf,origin)

    def test_caption_math_and_algorithm_math_keep_roles_content_and_assets(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);origin=root/'ocr';origin.mkdir();pdf=root/'input.pdf'
            with pymupdf.open() as document:
                p=document.new_page();p.insert_text((40,70),'Source figure and caption')
                p.get_pixmap().save(origin/'figure.png');document.save(pdf)
            math=lambda value:dict(type='equation_inline',content=value)
            save(origin/'middle_json.json',dict(schema='docvortex.middle',schema_version='2.0',pages=[dict(
                page_idx=0,blocks=[dict(index=0,type='image',bbox=[.1,.1,.8,.4],content=[
                    dict(type='image_body',content='',image_path='figure.png'),
                    dict(type='image_caption',content=[dict(type='text',content='Elements of ',
                        source_text_runs=[dict(start=0,end=8,color=[1,.5,0],method='native_span_ink_v1')]),math('R^2'),
                                                       dict(type='text',content=' are vectors.')]),
                    dict(type='image_footnote',content=[math('v'),dict(type='text',content=' denotes a vector.')])]),
                dict(index=1,type='code',bbox=[.1,.5,.8,.7],content=[dict(type='algorithm_body',content=[
                    dict(type='text',content='Set '),math('x=0')])]),
                dict(index=2,type='table',bbox=[.1,.8,.8,.9],content=[dict(type='table_caption',content=[
                    dict(type='text',content='Values of '),math('n')])]),
                dict(index=3,type='text',bbox=[.1,.1,.9,.9],writing_direction='vertical_rl',
                     source_prose_end_bbox=[.1,.1,.2,.3],content=[dict(type='text',content='Continued source prompt')])])]))
            save(origin/'manifest.json',dict(input_pdf=str(pdf),input_sha256=digest(pdf),mineru_version='test'))
            raw=SourceIndex(load_source(pdf,origin))
            raw_hash=digest(origin/'middle_json.json')
            def meanings(index):return [(a['kind'],a['role'],a['content'],a.get('asset_sha256'),a.get('source_text_runs'),a.get('source_prose_end_bbox')) for a in index.atoms]
            dest=materialize_source(raw,root/'saved')
            reloaded=SourceIndex(load_source(pdf,dest))
            self.assertEqual(meanings(raw),meanings(reloaded))
            again=materialize_source(reloaded,root/'saved-again')
            self.assertEqual(meanings(raw),meanings(SourceIndex(load_source(pdf,again))))
            self.assertEqual(raw_hash,digest(origin/'middle_json.json'))
            self.assertEqual(materialize_source(raw,dest),dest)
            manifest=read(dest/'manifest.json');manifest.pop('materialization_method');save(dest/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'cache does not match'):materialize_source(raw,dest)
