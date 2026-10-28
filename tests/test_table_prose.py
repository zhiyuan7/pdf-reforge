"""Ruled prose recovery keeps true tables, immutable OCR and source crops."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.config import digest, load_config, read, save
from book_pipeline.ingestion.source import SourceIndex, load_source, materialize_source
from book_pipeline.ingestion.table_prose import recover_table_prose, validate, candidates


class Client:
    def __init__(self, answer): self.answer, self.calls, self.ledger = answer, 0, self
    def summary(self): return {}
    def ask(self, task, system, payload, images, validator):
        self.calls += 1
        value = deepcopy(self.answer)
        validator(value)
        return value


def fixture(root):
    pdf = root/'source.pdf'
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((300, 110), 'Ruled prose column. Another paragraph.')
        page.insert_text((100, 110), 'Continuation with a formula.')
        document.save(pdf)
    origin = root/'ocr'; origin.mkdir()
    with pymupdf.open(pdf) as document:
        document[0].get_pixmap().save(origin/'table.png')
    save(origin/'middle_json.json', dict(schema='docvortex.middle', schema_version='2.0',
        pages=[dict(page_idx=0, blocks=[dict(index=0, type='table', bbox=[.05,.05,.95,.95], content=[
            dict(type='table_body',content='<table><tr><td>Ruled prose</td></tr></table>',image_path='table.png')])])]))
    save(origin/'manifest.json', dict(input_pdf=str(pdf),input_sha256=digest(pdf),mineru_version='test'))
    return pdf, origin


ANSWER = dict(classification='prose', reason='Continuous prose in vertical columns', direction='vertical_rl', blocks=[
    dict(type='text', bbox=[.5,.1,.9,.2], parts=[dict(kind='text',content='Ruled prose column. Another paragraph.')]),
    dict(type='image',bbox=[.1,.21,.3,.25],parts=[],reason='Uncertain mathematical notation'),
    dict(type='text',bbox=[.1,.3,.4,.4],parts=[dict(kind='text',content='Continuation with a formula.')])])


class TableProseTests(unittest.TestCase):
    def test_true_table_keeps_original_html_and_image_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf,origin=fixture(root)
            client=Client(dict(classification='table',reason='Rows and columns contain related values',blocks=[]))
            dest=recover_table_prose(pdf,origin,root/'reviewed',load_config(),client=client)
            raw=SourceIndex(load_source(pdf,origin));result=SourceIndex(load_source(pdf,dest))
            self.assertEqual(raw.atoms[0]['content'],result.atoms[0]['content'])
            self.assertEqual(raw.atoms[0]['asset_sha256'],result.atoms[0]['asset_sha256'])
            self.assertEqual(read(dest/'manifest.json')['table_prose_recovery']['recovered_regions'],0)

    def test_recovery_order_crops_cache_policy_and_raw_immutability(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf,origin=fixture(root)
            before={p.name:digest(p) for p in origin.iterdir()}
            client=Client(ANSWER);config=load_config()
            dest=recover_table_prose(pdf,origin,root/'reviewed',config,client=client)
            index=SourceIndex(load_source(pdf,dest))
            self.assertEqual([a['kind'] for a in index.atoms],['text','image','text'])
            self.assertEqual(index.atoms[0]['content'],ANSWER['blocks'][0]['parts'][0]['content'])
            self.assertTrue(all(a['writing_direction']=='vertical_rl' for a in index.atoms))
            self.assertTrue(Path(index.atoms[1]['asset']).is_file())
            with pymupdf.open(pdf) as document:
                box=ANSWER['blocks'][1]['bbox'];page=document[0]
                rect=pymupdf.Rect(*(v*(page.rect.width if k%2==0 else page.rect.height) for k,v in enumerate(box)))
                self.assertEqual(pymupdf.Pixmap(index.atoms[1]['asset']).samples,
                                 page.get_pixmap(dpi=config['image_dpi'],clip=rect).samples)
            # Metadata and exact assets survive later formula materialization.
            saved=materialize_source(index,root/'materialized')
            reloaded=SourceIndex(load_source(pdf,saved))
            self.assertEqual(reloaded.atoms[1]['asset_sha256'],index.atoms[1]['asset_sha256'])
            self.assertEqual(reloaded.atoms[1]['recovery_issue'],'formula_source_crop_requires_transcription')
            self.assertEqual(recover_table_prose(pdf,origin,dest,config,client=client),dest)
            self.assertEqual(client.calls,1)
            with self.assertRaisesRegex(ValueError,'cache changed'):
                recover_table_prose(pdf,origin,dest,{**config,'image_dpi':config['image_dpi']+1},client=client)
            from book_pipeline.ingestion.formula_crops import policy_signature
            with patch('book_pipeline.ingestion.formula_crops.policy_signature',
                       return_value={**policy_signature(),'method':'changed_geometry_policy'}):
                with self.assertRaisesRegex(ValueError,'cache changed'):
                    recover_table_prose(pdf,origin,dest,config,client=client)
            self.assertEqual(before,{p.name:digest(p) for p in origin.iterdir()})

    def test_rejects_body_images_outside_regions_and_empty_recovery(self):
        for mutate in [lambda a:a['blocks'][1].update(bbox=[.05,.05,.95,.95]),
                       lambda a:a['blocks'][0].update(bbox=[0,0,.5,.5]),
                       lambda a:a.update(blocks=[a['blocks'][1]]),
                       lambda a:a.update(classification='uncertain')]:
            answer=deepcopy(ANSWER);mutate(answer)
            with self.assertRaises(ValueError): validate(answer,[.05,.05,.95,.95])

    def test_small_table_skips_review_and_reused_ocr_still_enters_common_review(self):
        self.assertEqual(candidates(dict(pages=[dict(page_idx=0,blocks=[
            dict(type='table',bbox=[.1,.1,.4,.3])])])),[])
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf,origin=fixture(root)
            (root/'output').mkdir()
            origin=origin.rename(root/'output'/'ocr')
            from book_pipeline.application.pipeline import prepare_book_source
            with patch('book_pipeline.application.pipeline.ROOT',root), \
                 patch('book_pipeline.ingestion.table_prose.recover_table_prose',return_value=origin) as review, \
                 patch('book_pipeline.ingestion.formula_bands.correct_formula_bands',return_value=dict(records=[])), \
                 patch('book_pipeline.ingestion.translation.translate_source',return_value=origin):
                self.assertEqual(prepare_book_source(pdf,None,load_config(),root/'result'),origin)
                self.assertEqual(review.call_args.args[1],origin)

    def test_horizontal_formula_review_does_not_cross_vertical_prose_columns(self):
        from book_pipeline.ingestion.formula_bands import detect_bands
        atom=dict(id='eq',group='eq',kind='display_math',block_type='equation',page=0,
                  bbox=[.3,.2,.7,.23],content='x=y',length=3,extraction_method='ocr',writing_direction='vertical_rl')
        self.assertEqual(detect_bands(SourceIndex(dict(atoms=[atom]))),[])

    def test_formula_crops_keep_order_instead_of_merging_into_an_illustration_row(self):
        from book_pipeline.publishing.latex import Renderer
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf,origin=fixture(root)
            answer=deepcopy(ANSWER)
            second=deepcopy(answer['blocks'][1]);second['bbox']=[.45,.21,.65,.25]
            answer['blocks'].insert(2,second)
            answer['blocks'][0]['parts'].append(dict(kind='math',content=r'\\frac{a}{b}'))
            dest=recover_table_prose(pdf,origin,root/'reviewed',load_config(),client=Client(answer))
            index=SourceIndex(load_source(pdf,dest))
            self.assertEqual(index.atoms[1]['content'],r'\frac{a}{b}')
            (root/'latex').mkdir()
            renderer=Renderer(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],root/'latex',dict(float_assets=True))
            tex=''.join(renderer.render())
            self.assertEqual(tex.count(r'\BookFormulaImage['),2)
            self.assertNotIn(r'\begin{figure}',tex)
            self.assertFalse(renderer.image_rows)
            self.assertLess(tex.index('Ruled prose'),tex.index(r'\BookFormulaImage['))
            self.assertGreater(tex.index('Continuation'),tex.rindex(r'\BookFormulaImage['))
