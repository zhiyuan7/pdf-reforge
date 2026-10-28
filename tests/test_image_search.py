"""Searchable image captions preserve pixels and stay inside their owner."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.image_search import ImageSearch, caption_rows, searchable_raster, source_row_caption_rows
from book_pipeline.publishing.latex import write_project


class ImageSearchTests(unittest.TestCase):
    def row_fixture(self, root):
        index,_=self.fixture(root)
        caption=dict(id='caption',group='figure',page=0,kind='text',role='image_caption',
                     block_type='image',extraction_method='ocr',content='A vector.',length=9)
        index=SourceIndex({**index.source,'atoms':[*index.atoms,caption]})
        row=dict(asset=index.atoms[0]['asset'],members=['figure','caption'],
                 images=['figure'],bbox=[.2,.2,.8,.8])
        observed=[[[[110,168],[220,168],[220,197],[110,197]],['A vector.',.945]]]
        return index,row,observed

    def test_owned_row_caption_is_searchable_with_identical_pixels_and_cached_readings(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index,row,observed=self.row_fixture(root)
            before=deepcopy(index.source)
            model=Mock(ocr=Mock(side_effect=[[observed],[[('A vector.',.94)]]]))
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(model,'row-reader')):
                search=ImageSearch(index,root/'render')
                asset=search.row_asset(row)
                self.assertEqual(asset,search.row_asset(row))
            self.assertEqual(model.ocr.call_count,2)
            self.assertEqual(index.source,before)
            with pymupdf.open(asset) as pdf:
                self.assertEqual(pdf[0].get_pixmap().samples,pymupdf.Pixmap(row['asset']).samples)
                self.assertEqual(pdf[0].get_text().strip(),'A vector.')
                self.assertEqual(len(pdf[0].search_for('A vector.')),1)
            # Changed source text invalidates a previously successful cache.
            index.atoms[-1]['content']='A scalar.'
            model=Mock(ocr=Mock(return_value=[observed]))
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(model,'row-reader')):
                self.assertIsNone(ImageSearch(index,root/'render').row_asset(row))
            self.assertEqual(model.ocr.call_count,1)

    def test_row_caption_requires_source_ownership_position_and_two_literal_readings(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for defect in ('score','reread','wrong_text','inside_image','duplicate_line','wrong_group','math'):
                index,row,observed=self.row_fixture(root)
                model=Mock(ocr=Mock(return_value=[[('A vector.',.94)]]))
                if defect=='score':observed[0][1][1]=.89
                if defect=='reread':model.ocr.return_value=[[('A scalar.',.99)]]
                if defect=='wrong_text':observed[0][1][0]='A scalar.'
                if defect=='inside_image':observed[0][0]=[[110,30],[220,30],[220,55],[110,55]]
                if defect=='duplicate_line':observed*=2
                if defect=='wrong_group':index.atoms[-1]['group']='other'
                if defect=='math':index.atoms[-1]['kind']='math'
                rgb=np.full((210,340,3),255,dtype=np.uint8)
                self.assertEqual(source_row_caption_rows(index,row,rgb,observed,model),[],defect)

    def fixture(self, root, text='A vector.'):
        image = root/'figure.png'
        with pymupdf.open() as pdf:
            p = pdf.new_page(width=340, height=210)
            p.draw_line((10, 100), (320, 20), color=(1, 0, 0))
            p.insert_text((115, 190), text, fontsize=18)
            p.get_pixmap().save(image)
        atom = dict(id='figure', group='figure', page=0, kind='image', role='image_body',
                    block_type='image', extraction_method='ocr', asset=str(image),
                    content='', length=1, bbox=[.2, .2, .8, .6])
        rows = [[[[110, 168], [220, 168], [220, 197], [110, 197]], [text, .999]]]
        index = SourceIndex(dict(atoms=[atom], source_page_count=1))
        return index, Mock(ocr=Mock(return_value=[rows]))

    def test_bottom_prose_requires_confidence_clear_gap_and_upright_geometry(self):
        rgb = np.full((210, 340, 3), 255, dtype=np.uint8)
        box = [[110, 168], [220, 168], [220, 197], [110, 197]]
        for text in ('A vector.', 'Scalar multiplication.', '一个向量.', '两向量之和.'):
            self.assertEqual(caption_rows(rgb, [[box, [text, .999]]])[0]['text'], text)
        for text in ('x', 'u + v', '(a, b)', '2x', 'Fig. 2.1', r'\input{secret}', ''):
            self.assertEqual(caption_rows(rgb, [[box, [text, .999]]]), [])
        self.assertEqual(caption_rows(rgb, [[box, ['A vector.', .99]]]), [])
        self.assertEqual(caption_rows(rgb, [[box, ['A vector.', .999]]]*2), [])
        tilted = [[110, 168], [220, 180], [220, 209], [110, 197]]
        self.assertEqual(caption_rows(rgb, [[tilted, ['A vector.', .999]]]), [])
        rgb[165, 20] = 0
        self.assertEqual(caption_rows(rgb, [[box, ['A vector.', .999]]]), [])

    def test_wrapper_has_identical_pixels_and_correct_invisible_english_and_chinese(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); index, _ = self.fixture(root)
            for text in ('A vector.', '一个向量.'):
                target = root/'search.pdf'
                row = dict(text=text, bbox=[110, 168, 220, 197])
                searchable_raster(index.atoms[0]['asset'], target, [row])
                self.assertLess(target.stat().st_size, 60000)  # CJK font subset, not a multi-MB font.
                with pymupdf.open(target) as pdf:
                    original = pymupdf.Pixmap(index.atoms[0]['asset'])
                    self.assertEqual(pdf[0].get_pixmap().samples, original.samples)
                    found = pdf[0].search_for(text)
                    self.assertEqual(len(found), 1)
                    self.assertLess(max(abs(a-b) for a,b in zip(found[0], row['bbox'])), .1)
                    self.assertEqual(pdf[0].get_text().strip(), text)

    def test_ownership_duplicate_and_changed_model_cache(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); index, model = self.fixture(root)
            before = deepcopy(index.source)
            with patch('book_pipeline.publishing.image_search.local_ocr', return_value=(model, 'model-one')):
                result = ImageSearch(index, root/'render').asset(index.atoms[0])
                self.assertTrue(result.is_file())
                ImageSearch(index, root/'render').asset(index.atoms[0])
                self.assertEqual(model.ocr.call_count, 1)
                # A separately extracted caption must remain the single owner.
                caption = dict(id='caption', group='caption', page=0, kind='text', role='image_caption',
                               block_type='image_caption', content='A vector.', length=9)
                duplicate = SourceIndex({**index.source, 'atoms':[*index.atoms, caption]})
                self.assertIsNone(ImageSearch(duplicate, root/'render').asset(index.atoms[0]))
                for atom in ({**index.atoms[0], 'extraction_method':'native'},
                             {**index.atoms[0], 'recovery_issue':'formula_source_crop_requires_transcription'},
                             {**index.atoms[0], 'asset':str(root/'composite.png')}):
                    self.assertIsNone(ImageSearch(index, root/'render').asset(atom))
            with patch('book_pipeline.publishing.image_search.local_ocr', return_value=(model, 'model-two')):
                result2 = ImageSearch(index, root/'render').asset(index.atoms[0])
                self.assertNotEqual(result, result2)
                self.assertEqual(model.ocr.call_count, 2)
            self.assertEqual(index.source, before)

    def test_reused_image_on_another_page_keeps_independent_caption_ownership(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); index, model = self.fixture(root)
            second = {**index.atoms[0], 'id':'second', 'group':'second', 'page':1}
            caption = dict(id='caption', group='caption', page=1, kind='text', role='image_caption',
                           block_type='image_caption', content='A vector.', length=9)
            index = SourceIndex({**index.source, 'atoms':[*index.atoms, second, caption]})
            with patch('book_pipeline.publishing.image_search.local_ocr', return_value=(model, 'model')):
                search = ImageSearch(index, root/'render')
                self.assertIsNotNone(search.asset(index.atoms[0]))
                self.assertIsNone(search.asset(second))
                self.assertEqual(model.ocr.call_count, 1)

    def test_xelatex_embedding_preserves_dimensions_pixels_and_searchable_caption(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); index, model = self.fixture(root)
            projects = []
            for name, runtime in (('plain', None), ('search', (model, 'test-model'))):
                project = root/name
                with patch('book_pipeline.publishing.image_search.local_ocr', return_value=runtime):
                    audit = write_project(index, dict(corrections=[], joins=[], exclusions=[], bold_spans=[]),
                        [], [], dict(title='Image ownership', author=''), load_config(), project, sample_mode=True)
                self.assertTrue(audit['coverage_complete'])
                result = compile_project(project, {t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(result['success'], result['log_tail'])
                self.assertEqual(result['warnings'], [])
                projects.append(project/'main.pdf')
            with pymupdf.open(projects[0]) as plain, pymupdf.open(projects[1]) as search:
                self.assertEqual(len(plain), len(search))
                self.assertFalse(plain[0].search_for('A vector.'))
                self.assertEqual(len(search[0].search_for('A vector.')), 1)
                for p,q in zip(plain, search):
                    self.assertEqual(p.get_pixmap(dpi=120).samples, q.get_pixmap(dpi=120).samples)
                rect = pymupdf.Rect(search[0].get_image_info()[0]['bbox'])
                self.assertTrue(rect.contains(search[0].search_for('A vector.')[0]))
                self.assertEqual(plain.resolve_names(), search.resolve_names())


if __name__ == '__main__':
    unittest.main()
