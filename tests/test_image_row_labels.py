from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import write_project


class ImageRowLabelTests(unittest.TestCase):
    def test_translated_trailing_caption_is_typeset_outside_original_language_crop(self):
        from book_pipeline.config import executable
        from book_pipeline.publishing.compiler import compile_project
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); index=self.source(root)
            translated='图2.3 中文图注可检索。'
            source=root/'with-english-caption.pdf'
            with pymupdf.open(index.source['source_pdf']) as pdf:
                pdf[0].insert_text((110,295),'Figure 2.3 Original English caption stays out of the crop.')
                pdf.save(source)
            atoms=deepcopy(index.atoms)
            parent_box=list(atoms[-1]['bbox'])
            for identifier, text in [('caption-prefix','图2.3 '),('caption','中文图注可检索。')]:
                atoms.append(dict(id=identifier,group=atoms[-1]['group'],page=0,kind='text',
                    role='image_caption',block_type='image',content=text,length=len(text),
                    extraction_method='ocr',bbox=[.22,.358,.79,.41],block_bbox=parent_box))
            index=SourceIndex({**index.source,'atoms':atoms,'source_pdf':str(source),
                'translation':dict(provider='volcengine',target_language='zh')})
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Translated caption',author=''),{**load_config(),'float_assets':True},root/'latex',sample_mode=True)
            row=next(r for r in audit['source_order_layout'] if r['method']=='horizontal_source_image_row')
            self.assertEqual(row['native_caption_members'],['caption-prefix','caption'])
            self.assertNotIn('caption',row['members']);self.assertNotIn('caption-prefix',row['members'])
            self.assertLessEqual(row['bbox'][3],.358)
            self.assertGreaterEqual(row['bbox'][3],250/700)
            tex=''.join(p.read_text() for p in (root/'latex/chapters').glob('*.tex'))
            self.assertIn('图2.3 ',tex);self.assertIn('中文图注可检索。',tex)
            self.assertLess(tex.index('中文图注可检索。'),tex.index(r'\end{figure}'))
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                names=pdf.resolve_names()
                self.assertEqual(len({names['source-'+a['id']]['page'] for a in index.atoms}),1)
                self.assertIn('中文图注',pdf[0].get_text())

    def stacked_source(self, root):
        index=self.source(root)
        atoms=deepcopy(index.atoms)
        for atom in index.atoms:
            other=deepcopy(atom);other.update(id=atom['id']+'-lower',group=atom['group']+'-lower')
            other['bbox']=[v+.19 if i in (1,3) else v for i,v in enumerate(other['bbox'])]
            atoms.append(other)
        text='Figure 3.4 Four panels share this complete caption.'
        atoms.append(dict(id='caption',group='caption',page=0,kind='text',role='image_caption',
            block_type='image_caption',content=text,length=len(text),extraction_method='native',
            bbox=[.22,.58,.78,.62]))
        return SourceIndex({**index.source,'atoms':atoms})

    def test_common_caption_merges_two_rows_into_one_compiled_figure(self):
        from book_pipeline.config import executable
        from book_pipeline.publishing.compiler import compile_project
        from book_pipeline.publishing.image_rows import detect_image_rows
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.stacked_source(root)
            gap_source=root/'with-gap.pdf'
            with pymupdf.open(index.source['source_pdf']) as pdf:
                # Exactly between the rows: local rectangle ownership cannot
                # assign this ink, but a complete shared figure must retain it.
                pdf[0].draw_rect((103,265.5,107,267.5),fill=(0,0,0))
                pdf.save(gap_source)
            index=SourceIndex({**index.source,'source_pdf':str(gap_source)})
            rows=detect_image_rows(index)
            self.assertEqual(len(rows),1)
            self.assertTrue(rows[0]['shared_caption_stack'])
            self.assertEqual(len(rows[0]['images']),4)
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Shared figure',author=''),{**load_config(),'float_assets':True},root/'latex',sample_mode=True)
            self.assertEqual(sum(r.get('shared_caption_stack',False) for r in audit['source_order_layout']),1)
            shared=next(r for r in audit['source_order_layout'] if r.get('shared_caption_stack'))
            self.assertEqual(shared['crop_method'],'source_rectangle')
            from book_pipeline.publishing.project_assets import verify_asset_manifest
            self.assertEqual(verify_asset_manifest(root/'latex', audit['asset_files']), [])
            included = audit['assets'][shared['asset']]
            tex = ''.join(p.read_text() for p in (root/'latex/chapters').glob('*.tex'))
            self.assertIn('{'+included+'}', tex)
            self.assertEqual((root/'latex'/included).read_bytes(), Path(shared['asset']).read_bytes())
            pix=pymupdf.Pixmap(shared['asset']);b=shared['bbox']
            x=round((105/500-b[0])/(b[2]-b[0])*pix.width)
            y=round((266.5/700-b[1])/(b[3]-b[1])*pix.height)
            self.assertLess(sum(pix.pixel(x,y)[:3]),100)
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                names=pdf.resolve_names()
                self.assertEqual(len({names['source-'+a['id']]['page'] for a in index.atoms}),1)
                self.assertEqual(sum(p.get_text().count('Four panels share') for p in pdf),1)

    def test_stack_rejects_separate_captions_text_gaps_pages_and_registry_boundaries(self):
        from book_pipeline.publishing.image_rows import detect_image_rows,recover_horizontal_images
        for defect in ('caption','prose','gap','page','ocr','semantic','lateral_caption','above_caption'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as name:
                root=Path(name);index=self.stacked_source(root);atoms=index.atoms
                if defect in {'caption','prose'}:
                    extra=deepcopy(atoms[-1]);extra.update(id='intervening',group='intervening')
                    if defect=='prose':extra.update(role='text',block_type='text')
                    atoms.insert(2,extra)
                if defect in {'lateral_caption','above_caption'}:
                    extra=deepcopy(atoms[-1]);extra.update(id='own-caption',group='own-caption',
                        bbox=[0,.20,.17,.32] if defect=='lateral_caption' else [.2,.15,.8,.20])
                    atoms.insert(0,extra)
                if defect=='gap':
                    for atom in atoms[2:4]:atom['bbox'][1]+=.15;atom['bbox'][3]+=.15
                if defect=='page':atoms[2]['page']=1
                if defect=='ocr':atoms[2]['extraction_method']='ocr'
                index=SourceIndex({**index.source,'atoms':atoms})
                if defect=='semantic':
                    rows=recover_horizontal_images(index,[dict(kind='example',lo=0,hi=index.prefix[-1])],root/'latex')
                    self.assertFalse(any(r.get('shared_caption_stack') for r in rows.values()))
                else:
                    self.assertFalse(any(r.get('shared_caption_stack') for r in detect_image_rows(index)))

    def source(self, root):
        source = root / 'source.pdf'
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=500, height=700)
            for x in (100, 280):
                page.draw_rect(pymupdf.Rect(x, 150, x+120, 250))
                # Outlined panel letters are absent from the text layer and
                # sit above the extraction rectangles, like vector plot labels.
                page.draw_rect(pymupdf.Rect(x+3, 136, x+7, 144), fill=(0, 0, 0))
            pdf.save(source)
            page.get_pixmap().save(root / 'image.png')
        atoms = [dict(id=f'image{i}', group=f'g{i}', page=0, kind='image',
            role='image_body', block_type='image', content='', length=1,
            asset=str(root/'image.png'), extraction_method='native',
            bbox=[x/500, 150/700, (x+120)/500, 250/700])
            for i, x in enumerate((100, 280))]
        return SourceIndex(dict(source_pdf=str(source), atoms=atoms, source_page_count=1))

    def test_shared_crop_preserves_recovered_panel_ink_and_all_anchors(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            index = self.source(root)
            before = deepcopy(index.source)
            audit = write_project(index, dict(corrections=[], joins=[], exclusions=[], bold_spans=[]),
                [], [], dict(title='Panels', author=''), load_config(), root/'latex', sample_mode=True)
            rows = [r for r in audit['source_order_layout'] if r['method'] == 'horizontal_source_image_row']
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertLess(row['bbox'][1], 136/700)
            with pymupdf.open(row['asset']) as raster:
                pix = raster[0].get_pixmap()
                # Actual outlined ink is present at both panel positions,
                # above the original image boxes, rather than just extra white.
                for x in (105, 285):
                    px = round((x/500-row['bbox'][0])/(row['bbox'][2]-row['bbox'][0])*pix.width)
                    py = round((140/700-row['bbox'][1])/(row['bbox'][3]-row['bbox'][1])*pix.height)
                    self.assertLess(sum(pix.pixel(px, py)[:3]), 100)
            self.assertTrue(audit['coverage_complete'])
            self.assertEqual(set(audit['source_anchors']), {'image0', 'image1'})
            tex = ''.join(p.read_text() for p in (root/'latex/chapters').glob('*.tex'))
            for identifier in ('image0', 'image1'):
                self.assertEqual(tex.count('source-'+identifier+'}'), 1)
            self.assertEqual(index.source, before)

    def test_edited_member_still_prevents_rasterizing_the_row(self):
        from book_pipeline.publishing.image_rows import recover_horizontal_images
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            index = self.source(root)
            rows = recover_horizontal_images(index, [], root/'latex', {'image0'},
                image_boxes={'image0': [.2, .19, .44, .36]})
            self.assertEqual(rows, {})

    def test_neighboring_rows_do_not_duplicate_the_lower_panels_letter(self):
        from book_pipeline.publishing.figure_recovery import recover_native_vector_labels
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = root/'rows.pdf'
            bounds = [(100,150,220,250), (280,150,400,260),
                      (100,270,220,370), (280,290,400,390)]
            with pymupdf.open() as pdf:
                page = pdf.new_page(width=500, height=700)
                for box in bounds:
                    page.draw_rect(box)
                for x in (100,280):
                    page.draw_rect((x+3,136,x+7,144), fill=(0,0,0))
                # Lower left's panel letter and upper right's axis label share
                # vertical space. A rectangular crop of the row includes both.
                page.draw_rect((103,258,107,266), fill=(0,0,0))
                page.draw_rect((330,262,334,268), fill=(0,0,0))
                pdf.save(source); page.get_pixmap().save(root/'image.png')
            atoms = [dict(id=f'image{i}',group=f'g{i}',page=0,kind='image',role='image_body',
                block_type='image',content='',length=1,extraction_method='native',
                bbox=[v/d for v,d in zip(box,[500,700]*2)],asset=str(root/'image.png'))
                for i,box in enumerate(bounds)]
            index = SourceIndex(dict(source_pdf=str(source),atoms=atoms,source_page_count=1))
            recovered = recover_native_vector_labels(index, root/'recovered')
            self.assertLess(recovered['image0']['bbox'][3],258/700)
            self.assertLess(recovered['image2']['bbox'][1],258/700)
            audit = write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Rows',author=''),load_config(),root/'latex',sample_mode=True)
            row = next(r for r in audit['source_order_layout']
                       if r['method']=='horizontal_source_image_row' and 'image0' in r['images'])
            self.assertEqual(row['crop_method'],'verified_image_bounds_composite')
            with pymupdf.open(row['asset']) as raster:
                pix = raster[0].get_pixmap()
                def pixel(x,y):
                    px = round((x/500-row['bbox'][0])/(row['bbox'][2]-row['bbox'][0])*pix.width)
                    py = round((y/700-row['bbox'][1])/(row['bbox'][3]-row['bbox'][1])*pix.height)
                    return sum(pix.pixel(px,py)[:3])
                self.assertGreater(pixel(105,262),740)
                self.assertLess(pixel(332,265),100)

    def test_small_panel_letter_and_ambiguous_neighbor_have_distinct_owners(self):
        from book_pipeline.publishing.figure_recovery import recover_native_vector_labels
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); source = root/'small.pdf'
            with pymupdf.open() as pdf:
                page = pdf.new_page(width=500, height=700)
                page.draw_rect((100,150,220,250))
                page.draw_rect((103,145,107,149), fill=(0,0,0))
                pdf.save(source)
            atom = dict(id='plot',group='plot',page=0,kind='image',role='image_body',
                block_type='image',content='',length=1,extraction_method='native',
                bbox=[.2,150/700,.44,250/700])
            index = SourceIndex(dict(source_pdf=str(source),atoms=[atom]))
            recovered = recover_native_vector_labels(index, root/'one')
            self.assertLess(recovered['plot']['bbox'][1],145/700)
            # An equally near image makes this external ink ambiguous.
            other = dict(atom,id='neighbor',group='neighbor',bbox=[.2,40/700,.44,144/700])
            index = SourceIndex(dict(source_pdf=str(source),atoms=[atom,other]))
            self.assertEqual(recover_native_vector_labels(index, root/'two'), {})

    def test_partially_clipped_filled_label_survives_small_overshoot_threshold(self):
        from book_pipeline.publishing.figure_recovery import recover_native_vector_labels
        for defect in ('none','thin_stroke','owned_text','ambiguous'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as name:
                root=Path(name);source=root/'tick.pdf'
                with pymupdf.open() as pdf:
                    page=pdf.new_page(width=600,height=800)
                    page.draw_rect((130,140,510,290))
                    # Two points above the crop is less than .003 page height;
                    # half of a real outlined tick glyph is nevertheless lost.
                    if defect=='thin_stroke':page.draw_line((250,138),(253,142))
                    else:page.draw_rect((250,138,253,142),fill=(0,0,0))
                    pdf.save(source)
                atom=dict(id='plot',group='plot',page=0,kind='image',role='image_body',
                    block_type='image',content='',length=1,extraction_method='native',
                    bbox=[130/600,140/800,510/600,290/800])
                atoms=[atom]
                if defect=='owned_text':
                    atoms.append(dict(atom,id='caption',group='caption',kind='text',role='text',
                        content='Owned prose',length=11,bbox=[249/600,137/800,254/600,143/800]))
                if defect=='ambiguous':
                    atoms.append(dict(atom,id='other',group='other',bbox=[130/600,100/800,510/600,140/800]))
                recovered=recover_native_vector_labels(SourceIndex(dict(source_pdf=str(source),atoms=atoms)),root/'out')
                if defect=='none':self.assertLess(recovered['plot']['bbox'][1],138/800)
                else:self.assertEqual(recovered,{})
