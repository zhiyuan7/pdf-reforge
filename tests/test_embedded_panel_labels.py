from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.image_rows import detect_image_rows, recover_horizontal_images
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.project_assets import verify_asset_manifest


class EmbeddedPanelLabelTests(unittest.TestCase):
    def source(self, root):
        source = root/'source.pdf'
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=500, height=700)
            page.draw_rect((100, 150, 400, 350), fill=(.2, .5, .8))
            page.insert_text((100, 142), 'a) Original')
            page.insert_text((250, 142), 'b) Flip')
            # OCR omitted this third label entirely. The complete source row
            # must preserve it too, rather than repainting only known labels.
            page.draw_rect((380, 135, 385, 141), fill=(0, 0, 0))
            page.insert_text((110, 390), 'Figure 2.1 Original English caption.')
            pdf.save(source)
            page.get_pixmap(clip=pymupdf.Rect(100,150,400,350)).save(root/'image.png')
        def text(identifier, content, box):
            return dict(id=identifier, group='figure', page=0, kind='text',
                        role='image_caption', block_type='image', content=content,
                        length=len(content), bbox=box, extraction_method='ocr')
        atoms = [text('label-b', '翻转', [.5,.184,.6,.204]),
                 text('label-a', '一个', [.2,.184,.36,.204]),
                 dict(id='image', group='figure', page=0, kind='image',
                      role='image_body', block_type='image', content='', length=1,
                      bbox=[.2,150/700,.8,.5], extraction_method='ocr', asset=str(root/'image.png')),
                 text('caption', '图2.1 完整的中文图注仍可检索。', [.22,.54,.78,.58])]
        return SourceIndex(dict(source_pdf=str(source), source_page_count=1, atoms=atoms,
                                translation=dict(provider='volcengine')))

    def test_label_positions_pixels_anchors_and_searchable_caption_in_pdf(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); index=self.source(root); before=deepcopy(index.source)
            audit=write_project(index, dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Panel labels',author=''),{**load_config(),'float_assets':True},
                root/'latex',sample_mode=True)
            row=next(r for r in audit['source_order_layout'] if r.get('embedded_panel_labels'))
            self.assertEqual(row['members'],['label-b','label-a','image'])
            self.assertEqual(row['native_caption_members'],['caption'])
            self.assertLess(row['bbox'][1],135/700)
            self.assertLess(row['bbox'][3],.54)
            pix=pymupdf.Pixmap(row['asset']); b=row['bbox']
            x=round((382/500-b[0])/(b[2]-b[0])*pix.width)
            y=round((138/700-b[1])/(b[3]-b[1])*pix.height)
            self.assertLess(sum(pix.pixel(x,y)[:3]),100)
            self.assertEqual(verify_asset_manifest(root/'latex',audit['asset_files']),[])
            tex=''.join(p.read_text() for p in (root/'latex/chapters').glob('*.tex'))
            self.assertNotIn('一个',tex); self.assertNotIn('翻转',tex)
            self.assertEqual(tex.count('完整的中文图注仍可检索。'),1)
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                names=pdf.resolve_names()
                self.assertEqual(len({names['source-'+a['id']]['page'] for a in index.atoms}),1)
                self.assertEqual(sum(p.get_text().count('完整的中文图注仍可检索。') for p in pdf),1)
                self.assertEqual(sum(len(p.get_image_info()) for p in pdf),1)
            self.assertTrue(audit['coverage_complete'])
            self.assertEqual(index.source,before)

    def test_ambiguous_labels_and_structural_edits_are_not_rasterized(self):
        for defect in ('missing_box','long','far','other_group','noncaption','tail_above',
                       'second_image','outside','edit','registry','label_registry'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                root=Path(name); index=self.source(root); atoms=deepcopy(index.atoms)
                if defect=='missing_box':atoms[0]['bbox']=None
                if defect=='long':atoms[0]['content']='long explanatory prose '*6
                if defect=='far':atoms[0]['bbox']=[.5,.1,.6,.12]
                if defect=='outside':atoms[0]['bbox']=[.1,.184,.15,.204]
                if defect=='noncaption':atoms[0]['role']='text'
                if defect=='tail_above':atoms[-1]['bbox']=[.22,.48,.78,.52]
                if defect=='second_image':
                    extra=deepcopy(atoms[2]);extra.update(id='other-image',bbox=[.2,.6,.8,.8]);atoms.append(extra)
                if defect=='other_group':
                    extra=deepcopy(atoms[0]);extra.update(id='other',group='other');atoms.append(extra)
                index=SourceIndex({**index.source,'atoms':atoms})
                registry=[dict(kind='theorem',lo=0,hi=index.prefix[-1])] if defect=='registry' else []
                if defect=='label_registry':registry=[dict(kind='theorem',lo=0,hi=index.prefix[1])]
                rows=recover_horizontal_images(index,registry,root/'latex',{'label-a'} if defect=='edit' else ())
                self.assertFalse(any(r.get('embedded_panel_labels') for r in rows.values()))

    def test_short_caption_is_searchable_even_without_translation(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root)
            source=deepcopy(index.source);source.pop('translation')
            source['atoms'][-1].update(content='Figure 2.1',length=10)
            rows=recover_horizontal_images(SourceIndex(source),[],root/'latex')
            row=next(r for r in rows.values() if r.get('embedded_panel_labels'))
            self.assertEqual(row['native_caption_members'],['caption'])
            self.assertNotIn('caption',row['members'])
