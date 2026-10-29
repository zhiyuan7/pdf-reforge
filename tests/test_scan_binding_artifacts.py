from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pymupdf
from PIL import Image,ImageDraw

from book_pipeline.config import digest,read,save
from book_pipeline.ingestion.source import SourceIndex,load_source
from book_pipeline.ingestion.scan_binding_artifacts import binding_groups
from book_pipeline.ingestion.scan_page_frames import recover_scan_page_frames,recognition_bounds


class ScanBindingArtifactTests(unittest.TestCase):
    def source(self,defect='none'):
        image=Image.new('L',(1000,1400),255);draw=ImageDraw.Draw(image)
        draw.line((966,10,984,500 if defect=='short' else 1390),fill=0,width=6)
        if defect=='real_ink':draw.rectangle((964,700,970,718),fill=0)
        atoms=[]
        for n in range(10):
            text='An independently located source paragraph with ordinary horizontal text. '*3
            box=[.2,.1+n*.06,.8,.14+n*.06]
            atoms.append(dict(id=str(n),group=str(n),kind='text',role='text',block_type='text',
                content=text,length=len(text),page=0,bbox=box,block_bbox=box,extraction_method='ocr'))
        text='Unsupported margin transcription'
        atoms.append(dict(id='ghost',group='ghost',kind='text',role='aside_text',block_type='aside_text',
            content=text,length=len(text),page=0,bbox=[.963,.001,.988,.999],block_bbox=[.963,.001,.988,.999],extraction_method='ocr'))
        if defect=='native':atoms[-1]['extraction_method']='native'
        if defect=='image':atoms[-1]['kind']='image'
        if defect=='footnote':atoms[-1]['role']='page_footnote'
        if defect=='vertical':atoms[-1]['writing_direction']='vertical_rl'
        if defect=='body_crossing':atoms[0]['block_bbox'][2]=.96
        return image,SourceIndex(dict(atoms=atoms))

    def test_only_bare_outer_stroke_can_own_a_tall_ocr_ghost(self):
        for defect in ('none','real_ink','short','native','image','footnote','vertical','body_crossing'):
            with self.subTest(defect=defect):
                image,index=self.source(defect);before=deepcopy(index.source)
                result=binding_groups(index,0,np.array(image))
                self.assertEqual(bool(result),defect=='none')
                self.assertEqual(index.source,before)
                if result:self.assertEqual(result['excluded_groups'],['ghost'])
                if result:
                    left,right=recognition_bounds(result,.2,.8)
                    self.assertEqual(left,0);self.assertLess(right,.963)
                    self.assertGreater(right,.9)

    def test_structural_derivative_preserves_original_scan_ocr_and_old_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);image,index=self.source();scan=root/'scan.png';image.save(scan)
            pdf=root/'source.pdf';origin=root/'ocr';origin.mkdir()
            with pymupdf.open() as d:
                p=d.new_page(width=500,height=700);p.insert_image(p.rect,filename=str(scan));d.save(pdf)
            blocks=[dict(index=n,type=a['block_type'],bbox=a['bbox'],content=a['content']) for n,a in enumerate(index.atoms)]
            save(origin/'middle_json.json',dict(schema='docvortex.middle',schema_version='2.0',pages=[dict(page_idx=0,blocks=blocks)]))
            save(origin/'manifest.json',dict(input_sha256=digest(pdf),mineru_version='test',source_page_numbers=[1]))
            before=digest(pdf),digest(origin/'middle_json.json')
            target=recover_scan_page_frames(pdf,origin,root/'derived')
            self.assertNotEqual(target,origin)
            self.assertEqual(len(load_source(pdf,target)['atoms']),10)
            self.assertEqual(before,(digest(pdf),digest(origin/'middle_json.json')))
            manifest=read(target/'manifest.json');self.assertIn('binding_artifacts',manifest['scan_page_frames']['pages']['0'])
            manifest['scan_page_frames']['method']='obsolete';save(target/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'policy changed'):
                recover_scan_page_frames(pdf,target,root/'next')


if __name__=='__main__':unittest.main()
