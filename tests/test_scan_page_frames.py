from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from book_pipeline.config import digest, read, save
from book_pipeline.ingestion.scan_page_frames import (detect_frame, frame_groups,
    horizontal_bounds, recover_scan_page_frames)
from book_pipeline.ingestion.source import SourceIndex, load_source
from book_pipeline.ingestion.formula_bands import band_image


class ScanPageFrameTests(unittest.TestCase):
    def leaf(self, paired=True):
        image=Image.new('L',(1000,1400),245);draw=ImageDraw.Draw(image)
        draw.line((115,50,130,1350),fill=45,width=5)
        if paired: draw.line((920,50,905,1350),fill=45,width=5)
        return image

    def index(self):
        atoms=[]
        for n in range(12):
            text=('External fragment' if n==0 else 'Ordinary source paragraph '*4)
            box=([0,.4,.07,.42] if n==0 else [.2,.1+n*.05,.8,.13+n*.05])
            atoms.append(dict(id=str(n),group=str(n),kind='text',role='text',block_type='text',
                              content=text,length=len(text),page=0,block_bbox=box,bbox=box,extraction_method='ocr'))
        return SourceIndex(dict(atoms=atoms))

    def test_paired_edges_isolate_only_external_ocr(self):
        frame=detect_frame(np.array(self.leaf()))
        self.assertIsNotNone(frame)
        left,right=horizontal_bounds(frame,.2,.8)
        self.assertTrue(.11<left<.15)
        self.assertTrue(.88<right<.93)
        result=frame_groups(self.index(),0,frame)
        self.assertEqual(result['excluded_groups'],['0'])
        inside=self.index();inside.atoms[0].update(block_bbox=[.16,.4,.19,.42])
        self.assertIsNone(frame_groups(inside,0,frame))  # A true side note is retained.
        asset=self.index();asset.atoms[0]['kind']='image'
        self.assertIsNone(frame_groups(asset,0,frame))
        footnote=self.index();footnote.atoms[0]['role']='page_footnote'
        self.assertIsNone(frame_groups(footnote,0,frame))

    def test_unframed_and_single_edge_pages_abstain(self):
        self.assertIsNone(detect_frame(np.full((1400,1000),255,dtype=np.uint8)))
        self.assertIsNone(detect_frame(np.array(self.leaf(paired=False))))

    def test_derivative_roundtrip_retains_original_and_physical_page_scope(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);origin=root/'ocr';origin.mkdir();pdf=root/'input.pdf'
            image=root/'leaf.png';self.leaf().save(image)
            with pymupdf.open() as document:
                page=document.new_page(width=500,height=700)
                page.insert_image(page.rect,filename=str(image))
                page.insert_text((10,20),'Old OCR overlay',render_mode=3)
                document.save(pdf)
            original_hash=digest(pdf)
            blocks=[dict(index=n,type='text',bbox=a['bbox'],content=a['content'])
                    for n,a in enumerate(self.index().atoms)]
            save(origin/'middle_json.json',dict(schema='docvortex.middle',schema_version='2.0',
                                               pages=[dict(page_idx=0,blocks=blocks)]))
            raw_hash=digest(origin/'middle_json.json')
            save(origin/'manifest.json',dict(input_sha256=original_hash,mineru_version='test',source_page_numbers=[1]))
            target=recover_scan_page_frames(pdf,origin,root/'derivative')
            result=load_source(pdf,target)
            self.assertEqual(len(result['atoms']),11)
            self.assertIn('scan_page_frames',result)
            self.assertEqual(digest(pdf),original_hash)
            self.assertEqual(digest(origin/'middle_json.json'),raw_hash)
            self.assertEqual(recover_scan_page_frames(pdf,origin,root/'derivative'),target)
            self.assertEqual(recover_scan_page_frames(pdf,target,root/'again'),target)

    def test_formula_band_honors_horizontal_crop(self):
        with tempfile.TemporaryDirectory() as name, pymupdf.open() as document:
            page=document.new_page(width=500,height=700)
            image=Path(name)/'band.png'
            band_image(page,dict(bbox=[0,.2,1,.3],crop_bbox=[.15,.2,.9,.3]),image,False)
            with Image.open(image) as cropped:
                self.assertAlmostEqual(cropped.width,500*.75*220/72,delta=2)


if __name__=='__main__': unittest.main()
