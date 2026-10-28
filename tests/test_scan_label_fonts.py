"""Local label fonts must retain independent ink and literal ownership."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pymupdf
from book_pipeline.publishing import scan_font_calibration as scan
from book_pipeline.publishing.raster_body_styles import is_bold, is_italic, owned_ranges


class ScanLabelFontTests(unittest.TestCase):
    @unittest.skipUnless((scan.FONTDIR/'cmssbx10.pfb').is_file(), 'TeX fonts required')
    def test_scan_gain_regular_slanted_and_serif_labels_cannot_acquire_sans_weight(self):
        names=(*scan.LABEL_FONTS,*(n for gs in scan.FAMILIES.values() for g in gs for n in g))
        for name in dict.fromkeys(names):
            for text in ('Example.', 'Property.', 'Definition.', 'Proposition.'):
                with pymupdf.open() as pdf:
                    page = pdf.new_page(width=150, height=45)
                    kwargs = (dict(fontname='source', fontfile=str(scan.FONTDIR/(name+'.pfb')))
                              if name.startswith('cm') else dict(fontname=name))
                    page.insert_text((10,30), text, fontsize=13, **kwargs)
                    pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
                    ink = 255-np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width)
                ink=cv2.resize(ink,None,fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
                for gain in (0,.5,1):
                    gray=cv2.addWeighted(ink,1-gain,cv2.dilate(ink,np.ones((3,3),np.uint8)),gain,0)
                    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
                    scores=scan.label_bold_scores(binary,gray,text)
                    with self.subTest(font=name,text=text,gain=gain):
                        self.assertFalse(is_italic(scores))
                        if is_bold(scores): self.assertEqual(name,'cmssbx10')
                        if name=='cmssbx10' and gain==0:self.assertTrue(is_bold(scores))

    def test_local_family_and_weight_require_separate_dual_view_margins(self):
        # A losing serif face can be closer than the weight margin, while
        # the independently established sans family has unambiguous weight.
        names=(*scan.LABEL_FONTS,*(n for gs in scan.FAMILIES.values() for g in gs for n in g))
        good={n:.3 for n in names}
        good.update(cmss10=.60,cmssi10=.50,cmssbx10=.91,cmr10=.78)
        for defect in ('none','family','weight','absolute','inconsistent_views'):
            second=dict(good)
            if defect=='family':second['cmr10']=.84
            if defect=='weight':second['cmss10']=.75
            if defect=='absolute':second['cmssbx10']=.84
            if defect=='inconsistent_views':
                # Both views pass separately but their common evidence does not.
                second.update(cmss10=.77,cmssbx10=.96)
            for views in ((good,second),(second,good)):
                with patch.object(scan,'metric_face_scores',side_effect=views):
                    result=scan.label_bold_scores(None,None,'Property.')
                self.assertEqual(is_bold(result),defect=='none',defect)
                self.assertFalse(is_italic(result))

    def test_label_ink_cannot_supply_reading_context_or_source_ownership(self):
        word=dict(text='Property.',score=.999,bbox=[10,10,100,40],scores={})
        row=dict(text='2.7 Property.',score=.999,bbox=[0,10,100,40],words=[word])
        rgb=np.full((60,120,3),255,np.uint8)
        bold=dict(tiro=.4,tibo=.94,tiit=.4,tibi=.94)
        with patch.object(scan,'select_family',return_value=('computer_modern',{})), \
             patch.object(scan,'shape_scores',return_value={}), \
             patch.object(scan,'metric_scores',return_value=[]), \
             patch.object(scan,'label_bold_scores',return_value=bold) as local:
            calibrated,_=scan.calibrate_observations(rgb,[row])
            self.assertEqual(calibrated[0]['words'][0]['scores'],bold)
            item=dict(atom='a',text='2.7 Property. ',bbox=[0,0,1,1])
            result=owned_ranges(calibrated,[item],[120,60]*2)
            self.assertEqual(result['a'][0]['quote'],'Property.')
            self.assertTrue(result['a'][0]['bold'])
            self.assertEqual(owned_ranges(calibrated,[item,dict(item,atom='duplicate')],[120,60]*2),{})
            self.assertEqual(owned_ranges(calibrated,[dict(item,text='2.8 Property.')],[120,60]*2),{})
            for defect in ('uncertain_word','uncertain_line','missing_reading','unlabeled','repeated_label','outside'):
                altered=deepcopy(row)
                if defect=='uncertain_word':altered['words'][0]['score']=.98
                if defect=='uncertain_line':altered['score']=.98
                if defect=='missing_reading':altered['words'][0]['text']='Unknown.'
                if defect=='unlabeled':altered['text']='Ordinary Property. prose'
                if defect=='repeated_label':altered['text']+=' Property.'
                if defect=='outside':altered['words'][0]['bbox'][2]=121
                local.reset_mock()
                self.assertEqual(scan.calibrate_observations(rgb,[altered])[0],[],defect)
                local.assert_not_called()


if __name__=='__main__': unittest.main()
