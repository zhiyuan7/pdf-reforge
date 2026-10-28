from copy import deepcopy
import hashlib
import io
import unittest
from unittest.mock import Mock

import numpy as np
import pymupdf
from PIL import Image

from book_pipeline.ingestion.prose_ink_exclusions import exclusions,render_crop


class ProseInkExclusionTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.full((150,150,3),255,np.uint8)
        self.rgb[10:46,52:67]=0
        self.rgb[43:60,90:100]=0  # mathematics overlaps prose in y only
        self.first=dict(bbox=[50,0,70,50])
        self.second=dict(bbox=[50,100,70,149])
        self.bounds=[20,42,110,100]

    def record(self):
        result=exclusions(self.rgb,self.bounds,self.first,self.second,20)
        if result:
            result.update(pixel_bbox=self.bounds,page_pixel_size=[150,150],
                          page_rgb_sha256=hashlib.sha256(self.rgb.tobytes()).hexdigest())
        return result

    def page(self):
        stream=io.BytesIO();Image.fromarray(self.rgb).save(stream,format='PNG')
        return Mock(get_pixmap=Mock(return_value=pymupdf.Pixmap(stream.getvalue())))

    def test_remove_only_witnessed_prose_pixels_and_keep_all_mathematics(self):
        original=self.rgb.copy()
        record=self.record()
        output=np.array(render_crop(self.page(),72,record))
        self.assertTrue((output[:5,31:48]==255).all())
        np.testing.assert_array_equal(output[1:18,70:80],self.rgb[43:60,90:100])
        np.testing.assert_array_equal(self.rgb,original)
        mask=np.zeros(output.shape[:2],bool)
        for y,x,z in record['runs']:mask[y,x:z]=True
        crop=self.rgb[42:100,20:110]
        np.testing.assert_array_equal(output[~mask],crop[~mask])

    def test_wholly_inside_formula_or_ambiguous_connected_ink_cannot_be_erased(self):
        self.assertIsNone(exclusions(self.rgb,[20,0,110,100],self.first,self.second,20))
        self.rgb[43:60,60:100]=0
        self.assertIsNone(self.record())

    def test_disconnected_narrow_prose_radical_needs_a_witnessed_glyph_neighbor(self):
        # Narrow radical and ordinary component are one physical glyph. Only
        # their ends intrude into the formula; unrelated mathematics survives.
        rgb=np.full((200,200,3),255,np.uint8)
        rgb[100:158,80:112]=0
        rgb[108:151,66:73]=0
        rgb[105:129,140:147]=0
        first=dict(bbox=[60,0,120,40]);second=dict(bbox=[60,95,120,190])
        bounds=[40,50,170,130]
        mask=exclusions(rgb,bounds,first,second,40)
        self.assertEqual(len(mask['components']),2)
        self.assertIn([66,108,73,151],mask['components'])
        self.assertNotIn([140,105,147,129],mask['components'])
        # The narrow component alone is never proof of prose ownership.
        rgb[100:158,80:112]=255
        self.assertIsNone(exclusions(rgb,bounds,first,second,40))

    def test_source_size_digest_and_mask_bounds_are_validated(self):
        record=self.record()
        for changes in (dict(page_rgb_sha256='changed'),dict(page_pixel_size=[149,150]),
                        dict(pixel_bbox=[-1,42,110,100]),dict(runs=[[0,0,1000]])):
            with self.assertRaises(ValueError):
                render_crop(self.page(),72,{**deepcopy(record),**changes})


if __name__ == '__main__':
    unittest.main()
