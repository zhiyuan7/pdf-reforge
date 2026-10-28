import unittest

import numpy as np

from book_pipeline.ingestion.formula_gap_ink import boundary_ink_clear, extend_scripts


class FormulaGapInkTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((170,160,3),255,np.uint8)
        self.rgb[75:90,50:60] = 0
        self.rgb[93:103,50:60] = 0
        self.first = dict(bbox=[90,0,120,70])
        self.second = dict(bbox=[90,110,120,160])
        self.bounds = [40,70,80,91]
        self.boxes = [self.bounds]

    def extend(self, **kw):
        args = dict(rgb=self.rgb,boxes=self.boxes,selected=[0],bounds=self.bounds,
                    first=self.first,second=self.second,pitch=30)
        return extend_scripts(**{**args,**kw})

    def test_detector_missed_denominator_retained_at_both_scales(self):
        bounds,tails = self.extend()
        self.assertEqual(bounds,[40,70,80,105])
        self.assertEqual(tails,[[50,93,60,103]])
        scale = lambda b:[v*2 for v in b]
        enlarged = np.repeat(np.repeat(self.rgb,2,axis=0),2,axis=1)
        second,_ = self.extend(rgb=enlarged,boxes=[scale(self.bounds)],bounds=scale(self.bounds),
            first=dict(bbox=scale(self.first['bbox'])),second=dict(bbox=scale(self.second['bbox'])),pitch=60)
        self.assertLessEqual(max(abs(a-b/2) for a,b in zip(bounds,second)),1)

    def test_prose_another_detector_owner_and_distant_ink_cannot_extend(self):
        for kw in (dict(second=dict(bbox=[45,92,65,140])),
                   dict(boxes=[self.bounds,[49,92,61,104]])):
            with self.subTest(kw=kw):
                self.assertEqual(self.extend(**kw),(self.bounds,[]))
        rgb = self.rgb.copy();rgb[93:103,50:60] = 255;rgb[118:128,50:60] = 0
        self.assertEqual(self.extend(rgb=rgb),(self.bounds,[]))
        rgb[118:128,50:60] = 255;rgb[93:94,50:52] = 0
        self.assertEqual(self.extend(rgb=rgb),(self.bounds,[]))

    def test_boundary_padding_requires_every_prose_component_removed(self):
        rgb = np.full((20,30,3),255,np.uint8);rgb[6:8,12:18] = 0
        bounds = [10,5,20,10];row = dict(bbox=[12,0,18,20])
        mask = dict(runs=[[1,2,8],[2,2,8]])
        self.assertTrue(boundary_ink_clear(rgb,bounds,row,mask))
        self.assertFalse(boundary_ink_clear(rgb,bounds,row,None))
        self.assertFalse(boundary_ink_clear(rgb,bounds,row,dict(runs=[[1,2,8]])))
