from copy import deepcopy
import unittest

import numpy as np

from book_pipeline.ingestion.vertical_prose_duplicates import redundant_owner


class CoveredDuplicateOwnerTests(unittest.TestCase):
    def setUp(self):
        self.text='因而可求此積分式'
        self.rgb=np.full((500,300,3),255,np.uint8)
        self.rgb[370:410,150:190]=0
        self.first=dict(text=self.text[:2],bbox=[90,40,130,100])
        self.second=dict(text=self.text[2:],bbox=[90,150,130,330])
        self.blocks=[dict(type='text',bbox=[.6,.4,.74,.6],parts=[dict(kind='text',content=self.text)]),
                     dict(type='image',bbox=[.5,.7,.65,.84],parts=[]),
                     dict(type='text',bbox=[.2,.01,.4,.07],parts=[dict(kind='text',content='得')]),
                     dict(type='text',bbox=[.25,.08,.45,.66],parts=[dict(kind='text',content=self.text)]),
                     dict(type='image',bbox=[.15,.67,.67,.85],parts=[])]
        self.preserved=[[48,338,202,425],[90,430,130,490]]

    def witness(self,**kwargs):
        args=dict(rgb=self.rgb,blocks=self.blocks,conflict=1,paragraph=3,full_text=self.text,
                  first=self.first,second=self.second,pitch=30,cell_box=[0,0,300,500],
                  page_size=[300,500],preserved_bounds=self.preserved)
        return redundant_owner(**{**args,**kwargs})

    def test_repeated_complete_prose_and_entire_image_ink_must_survive(self):
        original=deepcopy(self.blocks)
        result=self.witness()
        self.assertEqual(result['removed_blocks'],[0,1])
        self.assertEqual(result['covered_ink_pixels'],1600)
        self.assertTrue(result['requires_visual_review'])
        self.assertEqual(self.blocks,original)

    def test_unique_text_large_prose_box_and_actual_column_cannot_be_removed(self):
        for change in (dict(parts=[dict(kind='text',content='另一段正文')]),
                       dict(parts=[dict(kind='math',content=self.text)]),
                       dict(bbox=[.6,.1,.74,.6]),dict(bbox=[.3,.4,.43,.6])):
            blocks=deepcopy(self.blocks);blocks[0].update(change)
            self.assertIsNone(self.witness(blocks=blocks))
        blocks=deepcopy(self.blocks);blocks.append(deepcopy(blocks[0]))
        self.assertIsNone(self.witness(blocks=blocks))
        self.assertIsNone(self.witness(conflict=0))
        self.assertIsNone(self.witness(paragraph=1))

    def test_uncovered_formula_ink_or_different_cell_abstains(self):
        # A duplicate sentence never authorizes dropping unique mathematics.
        self.rgb[390:415,180:210]=0
        self.assertIsNone(self.witness(preserved_bounds=[[48,338,170,425]]))
        self.assertIsNone(self.witness(cell_box=[100,0,400,500]))
        self.assertIsNone(self.witness(rgb=np.full_like(self.rgb,255)))

    def test_cell_rule_and_tiny_isolated_paper_specks_are_recorded(self):
        blocks=deepcopy(self.blocks);blocks[1]['bbox']=[.5,.7,.65,1.]
        self.rgb[495:500,:]=0
        self.rgb[460:462,175:177]=0
        result=self.witness(blocks=blocks)
        self.assertEqual(result['covered_ink_pixels'],1600)
        self.assertGreater(result['ignored_rule_or_speck_pixels'],0)


if __name__ == '__main__':
    unittest.main()
