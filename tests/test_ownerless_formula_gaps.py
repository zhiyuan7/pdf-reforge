"""Missing images need a unique suffix, full-cell ownership and two scales."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np

from book_pipeline.ingestion.ownerless_formula_gaps import recover_cell,suffix_witness,gap_bounds


class OwnerlessFormulaGapTests(unittest.TestCase):
    def setUp(self):
        self.prefix='欲留其式中之未知數而令'
        self.suffix='將此式之右邊通分齊同'
        self.text=self.prefix+self.suffix
        self.pitch=30
        self.first=[180,20,220,20+len(self.prefix)*30]
        self.second=[180,self.first[3]+65,220,self.first[3]+65+len(self.suffix)*30]
        self.rows=[dict(text=self.suffix,score=.999,bbox=self.second,reader_signature='cpu')]
        self.math=[120,self.first[3]-5,280,self.second[1]+5]
        self.boxes=[self.first,self.math,self.second]
        self.answer=dict(blocks=[dict(type='text',bbox=[.65,.01,.85,.99],parts=[dict(kind='text',content=self.text)])])
        self.cell=dict(cell=1,pixel_bbox=[0,0,400,900],image_blocks=[],regions=self.boxes)
        self.rgb=np.full((900,400,3),255,np.uint8)
        self.secondary=[dict(text='無可靠轉錄',score=.1,bbox=[v*2 for v in self.first]),
                        dict(text='',score=1,bbox=[v*2 for v in self.math]),
                        {**self.rows[0],'bbox':[v*2 for v in self.second]}]

    def recover(self, answer=None, cell=None, second=None, bounds=None):
        answer=self.answer if answer is None else answer
        with patch('book_pipeline.ingestion.ownerless_formula_gaps.observe_layout',return_value=self.secondary if second is None else second), \
             patch('book_pipeline.ingestion.formula_crops.ink_bounds',side_effect=bounds or [self.math,[v*2 for v in self.math]]), \
             patch('book_pipeline.ingestion.prose_ink_exclusions.exclusions',return_value=None):
            return recover_cell(self.rgb,self.cell if cell is None else cell,answer,answer,[400,900],self.rows)

    def test_displaced_paragraph_and_two_row_gap_preserve_literal_transcription(self):
        original=deepcopy(self.answer)
        result,evidence=self.recover()
        self.assertEqual(self.answer,original)
        self.assertEqual([b['type'] for b in result[0]],['text','image','text'])
        self.assertEqual(''.join(p['content'] for b in result[0] for p in b['parts']),self.text)
        self.assertFalse(evidence[0]['prefix_transcription_verified'])
        self.assertTrue(evidence[0]['requires_visual_review'])
        self.assertLess(result[0][0]['bbox'][0],self.answer['blocks'][0]['bbox'][0])

    def test_uncertain_wrong_duplicate_or_short_suffix_abstains(self):
        for change in (dict(score=.98),dict(text='此式通分'),dict(reader_signature=''),dict(text='將彼式之右邊通分齊同')):
            self.assertIsNone(suffix_witness(self.text,[{**self.rows[0],**change}],self.boxes))
        self.assertIsNone(suffix_witness(self.text,self.rows*2,self.boxes))
        self.assertIsNone(suffix_witness(self.text,self.rows,self.boxes+[self.first]))

    def test_prefix_geometry_must_match_column_count_and_gap(self):
        for box in ([300,*self.first[1:]], [180,20,220,100], [180,20,220,self.first[3]-150],
                    [20,20,60,self.first[3]]):
            self.assertIsNone(suffix_witness(self.text,self.rows,[box,self.math,self.second]))
        second=deepcopy(self.rows);second[0]['bbox']=[180,700,220,1000]
        self.assertIsNone(suffix_witness(self.text,second,self.boxes))

    def test_other_owners_mixed_math_existing_images_and_duplicate_suffix_abstain(self):
        self.assertEqual(self.recover(cell={**self.cell,'image_blocks':[1]}),({},[]))
        for mutate in (lambda a:a['blocks'].append(deepcopy(a['blocks'][0])),
                       lambda a:a['blocks'][0]['parts'][0].update(kind='math'),
                       lambda a:a['blocks'].append(dict(type='text',bbox=[.9,.1,1,.9],parts=[dict(kind='text',content=self.suffix)])),
                       lambda a:a['blocks'].append(dict(type='image',bbox=[.3,.39,.7,.49],parts=[]))):
            answer=deepcopy(self.answer);mutate(answer)
            self.assertEqual(self.recover(answer=answer),({},[]))
        self.assertEqual(self.recover(cell={**self.cell,'pixel_bbox':[0,0,100,900]}),({},[]))

    def test_second_scale_must_agree_on_suffix_prefix_geometry_and_formula_extent(self):
        for mutate in (lambda r:r[-1].update(text='將此式之右邊通分通同'),
                       lambda r:r[0]['bbox'].__setitem__(0,400),
                       lambda r:r[-1]['bbox'].__setitem__(1,r[-1]['bbox'][1]+30)):
            rows=deepcopy(self.secondary);mutate(rows)
            self.assertEqual(self.recover(second=rows),({},[]))
        self.assertEqual(self.recover(bounds=[self.math,[0,0,600,1500]]),({},[]))

    def test_unowned_neighboring_prose_inside_crop_abstains(self):
        boxes=self.boxes+[[90,self.first[3]-15,130,self.second[1]+45]]
        with patch('book_pipeline.ingestion.formula_crops.ink_bounds',return_value=[80,self.first[3]-5,280,self.second[1]+5]):
            self.assertIsNone(gap_bounds(self.rgb,boxes,dict(bbox=self.first),self.rows[0],30))

    def test_clipped_cell_edge_sliver_cannot_expand_formula_height_or_edit_source(self):
        rgb=self.rgb.copy()
        rgb[355:385,145:155]=0
        rgb[300:366,-1]=0
        original=rgb.copy()
        boxes=[self.first,[140,self.first[3]-5,400,self.second[1]+5],self.second]
        bounds=gap_bounds(rgb,boxes,dict(bbox=self.first),self.rows[0],30)
        self.assertIsNotNone(bounds)
        self.assertEqual(bounds[1],353)
        np.testing.assert_array_equal(rgb,original)


if __name__=='__main__':
    unittest.main()
