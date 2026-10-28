import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from book_pipeline.ingestion.vertical_register_continuation import join_registers


class RegisterContinuationTests(unittest.TestCase):
    def setUp(self):
        self.before = '取兩邊各項之係數為相'
        self.after = '等而得'
        self.cells = [dict(cell=1,pixel_bbox=[200,0,400,500],regions=[]),
                      dict(cell=2,pixel_bbox=[0,0,200,500],regions=[]),
                      dict(cell=3,pixel_bbox=[200,500,400,1000],regions=[]),
                      dict(cell=4,pixel_bbox=[0,500,200,1000],regions=[])]
        self.answer = dict(blocks=[
            dict(type='text',bbox=[.1,.1,.3,.49],parts=[dict(kind='text',content=self.before)]),
            dict(type='text',bbox=[.6,.55,.8,.67],parts=[dict(kind='text',content=self.after)]),
            dict(type='image',bbox=[.55,.68,.85,.8],parts=[],reason='Existing source formula')])
        self.rgb = np.full((1000,400,3),255,np.uint8)
        self.rows = []
        for text,box in ((self.before,[50,40,100,480]),(self.after,[50,50,100,170])):
            row = dict(text=text,bbox=box,score=.999,reader_signature='cpu')
            self.rows.extend([[row],[dict(row,bbox=[v*2 for v in box])]])

    def run_join(self,answer=None,cells=None,rows=None):
        with patch('book_pipeline.ingestion.ownerless_formula_gaps.observe_layout',side_effect=rows or self.rows):
            return join_registers(self.rgb,cells or self.cells,answer or self.answer,[400,1000])

    def test_reading_order_join_preserves_literal_parts_images_and_original(self):
        original = deepcopy(self.answer)
        output,evidence = self.run_join()
        self.assertEqual(self.answer,original)
        self.assertEqual(len(output['blocks']),2)
        self.assertEqual(''.join(p['content'] for p in output['blocks'][0]['parts']),self.before+self.after)
        self.assertEqual(output['blocks'][1],self.answer['blocks'][2])
        self.assertTrue(evidence[0]['text_unchanged'])
        self.assertTrue(evidence[0]['requires_visual_review'])

    def test_titles_math_punctuation_duplicate_text_and_nonadjacent_blocks_are_barriers(self):
        for mutate in (lambda b:b[1].update(type='paragraph_title'),
                       lambda b:b[0]['parts'][0].update(kind='math'),
                       lambda b:b[0]['parts'][0].update(content=self.before+'。'),
                       lambda b:b.append(deepcopy(b[0])),
                       lambda b:b.insert(1,deepcopy(b[-1]))):
            answer = deepcopy(self.answer);mutate(answer['blocks'])
            self.assertEqual(self.run_join(answer=answer),(answer,[]))

    def test_readings_must_agree_completely_and_geometrically_at_both_scales(self):
        for mutate in (lambda r:r[1][0].update(text=self.before[:-1]),
                       lambda r:r[0][0].update(score=.97),
                       lambda r:r[3][0].update(reader_signature=''),
                       lambda r:r[3][0].update(bbox=[100,140,200,380]),
                       lambda r:r[0].append(deepcopy(r[0][0]))):
            rows = deepcopy(self.rows);mutate(rows)
            output,evidence = self.run_join(rows=rows)
            self.assertEqual(output,self.answer)
            self.assertFalse(evidence)

    def test_only_last_and_first_columns_of_adjacent_full_registers_can_join(self):
        for mutate in (lambda c:c[2]['pixel_bbox'].__setitem__(1,520),
                       lambda c:c[3]['pixel_bbox'].__setitem__(0,20),
                       lambda c:c.pop(0),
                       lambda c:c[1].update(regions=[[60,485,90,495]]),
                       lambda c:c[2].update(regions=[[60,5,90,40]])):
            cells = deepcopy(self.cells);mutate(cells)
            output,evidence = self.run_join(cells=cells)
            self.assertEqual(output,self.answer)
            self.assertFalse(evidence)

    def test_interior_paragraph_end_or_late_next_start_cannot_join(self):
        for k,box in ((0,[50,40,100,400]),(2,[50,160,100,280])):
            rows=deepcopy(self.rows);rows[k][0]['bbox']=box;rows[k+1][0]['bbox']=[v*2 for v in box]
            self.assertEqual(self.run_join(rows=rows),(self.answer,[]))


if __name__ == '__main__':
    unittest.main()
