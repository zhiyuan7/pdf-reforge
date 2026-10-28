from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from book_pipeline.ingestion.vertical_formula_tail import bounded_tail, continue_in_next_column
from book_pipeline.ingestion.vertical_formula_gaps import recover_gaps


class VerticalFormulaTailTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.full((500,300,3),255,np.uint8)
        self.preceding=dict(text='故可求之',score=.999,bbox=[90,120,130,300],reader_signature='cpu')
        self.terminal=dict(text='由此',score=.999,bbox=[90,430,130,490],reader_signature='cpu')
        self.rows=[self.preceding,self.terminal]
        self.boxes=[[50,310,180,350],[50,370,180,420]]
        self.original=[80,350,160,420]

    def tail(self,rows=None,second=None,original=None,bounds=None,source_texts=None):
        rows=self.rows if rows is None else rows
        with patch('book_pipeline.ingestion.formula_crops.ink_bounds',return_value=bounds or [48,308,182,422]):
            return bounded_tail(self.rgb,self.boxes,self.preceding,rows,
                deepcopy(rows) if second is None else second,30,
                self.original if original is None else original,source_texts or ['完整的前文'])

    def test_two_scale_terminal_bounds_a_multirow_formula(self):
        original=deepcopy(self.rows)
        result=self.tail()
        self.assertEqual(result['bounds'],[48,308,182,422])
        self.assertEqual(result['terminal']['text'],'由此')
        self.assertTrue(result['requires_visual_review'])
        self.assertEqual(self.rows,original)

    def test_disagreement_duplicate_and_existing_prose_abstain(self):
        for change in (dict(text='由彼'),dict(score=.98),dict(reader_signature='other'),
                       dict(bbox=[120,430,160,490])):
            self.assertIsNone(self.tail(second=[self.preceding,{**self.terminal,**change}]))
        self.assertIsNone(self.tail(second=self.rows+[self.terminal]))
        self.assertIsNone(self.tail(rows=self.rows+[self.terminal]))
        self.assertIsNone(self.tail(source_texts=['前文由此求得']))

    def test_wrong_column_distant_formula_and_intervening_prose_abstain(self):
        self.assertIsNone(self.tail(original=[210,350,290,420]))
        self.assertIsNone(self.tail(original=[80,10,160,80]))
        for changes in (dict(bbox=[170,430,210,490]),dict(bbox=[90,330,130,390]),
                        dict(bbox=[90,430,130,470]),dict(score=.98),dict(text='甲')):
            terminal={**self.terminal,**changes}
            self.assertIsNone(self.tail(rows=[self.preceding,terminal]))
        row=dict(text='其他正文',score=.999,bbox=[90,315,130,415],reader_signature='cpu')
        self.assertIsNone(self.tail(rows=self.rows+[row]))
        for bounds in ([48,200,182,422],[48,308,182,480],[200,308,282,422]):
            self.assertIsNone(self.tail(bounds=bounds))

    def test_terminal_fragment_continues_only_at_adjacent_column_top(self):
        terminal=dict(type='text',bbox=[.55,.85,.65,.98],parts=[dict(kind='text',content='由此')])
        following=dict(type='text',bbox=[.35,.32,.48,.56],parts=[dict(kind='text',content='可知其理')])
        # Cell top=150; the next column begins ten pixels below it.
        original=deepcopy(following)
        result=continue_in_next_column(terminal,following,[250,150,400,500],[500,500],30)
        self.assertEqual(''.join(p['content'] for p in result['parts']),'由此可知其理')
        self.assertEqual(following,original)
        self.assertEqual(result['source_prose_end_bbox'],following['bbox'])
        for changes in (dict(type='paragraph_title'),dict(parts=[dict(kind='math',content='x=1')]),
                        dict(bbox=[.39,.5,.48,.74]),dict(bbox=[.1,.32,.19,.56]),
                        dict(bbox=[.51,.32,.6,.56]),dict(parts=[dict(kind='text',content='可知其理。')])):
            self.assertIsNone(continue_in_next_column(terminal,{**following,**changes},[250,150,400,500],[500,500],30))

    def test_complete_prose_witness_integration_and_other_image_veto(self):
        prefix='欲求此式之乘數可令';suffix='開得其方根為'
        rows=[dict(text=prefix,score=.999,bbox=[90,30,130,330],reader_signature='cpu'),
              dict(text=suffix,score=.999,bbox=[90,360,130,540],reader_signature='cpu'),
              dict(text='由此',score=.999,bbox=[90,690,130,750],reader_signature='cpu')]
        second=deepcopy(rows)
        for row in second:row['bbox']=[2*v for v in row['bbox']]
        answer=dict(blocks=[dict(type='text',bbox=[.1,.01,.5,.95],parts=[dict(kind='text',content=prefix+suffix)]),
                            dict(type='image',bbox=[.2,.7,.45,.95],parts=[])])
        cells=[dict(cell=1,pixel_bbox=[0,0,400,760],regions=[[70,335,150,355],[50,545,180,600],[50,610,180,680]])]
        rgb=np.full((760,400,3),255,np.uint8)
        original=deepcopy(answer)
        def run(output):
            with patch('book_pipeline.ingestion.vertical_formula_gaps.observe_prose',side_effect=[rows,deepcopy(second)]), \
                 patch('book_pipeline.ingestion.formula_crops.ink_bounds',side_effect=[[68,333,152,357],[48,543,182,682]]):
                return recover_gaps(rgb,cells,answer,output,[400,760])
        result,evidence=run(answer)
        self.assertEqual(answer,original)
        self.assertEqual([b['type'] for b in result['blocks']],['text','image','text','image','text'])
        self.assertEqual(''.join(p['content'] for b in result['blocks'] for p in b['parts']),prefix+suffix+'由此')
        self.assertEqual(evidence[-1]['action'],'bounded_following_image_and_terminal_prose')
        self.assertEqual(result['blocks'][-2]['bbox'],[.12,543/760,.455,682/760])
        output=deepcopy(answer)
        output['blocks'].append(dict(type='image',bbox=[.1,.8,.2,.88],parts=[]))
        # The old single-row fallback must not consume another mocked result.
        with patch('book_pipeline.ingestion.vertical_formula_gaps.compact_bounds',side_effect=[[68,333,152,357],None]):
            result,evidence=run(output)
        self.assertEqual(len(evidence),1)
        self.assertEqual(result['blocks'][-2:],output['blocks'][-2:])


if __name__ == '__main__':
    unittest.main()
