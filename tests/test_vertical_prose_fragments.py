import unittest
from unittest.mock import Mock, patch

import numpy as np

from book_pipeline.ingestion.vertical_prose_fragments import observe_fragments


def detected(text, box, score=.999):
    x,y,z,t=box
    return ([[x,y],[z,y],[z,t],[x,t]],(text,score))


class VerticalProseFragmentTests(unittest.TestCase):
    def observe(self, reread=None, raw=None):
        model=Mock()
        raw = raw if raw is not None else [detected('因',[90,30,130,70]),
            detected('而',[90,65,130,105],.7),
            detected('可得此積分式',[90,150,130,330])]
        model.ocr.side_effect = [[raw],[reread if reread is not None else [detected('因而',[40,40,120,190])]]]
        with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(model,'cpu')):
            rows=observe_fragments(np.full((400,300,3),255,np.uint8))
        return rows,model

    def test_adjacent_glyphs_require_a_blind_complete_rereading(self):
        rows,model=self.observe()
        self.assertEqual([r['text'] for r in rows],['可得此積分式','因而'])
        self.assertEqual(rows[1]['bbox'],[90,30,130,105])
        self.assertEqual(rows[1]['reader_signature'],'cpu')
        for call in model.ocr.call_args_list:
            self.assertEqual(len(call.args),1)
            self.assertEqual(call.kwargs,{})
            self.assertIsInstance(call.args[0],np.ndarray)

    def test_weak_multiple_wrong_length_and_distant_readings_cannot_join(self):
        for reread in ([detected('因而',[40,40,120,190],.98)],
                       [detected('因',[40,40,120,190])],
                       [detected('因而',[40,40,120,190])]*2,[]):
            rows,_=self.observe(reread=reread)
            self.assertEqual([r['text'] for r in rows],['可得此積分式'])
        for box in ([180,65,220,105],[90,110,130,150],[90,65,170,105]):
            rows,model=self.observe(raw=[detected('因',[90,30,130,70]),detected('而',box)])
            self.assertEqual(rows,[])
            model.ocr.assert_called_once()


if __name__ == '__main__':
    unittest.main()
