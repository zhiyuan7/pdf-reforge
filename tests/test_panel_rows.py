from pathlib import Path
from copy import deepcopy
import tempfile
import unittest

import pymupdf

from book_pipeline.ingestion.panel_rows import anchor_rows,terminal_display_bounds


class PanelRowTests(unittest.TestCase):
    def lines(self):
        return [[dict(atom=dict(kind='text'),content=t)] for t in
                ('First independent paragraph.','Another separate sentence.','x=1')]

    def test_prose_anchors_require_words_unique_order_and_confidence(self):
        rows=[dict(bbox=[0,y,200,y+10],words=[dict(text=t,score=.99) for t in text.split()])
              for y,text in [(10,'First independent paragraph.'),(40,'Another separate sentence.')]]
        self.assertEqual(anchor_rows(self.lines(),rows),rows)
        for mutate in (lambda r:r.append(deepcopy(r[0])),lambda r:r[0]['words'][0].update(score=.4),
                       lambda r:r[1].update(bbox=[0,1,200,10])):
            bad=deepcopy(rows);mutate(bad);self.assertIsNone(anchor_rows(self.lines(),bad))
        self.assertIsNone(anchor_rows([[dict(atom=dict(kind='text'),content='1')]]+self.lines()[1:],rows))

    def test_terminal_math_owns_upper_limits_below_last_observed_prose(self):
        with tempfile.TemporaryDirectory() as name,pymupdf.open() as d:
            p=d.new_page(width=500,height=700)
            p.draw_rect((40,90,460,280),color=(0,0,0),width=1)
            p.insert_text((60,120),'First independent paragraph.',fontsize=11)
            p.insert_text((60,160),'Another separate sentence.',fontsize=11)
            p.insert_text((220,200),'oo',fontsize=7)
            p.insert_text((190,225),'x = 1',fontsize=15)
            panel=dict(bbox=[.08,90/700,.92,280/700],title_bbox=None)
            original=dict(classification='prose_panel',reason='image',lines=[
                dict(id=0,kind='text',bbox=[.1,.14,.8,.19]),
                dict(id=1,kind='text',bbox=[.1,.20,.8,.26]),
                dict(id=2,kind='display',bbox=[.3,210/700,.7,240/700])])
            result,evidence=terminal_display_bounds(p,panel,self.lines(),original,Path(name))
            self.assertEqual(evidence['status'],'bounded_by_cpu_prose')
            self.assertLess(result['lines'][-1]['bbox'][1],190/700)
            self.assertGreater(result['lines'][-1]['bbox'][3],240/700)
            self.assertEqual([r['kind'] for r in result['lines']],[r['kind'] for r in original['lines']])
