from copy import deepcopy
import unittest
from unittest.mock import Mock, patch
import numpy as np

from book_pipeline.ingestion.vertical_mixed_math import (TIGHT_METHOD, anchored_span, complete_bounds, complete_prose_bounds, math_spans,
                                                         recover_mixed_math, tight_prose)


def row(text, box):
    return dict(text=text,bbox=box,score=.999,reader_signature='cpu',tight_method=TIGHT_METHOD)


class VerticalMixedMathTests(unittest.TestCase):
    def setUp(self):
        self.first=row('某題已有公式',[170,40,210,220])
        self.second=row('欲求此式積分',[170,300,210,480])
        self.rows=[self.first,self.second]
        self.parts=[dict(kind='text',content='前文某題已有公式'),dict(kind='math',content=r'\frac{a}{b}'),
                    dict(kind='text',content='從'),dict(kind='math',content=r'\frac{x}{y}'),
                    dict(kind='text',content='欲求此式積分而後另議')]
        self.answer=dict(blocks=[dict(type='text',bbox=[.4,.01,.6,.99],parts=self.parts)])
        self.rgb=np.full((600,400,3),255,np.uint8)
        self.cell=dict(cell=1,pixel_bbox=[0,0,400,600],regions=[[130,230,260,290]])
        self.bounds=[130,230,260,290]

    def recover(self, rows=None, secondary=None, bounds=None, answer=None, cell=None):
        initial=deepcopy(self.rows if rows is None else rows)
        second=deepcopy(initial if secondary is None else secondary)
        for r in second:r['bbox']=[v*2 for v in r['bbox']]
        with patch('book_pipeline.ingestion.vertical_mixed_math.tight_prose',side_effect=[initial,second]), \
             patch('book_pipeline.ingestion.vertical_formula_gaps.compact_bounds',side_effect=
                   bounds or [self.bounds,[v*2 for v in self.bounds]]), \
             patch('book_pipeline.ingestion.prose_ink_exclusions.exclusions',return_value=None):
            return recover_mixed_math(self.rgb,[self.cell if cell is None else cell],
                                      self.answer if answer is None else answer,[400,600])

    def test_source_fallback_retains_original_parts_and_outside_prose_without_mutation(self):
        original=deepcopy(self.answer)
        result,evidence=self.recover()
        self.assertEqual(self.answer,original)
        self.assertEqual([b['type'] for b in result['blocks']],['text','image','text'])
        self.assertEqual(result['blocks'][0]['parts'],self.parts[:1])
        self.assertEqual(result['blocks'][2]['parts'],self.parts[4:])
        self.assertEqual(evidence[0]['original_parts'],self.parts[1:4])
        self.assertFalse(evidence[0]['transcription_verified'])
        self.assertTrue(evidence[0]['requires_visual_review'])
        self.assertEqual(result['blocks'][1]['bbox'],[.325,230/600,.65,290/600])

    def test_ordinary_long_interstitial_text_is_never_swallowed(self):
        parts=deepcopy(self.parts);parts[2]['content']='而其後又可得'
        self.assertEqual(list(math_spans(parts)),[(1,2),(3,4)])
        # Neither side of either smaller span has two complete long anchors.
        answer=deepcopy(self.answer);answer['blocks'][0]['parts']=parts
        result,evidence=self.recover(answer=answer)
        self.assertEqual(result,answer);self.assertFalse(evidence)

    def test_no_short_weak_duplicate_changed_or_unverified_anchor(self):
        for change in (dict(score=.98),dict(text='已有式'),dict(reader_signature=''),
                       dict(tight_method=''),dict(text='某題已有公理')):
            rows=deepcopy(self.rows);rows[0].update(change)
            self.assertIsNone(anchored_span(self.parts[0]['content'],self.parts[-1]['content'],rows))
        self.assertIsNone(anchored_span(self.parts[0]['content'],self.parts[-1]['content'],self.rows*2))
        self.assertIsNone(anchored_span('欲求此式積分','某題已有公式',self.rows))

    def test_other_column_nonadjacency_changed_scales_and_existing_images_abstain(self):
        for change in (dict(bbox=[290,40,330,220]),dict(bbox=[170,5,210,185])):
            rows=deepcopy(self.rows);rows[0].update(change)
            result,evidence=self.recover(rows=rows)
            self.assertEqual(result,self.answer);self.assertFalse(evidence)
        second=deepcopy(self.rows);second[0]['bbox'][3]+=20
        self.assertEqual(self.recover(secondary=second),(self.answer,[]))
        self.assertEqual(self.recover(bounds=[self.bounds,[100,460,520,580]]),(self.answer,[]))
        answer=deepcopy(self.answer);answer['blocks'].append(dict(type='image',parts=[],bbox=[.35,.4,.6,.45]))
        self.assertEqual(self.recover(answer=answer),(answer,[]))
        self.assertEqual(self.recover(cell={**self.cell,'pixel_bbox':[0,0,100,600]}),(self.answer,[]))

    def test_unrelated_prose_in_formula_interval_is_protected(self):
        rows=self.rows+[row('獨立普通正文',[135,235,175,285])]
        self.assertEqual(self.recover(rows=rows),(self.answer,[]))

    def test_whole_gap_components_restore_clipped_tips_without_neighbor_prose_or_dust(self):
        rgb=self.rgb.copy()
        rgb[225:240,133:145]=0  # numerator tip owned by coarse prose padding
        rgb[280:294,142:151]=0  # denominator below the initial rectangle
        rgb[260:263,125:140]=0  # fraction-bar tip to the left
        rgb[212:226,180:205]=0  # crosses the complete prefix boundary
        rgb[299:320,180:205]=0  # crosses the complete successor boundary
        rgb[230:240,80:90]=0   # distant dust must not expand this crop
        before=rgb.copy()
        bounds,parts=complete_bounds(rgb,self.bounds,self.first,self.second,30)
        self.assertEqual(bounds,[123,223,260,296])
        self.assertEqual(len(parts),3)
        np.testing.assert_array_equal(rgb,before)
        big=np.repeat(np.repeat(rgb,2,axis=0),2,axis=1)
        first={**self.first,'bbox':[v*2 for v in self.first['bbox']]}
        second={**self.second,'bbox':[v*2 for v in self.second['bbox']]}
        enlarged,_=complete_bounds(big,[v*2 for v in self.bounds],first,second,60)
        self.assertLessEqual(max(abs(a-b/2) for a,b in zip(bounds,enlarged)),1)

    def test_no_mixed_math_is_a_noop_without_loading_ocr(self):
        answer=dict(blocks=[dict(type='text',bbox=[.4,.01,.6,.99],parts=[self.parts[0]])])
        with patch('book_pipeline.ingestion.vertical_mixed_math.tight_prose') as reader:
            self.assertEqual(recover_mixed_math(self.rgb,[self.cell],answer,[400,600]),(answer,[]))
            reader.assert_not_called()

    def test_projected_prose_bounds_complete_clipped_strokes_before_reading(self):
        # [left, top, width, height, area] includes a first glyph whose faint
        # top lies outside the projection and the analogous final descender.
        stats=np.array([[72,45,36,37,1100],[70,90,40,32,900],
                        [72,130,36,37,1100],[30,48,3,3,9],
                        [135,45,38,38,1000],[75,240,3,3,9]])
        self.assertEqual(complete_prose_bounds([60,50,120,162],stats,60,(300,180,3)),
                         [70,45,110,167])
        doubled=stats.copy();doubled[:,:4]*=2;doubled[:,4]*=4
        self.assertEqual(complete_prose_bounds([120,100,240,324],doubled,120,(600,360,3)),
                         [140,90,220,334])

    def test_ambiguous_large_boundary_completion_abstains(self):
        stats=np.array([[72,20,36,80,1800],[72,130,36,37,1100]])
        self.assertIsNone(complete_prose_bounds([60,50,120,162],stats,60,(300,180,3)))
        self.assertIsNone(complete_prose_bounds([60,50,120,162],np.empty((0,5)),60,(300,180,3)))

    def test_reader_never_sees_clipped_component_even_when_it_claims_complete_text(self):
        rgb=np.full((420,180,3),255,np.uint8)
        for y in (50,90,130,170,210):rgb[y:y+32,72:108]=0
        # A thin connected top stroke falls below the row projection cutoff.
        rgb[44:50,90:94]=0
        detected=row('已知普通文字',[60,40,120,310]);detected.pop('tight_method')
        model=Mock();counts=[]
        def blind_read(pixels):
            count=np.count_nonzero(pixels[:,:,0] == 0);counts.append(count)
            return [[[[[0,0],[60,0],[60,200],[0,200]],
                       ('已知普通文字' if count>=5*36*32 else '普通',.999)]]]
        model.ocr.side_effect=blind_read
        with patch('book_pipeline.ingestion.vertical_prose_fragments.observe_fragments',return_value=[detected]), \
             patch('book_pipeline.publishing.image_search.local_ocr',return_value=(model,'cpu')):
            rows=tight_prose(rgb)
        self.assertTrue(rows)
        self.assertEqual(rows[0]['bbox'],[72,44,108,242])
        self.assertEqual(counts[-1],5*36*32+24)
        # The call is blind: only source pixels are passed, without text hints.
        self.assertEqual(len(model.ocr.call_args.args),1)
        self.assertFalse(model.ocr.call_args.kwargs)

    def test_projection_reread_has_only_pixels_and_must_match_complete_text(self):
        rgb=np.full((420,180,3),255,np.uint8)
        # A complete five-glyph run and small mathematical ink inside its
        # detector's padding. Only the shorter complete reading is eligible.
        for y in (50,90,130,170,210):rgb[y:y+32,72:108]=0
        rgb[270:290,78:103]=0
        detected=row('已知普通文字',[60,40,120,310]);detected.pop('tight_method')
        model=Mock()
        def blind_read(pixels):
            complete = np.count_nonzero(pixels[:,:,0] == 0) >= 5*36*32
            return [[[[[0,0],[60,0],[60,200],[0,200]],
                       ('已知普通文字' if complete else '普通',.999)]]]
        model.ocr.side_effect=blind_read
        with patch('book_pipeline.ingestion.vertical_prose_fragments.observe_fragments',return_value=[detected]), \
             patch('book_pipeline.publishing.image_search.local_ocr',return_value=(model,'cpu')):
            rows=tight_prose(rgb)
        self.assertEqual(len(rows),1)
        self.assertLess(rows[0]['bbox'][3],270)
        self.assertEqual(rows[0]['text'],detected['text'])
        self.assertEqual(rows[0]['tight_method'],TIGHT_METHOD)
        args,kwargs=model.ocr.call_args
        self.assertEqual(len(args),1);self.assertFalse(kwargs)
        self.assertIsInstance(args[0],np.ndarray)
        model.ocr.side_effect=None
        model.ocr.return_value=[[[[[0,0],[60,0],[60,200],[0,200]],('已知普通',.999)]]]
        with patch('book_pipeline.ingestion.vertical_prose_fragments.observe_fragments',return_value=[detected]), \
             patch('book_pipeline.publishing.image_search.local_ocr',return_value=(model,'cpu')):
            self.assertEqual(tight_prose(rgb),[])


if __name__=='__main__':unittest.main()
