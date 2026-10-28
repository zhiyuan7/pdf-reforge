from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pymupdf

from book_pipeline.ingestion.vertical_formula_gaps import recover_gaps, split_witness, preceding_bounds


class VerticalFormulaGapTests(unittest.TestCase):
    def setUp(self):
        self.prefix = '欲求此式之乘數可令'
        self.suffix = '開得其方根為'
        self.text = self.prefix+self.suffix
        self.rows = [dict(text=self.prefix,score=.999,bbox=[90,30,130,330],reader_signature='cpu'),
                     dict(text=self.suffix,score=.999,bbox=[90,360,130,540],reader_signature='cpu')]
        self.answer = dict(blocks=[dict(type='text',bbox=[.1,.01,.5,.95],
                                      parts=[dict(kind='text',content=self.text)])])
        self.cells = [dict(cell=1,pixel_bbox=[0,0,400,600],regions=[[70,335,150,355]])]
        self.rgb = np.full((600,400,3),255,dtype=np.uint8)

    def recover(self, rows=None, second=None, bounds=None, answer=None, output=None, bounds_sequence=None):
        rows = self.rows if rows is None else rows
        second = deepcopy(rows) if second is None else second
        for row in second:
            row['bbox'] = [v*2 for v in row['bbox']]
        answer = answer or self.answer
        with patch('book_pipeline.ingestion.vertical_formula_gaps.observe_prose',side_effect=[rows,second]), \
             patch('book_pipeline.ingestion.formula_crops.ink_bounds',return_value=bounds or [68,333,152,357],
                   side_effect=bounds_sequence):
            return recover_gaps(self.rgb,self.cells,answer,output or answer,[400,600])

    def test_recover_missing_image_without_changing_prose_or_input(self):
        original = deepcopy(self.answer)
        result,evidence = self.recover()
        self.assertEqual(self.answer,original)
        self.assertEqual([b['type'] for b in result['blocks']],['text','image','text'])
        self.assertEqual(''.join(p['content'] for b in result['blocks'] for p in b['parts']),self.text)
        self.assertEqual(len(evidence),1)
        self.assertTrue(evidence[0]['requires_visual_review'])

    def test_formula_repair_checks_ownerless_cells_without_api_calls(self):
        from book_pipeline.ingestion.formula_crops import repair_formula_images
        detector = Mock(return_value=([[[70,335],[150,335],[150,355],[70,355]]], .01))
        for agreement in (True, False):
            second = deepcopy(self.rows)
            if not agreement:
                second[1]['text'] = second[1]['text'].replace('根', '數')
            for row in second:
                row['bbox'] = [v*2 for v in row['bbox']]
            client = Mock()
            original = deepcopy(self.answer)
            with self.subTest(agreement=agreement), tempfile.TemporaryDirectory() as name, \
                 pymupdf.open() as pdf, \
                 patch('book_pipeline.ingestion.formula_crops.detector_path',return_value=Path('cpu.onnx')), \
                 patch('book_pipeline.ingestion.formula_crops.partition_cells',return_value=[[0,0,400,600]]), \
                 patch('mineru.model.ocr.pp_ocr_v6_onnx.TextDetectorONNX',return_value=detector), \
                 patch('book_pipeline.ingestion.vertical_formula_gaps.observe_prose',side_effect=[self.rows,second]), \
                 patch('book_pipeline.ingestion.formula_crops.ink_bounds',return_value=[68,333,152,357]):
                page = pdf.new_page(width=400,height=600)
                result,evidence = repair_formula_images(page,[0,0,1,1],self.answer,Path(name),
                                                       dict(image_dpi=72),client)
            self.assertEqual(self.answer,original)
            client.ask.assert_not_called()
            self.assertEqual(evidence['cells'][0]['status'],'cpu_gap_check_only')
            self.assertEqual(evidence['reviewed'],0)
            self.assertEqual(len(evidence['prose_gaps']),int(agreement))
            self.assertEqual([b['type'] for b in result['blocks']],
                             ['text','image','text'] if agreement else ['text'])

    def test_disagreement_at_second_scale_cannot_split_or_invent_formula(self):
        second = deepcopy(self.rows)
        second[1]['text'] = second[1]['text'].replace('根','數')
        self.assertEqual(self.recover(second=second),(self.answer,[]))
        second = deepcopy(self.rows)
        second[1]['bbox'][0] += 30
        self.assertEqual(self.recover(second=second),(self.answer,[]))

    def test_displaced_source_box_requires_unique_long_same_cell_witness(self):
        from book_pipeline.ingestion.vertical_formula_gaps import displaced_unique_owner
        block = deepcopy(self.answer['blocks'][0])
        block['bbox'] = [.34,.05,.5,.9]
        w = split_witness(self.text,self.rows)
        self.assertTrue(displaced_unique_owner(block,self.text,w,[block],[0,0,400,600],[400,600]))
        for box in ([.5,.05,.6,.9],[.39,.91,.5,.99],[.7,.05,.8,.9]):
            bad = dict(block,bbox=box)
            self.assertFalse(displaced_unique_owner(bad,self.text,w,[bad],[0,0,400,600],[400,600]))
        self.assertFalse(displaced_unique_owner(block,self.text,w,[block,deepcopy(block)],
                                               [0,0,400,600],[400,600]))
        self.assertFalse(displaced_unique_owner(block,self.text,w,[block],[0,0,150,600],[400,600]))
        short = (w[0],w[1],2,w[3])
        self.assertFalse(displaced_unique_owner(block,self.text,short,[block],[0,0,400,600],[400,600]))

    def test_long_witness_does_not_suppress_another_paragraphs_short_witness(self):
        short_text = '因而可得此積分式'
        short_rows = [dict(text=short_text[:2],score=.999,bbox=[250,30,290,90],reader_signature='cpu'),
                      dict(text=short_text[2:],score=.999,bbox=[250,120,290,300],reader_signature='cpu')]
        answer = deepcopy(self.answer)
        answer['blocks'].append(dict(type='text',bbox=[.6,.01,.8,.6],
                                     parts=[dict(kind='text',content=short_text)]))
        cells = deepcopy(self.cells)
        cells[0]['regions'].append([230,95,310,115])
        for agree in (True, False):
            second_long, second_short = deepcopy(self.rows), deepcopy(short_rows)
            if not agree:
                second_short[1]['text'] = '完全不同的文字'
            for row in second_long+second_short:
                row['bbox'] = [v*2 for v in row['bbox']]
            with self.subTest(short_agreement=agree), \
                 patch('book_pipeline.ingestion.vertical_formula_gaps.observe_prose',
                       side_effect=[self.rows,second_long]), \
                 patch('book_pipeline.ingestion.vertical_prose_fragments.observe_fragments',
                       side_effect=[short_rows,second_short]), \
                 patch('book_pipeline.ingestion.formula_crops.ink_bounds',
                       side_effect=[[68,333,152,357],[228,93,312,117]]):
                result,evidence = recover_gaps(self.rgb,cells,answer,answer,[400,600])
            self.assertEqual([e['block'] for e in evidence],[0,1] if agree else [0])
            self.assertEqual(''.join(p['content'] for b in result['blocks'] for p in b['parts']),
                             self.text+short_text)
            self.assertEqual(sum(b['type']=='image' for b in result['blocks']),2 if agree else 1)

    def test_repeated_prose_and_existing_image_abstain(self):
        answer = deepcopy(self.answer)
        answer['blocks'] *= 2
        self.assertEqual(self.recover(answer=answer),(answer,[]))
        output = deepcopy(self.answer)
        output['blocks'].append(dict(type='image',bbox=[.15,.54,.4,.61],parts=[]))
        self.assertEqual(self.recover(output=output),(output,[]))

    def test_wrong_column_tall_gap_duplicate_reading_and_literal_change_abstain(self):
        for mutate in (lambda rows: rows[1]['bbox'].__setitem__(0,220),
                       lambda rows: rows[1]['bbox'].__setitem__(1,400),
                       lambda rows: rows.append(deepcopy(rows[0])),
                       lambda rows: rows[1].update(text='無關的正文詞句')):
            rows=deepcopy(self.rows);mutate(rows)
            self.assertIsNone(split_witness(self.text,rows))
        self.assertEqual(self.recover(bounds=[68,300,152,400]),(self.answer,[]))

    def test_traditional_glyph_match_preserves_the_existing_character(self):
        rows=deepcopy(self.rows)
        rows[1]['text']=rows[1]['text'].replace('為','爲')
        result,evidence = self.recover(rows=rows)
        self.assertEqual(result['blocks'][2]['parts'][0]['content'],self.suffix)
        self.assertTrue(evidence)

    def test_following_image_needs_same_column_complete_bottom_ink(self):
        answer=deepcopy(self.answer)
        answer['blocks'].append(dict(type='image',bbox=[.2,.65,.45,.8],parts=[],reason='Uncertain math'))
        self.cells[0]['regions'].append([80,545,150,585])
        result,evidence=self.recover(answer=answer,bounds_sequence=[[68,333,152,357],[78,543,152,587]])
        self.assertEqual(len(result['blocks']),4)
        self.assertEqual([e['action'] for e in evidence],['inserted_gap','relocated_following_image'])
        self.assertEqual(result['blocks'][3]['bbox'],[.195,.905,.38,587/600])
        # A differently owned column cannot acquire this terminal formula.
        answer['blocks'][1]['bbox']=[.65,.65,.85,.8]
        result,evidence=self.recover(answer=answer)
        self.assertEqual(len(evidence),1)
        self.assertEqual(result['blocks'][-1],answer['blocks'][-1])
        # Ink further below means that the proposed tail is incomplete.
        answer['blocks'][1]['bbox']=[.2,.65,.45,.8]
        self.cells[0]['regions'].append([80,590,150,598])
        result,evidence=self.recover(answer=answer,bounds_sequence=[[68,333,152,357],[78,543,152,587]])
        self.assertEqual(len(evidence),1)
        self.assertEqual(result['blocks'][-1],answer['blocks'][-1])

    def test_short_fragments_need_complete_prose_and_unique_geometric_owner(self):
        text='因而可得此積分式'
        rows=[dict(text=text[:2],score=.999,bbox=[90,30,130,90],reader_signature='cpu'),
              dict(text=text[2:],score=.999,bbox=[90,120,130,300],reader_signature='cpu')]
        second=deepcopy(rows)
        for row in second:row['bbox']=[v*2 for v in row['bbox']]
        answer=dict(blocks=[dict(type='image',bbox=[.15,.005,.425,.05],parts=[]),
            dict(type='text',bbox=[.2,.01,.4,.6],parts=[dict(kind='text',content=text)]),
            dict(type='text',bbox=[.7,.01,.9,.6],parts=[dict(kind='text',content=text)])])
        cells=[dict(cell=1,pixel_bbox=[0,0,400,600],regions=[[70,5,150,25],[70,95,150,115]])]
        with patch('book_pipeline.ingestion.vertical_formula_gaps.observe_prose',return_value=[]), \
             patch('book_pipeline.ingestion.vertical_prose_fragments.observe_fragments',side_effect=[rows,second]), \
             patch('book_pipeline.ingestion.formula_crops.ink_bounds',side_effect=[[68,93,152,117],[68,5,152,27]]):
            result,evidence=recover_gaps(self.rgb,cells,answer,answer,[400,600])
        self.assertEqual([e['action'] for e in evidence],['inserted_gap','tightened_preceding_image'])
        self.assertEqual(''.join(p['content'] for b in result['blocks'] for p in b['parts']),text*2)
        self.assertEqual(result['blocks'][-1],answer['blocks'][-1])
        self.assertEqual(result['blocks'][0]['bbox'],[.17,5/600,.38,27/600])
        self.assertIsNone(split_witness('因而所以',rows))
        # Equal text in the same physical column is ambiguous, even if the
        # independent reader is certain. A short phrase cannot resolve it.
        answer['blocks'][-1]['bbox']=answer['blocks'][1]['bbox']
        with patch('book_pipeline.ingestion.vertical_formula_gaps.observe_prose',return_value=[]), \
             patch('book_pipeline.ingestion.vertical_prose_fragments.observe_fragments',return_value=rows):
            self.assertEqual(recover_gaps(self.rgb,cells,answer,answer,[400,600]),(answer,[]))

    def test_preceding_image_cannot_acquire_vertical_prose_or_another_column(self):
        first=dict(text='因而',bbox=[90,100,130,160])
        rgb=np.full((400,300,3),255,np.uint8)
        for boxes,original in (([[90,40,130,80]],[60,20,180,100]),
                               ([[60,40,180,80]],[210,20,290,100]),
                               ([[60,40,180,80]],[60,200,180,300])):
            self.assertIsNone(preceding_bounds(rgb,boxes,first,30,original))



if __name__ == '__main__':
    unittest.main()
