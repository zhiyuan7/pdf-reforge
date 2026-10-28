import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from book_pipeline.ingestion.ruled_prose import partition_cells, page_box
from book_pipeline.ingestion.formula_crops import validate_selection, validate_joint_selection, exclusive_bounds, ink_bounds


class RuledProseTests(unittest.TestCase):
    def test_skewed_rules_and_double_register_border_keep_rtl_order(self):
        im = Image.new('RGB', (800, 1100), '#e6dccb')
        draw = ImageDraw.Draw(im)
        for y in (540, 548):
            draw.line((0,y,799,y-7), fill='#333333', width=3)
        draw.line((300,0,305,540), fill='#333333', width=3)
        draw.line((500,0,505,540), fill='#333333', width=3)
        draw.line((400,548,406,1099), fill='#333333', width=3)
        # Stacked glyph-like strokes must not be mistaken for a ruled boundary.
        for y in range(95, 520, 24):
            draw.line((620,y,620,y+12), fill='black', width=3)
            draw.line((613,y,627,y), fill='black', width=2)
        cells = partition_cells(np.array(im))
        self.assertEqual(len(cells), 5)
        self.assertGreater(cells[0][0], cells[1][0])
        self.assertGreater(cells[1][0], cells[2][0])
        self.assertLess(abs(cells[0][3]-544), 12)
        self.assertEqual(cells[2][0], 0)
        self.assertGreater(cells[3][0], cells[4][0])
        self.assertEqual(cells[-1][3], 1100)

    def test_unruled_and_dense_table_grid_do_not_create_local_requests(self):
        im = Image.new('RGB', (800, 1100), 'white')
        self.assertEqual(partition_cells(np.array(im)), [])
        draw=ImageDraw.Draw(im)
        for y in range(50, 1100, 50):
            draw.line((0,y,799,y), fill='black', width=3)
        self.assertEqual(partition_cells(np.array(im)), [])

    def test_local_coordinates_use_integer_crop_and_actual_full_render_size(self):
        self.assertEqual(page_box([0,.25,1,.75], [80,100,480,900], [800,1000]), [.1,.3,.6,.7])

    def test_selection_cannot_invent_ids_reuse_ids_or_choose_prose(self):
        validate_selection(dict(regions=[1,2],uncertain=False,reason='Math fragments'),[1,2])
        validate_selection(dict(regions=[],uncertain=True,reason='Mixed owner'),[1,2])
        for ids in ([1,1],[0],[True],[],[1,3]):
            with self.assertRaises(ValueError):
                validate_selection(dict(regions=ids,uncertain=False,reason='Test'),[1,2])

    def test_ink_ownership_removes_adjacent_prose_and_recovers_equality_tail(self):
        im=Image.new('RGB',(300,600),'#e6dccb');draw=ImageDraw.Draw(im)
        # The detector's formula box overlaps the neighboring prose tips.
        draw.rectangle((130,40,160,196),fill='black')
        draw.rectangle((130,264,160,520),fill='black')
        draw.rectangle((50,205,65,240),fill='black')
        draw.rectangle((115,205,130,240),fill='black')
        draw.line((70,220,112,220),fill='black',width=3)
        # This separate equality tail was entirely missed by the detector.
        draw.ellipse((205,210,225,235),outline='black',width=4)
        # A broken ruled edge at the same height must not enlarge the crop.
        draw.line((0,208,0,240),fill='black',width=2)
        draw.line((299,0,299,599),fill='black',width=2)
        boxes=[[40,190,140,280],[125,30,165,200],[125,260,165,530]]
        box=ink_bounds(np.array(im),boxes,[0])
        self.assertEqual(box,[38,203,228,243])
        self.assertIsNone(ink_bounds(np.full((600,300,3),255,dtype=np.uint8),boxes,[0]))

    def test_joint_selection_requires_complete_exclusive_existing_owners(self):
        rows = [dict(block=4, regions=[0], uncertain=False, reason='First equation'),
                dict(block=7, regions=[1], uncertain=False, reason='Second equation')]
        validate_joint_selection(dict(owners=rows), [0, 1], [4, 7])
        from copy import deepcopy
        for mutate in (lambda a: a.pop(), lambda a: a[1].update(block=4),
                       lambda a: a[1].update(block=True), lambda a: a[1].update(block=9),
                       lambda a: a[1].update(regions=[0]), lambda a: a[0].update(regions=[2])):
            changed = deepcopy(rows); mutate(changed)
            with self.assertRaises(ValueError):
                validate_joint_selection(dict(owners=changed), [0, 1], [4, 7])

    def test_unobserved_tail_requires_nearby_math_not_only_the_same_row(self):
        im = Image.new('RGB', (700, 600), 'white')
        draw = ImageDraw.Draw(im)
        draw.rectangle((80, 205, 100, 240), fill='black')
        draw.rectangle((140, 205, 160, 240), fill='black')
        # An unobserved equality tail and terminal dot belong to this row.
        draw.ellipse((205, 210, 225, 235), outline='black', width=4)
        draw.rectangle((235, 230, 237, 232), fill='black')
        boxes = [[75, 200, 165, 245]]
        baseline = ink_bounds(np.array(im), boxes, [0])
        self.assertEqual(baseline, [73, 203, 240, 243])
        # Far dust and unknown ink in another column must not pull the crop
        # across intervening whitespace, despite matching its y coordinates.
        draw.rectangle((600, 220, 602, 222), fill='black')
        draw.rectangle((620, 208, 635, 238), fill='black')
        self.assertEqual(ink_bounds(np.array(im), boxes, [0]), baseline)

    def test_joint_rectangles_cannot_enclose_prose_or_other_equation(self):
        boxes = [[20,20,60,40], [20,100,60,120], [25,50,55,90]]
        choices = [dict(block=4,regions=[0],uncertain=False), dict(block=7,regions=[1],uncertain=False)]
        rgb = np.full((600,300,3),255,dtype=np.uint8)
        with patch('book_pipeline.ingestion.formula_crops.ink_bounds',side_effect=[[18,18,62,42], [18,98,62,122]]):
            bounds=exclusive_bounds(rgb,boxes,choices,[0,1])
        self.assertTrue(all(bounds.values()))
        # Even disjoint region IDs are unsafe if one rectangle crosses prose.
        with patch('book_pipeline.ingestion.formula_crops.ink_bounds',side_effect=[[18,18,62,95], [18,98,62,122]]):
            bounds=exclusive_bounds(rgb,boxes,choices,[0,1])
        self.assertIsNone(bounds[4]); self.assertIsNotNone(bounds[7])
        # A short prose region is still protected when it passes the height
        # filter and is offered to the model as an allowed region.
        with patch('book_pipeline.ingestion.formula_crops.ink_bounds',side_effect=[[18,18,62,95], [18,98,62,122]]):
            bounds=exclusive_bounds(rgb,boxes,choices,[0,1,2])
        self.assertIsNone(bounds[4]); self.assertIsNotNone(bounds[7])
        with patch('book_pipeline.ingestion.formula_crops.ink_bounds',side_effect=[[18,18,62,110], [18,98,62,122]]):
            self.assertTrue(all(b is None for b in exclusive_bounds(rgb,boxes,choices,[0,1]).values()))

    def test_disjoint_selection_cannot_swap_existing_equation_owners(self):
        boxes=[[20,20,60,40],[20,100,60,120]]
        choices=[dict(block=4,regions=[1],uncertain=False),dict(block=7,regions=[0],uncertain=False)]
        anchors={4:boxes[0],7:boxes[1]}
        rgb=np.full((600,300,3),255,dtype=np.uint8)
        with patch('book_pipeline.ingestion.formula_crops.ink_bounds',side_effect=[boxes[1],boxes[0]]):
            self.assertTrue(all(b is None for b in exclusive_bounds(rgb,boxes,choices,[0,1],anchors).values()))

    def test_uncertain_joint_review_keeps_all_content_and_bounds(self):
        import pymupdf
        from book_pipeline.ingestion.formula_crops import repair_formula_images
        answer=dict(classification='prose',direction='vertical_rl',blocks=[
            dict(type='image',bbox=[.1,.1,.3,.2],parts=[],reason='First old formula'),
            dict(type='image',bbox=[.1,.3,.3,.4],parts=[],reason='Second old formula'),
            dict(type='text',bbox=[.7,.1,.8,.8],parts=[dict(kind='text',content='Keep original text')])])
        client=Mock()
        client.ask.return_value = dict(owners=[dict(block=n,regions=[],uncertain=True,reason='Ambiguous') for n in (0,1)])
        detector=Mock(return_value=([[[10,20],[40,20],[40,40],[10,40]]], .01))
        with TemporaryDirectory() as name,pymupdf.open() as document:
            page=document.new_page(width=200,height=200)
            with (patch('book_pipeline.ingestion.formula_crops.detector_path',return_value=Path('/unused')),
                  patch('mineru.model.ocr.pp_ocr_v6_onnx.TextDetectorONNX',return_value=detector),
                  patch('book_pipeline.ingestion.formula_crops.partition_cells',return_value=[[0,0,200,417],[200,0,417,417]])):
                output,review=repair_formula_images(page,[0,0,1,1],answer,Path(name),dict(image_dpi=150),client)
            self.assertEqual(output,answer)
            self.assertIsNot(output,answer)
            self.assertEqual(review['reviewed'],1)
            self.assertEqual(review['repaired'],0)
            client.ask.assert_called_once()
            self.assertEqual(client.ask.call_args.args[0], 'review_joint_formula_crop_geometry')
            self.assertEqual(review['cells'][1]['status'],'cpu_gap_check_only')
            self.assertEqual(review['cells'][1]['image_blocks'],[])
