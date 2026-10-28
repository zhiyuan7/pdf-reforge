"""Text panels must not become duplicate screenshots; diagrams remain figures."""
import unittest
import pymupdf
from book_pipeline.ingestion.native import visual_regions, validate_layout, bounded_image_box


class NativePanelTests(unittest.TestCase):
    def test_crop_grazing_neighbor_panel_moves_only_through_proven_empty_gap(self):
        for condition in ('clear', 'crossing_stroke', 'owned_label', 'embedded_image', 'no_art'):
            with self.subTest(condition=condition), pymupdf.open() as pdf:
                p=pdf.new_page(width=400,height=700)
                p.draw_rect((40,120,360,220),fill=(.9,.9,.9))
                p.insert_text((60,160),'A note with source-owned explanatory prose.')
                if condition != 'no_art':p.draw_line((80,380),(320,260),color=(1,0,0))
                if condition != 'no_art':p.draw_polyline([(315,262),(320,260),(318,267)],fill=(1,0,0),color=None,closePath=True)
                if condition=='crossing_stroke':p.draw_line((100,219),(110,270))
                spans={};parts=[]
                if condition=='owned_label':
                    spans={'label':dict(bbox=[.3,219/700,.4,230/700])}
                    parts=[dict(spans=['label'])]
                if condition=='embedded_image':
                    pix=pymupdf.Pixmap(pymupdf.csRGB,(0,0,2,2));pix.clear_with(0)
                    p.insert_image((100,219,110,270),pixmap=pix)
                row=dict(bbox=[.1,218/700,.9,.6],parts=parts)
                bbox,evidence=bounded_image_box(p,row,spans)
                if condition=='clear':
                    self.assertAlmostEqual(bbox[1],221/700,delta=.01/700)
                    self.assertEqual(evidence['original_bbox'],row['bbox'])
                    self.assertEqual(bbox[0],row['bbox'][0]);self.assertEqual(bbox[2:],row['bbox'][2:])
                    self.assertEqual(p.get_pixmap(clip=pymupdf.Rect(40,bbox[1]*700,360,420)).pixel(1,1),(255,255,255))
                else:
                    self.assertEqual(bbox,row['bbox']);self.assertIsNone(evidence)

    def test_filled_text_panel_is_not_an_illustration_candidate(self):
        with pymupdf.open() as document:
            page = document.new_page()
            page.draw_rect((50,50,400,150), color=(.2,.4,.2), fill=(.95,1,.95))
            page.insert_text((60,90), 'Definition: a vector space is a set with two operations.')
            self.assertEqual(visual_regions(page), [])

    def test_unlabelled_vector_diagram_is_still_a_candidate(self):
        with pymupdf.open() as document:
            page = document.new_page()
            page.draw_rect((50,50,300,250), color=(0,0,0))
            self.assertTrue(visual_regions(page))

    def test_short_filled_title_strip_is_not_a_figure(self):
        with pymupdf.open() as document:
            page = document.new_page()
            page.draw_rect((50,50,400,70), color=(0,.5,0), fill=(0,.5,0))
            page.insert_text((60,65), '1.15 Zero', color=(1,1,1))
            self.assertEqual(visual_regions(page), [])

    def test_image_cannot_duplicate_a_separately_extracted_span(self):
        spans = [dict(id='s0', bbox=[.2,.2,.3,.3])]
        value = dict(blocks=[dict(type='image', bbox=[.1,.1,.5,.5], parts=[]),
                             dict(type='text', bbox=[.2,.2,.3,.3], parts=[dict(kind='text', spans=['s0'])])])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_layout(value, spans)
