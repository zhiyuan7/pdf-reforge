"""Text panels must not become duplicate screenshots; diagrams remain figures."""
import unittest
import tempfile
from pathlib import Path
import pymupdf
from book_pipeline.ingestion.native import visual_regions, validate_layout, bounded_image_box
from book_pipeline.ingestion.panels import text_panels, panel_environments
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.config import load_config, executable


class NativePanelTests(unittest.TestCase):
    def test_white_rectangular_knockout_is_not_an_illustration_or_panel(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=500, height=700)
            page.draw_rect((50, 150, 450, 300), fill=(1, 1, 1), color=None)
            page.insert_text((65, 200), 'Ordinary text on a white knockout background.')
            self.assertEqual(text_panels(page), [])
            self.assertEqual(visual_regions(page), [])
            # White paper must not suppress actual diagram strokes above it.
            page.draw_line((80, 240), (200, 270))
            self.assertTrue(visual_regions(page))

    def test_small_vector_graph_inside_white_knockout_keeps_its_own_bounds(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=504, height=720)
            page.draw_rect((230, 550, 273, 601), fill=(1, 1, 1), color=None)
            page.draw_line((237, 590), (268, 555))
            regions = visual_regions(page)
            self.assertEqual(len(regions), 1)
            self.assertGreater(regions[0][0], 230/504)
            self.assertLess(regions[0][2], 273/504)

    def test_filled_circle_with_formula_labels_remains_an_illustration(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=500, height=700)
            page.draw_circle((250, 250), 90, fill=(.75, .75, .75))
            page.insert_text((200, 250), 'radius and norm labels')
            self.assertEqual(text_panels(page), [])
            self.assertTrue(visual_regions(page))

    def test_short_overhanging_title_and_separate_outline_share_one_panel(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=500, height=700)
            page.draw_rect((50, 180, 450, 300), fill=(.88, .95, .98), color=None)
            page.draw_rect((50.1, 180.1, 449.9, 299.9), color=(.7, .88, .95), width=1)
            page.draw_rect((65, 170, 260, 195), fill=(.7, .88, .95), color=None)
            page.insert_text((75, 187), 'A short theorem title')
            page.insert_text((65, 230), 'A source-owned theorem with its original body text.')
            panels = text_panels(page)
            self.assertEqual(len(panels), 1)
            panel = panels[0]
            self.assertTrue(panel['frame_visible'])
            self.assertEqual(panel['frame_color'], [.7, .88, .95])
            for actual, expected in zip(panel['title_bbox'], [65/500, 170/700, 260/500, 195/700]):
                self.assertAlmostEqual(actual, expected, places=5)
            self.assertEqual(panel['background_color'], [.88, .95, .98])
            self.assertEqual(visual_regions(page), [])

    def test_white_rounded_panel_and_white_title_keep_colored_outline(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=500, height=700)
            page.draw_rect((50, 180, 450, 340), fill=(1, 1, 1), color=None, radius=.08)
            page.draw_rect((49.7, 179.7, 450.3, 340.3), color=(0, .23, .23), width=2, radius=.08)
            page.draw_rect((65, 170, 160, 195), fill=(1, 1, 1), color=None)
            page.insert_text((75, 187), 'Investigate!')
            page.insert_text((65, 230), 'Explore the planar graph using the original argument.')
            panels = text_panels(page)
            self.assertEqual(len(panels), 1)
            self.assertTrue(panels[0]['frame_visible'])
            self.assertEqual(panels[0]['frame_color'], [0, .23, .23])
            self.assertEqual(panels[0]['background_color'], [1, 1, 1])
            self.assertIsNotNone(panels[0]['title_bbox'])
            self.assertEqual(visual_regions(page), [])

    def test_saved_panel_without_title_rechecks_live_outline_before_compilation(self):
        for outlined in (False,True):
            with self.subTest(outlined=outlined),tempfile.TemporaryDirectory() as name:
                root=Path(name)
                with pymupdf.open() as pdf:
                    page=pdf.new_page(width=500,height=700)
                    page.draw_rect((60,100,440,220),fill=(.98,.95,.93),
                                   color=(.8,.2,.18) if outlined else None,width=.8)
                    page.insert_text((75,150),'Source-owned theorem body with no separate title strip.')
                    panel=text_panels(page)[0]
                    self.assertEqual(panel['frame_visible'],outlined)
                    # Model a saved extraction made before the evidence field.
                    panel.pop('frame_visible')
                    pdf.save(root/'source.pdf')
                text='Source-owned theorem body with no separate title strip.'
                index=SourceIndex(dict(source_pdf=str(root/'source.pdf'),atoms=[dict(id='body',group='body',
                    page=0,kind='text',role='text',block_type='text',content=text,length=len(text),
                    bbox=[.15,.19,.85,.22],extraction_method='native',source_panel=panel)]))
                wrappers,_=panel_environments(index,[])
                self.assertEqual(bool(wrappers[0].get('panel_frame_visible')),outlined)
                audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                    [],[],dict(title='Panel',author=''),load_config(),root/'latex',sample_mode=True)
                self.assertTrue(audit['coverage_complete'])
                result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(result['success'],result['log_tail'])
                with pymupdf.open(root/'latex/main.pdf') as pdf:
                    frames=[d for d in pdf[0].get_drawings() if (d.get('color') or d.get('fill'))
                        and (d.get('color') or d['fill'])[0]>.7
                        and (d.get('color') or d['fill'])[1]<.3 and d['rect'].height>10]
                    self.assertEqual(bool(frames),outlined)

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
