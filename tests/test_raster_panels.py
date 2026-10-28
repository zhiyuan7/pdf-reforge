"""Image evidence and source ownership, including immutable OCR reuse."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import digest, read, save
from book_pipeline.ingestion.raster_panels import annotate_raster_panels, recover_raster_panels
from book_pipeline.ingestion.source import load_source


class RasterPanelTests(unittest.TestCase):
    def image_page(self, document):
        with pymupdf.open() as source:
            p = source.new_page(width=500, height=700)
            p.draw_rect((40, 100, 460, 230), color=(0, .6, .3), fill=(.94, .98, .96))
            p.draw_rect((50, 90, 450, 120), color=(0, .6, .3), fill=(0, .6, .3))
            p.insert_text((60, 110), 'A source title', fontsize=10, color=(1, 1, 1))
            p.insert_text((60, 155), 'Body text remains searchable.', fontsize=10)
            p.insert_text((180, 200), 'x + 0 = x', fontsize=10)
            page = document.new_page(width=500, height=700)
            page.insert_image(page.rect, stream=p.get_pixmap(dpi=150).tobytes('png'))
            return page

    def blocks(self):
        return [dict(index=n, type=kind, bbox=[x/500, y/700, z/500, h/700], content=text)
                for n, (kind, x, y, z, h, text) in enumerate([
                    ('paragraph_title', 60, 98, 250, 113, 'A source title'),
                    ('text', 60, 140, 260, 160, 'Body text remains searchable.'),
                    ('equation', 180, 185, 260, 205, 'x+0=x')])]

    def test_observed_strip_and_contiguous_title_prose_and_formula_have_one_owner(self):
        with pymupdf.open() as d:
            p = self.image_page(d)
            blocks = self.blocks()
            text = [b['content'] for b in blocks]
            rows = annotate_raster_panels(p, blocks)
            self.assertEqual([r['status'] for r in rows], ['annotated'])
            self.assertEqual([b['source_panel_title'] for b in blocks], [True, False, False])
            self.assertEqual(len({b['source_panel']['id'] for b in blocks}), 1)
            self.assertEqual([b['content'] for b in blocks], text)
            self.assertAlmostEqual(blocks[0]['source_panel']['color'][1], .6, delta=.015)

    def test_assets_code_partial_and_intervening_content_are_rejected(self):
        with pymupdf.open() as d:
            p = self.image_page(d)
            for problem in ('image', 'table', 'code', 'partial', 'intervening', 'no_title'):
                with self.subTest(problem=problem):
                    blocks = self.blocks()
                    if problem in {'image', 'table', 'code'}:
                        blocks[1]['type'] = problem
                    elif problem == 'partial':
                        blocks[1]['bbox'][0] = 0
                        blocks[1]['bbox'][2] = .99
                    elif problem == 'intervening':
                        blocks.insert(1, dict(index=5, type='text', bbox=[.05,.5,.6,.55], content='Outside'))
                    else:
                        blocks[0]['type'] = 'text'
                    before = deepcopy(blocks)
                    rows = annotate_raster_panels(p, blocks)
                    self.assertEqual(rows[0]['status'], 'rejected')
                    self.assertEqual(blocks, before)

    def test_live_text_is_not_reprocessed_as_a_raster_panel(self):
        with pymupdf.open() as d:
            p = self.image_page(d)
            p.insert_text((40, 400), 'Native text layer')
            blocks = self.blocks()
            self.assertEqual(annotate_raster_panels(p, blocks), [])
            self.assertFalse(any(b.get('source_panel') for b in blocks))

    def test_old_ocr_reuse_is_derived_and_signature_checked_without_changing_raw_source(self):
        with tempfile.TemporaryDirectory() as name, pymupdf.open() as d:
            root = Path(name)
            pdf = root/'input.pdf'
            self.image_page(d)
            d.save(pdf)
            origin = root/'ocr'; derived = root/'panels'
            blocks = self.blocks()
            blocks[-1]['content'] = [dict(type='equation', content='x+0=x')]
            save(origin/'middle_json.json', dict(schema='docvortex.middle', schema_version='2.0',
                 pages=[dict(page_idx=0, blocks=blocks)]))
            save(origin/'manifest.json', dict(input_pdf=str(pdf), input_sha256=digest(pdf), mineru_version='test',
                 middle_sha256=digest(origin/'middle_json.json')))
            hashes = [digest(pdf), digest(origin/'middle_json.json'), digest(origin/'manifest.json')]
            before = load_source(pdf, origin)
            self.assertEqual(recover_raster_panels(pdf, origin, derived), derived)
            after = load_source(pdf, derived)
            self.assertNotEqual(before['mineru_sha256'], after['mineru_sha256'])
            self.assertEqual([(a['kind'], a['content']) for a in before['atoms']],
                             [(a['kind'], a['content']) for a in after['atoms']])
            self.assertTrue(all(a.get('source_panel') for a in after['atoms']))
            from book_pipeline.ingestion.panels import panel_environments
            from book_pipeline.ingestion.source import SourceIndex
            wrappers, evidence = panel_environments(SourceIndex(after), [])
            self.assertEqual(len(wrappers), 1)
            self.assertEqual(evidence[0]['method'], 'raster_color_text_panels_v3')
            self.assertEqual(hashes, [digest(pdf), digest(origin/'middle_json.json'), digest(origin/'manifest.json')])
            self.assertEqual(recover_raster_panels(pdf, origin, derived), derived)
            saved = read(derived/'manifest.json'); saved['raster_panel_recovery']['signature'] = 'old-version'
            save(derived/'manifest.json', saved)
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                recover_raster_panels(pdf, origin, derived)

    def test_dark_title_separator_joins_but_white_gutter_does_not(self):
        from book_pipeline.ingestion.raster_panels import color_regions
        with pymupdf.open() as document, pymupdf.open() as raster:
            page = document.new_page(width=500, height=700)
            for y, gap_color in [(80, (0,0,0)), (350, (1,1,1))]:
                body_top=y if gap_color==(0,0,0) else y+33
                page.draw_rect((40,body_top,460,y+130), color=None, fill=(1,.98,.76))
                page.draw_rect((40,y,460,y+30), color=None, fill=(1,.97,.6))
                if gap_color==(0,0,0):
                    page.draw_line((40,y+30),(460,y+30),color=gap_color,width=.7)
            p=raster.new_page(width=500,height=700)
            p.insert_image(p.rect,stream=page.get_pixmap(dpi=150).tobytes('png'))
            regions=color_regions(p)
            joined=[r for r in regions if r['bbox'][1]<.3]
            separate=[r for r in regions if r['bbox'][1]>.4]
            self.assertEqual(len(joined),1)
            self.assertIsNotNone(joined[0]['title_bbox'])
            self.assertEqual(len(separate),2)
