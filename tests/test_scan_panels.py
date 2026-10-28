"""Closed scan frames and hidden OCR must not depend on book labels."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config, read, save
from book_pipeline.ingestion.raster_panels import recover_raster_panels
from book_pipeline.ingestion.scan_panels import monochrome_regions, raster_page
from book_pipeline.ingestion.source import load_source, SourceIndex
from book_pipeline.ingestion.panels import panel_environments
from test_raster_prose import fixture


class MonoClient:
    calls = 0
    ledger = None

    def __init__(self): self.ledger = self
    def summary(self): return {}
    def ask(self, task, system, payload, images, validator, **kwargs):
        self.calls += 1
        assert payload['title_strip'] is None
        answer = dict(classification='prose_panel', reason='Observed prose and standalone math', lines=[
            dict(id=n, kind=kind, bbox=box) for n, (kind, box) in enumerate([
                ('text', [.12,.143,.8,.218]), ('display', [.36,.25,.53,.28]),
                ('text', [.12,.3,.8,.326])])])
        validator(answer)
        return answer


class ScanPanelTests(unittest.TestCase):
    def page(self, d, incomplete=False, grid=False):
        with pymupdf.open() as original:
            p = original.new_page(width=500, height=700)
            if incomplete:
                p.draw_line((40,90),(460,90),color=(0,0,0),width=1.5)
                p.draw_line((40,90),(40,240),color=(0,0,0),width=1.5)
            else:
                p.draw_rect((40,90,460,240),color=(0,0,0),width=1.5)
            if grid: p.draw_line((40,160),(460,160),color=(0,0,0),width=1.5)
            for x,y,text in [(60,110,'Inline label. Body text remains in its paragraph.'),
                             (60,150,'A source paragraph states that'),(180,190,'x + 0 = x.'),
                             (60,225,'A following source paragraph.')]:
                p.insert_text((x,y),text,fontsize=10)
            page = d.new_page(width=500,height=700)
            page.insert_image(page.rect,stream=p.get_pixmap(dpi=150).tobytes('png'))
            return page

    def test_only_hidden_ocr_over_full_page_raster_is_allowed(self):
        with pymupdf.open() as d:
            p=self.page(d)
            self.assertTrue(raster_page(p))
            p.insert_text((20,350),'Old inaccurate OCR layer',render_mode=3)
            self.assertTrue(raster_page(p))
            p.insert_text((20,400),'Visible body text')
            self.assertFalse(raster_page(p))
            p=d.new_page();p.insert_text((20,40),'Hidden text without scan',render_mode=3)
            self.assertFalse(raster_page(p))

    def test_four_closed_sides_required_and_grid_rejected(self):
        with pymupdf.open() as d:
            rows=monochrome_regions(self.page(d))
            self.assertEqual(len(rows),1)
            self.assertIsNone(rows[0]['title_bbox'])
            self.assertTrue(rows[0]['frame_visible'])
            self.assertEqual(monochrome_regions(self.page(d,incomplete=True)),[])
            self.assertEqual(monochrome_regions(self.page(d,grid=True)),[])

    def test_monochrome_algorithm_is_reviewed_without_changing_original_prose(self):
        with tempfile.TemporaryDirectory() as name, pymupdf.open() as d:
            root=Path(name);pdf,origin=fixture(root)
            p=self.page(d);p.insert_text((20,350),'Hidden archive OCR',render_mode=3)
            pdf.unlink();d.save(pdf)
            manifest=read(origin/'manifest.json')
            from book_pipeline.config import digest
            manifest.update(input_sha256=digest(pdf));save(origin/'manifest.json',manifest)
            middle=read(origin/'middle_json.json')
            content=middle['pages'][0]['blocks'][0]['content'][0]['content']
            content[0]['content']='Inline label. Body text remains in its paragraph. A source paragraph states that\n    '
            save(origin/'middle_json.json',middle)
            raw=load_source(pdf,origin)
            client=MonoClient();dest=recover_raster_panels(pdf,origin,root/'derived',load_config(),client=client)
            self.assertNotEqual(dest,origin)
            after=load_source(pdf,dest)
            self.assertEqual(''.join(a['content'] for a in raw['atoms']),''.join(a['content'] for a in after['atoms']))
            self.assertTrue(all(a.get('source_panel') for a in after['atoms']))
            self.assertFalse(any(a.get('source_panel_title') for a in after['atoms']))
            wrappers,rows=panel_environments(SourceIndex(after),[])
            self.assertEqual(len(wrappers),1)
            self.assertTrue(wrappers[0]['panel_frame_visible'])
            self.assertNotIn('body_start',wrappers[0])
            self.assertEqual(client.calls,1)
