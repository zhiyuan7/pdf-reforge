"""CJK family contrast needs unchanged native text and a calligraphic note."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.native_note_styles import font_family, note_ranges, recover_native_note_styles


class NativeNoteStyleTests(unittest.TestCase):
    def fixture(self):
        note = [.1, .1, .9, .4]
        spans = [dict(text='这里是使用楷体书写的普通说明文字。', font='ABCDEF+FZKTK--GBK1-0', bbox=[.15,.15,.85,.2]),
                 dict(text='概率分布', font='FZHTK--GBK1-0', bbox=[.4,.25,.6,.3])]
        text = spans[0]['text'] + spans[1]['text'] + '保留。'
        items = [dict(atom='a', text=text, bbox=note)]
        return spans, items, note

    def test_local_heiti_in_calligraphic_prose_keeps_literal_offsets(self):
        spans, items, note = self.fixture()
        runs = note_ranges(spans, items, note)['a']
        self.assertEqual([(r['quote'], r['italic'], r['bold']) for r in runs], [('概率分布', False, True)])
        self.assertEqual(items[0]['text'][runs[0]['start']:runs[0]['end']], '概率分布')
        self.assertEqual(note_ranges(spans*2, items, note)['a'], runs)
        self.assertEqual(font_family('ABCDEF+NotoSansCJKsc-Regular'), 'sans')
        self.assertEqual(font_family('LXGWWenKaiGB-Regular'), 'calligraphic')
        self.assertIsNone(font_family('OtherSans-Regular'))

    def test_broad_main_paragraph_does_not_veto_exclusive_note_glyphs(self):
        spans, items, note = self.fixture()
        crossing = dict(atom='main', text='其他内容跨在侧注附近。', bbox=[.05,.25,.95,.55])
        self.assertIn('a', note_ranges(spans, [*items, crossing], note))
        crossing['text'] = '概率分布也在主文中出现。'
        self.assertEqual(note_ranges(spans, [*items, crossing], note), {})
        self.assertEqual(note_ranges(spans, [crossing], note), {})

    def test_uniform_sans_wrong_geometry_unknown_family_and_ambiguous_text_rejected(self):
        spans, items, note = self.fixture()
        for defect in ('uniform_sans', 'unowned', 'missing', 'unknown', 'duplicate', 'repeated', 'math', 'single'):
            altered = deepcopy(spans); candidates = deepcopy(items)
            if defect == 'uniform_sans': altered[0]['font'] = 'SimHei'
            if defect == 'unowned': altered[1]['bbox'] = [.8,.25,1,.3]
            if defect == 'missing': candidates[0]['text'] = 'Different prose'
            if defect == 'unknown': altered[1]['font'] = 'UnknownChinese'
            if defect == 'duplicate': candidates *= 2
            if defect == 'repeated': candidates[0]['text'] *= 2
            if defect == 'math': altered[1]['text'] = 'X概率分布'
            if defect == 'single': altered[1]['text'] = '概'
            self.assertEqual(note_ranges(altered, candidates, note), {}, defect)

    def test_saved_native_page_requires_original_bytes_and_rejects_scan_overlay(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'native.pdf'
            with pymupdf.open() as pdf:
                pdf.new_page().insert_text((60,100), 'Native PDF'); pdf.save(path)
            atom = dict(id='a', group='g', page=0, kind='text', role='text', block_type='text',
                        extraction_method='native', content='这是普通文字', length=6, bbox=[.1,.1,.9,.4])
            raw = SourceIndex(dict(atoms=[atom], source_pdf=str(path)))
            edited = SourceIndex(dict(raw.source, atoms=[dict(atom, original_atom='a',
                original_start_offset=0, original_end_offset=6)]))
            row = dict(a=[dict(start=2,end=4,quote='普通',bold=True,italic=False)])
            module = 'book_pipeline.publishing.native_note_styles.'
            with patch('book_pipeline.publishing.raster_decorations.gray_regions', return_value=[dict(bbox=atom['bbox'])]), \
                 patch(module+'note_ranges', return_value=row) as compare:
                self.assertEqual(recover_native_note_styles(edited, raw)[0], row)
                compare.assert_called_once()
                compare.reset_mock()
                self.assertEqual(recover_native_note_styles(edited, raw, {'a'})[0], {})
                changed = SourceIndex(dict(edited.source, atoms=[dict(edited.atoms[0], content='这是改写文字')]))
                self.assertEqual(recover_native_note_styles(changed, raw)[0], {})
                translated = SourceIndex(dict(edited.source, translation=True))
                self.assertEqual(recover_native_note_styles(translated, raw)[0], {})
                ocr = SourceIndex(dict(raw.source, atoms=[dict(atom, extraction_method='ocr')]))
                self.assertEqual(recover_native_note_styles(edited, ocr)[0], {})
                compare.assert_not_called()

    def test_real_native_font_families_recover_gray_note_without_bold_metadata(self):
        from book_pipeline.config import ROOT
        fontdir = ROOT/'.tools/.TinyTeX/texmf-dist/fonts/opentype/public/fandol'
        if not (fontdir/'FandolKai-Regular.otf').is_file():
            self.skipTest('Chinese source font fixture requires bundled TeX fonts')
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'native.pdf'
            ordinary, emphasis = '这是书中使用楷体书写的普通说明文字', '随机变量'
            with pymupdf.open() as pdf:
                page = pdf.new_page(width=600, height=800)
                page.draw_rect(pymupdf.Rect(300,80,570,240), color=None, fill=(.92,.92,.92))
                writer = pymupdf.TextWriter(page.rect)
                writer.append((310,115), ordinary, font=pymupdf.Font(fontfile=str(fontdir/'FandolKai-Regular.otf')), fontsize=10)
                writer.append((310,140), emphasis, font=pymupdf.Font(fontfile=str(fontdir/'FandolHei-Regular.otf')), fontsize=10)
                writer.write_text(page); pdf.save(path)
            text = ordinary + '\n' + emphasis
            atom = dict(id='a', group='g', page=0, kind='text', role='text', block_type='text',
                        extraction_method='native', content=text, length=len(text), bbox=[.5,.1,.95,.3])
            raw = SourceIndex(dict(atoms=[atom], source_pdf=str(path)))
            edited = SourceIndex(dict(raw.source, atoms=[dict(atom, original_atom='a',
                original_start_offset=0, original_end_offset=len(text))]))
            before = deepcopy(edited.source)
            styles, report = recover_native_note_styles(edited, raw)
            self.assertEqual([r['quote'] for r in styles['a']], [emphasis])
            self.assertEqual(styles['a'][0]['source_font'], 'FandolHei-Regular')
            self.assertEqual(report['status'], 'requires_visual_review')
            self.assertEqual(edited.source, before)


if __name__ == '__main__':
    unittest.main()
