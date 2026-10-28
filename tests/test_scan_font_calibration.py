"""Archive typography requires page evidence and independent metric fits."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pymupdf

from book_pipeline.publishing import scan_font_calibration as scan
from book_pipeline.publishing.raster_body_styles import is_bold, is_italic


def samples():
    return [dict(text='word'+chr(97+n), bbox=[0, n*15, 60, n*15+12],
        scores={'computer_modern': [.91, .8, .3, .3], 'times': [.7, .65, .3, .3]})
        for n in range(16)]


class ScanFontCalibrationTests(unittest.TestCase):
    def test_metric_weight_requires_both_views_and_independent_axis_margins(self):
        bold = [.4, .94, .4, .6]
        scores = scan.metric_style_scores(bold, bold)
        self.assertTrue(is_bold(scores))
        self.assertFalse(is_italic(scores))
        bold_italic = [.4, .5, .6, .95]
        scores = scan.metric_style_scores(bold_italic, bold_italic)
        self.assertTrue(is_bold(scores))
        self.assertTrue(is_italic(scores))
        for negative in ([], [.4, .84, .4, .5], [.8, .94, .4, .5], [.94, .4, .5, .4]):
            self.assertEqual(scan.metric_style_scores(bold, negative), {})
            self.assertEqual(scan.metric_style_scores(negative, bold), {})
        italic = [.4, .5, .94, .6]
        self.assertTrue(is_italic(scan.metric_style_scores(italic, italic)))
        self.assertFalse(is_bold(scan.metric_style_scores(italic, italic)))

    def test_metric_fit_rejects_ink_at_each_crop_boundary(self):
        ink = np.zeros((25, 100), np.uint8)
        ink[4:21, 5:95] = 220
        self.assertTrue(scan.complete_word_ink(ink))
        for y, x in ((0, 10), (-1, 10), (10, 0), (10, -1)):
            clipped = ink.copy()
            clipped[y, x] = 220
            self.assertFalse(scan.complete_word_ink(clipped))
            self.assertEqual(scan.metric_scores(clipped, 'appears', 'computer_modern'), [])

    @unittest.skipUnless((scan.FONTDIR/'cmr10.pfb').is_file(), 'TeX fonts required')
    def test_scan_gain_does_not_replace_bold_width_and_real_bold_is_recovered(self):
        for name in ('cmr10', 'cmbx10', 'cmti10', 'cmbxti10', 'tiro', 'tibo', 'tiit', 'tibi'):
            for text in ('event', 'sequence', 'number'):
                with pymupdf.open() as pdf:
                    page = pdf.new_page(width=180, height=45)
                    kwargs = (dict(fontname='source', fontfile=str(scan.FONTDIR/(name+'.pfb')))
                              if name.startswith('cm') else dict(fontname=name))
                    page.insert_text((10, 30), text, fontsize=13, **kwargs)
                    pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
                    original = 255-np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
                original = cv2.resize(original, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                spread = cv2.dilate(original, np.ones((3, 3), np.uint8))
                for gain in (0, .5, 1):
                    gray = cv2.addWeighted(original, 1-gain, spread, gain, 0)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    family = 'computer_modern' if name.startswith('cm') else 'times'
                    scores = scan.metric_style_scores(scan.metric_scores(binary, text, family),
                                                       scan.metric_scores(gray, text, family))
                    with self.subTest(name=name, text=text, gain=gain):
                        # Ambiguous gain and short Times words may abstain;
                        # none may acquire an unsupported style axis.
                        if is_bold(scores):
                            self.assertIn(name, {'cmbx10', 'cmbxti10', 'tibo', 'tibi'})
                        if is_italic(scores):
                            self.assertIn(name, {'cmti10', 'cmbxti10', 'tiit', 'tibi'})
                        # All four CM faces remain distinguishable after
                        # independent low-resolution rendering/resampling.
                        if name.startswith('cm') and gain == 0:
                            self.assertEqual(is_bold(scores), name in {'cmbx10', 'cmbxti10'})
                            self.assertEqual(is_italic(scores), name in {'cmti10', 'cmbxti10'})

    def test_family_needs_diverse_words_locations_and_competing_font_margin(self):
        data = samples()
        family, report = scan.select_family(data)
        self.assertEqual(family, 'computer_modern')
        self.assertEqual(report['weight'], 'abstained_scan_gain_ambiguous')
        for defect in ('few', 'duplicates', 'one_line', 'unmatched', 'ambiguous', 'minority'):
            altered = deepcopy(data)
            if defect == 'few': altered = altered[:11]
            if defect == 'duplicates': altered = altered[:1]*25
            if defect == 'one_line':
                for row in altered: row['bbox'][1] = 10
            if defect == 'unmatched':
                for row in altered: row['scores']['computer_modern'][:2] = [.77, .76]
            if defect == 'ambiguous':
                for row in altered: row['scores']['times'][0] = .88
            if defect == 'minority':
                for row in altered[:7]: row['scores']['times'][0] = .95
            self.assertIsNone(scan.select_family(altered)[0], defect)

    def test_weight_remains_unknown_even_when_bold_template_wins(self):
        binary = {'computer_modern': [.2, .3, .82, .95]}
        scores = scan.slant_scores(binary, binary, 'computer_modern')
        self.assertTrue(is_italic(scores))
        self.assertFalse(is_bold(scores))
        for bad in ({}, {'computer_modern': [.7, .8, .85, .9]},
                    {'computer_modern': [.2, .3, .74, .73]}):
            self.assertEqual(scan.slant_scores(binary, bad, 'computer_modern'), {})
            self.assertEqual(scan.slant_scores(bad, binary, 'computer_modern'), {})

    @unittest.skipUnless((scan.FONTDIR/'cmr10.pfb').is_file(), 'TeX fonts required')
    def test_degraded_roman_bold_and_italic_have_separate_shape_controls(self):
        # Render at a different size/resolution from the classifier templates.
        for name in ('cmr10', 'cmbx10', 'cmti10', 'cmbxti10', 'tiro', 'tibo', 'tiit', 'tibi'):
            for text in ('event', 'sequence', 'assigns', 'number', 'cross-entropy', "can't"):
                with pymupdf.open() as pdf:
                    page = pdf.new_page(width=130, height=45)
                    kwargs = (dict(fontname='source', fontfile=str(scan.FONTDIR/(name+'.pfb')))
                              if name.startswith('cm') else dict(fontname=name))
                    page.insert_text((10, 30), text, fontsize=13, **kwargs)
                    pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
                    soft = 255-np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
                soft = cv2.resize(soft, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                _, binary = cv2.threshold(soft, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                family = 'computer_modern' if name.startswith('cm') else 'times'
                score = scan.slant_scores(scan.shape_scores(binary, text), scan.shape_scores(soft, text), family)
                with self.subTest(name=name, text=text):
                    self.assertFalse(is_bold(score))
                    italic = name in {'cmti10', 'cmbxti10', 'tiit', 'tibi'}
                    if text in {'cross-entropy', "can't"}:
                        # Punctuation may reduce confidence; abstention is
                        # valid, but upright ink cannot acquire slant.
                        if is_italic(score): self.assertTrue(italic)
                    else:
                        self.assertEqual(is_italic(score), italic)

    def test_unconfirmed_words_repeated_tokens_and_changed_pixels_cannot_acquire_style(self):
        rgb = np.full((100, 200, 3), 255, np.uint8)
        word = dict(text='sample', score=.999, bbox=[10, 10, 100, 30])
        row = dict(text='A sample of unchanged prose', score=.999, words=[word])
        good = {'computer_modern': [.2, .3, .9, .8], 'times': [.2, .3, .7, .6]}
        with patch.object(scan, 'select_family', return_value=('computer_modern', {})), \
             patch.object(scan, 'shape_scores', return_value=good):
            self.assertEqual(scan.calibrate_observations(rgb, [row])[0][0]['words'][0]['text'], 'sample')
            for defect in ('confidence', 'line', 'duplicate', 'outside', 'missing_pixels'):
                altered = deepcopy(row)
                if defect == 'confidence': altered['words'][0]['score'] = .98
                if defect == 'line': altered['score'] = .98
                if defect == 'duplicate': altered['text'] = 'A sample of sample prose'
                if defect == 'outside': altered['words'][0]['bbox'][2] = 201
                if defect == 'missing_pixels': altered['text'] = 'A changed line of prose'
                self.assertEqual(scan.calibrate_observations(rgb, [altered])[0], [], defect)

    def test_missing_font_resources_cannot_use_legacy_times_weights(self):
        scan.runtime_signature.cache_clear()
        with patch.object(scan, 'FONTDIR', Path('/no-font-resources')):
            self.assertIsNone(scan.runtime_signature())
        scan.runtime_signature.cache_clear()

    def test_calibration_keeps_repeated_word_only_with_independent_touching_context(self):
        rgb=np.full((60,220,3),255,np.uint8)
        row=dict(text='a probability distribution or a probability measure',score=.999,words=[
            dict(text='probability',score=.999,bbox=[10,10,80,40]),
            dict(text='distribution',score=.999,bbox=[80,10,170,40])])
        good={'computer_modern':[.3,.94,.3,.4], 'times':[.2,.6,.2,.3]}
        with patch.object(scan,'select_family',return_value=('computer_modern',{})), \
             patch.object(scan,'shape_scores',return_value=good), \
             patch.object(scan,'metric_scores',return_value=[.3,.94,.3,.4]):
            result,_=scan.calibrate_observations(rgb,[row])
            self.assertEqual([w['text'] for w in result[0]['words']],['probability','distribution'])
            self.assertTrue(all(is_bold(w['scores']) for w in result[0]['words']))
            for defect in ('confidence','detached','duplicate_phrase'):
                bad=deepcopy(row)
                if defect=='confidence':bad['words'][1]['score']=.98
                if defect=='detached':bad['words'][1]['bbox'][0]+=4
                if defect=='duplicate_phrase':bad['text']+=' probability distribution'
                result,_=scan.calibrate_observations(rgb,[bad])
                self.assertNotIn('probability',[w['text'] for r in result for w in r['words']],defect)
    def test_family_only_policy_skips_metric_work_without_supplying_styles(self):
        rgb=np.full((100,200,3),255,np.uint8)
        row=dict(text='A sample of unchanged prose',score=.999,words=[
            dict(text='sample',score=.999,bbox=[10,10,100,30])])
        with patch.object(scan,'shape_scores',return_value={'times':[.9,.3,.3,.3]}), \
             patch.object(scan,'select_family',return_value=('times',dict(status='calibrated_slant_only',family='times'))), \
             patch.object(scan,'metric_scores') as metric:
            result,report=scan.calibrate_observations(rgb,[row],metric_families={'computer_modern'})
            self.assertEqual(result,[])
            self.assertEqual(report['status'],'calibrated_font_family')
            metric.assert_not_called()
