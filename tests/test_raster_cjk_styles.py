"""Independent CJK line, phrase, font competition and ownership controls."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pymupdf

from book_pipeline.publishing.raster_cjk_styles import (FONTDIR, NORMAL, EMPHASIS,
    analyze_note, emphasis_spans, glyph_grid, glyph_scores, glyph_templates, owned_note_ranges)
from book_pipeline.publishing.raster_styles import merge_style_evidence


class RasterCJKStyleTests(unittest.TestCase):
    def scores(self, heavy):
        row = dict.fromkeys((*NORMAL,*EMPHASIS), .3)
        row['FandolHei-Regular' if heavy else 'FandolKai-Regular'] = .85
        return row

    def observation(self):
        text = '普通说明随机变量'
        scores = [self.scores(False) for _ in range(4)] + [self.scores(True) for _ in range(4)]
        word = dict(text=text[4:], score=.999, start=4,end=8,bbox=[400,200,600,250],glyph_scores=scores[4:])
        return [dict(text=text,score=.999,detection_score=.999,bbox=[200,200,600,250],
                     words=[word],glyph_scores=scores)]

    def test_each_emphasized_glyph_needs_strong_shape_margin_and_nearby_kai(self):
        scores = [self.scores(False) for _ in range(4)] + [self.scores(True) for _ in range(4)]
        self.assertEqual(emphasis_spans(scores),[(4,8)])
        for defect in ('uniform_heavy','weak','normal_competitor','unknown','single','distant'):
            rows = deepcopy(scores)
            if defect == 'uniform_heavy': rows = rows[4:]
            if defect == 'weak':
                for r in rows[4:]: r['FandolHei-Regular'] = .74
            if defect == 'normal_competitor':
                for r in rows[4:]: r['FandolSong-Regular'] = .8
            if defect == 'unknown':
                for r in rows[4:]: r.pop('FandolKai-Regular')
            if defect == 'single': rows=rows[:5]
            if defect == 'distant': rows=rows[:4]+[dict.fromkeys((*NORMAL,*EMPHASIS),.3)]*3+rows[4:]
            self.assertEqual(emphasis_spans(rows),[],defect)

    def test_native_font_hypotheses_do_not_mark_regular_controls(self):
        if not (FONTDIR/'FandolKai-Regular.otf').is_file():self.skipTest('Bundled CJK fonts unavailable')
        for char in '随机变量说明':
            for name, ink in glyph_templates(char).items():
                resized=cv2.resize(ink,None,fx=.7,fy=.7,interpolation=cv2.INTER_AREA)
                _,binary=cv2.threshold(resized,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
                scores=glyph_scores(binary,char)
                if name in NORMAL:
                    self.assertEqual(emphasis_spans([self.scores(False)]*2+[scores]*2),[],(char,name,scores))

    def test_grid_cannot_cut_connected_glyphs_or_use_unbounded_advance(self):
        ink=np.zeros((30,160),np.uint8)
        for x in (3,43,83,123):ink[3:27,x:x+34]=255
        self.assertEqual(len(glyph_grid(ink,4)),4)
        ink[:,36:85]=255
        self.assertEqual(glyph_grid(ink,4),[])
        self.assertEqual(glyph_grid(np.ones((30,160),np.uint8),4),[])
        self.assertEqual(glyph_grid(np.zeros((5,20),np.uint8),4),[])

    def test_raster_font_switch_requires_independent_phrase_and_both_line_readings(self):
        if not (FONTDIR/'FandolKai-Regular.otf').is_file():self.skipTest('Bundled CJK fonts unavailable')
        text='普通说明随机变数'
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=230,height=85)
            page.draw_rect(page.rect,color=None,fill=(.92,.92,.92))
            writer=pymupdf.TextWriter(page.rect)
            for x,part,font in ((15,text[:4],'FandolKai-Regular'),(103,text[4:],'FandolHei-Regular')):
                writer.append((x,55),part,font=pymupdf.Font(fontfile=str(FONTDIR/(font+'.otf'))),fontsize=22)
            writer.write_text(page)
            pix=page.get_pixmap(dpi=144,alpha=False,colorspace=pymupdf.csRGB)
            rgb=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
        class Model:
            def __init__(self,confidence=.999,word=text[4:],line_confidence=.999):
                self.calls=0;self.confidence=confidence;self.word=word;self.line_confidence=line_confidence
            def ocr(self,image,det=True):
                if det:return [[[[[28,66],[384,66],[384,120],[28,120]],(text,self.line_confidence)]]]
                self.calls+=1
                return [[(text,.999)]] if self.calls==1 else [[(self.word,self.confidence)]]
        rows=analyze_note(rgb,Model())
        self.assertEqual([w['text'] for r in rows for w in r['words']],[text[4:]])
        self.assertEqual(analyze_note(rgb,Model(confidence=.98)),[])
        self.assertEqual(analyze_note(rgb,Model(word='不同文字')),[])
        self.assertEqual(analyze_note(rgb,Model(line_confidence=.98)),[])

    def test_ownership_revalidates_both_ocr_readings_each_glyph_and_context(self):
        obs=self.observation();items=[dict(atom='a',text='前文。普通说明随机变量。后文。',bbox=[.1,.1,.7,.4])]
        runs=owned_note_ranges(obs,items,[.1,.1,.7,.4],[1000]*4)['a']
        self.assertEqual(runs[0]['quote'],'随机变量')
        for defect in ('line_confidence','detection','word_confidence','word_text','ownership','geometry','context','glyph','duplicate','repeated'):
            rows=deepcopy(obs);batch=deepcopy(items)
            if defect=='line_confidence':rows[0]['score']=.98
            if defect=='detection':rows[0]['detection_score']=.98
            if defect=='word_confidence':rows[0]['words'][0]['score']=.98
            if defect=='word_text':rows[0]['words'][0]['text']='随机变更'
            if defect=='ownership':batch[0]['text']='改写后的正文。'
            if defect=='geometry':rows[0]['bbox']=[700,200,900,250]
            if defect=='context':rows[0]['glyph_scores']=[self.scores(True)]*8
            if defect=='glyph':rows[0]['words'][0]['glyph_scores'][0]['FandolHei-Regular']=.74
            if defect=='duplicate':batch*=2
            if defect=='repeated':batch[0]['text']*=2
            self.assertEqual(owned_note_ranges(rows,batch,[.1,.1,.7,.4],[1000]*4),{},defect)

    def test_agreeing_api_and_cpu_styles_coalesce_and_conflicts_fail(self):
        text='普通说明随机变量。'
        def run(a,b,bold=True):return dict(method='test',start=a,end=b,quote=text[a:b],italic=False,bold=bold)
        self.assertEqual(merge_style_evidence(text,[run(4,8)],[run(4,8)]),[run(4,8)])
        self.assertEqual(merge_style_evidence(text,[run(4,7)],[run(6,8)]),[run(4,8)])
        self.assertEqual(merge_style_evidence(text,[run(4,8)],[]),[run(4,8)])
        with self.assertRaises(ValueError):merge_style_evidence(text,[run(4,8)],[run(6,8,False)])


if __name__=='__main__':unittest.main()
