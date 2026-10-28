"""A second CPU reading cannot bypass font, context or confidence guards."""
from copy import deepcopy
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from book_pipeline.publishing.latin_word_ocr import METHOD, confirmed_word, read_word, single_word
from book_pipeline.publishing.raster_body_styles import analyze_page, supported_words


def tsv(text='firm', confidence='96.3'):
    return ('level\tconf\ttext\n1\t-1\t\n5\t'+confidence+'\t'+text+'\n')


def evidence(text='firm'):
    return dict(text=text, score=.96,
                recognition=dict(method=METHOD, signature='model-sha', texts=[text]*2, scores=[.97, .96]))


class LatinWordOCRTests(unittest.TestCase):
    def test_raw_confidence_and_two_literal_readings_are_required(self):
        self.assertEqual(single_word(tsv()), ('firm', .963))
        for value in (tsv(confidence='94.99'), tsv(confidence='nan'), tsv(confidence='101'),
                      tsv('fi rm'), tsv('rm'), tsv('x=1'), tsv()+'5\t99\textra\n', 'bad header'):
            self.assertIsNone(single_word(value), value)
        word=evidence()
        self.assertTrue(confirmed_word(word))
        self.assertFalse(confirmed_word(dict(text='firm', score=.96)))
        for change in (dict(texts=['firm','form']), dict(scores=[.97,.94]),
                       dict(scores=[.97]), dict(signature=''), dict(method='other-reader')):
            altered=deepcopy(word);altered['recognition'].update(change)
            self.assertFalse(confirmed_word(altered),change)
        self.assertFalse(confirmed_word(dict(word,score=.999)))

    def test_image_only_engine_agreement_and_failures(self):
        module='book_pipeline.publishing.latin_word_ocr.'
        rgb=np.full((30,80,3),255,np.uint8)
        with patch(module+'runtime_signature',return_value='sig'), patch(module+'subprocess.run') as run:
            run.side_effect=[SimpleNamespace(stdout=tsv().encode()), SimpleNamespace(stdout=tsv(confidence='95.2').encode())]
            result=read_word(rgb)
            self.assertEqual(result['text'],'firm')
            self.assertTrue(confirmed_word(result))
            for call in run.call_args_list:
                self.assertEqual(call.args[0][:3],['tesseract','stdin','stdout'])
                self.assertNotIn('firm',call.args[0])
            for second in (tsv('form'), tsv(confidence='90'), tsv()+'5\t99\textra\n'):
                run.side_effect=[SimpleNamespace(stdout=tsv().encode()),SimpleNamespace(stdout=second.encode())]
                self.assertIsNone(read_word(rgb))
            run.side_effect=subprocess.TimeoutExpired('tesseract',5)
            self.assertIsNone(read_word(rgb))
        with patch(module+'runtime_signature',return_value=None),patch(module+'subprocess.run') as run:
            self.assertIsNone(read_word(rgb));run.assert_not_called()

    def test_ligature_recovery_still_needs_complete_line_and_italic_ink(self):
        module='book_pipeline.publishing.raster_body_styles.'
        text='The output requires a firm categorical choice.'
        rgb=np.full((40,200,3),255,np.uint8)
        class Reader:
            def ocr(self, image, det=True):
                if det:
                    return [[([[0,0],[200,0],[200,40],[0,40]],(text,.999))]]
                return [[('rm',.999)]]
        scores=dict(tiro=.2,tibo=.3,tiit=.8,tibi=.6)
        with patch(module+'word_boxes',return_value=[(0,200)]),patch(module+'read_word',return_value=evidence()) as reader, \
             patch(module+'scored_scan_word',return_value=('firm',scores)):
            observations=analyze_page(rgb,Reader())
            self.assertEqual(observations[0]['words'][0]['text'],'firm')
            self.assertEqual(observations[0]['words'][0]['score'],.96)
            self.assertTrue(reader.called)
            for replacement in ('form','firm firm'):
                text='The output requires a '+replacement+' categorical choice.'
                self.assertEqual(analyze_page(rgb,Reader()),[])
        word=dict(evidence(),scores=scores,bbox=[0,0,80,30])
        self.assertEqual(supported_words([word],'a firm choice'),[word])
        word['scores']=dict(scores,tiro=.9)
        self.assertEqual(supported_words([word],'a firm choice'),[])


if __name__=='__main__':
    unittest.main()
