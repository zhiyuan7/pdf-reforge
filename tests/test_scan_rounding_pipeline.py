"""A missing local edit must not discard independent rounding agreement."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.ingestion.formula_bands import correct_formula_bands
from book_pipeline.ingestion.source import SourceIndex


class RoundingPipelineTests(unittest.TestCase):
    def test_cpu_and_blind_formula_agreement_triggers_and_invalidates_cache(self):
        base = dict(items=[dict(kind='formula',latex=r'T_p\leq[\frac{n_\tau}{p}]',number='')])
        witness = deepcopy(base)
        witness['items'][0]['latex'] = r'T_{p}\le\lceil\frac{n_{\tau}}{p}\rceil'
        calls=[]
        outer=self
        class Client:
            _api=None
            def __init__(self,*args,**kwargs):
                self.ledger=self
                self.reads=0
            def summary(self):return {}
            def ask(self,task,prompt,payload,**kwargs):
                calls.append(task)
                if task.startswith('literal_scan_band'):
                    output=dict(edits=[],unresolved=[])
                else:
                    self.reads+=1
                    output=deepcopy(base if self.reads==1 else witness)
                    outer.assertNotIn('items',payload)
                kwargs['validator'](output)
                return output
        with tempfile.TemporaryDirectory() as folder, pymupdf.open() as pdf:
            path=Path(folder)/'input.pdf'
            pdf.new_page(width=400,height=300)
            pdf.save(path)
            content=r'T _ {p}\leq\lceil\frac{n _ {\tau}}{p}\rceil'
            index=SourceIndex(dict(atoms=[dict(id='a',group='a',kind='display_math',page=0,
                content=content,length=len(content),extraction_method='ocr')],
                input_sha256='p',mineru_sha256='m',source_pdf=str(path)))
            band=dict(id='b',page=0,bbox=[0,.2,1,.5],groups=['a'],preserve=[])
            config=dict(model='fake',base_url='https://example.invalid',thinking='disabled',concurrency=1)
            prefix='book_pipeline.ingestion.formula_bands.'
            with patch(prefix+'detect_bands',side_effect=lambda *args:[dict(band)]), \
                    patch(prefix+'DeepSeekClient',Client), \
                    patch('book_pipeline.ingestion.scan_formula_anchors.recover_anchors',
                          side_effect=lambda answer,*args:(deepcopy(answer),{})):
                destination=Path(folder)/'output'
                report=correct_formula_bands(index,config,destination)
                record=report['records'][0]
                self.assertEqual(record['answer']['items'][0]['latex'],r'T_p\leq\lceil\frac{n_\tau}{p}\rceil')
                self.assertEqual(len(record['rounding_consensus']['changes']),1)
                self.assertEqual(len(calls),3)
                self.assertEqual(index.atoms[0]['content'],content)
                correct_formula_bands(index,config,destination)
                self.assertEqual(len(calls),3)
                with patch('book_pipeline.ingestion.scan_rounding_consensus.METHOD','changed-policy'):
                    correct_formula_bands(index,config,destination)
                self.assertEqual(len(calls),6)
