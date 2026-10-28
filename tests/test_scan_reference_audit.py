"""Conflicting digits must survive successful extraction as a quality issue."""
from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from book_pipeline.config import digest, fingerprint
from book_pipeline.ingestion.scan_reference_audit import observe_reasons, reference_conflicts


class ScanReferenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.answer = dict(items=[dict(kind='formula', number='',
            latex=r'\text{Therefore } x/y=2 \text{by (2.3).}')])
        self.rows = [dict(id=4, pixel_bbox=[0,0,1400,50], words=[
            dict(text='Therefore', score=.99, bbox=[100,5,160,30]),
            dict(text='by', score=.92, bbox=[1100,5,40,35]),
            dict(text='(2-8).', score=.91, bbox=[1170,5,80,35])])]
        self.observation = dict(row=4, reader_signature='cpu-runtime', readings=[
            dict(text='(2-8).', score=.91, scale=1), dict(text='(2.8).', score=.89, scale=3)])

    def test_unique_word_owned_multiscale_disagreement_flags_without_editing(self):
        original = deepcopy(self.answer)
        conflicts = reference_conflicts(self.answer, self.rows, [self.observation])
        self.assertEqual(self.answer, original)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]['reference'], '2.3')
        self.assertEqual(conflicts[0]['observed_digits'], [['2','8']])
        self.assertEqual(conflicts[0]['anchors'], ['therefore'])
        self.assertEqual(conflicts[0]['reason'], 'scan_numeric_reference_disagreement')

    def test_missing_or_ambiguous_ownership_and_single_scale_do_not_flag(self):
        for change in (dict(reader_signature=''), dict(row=9),
                       dict(readings=self.observation['readings'][:1]),
                       dict(readings=[self.observation['readings'][0]]*2),
                       dict(readings=[dict(text='(2.8).', score=.79, scale=2)]*2),
                       dict(readings=[dict(text='(2.3).', score=.99, scale=s) for s in (1,2)])):
            with self.subTest(change=change):
                self.assertEqual(reference_conflicts(self.answer, self.rows,
                    [{**self.observation, **change}]), [])
        for rows in ([*self.rows, {**self.rows[0], 'id': 5}],
                     [{**self.rows[0], 'words': self.rows[0]['words'][1:]}]):
            self.assertEqual(reference_conflicts(self.answer, rows, [self.observation]), [])
        self.assertEqual(reference_conflicts(dict(items=self.answer['items']*2),self.rows,[self.observation]),[])
        self.assertEqual(reference_conflicts(self.answer,self.rows,[self.observation]*2),[])
        for latex in (r'\text{Therefore } x=2 \quad (2.3)', r'\text{Therefore \textbf{by (2.3)}}'):
            self.assertEqual(reference_conflicts(dict(items=[dict(kind='formula',latex=latex)]),
                                                 self.rows,[self.observation]),[])

    def test_reader_is_blind_rechecks_image_hash_and_tolerates_optional_runtime_failure(self):
        def tsv(reference):
            return ('level\ttext\tconf\n5\tby\t92\n5\t'+reference+'\t91\n').encode()
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'row.png';Image.new('RGB',(1400,50),'white').save(path)
            row={**self.rows[0], 'image': str(path), 'image_sha256': digest(path)}
            with patch('book_pipeline.publishing.latin_word_ocr.runtime_signature',return_value='cpu-runtime'), patch(
                'book_pipeline.ingestion.scan_reference_audit.subprocess.run',
                return_value=subprocess.CompletedProcess([],0,tsv('(2.8).'))) as reader:
                records=observe_reasons([row],Path(name)/'observations')
                self.assertEqual(len(records),1)
                self.assertEqual(len(records[0]['readings']),3)
                for call in reader.call_args_list:
                    self.assertEqual(call.args[0],['tesseract','stdin','stdout','-l','eng','--psm','7','tsv'])
                    self.assertTrue(call.kwargs['input'].startswith(b'\x89PNG'))
                reader.side_effect=subprocess.TimeoutExpired('tesseract',10)
                self.assertEqual(observe_reasons([row],Path(name)/'failed'),[])
                reader.reset_mock()
                Image.new('RGB',(1400,50),'black').save(path)
                self.assertEqual(observe_reasons([row],Path(name)/'tampered'),[])
                reader.assert_not_called()

    def test_dotted_references_without_labels_are_audited_and_policy_invalidates_cache(self):
        from book_pipeline.ingestion.scan_formula_anchors import recover_anchors, policy_signature
        with patch('book_pipeline.ingestion.scan_formula_anchors.physical_rows',return_value=self.rows), patch(
            'book_pipeline.ingestion.scan_reference_audit.observe_reasons',return_value=[self.observation]):
            output,evidence=recover_anchors(self.answer,'scan.png','rows')
        self.assertEqual(output,self.answer)
        self.assertEqual(len(evidence['reference_conflicts']),1)
        with patch('book_pipeline.publishing.latin_word_ocr.runtime_signature',return_value='runtime'), patch(
            'book_pipeline.ingestion.formula_text_anchors.policy_signature',return_value={'reader':'hash'}):
            old=fingerprint(policy_signature())
            with patch('book_pipeline.ingestion.scan_reference_audit.METHOD','new-audit'):
                self.assertNotEqual(old,fingerprint(policy_signature()))

    def test_applied_conflict_survives_atom_creation_and_materialization(self):
        import json
        from book_pipeline.ingestion.formula_bands import apply_formula_corrections, METHOD
        import pymupdf
        from book_pipeline.ingestion.source import SourceIndex, materialize_source, load_source
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);origin=root/'origin';origin.mkdir();pdf=root/'input.pdf'
            with pymupdf.open() as document:
                document.new_page();document.save(pdf)
            source=dict(input_sha256=digest(pdf),mineru_sha256='ocr',source_pdf=str(pdf),
                mineru_dir=str(origin),pages=[dict(page=0,atoms=['a'])],atoms=[dict(
                    id='a',group='g',page=0,kind='display_math',role='equation',block_type='equation',
                    content='x/y=2',length=5,bbox=[0,0,1,1],block_bbox=[0,0,1,1])])
            record=dict(id='b',page=0,bbox=[0,0,1,1],groups=['g'],preserve=[],answer=self.answer,
                        scan_anchors=dict(reference_conflicts=reference_conflicts(self.answer,self.rows,[self.observation])))
            report=dict(status='complete',method=METHOD,source=source,records=[record],usage={})
            index=apply_formula_corrections(SourceIndex(source),report)
            self.assertEqual(index.atoms[0]['content'],self.answer['items'][0]['latex'])
            self.assertEqual(index.atoms[0]['recovery_issue'],'scan_numeric_reference_disagreement')
            (origin/'manifest.json').write_text(json.dumps(dict(input_pdf=str(pdf),input_sha256=digest(pdf))))
            target=materialize_source(index,root/'corrected')
            recovered=SourceIndex(load_source(pdf,target))
            self.assertEqual(recovered.atoms[0]['recovery_issue'],'scan_numeric_reference_disagreement')
            self.assertEqual(recovered.atoms[0]['content'],self.answer['items'][0]['latex'])


if __name__ == '__main__':
    unittest.main()
