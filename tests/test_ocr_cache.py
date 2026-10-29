"""Automatic OCR reuse cannot cross settings, package versions or changed assets."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.config import digest, load_config, read, save
from book_pipeline.ingestion.native import prepare_source, run_ocr
from book_pipeline.ingestion.ocr_cache import validate_ocr_cache, copy_ocr_cache
from mineru_checkpoint import export_signature, export_hashes


VERSIONS = dict(mineru_version='4.test', docvortex_version='2.test')


class OCRCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pdf = self.root/'sample.pdf'
        with pymupdf.open() as doc:
            doc.new_page().insert_text((50,80),'Modern mathematical text')
            doc.save(self.pdf)
        self.directory = self.root/'output'/'ocr'
        self.directory.mkdir(parents=True)
        self.config = load_config()
        (self.directory/'figure.png').write_bytes(b'original illustration')
        save(self.directory/'middle_json.json',dict(schema='docvortex.middle',schema_version='2.0',
            pages=[dict(page_idx=0,blocks=[dict(index=0,type='image',content=[
                dict(type='image_body',image_path='figure.png',content='')])])]))
        self.manifest = dict(input_sha256=digest(self.pdf),source_page_numbers=[1],**VERSIONS,
            tier=self.config['mineru_tier'],ocr_mode=self.config['mineru_ocr_mode'],
            model_source=self.config['mineru_model_source'])
        self.manifest.update(export_signature=export_signature(digest(self.pdf),[1],
            self.manifest['tier'],self.manifest['ocr_mode'],self.manifest['model_source'],VERSIONS),
            export_hashes=export_hashes(self.directory))
        save(self.directory/'manifest.json',self.manifest)

    def validate(self, config=None, pages=None, versions=None):
        return validate_ocr_cache(self.pdf,self.directory,config or self.config,
            [1] if pages is None else pages,versions=VERSIONS if versions is None else versions)

    def test_valid_cache_is_reused_without_ocr_or_writing_old_evidence(self):
        before = {p.name:p.read_bytes() for p in self.directory.iterdir()}
        with patch('book_pipeline.ingestion.ocr_cache.runtime_versions',return_value=VERSIONS), \
             patch('book_pipeline.ingestion.native.subprocess.Popen') as process:
            run_ocr(self.pdf,self.directory,[0],self.config)
        process.assert_not_called()
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.directory.iterdir()})

    def test_page_fallback_discovers_and_copies_a_verified_preflight_without_inference(self):
        before = {p.name:p.read_bytes() for p in self.directory.iterdir()}
        target=self.root/'render'/'ocr'
        with patch('book_pipeline.ingestion.native.ROOT',self.root), \
             patch('book_pipeline.ingestion.ocr_cache.runtime_versions',return_value=VERSIONS), \
             patch('book_pipeline.ingestion.native.subprocess.Popen') as process:
            run_ocr(self.pdf,target,[0],self.config)
        process.assert_not_called()
        self.assertEqual(before,{p.name:p.read_bytes() for p in target.iterdir()})
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.directory.iterdir()})

    def test_copied_cache_is_not_published_if_a_source_asset_changes(self):
        (self.directory/'figure.png').write_bytes(b'changed during copy')
        target=self.root/'render'/'ocr'
        with patch('book_pipeline.ingestion.ocr_cache.runtime_versions',return_value=VERSIONS), \
             self.assertRaises(ValueError):
            copy_ocr_cache(self.pdf,self.directory,target,self.config,[1])
        self.assertFalse(target.exists())
        self.assertEqual(list(target.parent.iterdir()),[])

    def test_each_changed_setting_or_runtime_rejects_the_old_result(self):
        for key,value in [('mineru_tier','advanced'),('mineru_ocr_mode','ocr'),
                          ('mineru_model_source','local')]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'签名'):
                self.validate({**self.config,key:value})
        for key in VERSIONS:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'签名'):
                self.validate(versions={**VERSIONS,key:'new-version'})
        with self.assertRaisesRegex(ValueError,'签名'):
            self.validate(pages=[2])
        self.pdf.write_bytes(self.pdf.read_bytes()+b'\n% changed input')
        with self.assertRaisesRegex(ValueError,'签名'):
            self.validate()

    def test_unsigned_legacy_export_and_changed_export_policy_are_rejected(self):
        for problem in ('legacy','method','missing_hashes','missing_middle_hash'):
            manifest=deepcopy(self.manifest)
            if problem=='legacy':manifest.pop('export_signature')
            elif problem=='method':manifest['export_signature']['method']='old-export'
            elif problem=='missing_hashes':manifest.pop('export_hashes')
            else:manifest['export_hashes'].pop('middle_json.json')
            save(self.directory/'manifest.json',manifest)
            with self.subTest(problem=problem),self.assertRaises(ValueError):self.validate()

    def test_changed_or_missing_middle_and_illustration_are_rejected(self):
        for name in ('middle_json.json','figure.png'):
            path=self.directory/name
            original=path.read_bytes()
            for problem in ('changed','missing'):
                if problem=='changed':path.write_bytes(original+b'changed')
                else:path.unlink()
                with self.subTest(name=name,problem=problem),self.assertRaisesRegex(ValueError,'改变或缺失'):
                    self.validate()
                path.write_bytes(original)

    def test_manifest_cannot_omit_referenced_image_or_traverse_directories(self):
        for problem in ('omitted','outside'):
            manifest=deepcopy(self.manifest)
            if problem=='omitted':manifest['export_hashes'].pop('figure.png')
            else:manifest['export_hashes']['../../sample.pdf']=digest(self.pdf)
            save(self.directory/'manifest.json',manifest)
            with self.subTest(problem=problem),self.assertRaises(ValueError):self.validate()

    def test_middle_page_indices_are_checked_besides_manifest(self):
        middle=read(self.directory/'middle_json.json');middle['pages'][0]['page_idx']=1
        save(self.directory/'middle_json.json',middle)
        save(self.directory/'manifest.json',{**self.manifest,'export_hashes':export_hashes(self.directory)})
        with self.assertRaisesRegex(ValueError,'正文页码'):self.validate()

    def test_automatic_cross_job_reuse_checks_current_policy(self):
        from book_pipeline.application.pipeline import prepare_book_source
        for changed in (False,True):
            config={**self.config,'mineru_tier':'advanced'} if changed else self.config
            with self.subTest(changed=changed),ExitStack() as stack:
                stack.enter_context(patch('book_pipeline.application.pipeline.ROOT',self.root))
                stack.enter_context(patch('book_pipeline.ingestion.ocr_cache.runtime_versions',return_value=VERSIONS))
                fresh=stack.enter_context(patch('book_pipeline.ingestion.native.prepare_source',
                    side_effect=RuntimeError('fresh extraction requested')))
                if changed:
                    with self.assertRaisesRegex(RuntimeError,'fresh extraction'):
                        prepare_book_source(self.pdf,None,config,self.root/'render')
                    fresh.assert_called_once()
                else:
                    for module,name in [('scan_page_frames','recover_scan_page_frames'),
                        ('table_prose','recover_table_prose'),('raster_panels','recover_raster_panels')]:
                        stack.enter_context(patch('book_pipeline.ingestion.'+module+'.'+name,
                            side_effect=lambda pdf,source,*args:source))
                    stack.enter_context(patch('book_pipeline.ingestion.formula_bands.correct_formula_bands',
                        return_value=dict(records=[])))
                    stack.enter_context(patch('book_pipeline.ingestion.translation.translate_source',
                        side_effect=lambda pdf,source,*args:source))
                    self.assertEqual(prepare_book_source(self.pdf,None,config,self.root/'render'),self.directory)
                    fresh.assert_not_called()

    def test_prepared_mixed_source_rechecks_ocr_runtime_before_resuming(self):
        dest=self.root/'prepared'
        # Use forced OCR so the source's cached success depends on this export.
        config={**self.config,'input_mode':'ocr'}
        from tests.test_auto_ingestion import Client
        import shutil
        def ocr(pdf,directory,*args):shutil.copytree(self.directory,directory)
        prepare_source(self.pdf,dest,config,client=Client(),ocr_runner=ocr)
        with patch('book_pipeline.ingestion.ocr_cache.runtime_versions',return_value=VERSIONS):
            prepare_source(self.pdf,dest,config,client=Client())
        for changed_config,versions in [(config,{**VERSIONS,'mineru_version':'changed'}),
                                       ({**config,'mineru_ocr_mode':'ocr'},VERSIONS)]:
            with self.subTest(config=changed_config['mineru_ocr_mode'],versions=versions), \
                 patch('book_pipeline.ingestion.ocr_cache.runtime_versions',return_value=versions), \
                 self.assertRaises(ValueError):
                prepare_source(self.pdf,dest,changed_config,client=Client())


if __name__=='__main__':unittest.main()
