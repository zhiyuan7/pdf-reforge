"""A versioned benchmark cannot replace the previous report or source manifest."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.benchmark import main, translation_items, translation_probe, excerpt, acquire, inspect_artifacts


class BenchmarkReportTests(unittest.TestCase):
    def test_successful_searchable_artifact_still_reports_scan_reference_conflict(self):
        import pymupdf
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);latex=root/'pdf/latex';reports=root/'pdf/reports'
            latex.mkdir(parents=True);reports.mkdir()
            (latex/'main.tex').write_text('generated project')
            with pymupdf.open() as document:
                document.new_page().insert_text((50,80),'Therefore x/y=2 by (2.3).')
                document.save(latex/'main.pdf')
            (reports/'edited_source.json').write_text(json.dumps(dict(atoms=[dict(
                id='a',kind='display_math',role='equation',content=r'x/y=2',
                recovery_issue='scan_numeric_reference_disagreement')])))
            result=inspect_artifacts(root,dict(status='complete'))
            self.assertTrue(result['artifact_success'])
            self.assertEqual(result['quality_status'],'issues_found')
            self.assertEqual(result['issues'],['scan_numeric_reference_disagreement'])

    def test_preserved_translation_placeholders_do_not_establish_semantic_quality(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'source';source.mkdir()
            pdf=root/'source.pdf';pdf.write_bytes(b'PDF identity')
            (source/'manifest.json').write_text(json.dumps(dict(input_pdf=str(pdf))))
            item=dict(id='p',text='If $x=2$, then $y=3$.')
            wrong_translation='若 $x=2$ 或 $y=3$。'
            with patch('book_pipeline.application.pipeline.load_book'), patch(
                'scripts.benchmark.translation_items',return_value=[item]), patch(
                'book_pipeline.ai.client.DeepSeekClient') as client, patch(
                'book_pipeline.ingestion.translation.Translator') as translator:
                translator.return_value.translate.return_value={'p':wrong_translation}
                translator.return_value.glossary={}
                client.return_value.ledger.summary.return_value={}
                result=translation_probe({},root,dict(_budget_job_id='original-sha'))
            self.assertEqual(result['placeholder_validation'],'passed')
            self.assertEqual(result['semantic_quality'],'not_evaluated')
            self.assertEqual(result['status'],'placeholders_validated_semantics_unreviewed')
            self.assertEqual(result['output']['p'],wrong_translation)

    def test_searchable_footer_cannot_make_an_image_only_body_successful(self):
        import pymupdf
        for partial in [False,True]:
            with self.subTest(partial=partial),tempfile.TemporaryDirectory() as name:
                root=Path(name);latex=root/'pdf/latex';reports=root/'pdf/reports'
                latex.mkdir(parents=True);reports.mkdir()
                (latex/'main.tex').write_text('generated project')
                with pymupdf.open() as d:
                    p=d.new_page();p.insert_text((50,800),'Book author footer')
                    if partial:p.insert_text((50,80),'Searchable reconstructed body prose.')
                    d.save(latex/'main.pdf')
                atoms=[dict(id='t',kind='table',role='table_body',content='<table><tr><td>Body</td></tr></table>')]
                if partial:atoms.append(dict(id='p',kind='text',role='text',content='Searchable reconstructed body prose.'))
                (reports/'edited_source.json').write_text(json.dumps(dict(atoms=atoms)))
                (reports/'coverage.json').write_text(json.dumps(dict(issues=[dict(reason='complex_table_preserved_as_image',atom='t')])))
                result=inspect_artifacts(root,dict(status='complete'))
                self.assertEqual(result['artifact_success'],partial)
                self.assertEqual('no_searchable_body_text' in result['issues'],not partial)

    def test_real_scan_excerpt_keeps_original_images_and_checks_complete_pdf_sha(self):
        import pymupdf
        from book_pipeline.config import digest
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); pdf=root/'scan.pdf'
            with pymupdf.open() as d:
                page=d.new_page()
                pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,50,50),False)
                pix.clear_with(130); page.insert_image(page.rect,stream=pix.tobytes('png'))
                d.new_page();d.save(pdf)
            book=dict(id='scan',pages=[1],sha256=digest(pdf))
            self.assertEqual(acquire(book,root),pdf)
            samples=root/'samples';samples.mkdir()
            sample=excerpt(book,pdf,'scan',samples)
            with pymupdf.open(pdf) as original,pymupdf.open(sample) as clipped:
                self.assertEqual(original[0].get_pixmap().samples,clipped[0].get_pixmap().samples)
                self.assertEqual(len(clipped[0].get_images()),1)
            with self.assertRaisesRegex(ValueError,'SHA'):
                acquire({**book,'sha256':'0'*64},root)

    def test_translation_probe_keeps_complete_paragraph_math_and_run_size_cap(self):
        from book_pipeline.ingestion.source import SourceIndex
        atoms = []
        for n in range(4):
            for k, content in [('text', 'A vector is represented by ' + 'its coordinate values ' * 4),
                               ('math', 'v=(2,3)'), ('text', ' and has exactly 2 coordinates.')]:
                atoms.append(dict(id=str(len(atoms)), group=str(n), kind=k, content=content,
                                  length=len(content), block_type='text', role='text'))
        selected = translation_items(SourceIndex(dict(atoms=atoms)))
        self.assertEqual(len(selected), 2)
        self.assertTrue(all('$v=(2,3)$ and has exactly 2 coordinates.' in item['text'] for item in selected))
        self.assertLessEqual(sum(len(item['text']) for item in selected), 2400)

    def test_run_id_preserves_prior_report_and_manifest(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = root / 'corpus.json'
            source = root / 'source.pdf'
            source.write_bytes(b'cached source identity')
            manifest.write_text(json.dumps(dict(books=[dict(id='math', variants=['native'], pages=[1]) ])))
            output = root / 'benchmark'
            output.mkdir()
            prior = output / 'results.json'
            prior.write_text('{"previous":true}')
            with patch('sys.argv', ['benchmark', '--manifest', str(manifest), '--output', str(output),
                                    '--run-id', 'revision-2', '--acquire-only']), \
                 patch('scripts.benchmark.acquire', return_value=source), \
                 patch('scripts.benchmark.excerpt', return_value=source):
                self.assertEqual(main(), 0)
            self.assertEqual(prior.read_text(), '{"previous":true}')
            report = json.loads((output / 'runs/revision-2/results.json').read_text())
            self.assertEqual(report['cases'][0]['status'], 'acquired')
            self.assertEqual(json.loads(manifest.read_text())['books'][0]['pages'], [1])

    def test_only_one_two_paragraph_probe_is_run_across_books_and_variants(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'source.pdf'
            source.write_bytes(b'cached source identity')
            manifest = root / 'corpus.json'
            manifest.write_text(json.dumps(dict(books=[dict(id='math', title='Math', author='Author',
                language='en', variants=['native', 'rasterized'], pages=[1]) ])))
            output = root / 'benchmark'
            with patch('sys.argv', ['benchmark', '--manifest', str(manifest), '--output', str(output)]), \
                 patch('scripts.benchmark.acquire', return_value=source), \
                 patch('scripts.benchmark.excerpt', return_value=source), \
                 patch('book_pipeline.application.pipeline.run_workflow', return_value=dict(status='complete')), \
                 patch('scripts.benchmark.inspect_artifacts', return_value=dict(artifact_success=True)), \
                 patch('scripts.benchmark.translation_probe', return_value=dict(status='validated_sample', paragraphs=2)) as probe:
                self.assertEqual(main(), 0)
            self.assertEqual(probe.call_count, 1)
            report = json.loads((output / 'results.json').read_text())
            self.assertEqual(report['cases'][1]['translation_probe']['case'], 'math-native')
