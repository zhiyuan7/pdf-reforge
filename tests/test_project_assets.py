from pathlib import Path
import shutil
import tempfile
import unittest

from book_pipeline.publishing.project_assets import asset_manifest, verify_asset_manifest


class ProjectAssetTests(unittest.TestCase):
    def test_replacing_only_review_intermediate_cannot_validate_included_copy(self):
        with tempfile.TemporaryDirectory() as name:
            project = Path(name)
            source = project / 'review-crop.png'
            included = project / 'included.png'
            source.write_bytes(b'old crop with missing axis labels')
            shutil.copy2(source, included)
            rows = asset_manifest(project, {str(source): included.name})
            self.assertEqual(verify_asset_manifest(project, rows), [])
            source.write_bytes(b'complete source rectangle')
            issues = verify_asset_manifest(project, rows)
            self.assertEqual([i['reason'] for i in issues], ['source_asset_changed'])
            shutil.copy2(source, included)
            self.assertEqual({i['reason'] for i in verify_asset_manifest(project, rows)},
                             {'included_asset_changed', 'source_asset_changed'})
            rows = asset_manifest(project, {str(source): included.name})
            self.assertEqual(verify_asset_manifest(project, rows), [])

    def test_transformed_asset_and_portable_export_keep_independent_digests(self):
        with tempfile.TemporaryDirectory() as name:
            project = Path(name)
            source = project / 'source.png'
            included = project / 'searchable.pdf'
            source.write_bytes(b'raster pixels')
            included.write_bytes(b'PDF pixels and search layer')
            rows = asset_manifest(project, {str(source): included.name})
            self.assertEqual(verify_asset_manifest(project, rows), [])
            source.unlink()
            self.assertEqual(verify_asset_manifest(project, rows), [])
            included.write_bytes(b'changed PDF')
            self.assertEqual(verify_asset_manifest(project, rows)[0]['reason'], 'included_asset_changed')
            included.unlink()
            self.assertEqual(verify_asset_manifest(project, rows)[0]['reason'], 'included_asset_missing')
