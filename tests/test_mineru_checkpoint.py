"""Regression checks for SDK serialization of nested directory entries."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from mineru.parser import ParseResult
from mineru.parser.writer import FileBasedDataWriter
from mineru_checkpoint import normalize_middle_json, read_checkpoint, write_checkpoint


def directory_document():
    return {
        'schema': 'docvortex.middle', 'schema_version': '2.0', 'is_full_document': True,
        'metadata': {'file_suffix': 'pdf', 'producer': {'name': 'mineru', 'version': '4.0.10'}},
        'extensions': {'mineru': {'tier': 'standard', 'parse_mode': 'ocr'},
                       'docvortex_layout': {'version': 1, 'pages': [
                           {'page_idx': p, 'width_pt': 600, 'height_pt': 800} for p in range(2)]}},
        'pages': [
            {'page_idx': 0, 'blocks': [{'type': 'index', 'index': 0, 'bbox': [0.1, 0.1, 0.9, 0.9], 'content': [
                {'type': 'text', 'content': [{'type': 'text', 'content': '第一章 绪论……1'}],
                 'continues_prev': None},
                {'type': 'index', 'content': [
                    {'type': 'text', 'content': [{'type': 'text', 'content': '1.1 基本概念……2'}],
                     'continues_prev': None}]}]}]},
            {'page_idx': 1, 'blocks': []},
        ],
    }


class DirectoryCacheRegression(unittest.TestCase):
    def test_existing_default_inclusive_checkpoint_can_be_loaded_without_rewriting_it(self):
        middle = directory_document()
        with self.assertRaisesRegex(ValueError, 'nested blocks must not contain continues_prev'):
            ParseResult.from_dict(middle)
        payload = {'middle_json': middle, 'model_output': None}
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            write_checkpoint(folder, {'fixture': 'legacy-directory'}, payload)
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            loaded = read_checkpoint(folder, {'fixture': 'legacy-directory'})
            normalized = normalize_middle_json(loaded['middle_json'])
            parsed = ParseResult.from_dict(normalized)
            self.assertEqual([p.page_idx for p in parsed.middle_json.pages], [0, 1])
            self.assertEqual(parsed.middle_json.pages[1].blocks, [])
            self.assertIn('第一章 绪论', parsed.markdown())
            self.assertIn('1.1 基本概念', parsed.markdown())
            self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
            self.assertEqual(loaded, payload)

    def test_normalized_new_cache_is_readable_and_preserves_all_other_data(self):
        middle = directory_document()
        original = deepcopy(middle)
        expected = deepcopy(middle)
        del expected['pages'][0]['blocks'][0]['content'][0]['continues_prev']
        del expected['pages'][0]['blocks'][0]['content'][1]['content'][0]['continues_prev']
        normalized = normalize_middle_json(middle)
        self.assertEqual(normalized, expected)
        self.assertEqual(middle, original)
        # Reproduce the SDK writer which originally introduced these defaults.
        emitted = ParseResult.from_dict(normalized).middle_json.to_dict(skip_defaults=False)
        stored = normalize_middle_json(emitted)
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            write_checkpoint(folder, {'fixture': 'new-directory'}, {'middle_json': stored})
            loaded = read_checkpoint(folder, {'fixture': 'new-directory'})
            ParseResult.from_dict(loaded['middle_json'])
        self.assertNotIn('continues_prev', stored['pages'][0]['blocks'][0]['content'][0])

    def test_real_nested_continuation_flags_are_not_discarded(self):
        for value in (True, False):
            with self.subTest(value=value):
                middle = directory_document()
                middle['pages'][0]['blocks'][0]['content'][0]['continues_prev'] = value
                normalized = normalize_middle_json(middle)
                self.assertIs(normalized['pages'][0]['blocks'][0]['content'][0]['continues_prev'], value)
                with self.assertRaisesRegex(ValueError, 'nested blocks must not contain continues_prev'):
                    ParseResult.from_dict(normalized)

    def test_directory_and_empty_page_survive_actual_sdk_export(self):
        parsed = ParseResult.from_dict(normalize_middle_json(directory_document()))
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            parsed.save(FileBasedDataWriter(str(folder)))
            exported = json.loads((folder / 'middle_json.json').read_text())
            reloaded = ParseResult.from_dict(exported)
            self.assertEqual([p.page_idx for p in reloaded.middle_json.pages], [0, 1])
            self.assertEqual(reloaded.middle_json.pages[1].blocks, [])
            self.assertIn('第一章 绪论', (folder / 'markdown.md').read_text())

    def test_top_level_flags_assets_text_and_extensions_are_untouched(self):
        middle = directory_document()
        middle['extensions']['application'] = {'continues_prev': None, 'pages': [{'continues_prev': None}]}
        blocks = middle['pages'][0]['blocks']
        for value in (True, False, None):
            blocks.append({'type': 'text', 'index': len(blocks), 'continues_prev': value,
                           'content': [{'type': 'text', 'content': '原有中文与公式 y_i y_j'}]})
        blocks.append({'type': 'image', 'index': len(blocks), 'bbox': [0.1, 0.2, 0.3, 0.4],
                       'content': [{'type': 'image_body', 'image_base64': 'opaque-original-image-bytes'}]})
        original = deepcopy(middle)
        normalized = normalize_middle_json(middle)
        self.assertEqual(normalized['pages'][0]['blocks'][1:], blocks[1:])
        self.assertEqual(normalized['extensions'], middle['extensions'])
        self.assertEqual(middle, original)

    def test_other_invalid_directory_data_still_fails_sdk_validation(self):
        middle = directory_document()
        middle['pages'][0]['blocks'].append(deepcopy(middle['pages'][0]['blocks'][0]))
        with self.assertRaisesRegex(ValueError, 'top-level block indices must be unique'):
            ParseResult.from_dict(normalize_middle_json(middle))


if __name__ == '__main__':
    unittest.main()
