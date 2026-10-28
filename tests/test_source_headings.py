"""Documents without contents pages retain auditable titles and body text."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from book_pipeline.config import load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.structure.headings import validate_metadata
from book_pipeline.structure.toc_catalog import SOURCE_METHOD
from book_pipeline.structure.toc_matching import analyze_toc_headings, verify_matching
from book_pipeline.structure.section_review import split_source


class SourceHeadingTests(unittest.TestCase):
    def test_excerpt_section_retains_depth_without_becoming_a_title_page(self):
        from book_pipeline.structure.toc_catalog import source_catalog
        from book_pipeline.structure.heading_layout import HeadingLayout
        atoms = []
        for n, text in enumerate(['2.4 Vector norms', '2.4.1 Euclidean norm']):
            atoms.append(dict(id=f'a{n}', group=f'g{n}', page=0, kind='text', role='paragraph_title',
                              block_type='paragraph_title', content=text, length=len(text),
                              bbox=[.1,.2+n*.1,.8,.25+n*.1]))
        index = SourceIndex(dict(atoms=atoms))
        entries = source_catalog(index, {})
        self.assertEqual([e['heading_level'] for e in entries], [2, 3])
        headings = [dict(kind='section' if n == 0 else 'subsection', **e) for n,e in enumerate(entries)]
        layout = HeadingLayout(headings)
        self.assertFalse(any(layout.standalone(h) for h in headings))
        self.assertEqual(entries[1]['parent_id'], entries[0]['id'])

    def test_heading_free_prose_survives_processing_and_publishing_index(self):
        from book_pipeline.ai.client import DeepSeekClient
        from book_pipeline.structure.page_content import run_page_content
        from book_pipeline.structure.book_mapping import build_book
        from book_pipeline.publishing.processed_book import EditedBook
        from book_pipeline.structure.toc_catalog import source_catalog
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            text = '没有目录或章节标题的正文也应该完整保留。段落中包含普通的数学讨论。'
            atom = dict(id='a', group='g', page=0, kind='text', role='text', block_type='text',
                        content=text, length=len(text), bbox=[.1,.2,.8,.4])
            index = SourceIndex(dict(atoms=[atom], source_pdf='source.pdf', source_page_count=1,
                                     selected_pages=[0], input_sha256='pdf', mineru_sha256='source',
                                     mineru_dir=str(directory)))
            outline = dict(headings=[], toc_entries=[], bold_spans=[], metadata=dict(title='测试',author='作者'),
                           metadata_config=dict(exclude_pages=[]))
            self.assertEqual(source_catalog(index, {}), [])
            config = load_config()
            def unexpected(*args):
                raise AssertionError('Plain one-page prose should not require a semantic API decision')
            client = DeepSeekClient(config, directory/'cache', transport=unexpected)
            report = run_page_content(index, outline, client, directory/'content')
            import json
            plan = json.loads((directory/'content/split_source.json').read_text())
            book = build_book(index, plan, report, directory/'content')
            edited = EditedBook(index, book, directory/'content')
            self.assertEqual(edited.index.text(0, edited.index.prefix[-1]), text)
            self.assertFalse(any(t['kind']=='heading' for t in book['targets']))
            from book_pipeline.structure.rule_mapping import run_rule_mapping
            mapping = run_rule_mapping(index, directory/'content', directory/'mapping')
            self.assertEqual(mapping['status'], 'complete')
            self.assertEqual(mapping['heading_correspondence'], [])

    def test_no_toc_uses_real_source_titles_without_excluding_body_pages(self):
        atoms = []
        for n, (text, kind) in enumerate([('第1章 向量', 'paragraph_title'),
                                         ('向量可以相加。', 'text'),
                                         ('1.1 加法', 'paragraph_title'), ('加法满足交换律。', 'text')]):
            atoms.append(dict(id=f'a{n}', group=f'g{n}', page=0, kind='text', role=kind,
                              block_type=kind, content=text, length=len(text), bbox=[.1,.2+n*.1,.8,.25+n*.1]))
        index = SourceIndex(dict(atoms=atoms, source_pdf='source.pdf', source_page_count=1,
                                 selected_pages=[0], input_sha256='pdf', mineru_sha256='source'))
        metadata = validate_metadata(dict(book=dict(title='标题测试', author='测试作者')))
        with tempfile.TemporaryDirectory() as directory, \
             patch('book_pipeline.structure.toc_catalog.analyze_toc', return_value=dict(pages=[], entries=[], regions=[], pure_pages=[], issues=[])), \
             patch('book_pipeline.structure.toc_matching.classify_exercise_headings', side_effect=lambda outline, *args: outline):
            outline = analyze_toc_headings(index, metadata, None, load_config(), Path(directory))
            self.assertEqual([e['catalog_source'] for e in outline['toc_entries']], [SOURCE_METHOD]*2)
            self.assertEqual(outline['toc_matching']['counts']['matched'], 2)
            verify_matching(outline, index)
            plan = split_source(index, outline)
            self.assertEqual(plan['excluded_pages'], [])
            self.assertTrue(any(s['nonempty'] for s in plan['sections']))
