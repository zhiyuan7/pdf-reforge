from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from book_pipeline.config import fingerprint
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import Renderer
from book_pipeline.publishing.reviewed_styles import validate_reviewed_styles, apply_reviewed_styles


class ReviewedStyleTests(unittest.TestCase):
    def test_visual_review_replaces_false_bold_only_inside_exact_reviewed_quote(self):
        text='Humidity and Wind are attributes.'
        auto=[dict(start=0,end=len(text),quote=text,italic=True,bold=True)]
        review=[dict(start=13,end=17,quote='Wind',italic=True,bold=False,evidence='source.png')]
        before=deepcopy(auto)
        runs=apply_reviewed_styles(text,auto,review)
        self.assertEqual([r['quote'] for r in runs],['Humidity and ','Wind',' are attributes.'])
        self.assertEqual([r['bold'] for r in runs],[True,False,True])
        self.assertEqual(auto,before)
        self.assertEqual(runs[1]['evidence'],'source.png')

    def setUp(self):
        self.index = SourceIndex(dict(atoms=[dict(id='a', group='g', page=0,
            kind='text', role='text', block_type='text', content='例 2.3 设 x 为实数。', length=14)]))
        self.review = dict(source_sha256=fingerprint(self.index.source), evidence='review.md: source crop',
            runs=[dict(atom='a', quote='例 2.3', occurrence=0, bold=True, italic=False)])

    def test_exact_style_keeps_source_and_surrounding_text(self):
        before = deepcopy(self.index.source)
        styles = validate_reviewed_styles(self.index, self.review)
        with tempfile.TemporaryDirectory() as name:
            renderer = Renderer(self.index, dict(corrections=[], joins=[], exclusions=[],
                bold_spans=[], source_styles=styles), [], [], Path(name))
            value = renderer.text_fragment(self.index.atoms[0], 0, 14)
        self.assertIn(r'{\bfseries 例 2.3}', value)
        self.assertTrue(value.endswith(' 设 x 为实数。'))
        self.assertEqual(self.index.source, before)

    def test_stale_source_missing_evidence_math_and_wrong_quotes_fail(self):
        for changes in (dict(source_sha256='stale'), dict(evidence=''), dict(runs=None),
                        dict(runs=[dict(self.review['runs'][0], quote='例 2.4')]),
                        dict(runs=[dict(self.review['runs'][0], atom='missing')])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_reviewed_styles(self.index, {**self.review, **changes})
        self.index.atoms[0]['kind'] = 'math'
        self.review['source_sha256'] = fingerprint(self.index.source)
        with self.assertRaises(ValueError):
            validate_reviewed_styles(self.index, self.review)

    def test_text_edits_must_not_overlap_style(self):
        row = dict(atom='a', start_offset=2, end_offset=3)
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            validate_reviewed_styles(self.index, self.review, [row])
        self.assertTrue(validate_reviewed_styles(self.index, self.review,
            [dict(row, start_offset=7, end_offset=8)]))

    def test_nested_emphasis_preserves_the_word_boundary_after_bold(self):
        import pymupdf
        from book_pipeline.config import load_config, executable
        from book_pipeline.publishing.latex import write_project
        from book_pipeline.publishing.compiler import compile_project
        text = 'The key term if it is valid.'
        atom = dict(self.index.atoms[0], content=text, length=len(text))
        index = SourceIndex(dict(atoms=[atom], source_page_count=1))
        review = dict(source_sha256=fingerprint(index.source), evidence='synthetic-font-review',
            runs=[dict(atom='a', quote=text, occurrence=0, italic=True, bold=False),
                  dict(atom='a', quote='key term', occurrence=0, italic=True, bold=True)])
        styles = validate_reviewed_styles(index, review)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_project(index, dict(corrections=[], joins=[], exclusions=[], bold_spans=[],
                source_styles=styles), [], [], dict(title='Nested emphasis', author=''),
                load_config(), root, sample_mode=True)
            compiled = compile_project(root, {t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(compiled['success'], compiled['log_tail'])
            with pymupdf.open(root/'main.pdf') as pdf:
                self.assertTrue(pdf[0].search_for('term if it'))
                self.assertNotIn('termif', pdf[0].get_text())
