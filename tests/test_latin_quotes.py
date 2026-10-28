"""Mixed-script quotes keep their source glyphs, fonts and line boundaries."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import Renderer, escape, write_project


DECISIONS = dict(corrections=[], joins=[], exclusions=[], bold_spans=[])


def quote_source():
    rows = [
        ('en', 'text', 'Gauss’s Method: say “true” or ‘false’. The quoted variable is ‘'),
        ('en', 'math', 'x'),
        ('en', 'text', '’.'),
        ('zh', 'text', '中文“成立”与‘不成立’，保留中文引号。'),
        ('mixed', 'text', '中文的 Gauss’s 方法。'),
    ]
    return SourceIndex(dict(source_page_count=1, atoms=[
        dict(id=f'atom-{n}', group=group, page=0, kind=kind, role='text',
             block_type='text', content=text, length=len(text),
             bbox=[.1, .1+n*.04, .9, .13+n*.04])
        for n, (group, kind, text) in enumerate(rows)]))


class LatinQuoteTests(unittest.TestCase):
    def test_chinese_quotes_stay_full_width_but_internal_apostrophes_do_not(self):
        chinese = '中文“true”与‘false’，保留中文引号。'
        self.assertEqual(escape(chinese), chinese)
        self.assertEqual(escape('中文的 Gauss’s 方法。'),
                         r'中文的 Gauss\BookLatinQuote{’}s 方法。')
        self.assertEqual(escape('‘x’ & “y”'),
                         r'\BookLatinQuote{‘}x\BookLatinQuote{’} \& \BookLatinQuote{“}y\BookLatinQuote{”}')

    def test_punctuation_fragment_inherits_paragraph_script(self):
        index = quote_source()
        with tempfile.TemporaryDirectory() as name:
            renderer = Renderer(index, DECISIONS, [], [], Path(name))
            atom = index.atoms[2]
            self.assertEqual(renderer.text_fragment(atom, 0, atom['length']),
                             r'\BookLatinQuote{’}.')
            zh = index.atoms[3]
            self.assertEqual(renderer.text_fragment(zh, 0, zh['length']), zh['content'])

    def test_actual_pdf_uses_latin_quotes_and_restores_chinese_punctuation(self):
        index = quote_source()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_project(index, DECISIONS, [], [], dict(title='Quotes', author=''),
                          load_config(), root, sample_mode=True)
            result = compile_project(root, {tool: executable(tool) for tool in ('xelatex','latexmk')})
            self.assertTrue(result['success'], result['log_tail'])
            self.assertEqual(result['warnings'], [])
            with pymupdf.open(root/'main.pdf') as pdf:
                self.assertEqual(len(pdf), 1)
                chars = [(c, s) for b in pdf[0].get_text('rawdict')['blocks'] if b['type']==0
                         for line in b['lines'] for s in line['spans'] for c in s['chars']]
                quotes = [(c, s) for c, s in chars if c['c'] in '‘’“”']
                self.assertEqual(len(quotes), 12)
                for n, (char, span) in enumerate(quotes):
                    width = char['bbox'][2]-char['bbox'][0]
                    if 7 <= n <= 10:
                        self.assertIn('NotoSerifCJK', span['font'])
                        self.assertGreater(width, 9)
                    else:
                        self.assertIn('TeXGyreTermes', span['font'])
                        self.assertLess(width, 7)
                text = ''.join(c['c'] for c, _ in chars)
                self.assertIn('Gauss’s Method: say “true” or ‘false’.', text)
                self.assertIn('中文“成立”与‘不成立’，保留中文引号。', text)
                # The closing quote and period cannot form an isolated line.
                closing = next(i for i, (c, _) in enumerate(chars)
                               if c['c']=='’' and i and chars[i-1][0]['c']=='𝑥')
                self.assertEqual(chars[closing+1][0]['c'], '.')
                self.assertLess(abs(chars[closing][0]['origin'][1]-chars[closing+1][0]['origin'][1]), 1)
                names = pdf.resolve_names()
                self.assertEqual(names['source-atom-2']['page'], 0)
