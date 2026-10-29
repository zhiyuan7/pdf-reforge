"""Long explanatory displays retain readable, searchable text and native math."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.prose_math import prose_math_parts


class ProseMathTests(unittest.TestCase):
    def test_nested_styled_numbered_and_ambiguous_math_keep_original_renderer(self):
        explanation = r'\text{the value of the function at time }t'
        for content in (r'x_{'+explanation+'}', r'x=\left('+explanation+r'\right)',
                        'x='+explanation+r'\tag{2}', r'\textbf{the value of the function at time }t',
                        r'\text{the value of the function at time {t}}',
                        r'\begin{cases}x&='+explanation+r'\end{cases}',
                        'x='+explanation+'}', r'x=\text{if }y>0',
                        r'\mathrm{Var}(X)='+explanation):
            with self.subTest(content=content):
                self.assertIsNone(prose_math_parts(content))
        content = r'q(t)=\text{“}q\text{ of }t\text{”}=\text{the value of a function at time }t'
        parts = prose_math_parts(content)
        self.assertIsNotNone(parts)
        self.assertEqual(''.join(value if kind == 'math' else r'\text{'+value+'}'
                                 for kind, value in parts), content)

    def test_explanation_wraps_at_body_size_with_all_tokens_and_destinations(self):
        content = (r'\lim_{h\to0}=\text{“limit as }h\text{ goes to zero”}='
                   r'\text{the process that starts with a sequence of numbers }'
                   r'\Delta q/\Delta t\text{ and produces the instantaneous rate }dq/dt.')
        rows = [('text', 'Before the explanation.'), ('display_math', content),
                ('display_math', r'x=y'), ('text', 'After the explanation.')]
        atoms = [dict(id=f'a{n}',group=f'g{n}',page=0,kind=kind,
                      role='text' if kind == 'text' else 'equation',
                      block_type='text' if kind == 'text' else 'equation',
                      bbox=[.1,.1+n*.05,.9,.14+n*.05],content=text,length=len(text))
                 for n,(kind,text) in enumerate(rows)]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            index = SourceIndex(dict(atoms=atoms,source_page_count=1))
            audit = write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                                  [],[],dict(title='Explanatory math',author=''),load_config(),root,sample_mode=True)
            result = compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                text = pdf[0].get_text()
                self.assertIn('the process that starts',text)
                self.assertIn('instantaneous rate',text)
                spans = [s for b in pdf[0].get_text('dict')['blocks'] if b['type'] == 0
                         for line in b['lines'] for s in line['spans']
                         if any(word in s['text'] for word in ('process','instantaneous','Before','After'))]
                self.assertGreaterEqual(len(spans),4)
                self.assertTrue(all(10.4 < s['size'] < 10.6 for s in spans),spans)
                self.assertTrue(all(30 < s['bbox'][0] < s['bbox'][2] < pdf[0].rect.width-30 for s in spans))
                names = pdf.resolve_names()
                self.assertTrue(all('source-'+a['id'] in names for a in atoms))
            self.assertEqual(audit['accounted_units'],index.prefix[-1])
