from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config, executable
from book_pipeline.ingestion.math_syntax import math_compatibility
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project


class BareAlignmentTests(unittest.TestCase):
    def test_only_complete_top_level_alignment_rows_are_wrapped_idempotently(self):
        source=r'x &= a+b \\ &= c+d,\tag{7.29}'
        rendered,evidence=math_compatibility(source)
        self.assertEqual(rendered,r'\begin{aligned}x &= a+b \\ &= c+d,\end{aligned}\tag{7.29}')
        self.assertTrue(evidence)
        self.assertEqual(math_compatibility(rendered),(rendered,[]))
        for unsafe in (r'x\&y',r'\text{a & b} \\ x',r'x &= y \\ z',
                       r'x &= {y \\ &= z',r'\begin{array}{cc}x&y\\z&w\end{array}'):
            self.assertEqual(math_compatibility(unsafe),(unsafe,[]))
        original='\f\f'+r'\left|\frac{df}{dx}\right|<1.'+'\v'+r'\tag{15.17}'
        clean,evidence=math_compatibility(original)
        self.assertEqual(clean,'  '+r'\left|\frac{df}{dx}\right|<1. '+r'\tag{15.17}')
        self.assertTrue(evidence)

    def test_heading_prose_wraps_do_not_break_macro_or_inline_math_comments(self):
        specs=[('a','text','A heading\n\nwith '),('m','math',r'\phi_i'),('b','text','\n\n and context')]
        atoms=[dict(id=i,group='heading',page=0,kind=k,role='paragraph_title',block_type='paragraph_title',
                    content=t,length=len(t),bbox=[.1,.2,.9,.3]) for i,k,t in specs]
        index=SourceIndex(dict(atoms=atoms,source_page_count=1))
        heading=dict(kind='section',anchor='heading',lo=0,hi=index.prefix[-1],heading_level=2,label='Heading',number='')
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[heading],[],
                dict(title='Heading',author=''),load_config(),root,sample_mode=True,
                heading_layout=dict(standalone_levels=[],centered_levels=[],page_break_after_levels=[]))
            self.assertTrue(audit['coverage_complete'])
            result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                self.assertIn('context',pdf[0].get_text())
                self.assertTrue({'source-a','source-m','source-b'}<=set(pdf.resolve_names()))

    def test_compiled_rows_and_equation_number_remain_searchable(self):
        tex=r'\mathbb{E}[x] &= \mathbb{E}[a+b] \\ &= \mathbb{E}[a]+\mathbb{E}[b]\tag{7.29}'
        atom=dict(id='formula',group='formula',page=0,kind='display_math',role='equation',
                  block_type='equation',content=tex,length=len(tex),bbox=[.1,.2,.9,.4])
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            audit=write_project(SourceIndex(dict(atoms=[atom],source_page_count=1)),
                dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],
                dict(title='Alignment',author=''),load_config(),root,sample_mode=True)
            self.assertTrue(audit['coverage_complete'])
            result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                self.assertIn('7.29',pdf[0].get_text())
                equations=pdf[0].search_for('=')
                self.assertEqual(len(equations),2)
                self.assertAlmostEqual(equations[0].x0,equations[1].x0,delta=2)
                self.assertGreater(equations[1].y0,equations[0].y0+5)
