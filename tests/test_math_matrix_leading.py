"""Displayed matrix geometry must not inherit wide prose line spacing."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config, executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.compiler import compile_project


class MathMatrixLeadingTests(unittest.TestCase):
    def test_matrix_row_pitch_is_independent_of_surrounding_prose_leading(self):
        atom=dict(id='a',group='a',page=0,kind='text',role='text',block_type='text',
                  content='Matrix comparison.',length=18,bbox=[.1,.2,.8,.3])
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            write_project(SourceIndex(dict(atoms=[atom])),dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Matrix leading',author=''),load_config(),root,sample_mode=True)
            chapter=root/'chapters/generated-001.tex'
            chapter.write_text(
                r'{\fontsize{10.5pt}{24pt}\selectfont\[\BookMath{\begin{bmatrix}11&12\\21&22\end{bmatrix}}\]}'+'\n'+
                r'{\fontsize{10.5pt}{12.6pt}\selectfont\[\BookMath{\begin{bmatrix}31&32\\41&42\end{bmatrix}}\]}')
            result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            with pymupdf.open(root/'main.pdf') as pdf:
                y=lambda t:pdf[0].search_for(t)[0].y0
                wide=y('21')-y('11');normal=y('41')-y('31')
                self.assertAlmostEqual(wide,normal,delta=.02)
                self.assertLess(wide,14)
