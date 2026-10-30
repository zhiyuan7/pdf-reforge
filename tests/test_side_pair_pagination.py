from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config, executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project


class SidePairPaginationTests(unittest.TestCase):
    def test_pending_top_floats_do_not_hide_the_side_pair_extent(self):
        self.check_pair(r'''
\begin{figure}[htp]\centering\rule{.8\linewidth}{.32\textheight}\par First figure.\end{figure}
\begin{figure}[htp]\centering\rule{.8\linewidth}{.27\textheight}\par Second figure.\end{figure}
''')

    def test_actual_minipage_depth_is_reserved_before_pair_is_shipped(self):
        self.check_pair(r'\par\vspace*{.70\textheight}')

    def check_pair(self, before):
        atom=dict(id='intro',group='intro',page=0,kind='text',role='text',
                  block_type='text',content='Introductory text.',length=18,bbox=[.1,.1,.8,.2])
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            write_project(SourceIndex(dict(atoms=[atom],source_page_count=1)),
                dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],
                dict(title='Side pair',author=''),load_config(),root,sample_mode=True)
            path=root/'main.tex'
            path.write_text(path.read_text().replace(r'\end{document}',before+r'''
\begin{BookSideBySide}{.45}{.45}{1}
PAIR-TOP\par\vspace*{.32\textheight}\noindent PAIR-BOTTOM
\BookSideSwitch
Right-hand caption.
\end{BookSideBySide}
\end{document}
'''))
            result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                self.assertGreaterEqual(len(pdf),2)
                self.assertIn('Introductory',pdf[0].get_text())
                pages=[p for p in pdf if 'PAIR-TOP' in p.get_text()]
                self.assertEqual(len(pages),1)
                page=pages[0]
                bottom=page.search_for('PAIR-BOTTOM')[0]
                self.assertLess(bottom.y1,page.rect.height-50)
