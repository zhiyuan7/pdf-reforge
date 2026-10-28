"""Source links follow the actual first prose line across a page break."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import Renderer


class SourceDestinationTests(unittest.TestCase):
    def test_prose_destination_moves_with_prompt_and_unbreakable_formula(self):
        atom = dict(id='prompt', group='prompt', page=0, kind='text',
                    role='text', block_type='text', content='UniquePrompt', length=12)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            index = SourceIndex(dict(atoms=[atom], source_page_count=1))
            renderer = Renderer(index, dict(corrections=[], joins=[], exclusions=[], bold_spans=[]),
                                [], [], root)
            body = renderer.fragment(atom, 0, atom['length'], False)
            # Leave enough room for the prompt but not its formula. TeX moves
            # both to page two; a vertical-mode hyperlink would remain on one.
            tex = r'''\documentclass{article}
\usepackage[paperwidth=220pt,paperheight=180pt,margin=40pt]{geometry}
\usepackage{hyperref}
\pagestyle{empty}
\begin{document}
\noindent\rule{1pt}{65pt}\par
\begingroup\interlinepenalty=10000
''' + body + r'''
\par\nobreak
\noindent\rule{30pt}{40pt}\par
\endgroup
\end{document}
'''
            (root / 'main.tex').write_text(tex)
            process = subprocess.run([executable('xelatex'), '-interaction=nonstopmode',
                                      '-halt-on-error', '-no-shell-escape', 'main.tex'],
                                     cwd=root, capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout[-3000:])
            with pymupdf.open(root / 'main.pdf') as pdf:
                text_pages = [n for n, page in enumerate(pdf) if 'UniquePrompt' in page.get_text()]
                self.assertEqual(text_pages, [1])
                self.assertEqual(pdf.resolve_names()['source-prompt']['page'], text_pages[0])
