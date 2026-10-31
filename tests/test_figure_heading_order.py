"""A deferred source figure must stay in the section that introduced it."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project


class FigureHeadingOrderTests(unittest.TestCase):
    def test_deferred_figure_is_drained_before_next_subsection(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            image = root/'panel.png'
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0,0,300,420), False)
            pix.clear_with(180)
            pix.save(image)
            def atom(key, kind, role, text='', **extra):
                return dict(id=key, group=key, page=0, kind=kind, role=role,
                            block_type=role, content=text, length=len(text) or 1, **extra)
            atoms = [atom('body', 'text', 'text', 'Prior discussion. '*220),
                     atom('image', 'image', 'image', asset=str(image)),
                     atom('caption', 'text', 'image_caption', 'Figure 1.1 Source figure.'),
                     atom('heading', 'text', 'paragraph_title', '1.1.2 Next example'),
                     atom('next', 'text', 'text', 'NEXT-SECTION begins here. '*5)]
            index = SourceIndex(dict(atoms=atoms, source_page_count=1))
            lo,hi = index.prefix[3:5]
            registry = [dict(kind='subsection', anchor='next-section', lo=lo, hi=hi,
                             label=atoms[3]['content'], name=atoms[3]['content'], heading_level=3)]
            cfg = load_config(); cfg['float_assets'] = True
            audit = write_project(index, dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                                  registry, [], dict(title='Figure order', author=''), cfg,
                                  root/'latex', sample_mode=True,
                                  heading_layout=dict(standalone_levels=[]))
            self.assertTrue(audit['coverage_complete'])
            result = compile_project(root/'latex', {t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'], result['log_tail'])
            self.assertEqual(result['warnings'], [])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                names = pdf.resolve_names()
                figure = names['source-image']; caption = names['source-caption']
                heading = names['source-heading']
                self.assertEqual(figure['page'], caption['page'])
                self.assertLessEqual(figure['page'], heading['page'])
                if figure['page'] == heading['page']:
                    self.assertGreater(figure['to'][1], heading['to'][1])
                self.assertEqual(sum(p.get_text().count('Figure 1.1') for p in pdf), 1)
