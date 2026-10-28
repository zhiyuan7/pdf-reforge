from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config,executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import table_tex,write_project
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.table_repair import table_template,expand_template

class SpanningHeaderTests(unittest.TestCase):
    html='<table><tr><td colspan="2">A shared heading</td></tr><tr><td><eq>A^c</eq></td><td>complement</td></tr><tr><td>∅</td><td>null event</td></tr></table>'

    def test_only_full_first_header_over_regular_body_is_supported(self):
        tex=table_tex(self.html)
        self.assertIn(r'\multicolumn{2}{c}{A shared heading}',tex)
        self.assertEqual(tex.count('complement'),1)
        self.assertIn('$A^c$',tex)
        for html in (self.html.replace('colspan="2"','rowspan="2"'),
                     self.html.replace('colspan="2"','colspan="3"'),
                     self.html.replace('colspan="2"','colspan="x"'),
                     '<table><tr><td>a</td><td>b</td></tr><tr><td colspan="2">body merge</td></tr></table>'):
            self.assertIsNone(table_tex(html))

    def test_repair_placeholders_preserve_heading_and_every_body_cell(self):
        data=table_template(self.html)
        replacement=dict(kind='table_template',template=data['template'],in_box=False)
        self.assertEqual(expand_template(self.html,replacement),data['latex'])
        self.assertEqual(len(data['cells']),5)
        self.assertIn(r'\multicolumn{2}{c}{@@CELL0001@@}',data['template'])

    def test_heading_and_cells_are_searchable_without_image_fallback(self):
        atom=dict(id='table',group='g0',page=0,kind='table',role='table',block_type='table',
                  content=self.html,length=len(self.html),bbox=[.1,.2,.9,.5])
        source=SourceIndex(dict(source_page_count=1,atoms=[atom]))
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            result=write_project(source,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],
                                 dict(title='Shared heading',author=''),load_config(),root,sample_mode=True)
            self.assertNotIn('complex_table_preserved_as_image',str(result))
            compiled=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(compiled['success'],compiled['log_tail'])
            self.assertEqual(compiled['warnings'],[])
            with pymupdf.open(root/'main.pdf') as d:
                text=''.join(p.get_text() for p in d)
                self.assertIn('A shared heading',text)
                self.assertIn('complement',text)
                self.assertIn('null event',text)
