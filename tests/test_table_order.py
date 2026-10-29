"""A searchable table stays before the prose that interprets its rows."""
from pathlib import Path
import tempfile
import unittest
import pymupdf
from book_pipeline.config import load_config,executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.compiler import compile_project

class TableOrderTests(unittest.TestCase):
    def test_bounded_table_keeps_source_order_spacing_symbols_and_searchable_cells(self):
        html='<table><tr><th>P</th><th>P ⇔ Q</th></tr>'+''.join(
            '<tr><td>Row'+str(n)+'</td><td>T</td></tr>' for n in range(9))+'</table>'
        body='The following table lists the possible outcomes. '*75
        values=[('text',body),('table',html),('text','After the table, Row8 is explained here.')]
        atoms=[dict(id=str(n),group=str(n),page=0,kind=k,role='table_body' if k=='table' else 'text',
                    block_type='table' if k=='table' else 'text',content=t,length=len(t),bbox=[.1,.2,.9,.4])
               for n,(k,t) in enumerate(values)]
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=SourceIndex(dict(atoms=atoms))
            coverage=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Table order',author=''),{**load_config(),'float_assets':True},root,sample_mode=True)
            tex=''.join(p.read_text() for p in (root/'chapters').glob('*.tex'))
            self.assertNotIn(r'\begin{table}',tex)
            self.assertIn(r'\par\addvspace{6pt}',tex)
            result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                text=''.join(p.get_text() for p in pdf)
                self.assertLess(text.index('Row8'),text.index('After the table'))
                self.assertIn('⇔',text)
                for page in pdf:
                    for word in page.get_text('words'):
                        self.assertGreaterEqual(word[0],25);self.assertLessEqual(word[2],page.rect.width-25)
            self.assertEqual(coverage['accounted_units'],index.prefix[-1])
