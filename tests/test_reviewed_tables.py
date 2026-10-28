from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import unicodedata

import pymupdf

from book_pipeline.config import executable, fingerprint, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project, table_tex
from book_pipeline.publishing.reviewed_tables import annotate_table_math, validate_reviewed_table_math, validate_reviewed_table_styles


class ReviewedTableTests(unittest.TestCase):
    def test_reviewed_italic_headers_keep_plain_html_and_longtable_natural_columns(self):
        html='<table><tr><td>Trial</td><td>Outcome</td></tr>'+''.join(
            f'<tr><td>D{i}</td><td>Yes</td></tr>' for i in range(14))+'</table>'
        atom=dict(id='table',group='g',page=0,kind='table',role='table',block_type='table',
                  content=html,length=len(html))
        index=SourceIndex(dict(atoms=[atom],source_page_count=1))
        review=dict(source_sha256=fingerprint(index.source),evidence='source.png',table_styles=[
            dict(atom='table',row=0,column=1,quote='Outcome',occurrence=0,italic=True,bold=False)])
        runs=validate_reviewed_table_styles(index,review)
        styled=annotate_table_math(html,runs['table'])
        self.assertIn('<td><i>Outcome</i></td>',styled)
        self.assertIn(r'{\itshape Outcome}',table_tex(styled))
        self.assertIn(r'\begin{longtable}{cc}',table_tex(styled))
        self.assertEqual(index.atoms[0]['content'],html)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[],table_styles=runs),
                          [],[],dict(title='Header styles',author='Test author'),load_config(),root,sample_mode=True)
            compiled=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(compiled['success'],compiled['log_tail'])
            with pymupdf.open(root/'main.pdf') as pdf:
                spans=[s for p in pdf for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]
                self.assertTrue(any(s['text']=='Outcome' and s['flags'] & 2 for s in spans))
                self.assertTrue(any(s['text']=='Trial' and not s['flags'] & 2 for s in spans))
        for change in (dict(italic=1),dict(bold=None),dict(quote='Missing'),dict(column=0),
                       dict(italic=False,bold=False)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_reviewed_table_styles(index,{**review,'table_styles':[{**review['table_styles'][0],**change}]})
        self.assertIsNone(table_tex('<table><tr><td><i>X</b></td></tr></table>'))

    def setUp(self):
        self.html = '<table>\n<tr><td>Expression</td><td>Meaning</td></tr>\n<tr><td>A∪B</td><td>A or B; A again</td></tr>\n<tr><td><eq>A^c</eq></td><td>ω and Ω</td></tr></table>'
        atom=dict(id='table',group='g',page=0,kind='table',role='table',block_type='table',
                  content=self.html,length=len(self.html),bbox=[.1,.1,.9,.4])
        self.index=SourceIndex(dict(atoms=[atom],source_page_count=1))
        self.review=dict(source_sha256=fingerprint(self.index.source),evidence='source.png: visual math review',
            table_math=[dict(atom='table',row=1,column=0,quote='A∪B',occurrence=0),
                        dict(atom='table',row=1,column=1,quote='A',occurrence=1),
                        dict(atom='table',row=2,column=1,quote='ω',occurrence=0),
                        dict(atom='table',row=2,column=1,quote='Ω',occurrence=0)])

    def test_cell_and_occurrence_ownership_preserves_original_html(self):
        before=deepcopy(self.index.source)
        runs=validate_reviewed_table_math(self.index,self.review)
        html=annotate_table_math(self.html,runs['table'])
        self.assertIn('<td><eq>A∪B</eq></td>',html)
        self.assertIn('<td>A or B; <eq>A</eq> again</td>',html)
        self.assertIn('<td><eq>A^c</eq></td>',html)
        self.assertIn(r'$\omega$ and $\Omega$',table_tex(html))
        self.assertEqual(self.index.source,before)

    def test_stale_source_wrong_cell_and_math_reannotation_fail(self):
        for row in (dict(row=0),dict(column=9),dict(occurrence=True),dict(occurrence=5),
                    dict(quote=r'\input{x}'),dict(quote='%'),dict(quote='#'),
                    dict(quote='A^c',row=2),dict(atom='missing'),dict(atom=[])):
            review=deepcopy(self.review);review['table_math']=[{**review['table_math'][0],**row}]
            with self.subTest(row=row),self.assertRaises(ValueError):
                validate_reviewed_table_math(self.index,review)
        for changes in (dict(source_sha256='old'),dict(evidence=''),dict(table_math=None),
                        dict(table_math=self.review['table_math']*2)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_reviewed_table_math(self.index,{**self.review,**changes})
        runs=validate_reviewed_table_math(self.index,self.review)
        with self.assertRaises(ValueError):
            annotate_table_math(self.html.replace('A∪B','A∩B'),runs['table'])

    def test_compiled_variables_are_italic_and_words_remain_searchable(self):
        runs=validate_reviewed_table_math(self.index,self.review)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            decisions=dict(corrections=[],joins=[],exclusions=[],bold_spans=[],table_math=runs)
            result=write_project(self.index,decisions,[],[],dict(title='Table review',author=''),
                                 load_config(),root,sample_mode=True)
            self.assertEqual(len([r for r in result['source_order_layout']
                                  if r['method']=='reviewed_literal_table_math_v1']),4)
            compiled=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(compiled['success'],compiled['log_tail'])
            self.assertEqual(compiled['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                text=''.join(p.get_text() for p in pdf)
                self.assertIn('Meaning',text);self.assertIn('again',text)
                spans=[s for p in pdf for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]
                self.assertTrue(any('A' in unicodedata.normalize('NFKC',s['text'])
                                    and ('MI' in s['font'] or s['flags'] & 2) for s in spans))
