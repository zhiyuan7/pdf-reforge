from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config, executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.source_footnotes import source_page_footnotes
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.compiler import compile_project


class SourceFootnoteTests(unittest.TestCase):
    def index(self, reference='An explanation.^{2} Continued prose.', page=0):
        specs=[('body','text','text',reference,0),('note','text','page_footnote','^{2} The definition is ',page),
               ('formula','math','page_footnote',r'z=\sum_i x_i',page),
               ('tail','text','page_footnote',' in this convention.',page)]
        atoms=[dict(id=i,group='note' if i!='body' else i,kind=k,role=r,block_type=r,
                    content=t,length=len(t),page=p,bbox=[.1,.7,.8,.8] if i!='body' else [.1,.2,.8,.3])
               for i,k,r,t,p in specs]
        return SourceIndex(dict(source_pdf='',input_sha256='fixture',mineru_sha256='fixture',
                                source_page_count=2,atoms=atoms))

    def test_matches_explicit_same_page_marker_without_mutating_source(self):
        index=self.index();before=deepcopy(index.source)
        note=source_page_footnotes(index)[0]
        self.assertEqual(note['status'],'matched')
        self.assertEqual(index.text(note['marker_lo'],note['marker_hi']),'2')
        self.assertEqual(note['reference']['edited_lo'],note['reference']['lo'])
        self.assertEqual(index.source,before)

    def test_ambiguous_flat_or_other_page_references_stay_unmatched(self):
        for ref,page in [('A reference.^{2} Another.^{2}',0),('There are 2 models.',0),
                         ('An explanation.^{2}',1)]:
            with self.subTest(ref=ref,page=page):
                self.assertEqual(source_page_footnotes(self.index(ref,page))[0]['status'],'unmatched')

    def test_exclusions_cannot_own_a_note_or_reference(self):
        index=self.index()
        self.assertEqual(source_page_footnotes(index,[dict(lo=index.prefix[1],hi=index.prefix[-1])]),[])
        note=source_page_footnotes(index,[dict(lo=0,hi=index.prefix[1])])[0]
        self.assertEqual(note['status'],'unmatched')

    def test_rendered_note_keeps_formula_anchors_and_lands_below_main_text(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.index();project=Path(name)/'latex'
            notes=source_page_footnotes(index)
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[],page_footnotes=notes),
                [],[],dict(title='Source notes',author=''),load_config(),project,sample_mode=True)
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            self.assertEqual(result['warnings'],[])
            self.assertEqual(audit['footnotes'][0]['status'],'matched')
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(project/'main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                self.assertGreater(pdf[0].search_for('The definition')[0].y0,
                                   pdf[0].search_for('Continued prose')[0].y1)
                self.assertEqual(pdf[0].get_text().count('The definition'),1)
                self.assertTrue(all('source-'+a['id'] in pdf.resolve_names() for a in index.atoms))
