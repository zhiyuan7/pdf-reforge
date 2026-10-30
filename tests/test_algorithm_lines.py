from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config, executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.algorithm_lines import algorithm_line_rows
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project


class AlgorithmLineTests(unittest.TestCase):
    def source(self):
        specs=[('a','text','Algorithm: sampling\nInput: data\nrepeat\n    '),
               ('m','math',r'x_i=\sqrt{\alpha_t}z_i'),
               ('b','text','  // Draw sample\n    Accumulate loss\nuntil converged')]
        atoms=[dict(id=i,group='algorithm',page=0,kind=k,role='algorithm_body',
                    block_type='code',content=t,length=len(t),bbox=[.1,.2,.9,.5]) for i,k,t in specs]
        return SourceIndex(dict(atoms=atoms,source_page_count=1))

    def test_requires_explicit_complete_unclaimed_algorithm_group(self):
        for defect in ('none','prose','different_page','edited','claimed','excluded','native_code'):
            with self.subTest(defect=defect):
                index=self.source();changed=[];registry=[];excluded=[];native=[]
                if defect=='prose':index.atoms[0]['role']='text'
                if defect=='different_page':index.atoms[1]['page']=1
                if defect=='edited':changed=['m']
                if defect=='claimed':registry=[dict(lo=0,hi=1)]
                if defect=='excluded':excluded=[dict(lo=0,hi=1)]
                if defect=='native_code':native=['a']
                before=deepcopy(index.source)
                self.assertEqual(len(algorithm_line_rows(index,registry,excluded,changed,native)),int(defect=='none'))
                self.assertEqual(index.source,before)

    def test_compiled_algorithm_retains_lines_indentation_math_and_all_destinations(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source()
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Algorithm',author=''),load_config(),root,sample_mode=True)
            self.assertTrue(audit['coverage_complete'])
            result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                self.assertEqual(len(pdf),1);page=pdf[0]
                repeat=page.search_for('repeat')[0];body=page.search_for('Accumulate')[0]
                end=page.search_for('until converged')[0]
                self.assertGreater(body.x0,repeat.x0+5)
                self.assertGreater(body.y0,repeat.y1)
                self.assertGreater(end.y0,body.y1)
                self.assertAlmostEqual(end.x0,repeat.x0,delta=2)
                self.assertEqual({'source-a','source-m','source-b'} & set(pdf.resolve_names()),
                                 {'source-a','source-m','source-b'})
                math_spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[])
                            for s in l['spans'] if 'Math' in s['font'] or 'NewTX' in s['font']]
                self.assertTrue(math_spans)
                self.assertTrue(any(d.get('fill') and min(d['fill'])>.9 for d in page.get_drawings()))
