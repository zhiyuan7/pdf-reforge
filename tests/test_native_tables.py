from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config, executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.native_tables import native_ruled_tables
from book_pipeline.publishing.native_lateral import native_side_notes


class NativeTableTests(unittest.TestCase):
    def source(self, root, defect='none'):
        atoms=[]
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=500,height=700)
            for n,(y,left,right) in enumerate([(140,'Symbol','Meaning'),(175,'x','Input vector'),
                                              (200,'w','Weight vector'),(225,'b','Bias vector')]):
                boxes=[]
                for x,text in [(80,left),(230,right)]:
                    page.insert_text((x,y),text,fontsize=10)
                    boxes.append(page.get_text('dict')['blocks'][-1]['bbox'])
                box=[min(b[0] for b in boxes)/500,min(b[1] for b in boxes)/700,
                     max(b[2] for b in boxes)/500,max(b[3] for b in boxes)/700]
                for j,text in enumerate([left,'\n',right]):
                    atoms.append(dict(id=f'{n}-{j}',group=str(n),page=0,
                        kind='math' if n and j==0 else 'text',role='text',block_type='text',
                        content=text,length=len(text),bbox=box,extraction_method='native'))
            for y in ((120,250) if defect=='two_rules' else (120,150,250)):
                page.draw_line((60,y),(375,y),width=.5)
            page.insert_text((400,185),'Shared note.',fontsize=7)
            b=page.get_text('dict')['blocks'][-1]['bbox']
            atoms.append(dict(id='note',group='note',page=0,kind='text',role='page_footnote',
                block_type='page_footnote',content='Shared note.',length=12,
                bbox=[v/(500 if k%2==0 else 700) for k,v in enumerate(b)],extraction_method='native'))
            page.insert_text((40,300),'Following main paragraph.',fontsize=10)
            atoms.append(dict(id='body',group='body',page=0,kind='text',role='text',block_type='text',
                content='Following main paragraph.',length=25,bbox=[.08,.41,.75,.44],extraction_method='native'))
            pdf.save(root/'source.pdf')
        if defect=='scan':atoms[3]['extraction_method']='ocr'
        if defect=='separator':atoms[4]['content']=' ';atoms[4]['length']=1
        if defect=='crossing':atoms[3]['bbox'][2]=.8
        if defect=='foreign':atoms.insert(6,dict(atoms[-1],id='foreign',group='foreign'))
        if defect=='math_note':
            old=atoms[12]
            atoms[12:13]=[dict(old,content='Shared ',length=7),
                dict(old,id='note-math',kind='math',content='n',length=1),
                dict(old,id='note-tail',content='ote.',length=4)]
        return SourceIndex(dict(source_pdf=str(root/'source.pdf'),atoms=atoms))

    def test_table_requires_rules_live_columns_and_complete_contiguous_rows(self):
        for defect in ('none','two_rules','scan','separator','crossing','foreign','excluded','semantic'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as name:
                index=self.source(Path(name),defect);before=deepcopy(index.source)
                tables=native_ruled_tables(index,
                    [dict(lo=0,hi=index.prefix[12],kind='example')] if defect=='semantic' else (),
                    [dict(lo=0,hi=index.prefix[12])] if defect=='excluded' else ())
                self.assertEqual(len(tables),int(defect=='none'),defect)
                self.assertEqual(index.source,before)

    def test_markerless_native_note_moves_only_with_live_small_font_and_no_callout(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name));tables=native_ruled_tables(index)
            lo,hi=index.prefix[12:14]
            note=dict(id='footnote',lo=lo,hi=hi,status='unmatched',marker=None)
            for defect in ('none','matched','marker','edited','excluded','wrong_text'):
                n={**note}
                index.atoms[index.ids['note']]['content']='Wrong text!!' if defect=='wrong_text' else 'Shared note.'
                if defect=='matched':n['status']='matched'
                if defect=='marker':n['marker']={'text':'1'}
                rows,moved=native_side_notes(index,tables,
                    exclusions=[dict(lo=lo,hi=hi)] if defect=='excluded' else (),footnotes=[n],
                    changed={'note'} if defect=='edited' else ())
                self.assertEqual(len(rows),int(defect=='none'),defect)
                self.assertEqual(moved,{'footnote'} if defect=='none' else set())

    def test_compiled_table_keeps_columns_math_note_and_each_destination_once(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);lo,hi=index.prefix[12:14]
            decisions=dict(corrections=[],joins=[],exclusions=[],bold_spans=[],
                page_footnotes=[dict(id='footnote',lo=lo,hi=hi,status='unmatched',marker=None)])
            audit=write_project(index,decisions,[],[],dict(title='Native table',author=''),
                load_config(),root/'latex',sample_mode=True)
            self.assertTrue(audit['coverage_complete'])
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                page=pdf[0];left=page.search_for('Symbol')[0];right=page.search_for('Meaning')[0]
                self.assertGreater(right.x0,left.x1+10)
                self.assertAlmostEqual(left.y0,right.y0,delta=1)
                symbol=(page.search_for('x') or page.search_for('𝑥'))[0];meaning=page.search_for('Input vector')[0]
                self.assertAlmostEqual(symbol.y1,meaning.y1,delta=3)
                self.assertGreater(page.search_for('Shared note.')[0].x0,right.x1)
                self.assertEqual(page.get_text().count('Shared note.'),1)
                names=pdf.resolve_names()
                self.assertEqual({f"source-{a['id']}" for a in index.atoms}&set(names),
                                 {f"source-{a['id']}" for a in index.atoms})

    def test_compiled_side_note_preserves_math_comment_newlines(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root,'math_note');lo,hi=index.prefix[12:16:3]
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[],
                page_footnotes=[dict(id='footnote',lo=lo,hi=hi,status='unmatched',marker=None)]),
                [],[],dict(title='Mixed note',author=''),load_config(),root/'latex',sample_mode=True)
            self.assertTrue(audit['coverage_complete'])
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                self.assertTrue(pdf[0].search_for('Shared'))
                self.assertTrue(pdf[0].search_for('ote.'))
                self.assertTrue({'source-note','source-note-math','source-note-tail'}<=set(pdf.resolve_names()))


if __name__=='__main__':
    unittest.main()
