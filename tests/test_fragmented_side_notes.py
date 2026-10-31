"""A margin note between serialized body fragments cannot own their text."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.native_prose import fragmented_native_prose
from book_pipeline.publishing.side_notes import explicit_side_note_rows


class FragmentedSideNoteTests(unittest.TestCase):
    def test_short_last_line_can_prove_ownership_of_an_interleaved_margin_note(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'final-line.pdf'
            first='The loss quantifies the sum of squared deviations between predictions and'
            last='every observed value of the source input.'
            specs=[('opening',first,150,100,10),
                   ('note','Appendix A',65,110,7),
                   ('note','Sets',65,118,7),
                   ('last',last,150,112,10)]
            boxes={};texts={}
            with pymupdf.open() as pdf:
                page=pdf.new_page(width=600,height=700)
                for group,text,x,y,size in specs:
                    page.insert_text((x,y),text,fontsize=size)
                    box=pymupdf.Rect(x,y-1.075*size,x+pymupdf.get_text_length(text,fontsize=size),y+.299*size)
                    boxes[group]=boxes[group]|box if group in boxes else box
                    texts[group]=texts.get(group,'')+' '+text
                # Tight native extraction bounds on a single source line.
                boxes['last'].y0=104;boxes['last'].y1=113
                pdf.save(path)
            atoms=[dict(id=g,group=g,page=0,kind='text',role='text',block_type='text',
                        content=texts[g].strip(),length=len(texts[g].strip()),
                        bbox=[v/(600 if n%2==0 else 700) for n,v in enumerate(boxes[g])],
                        extraction_method='native') for g in boxes]
            index=SourceIndex(dict(source_pdf=str(path),source_page_count=1,atoms=atoms))
            rows=explicit_side_note_rows(index)
            self.assertEqual(len(rows),1)
            self.assertEqual([a['id'] for bounds in rows[0]['main_ranges']
                              for a,_,_ in index.fragments(*bounds)],['opening','last'])
            self.assertEqual(rows[0]['inferred_note_groups'],['note'])

    def source(self, root, *, note_size=7, gap=5, leading=12):
        path = root/'source.pdf'
        end = 'The first sentence ends.'
        tail = 'With activation the input space is divided'
        split = 40+pymupdf.get_text_length(end, fontsize=10)+gap
        edge = split+pymupdf.get_text_length(tail, fontsize=10)
        specs = [('opening', 'A mathematical model with nonlinear input and output relations', 40,100,10),
                 ('opening', end,40,112,10),
                 ('note','Notebook 2.4',edge+15,109,note_size),
                 ('note','Activation',edge+15,118,note_size),
                 ('tail',tail,split,112,10),
                 ('continuation','into separate regions.',40,112+leading,10),
                 ('next','An independent paragraph follows.',55,170,10)]
        boxes = {}; texts = {}
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=600,height=700)
            for group,text,x,y,size in specs:
                page.insert_text((x,y),text,fontsize=size)
                box=pymupdf.Rect(x,y-1.075*size,x+pymupdf.get_text_length(text,fontsize=size),y+.299*size)
                boxes[group]=boxes[group]|box if group in boxes else box
                texts[group]=texts.get(group,'')+' '+text
            # Native multiline block unions can enclose a separate tail.
            boxes['opening'].x1=edge
            boxes['continuation'].x1=edge
            pdf.save(path)
        atoms=[dict(id=g,group=g,page=0,kind='text',role='text',block_type='text',
                    content=texts[g].strip(),length=len(texts[g].strip()),
                    bbox=[v/(600 if n%2==0 else 700) for n,v in enumerate(boxes[g])],
                    extraction_method='native') for g in boxes]
        return SourceIndex(dict(source_pdf=str(path),source_page_count=1,atoms=atoms))

    def test_note_with_fragmented_owners_is_attached_only_after_prose_proof(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name));before=deepcopy(index.source)
            rows=explicit_side_note_rows(index)
            self.assertEqual(len(rows),1)
            row=rows[0]
            body=[a['id'] for bounds in row['main_ranges'] for a,_,_ in index.fragments(*bounds)]
            self.assertEqual(body,['opening','tail','continuation'])
            self.assertEqual([a['id'] for a,_,_ in index.fragments(*row['note_ranges'][0])],['note'])
            self.assertEqual([(j['left'],j['right']) for j in row['prose_joins']],
                             [('opening','tail'),('tail','continuation')])
            self.assertEqual(index.source,before)
            for a in index.atoms:
                a.update(content='校订后的中文段落内容。'*5,length=len('校订后的中文段落内容。'*5))
            self.assertEqual(len(explicit_side_note_rows(SourceIndex(index.source))),1)

    def test_body_sized_note_or_structural_barrier_is_never_bypassed(self):
        for defect in ('body_font','edited','excluded','heading','image','large_gap'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as name:
                index=self.source(Path(name),note_size=10 if defect=='body_font' else 7,
                                  gap=30 if defect=='large_gap' else 5)
                registry=[];excluded=[];changed=[]
                n=index.ids['note'];lo,hi=index.prefix[n:n+2]
                if defect=='edited':changed=['note']
                if defect=='excluded':excluded=[dict(lo=lo,hi=hi)]
                if defect=='heading':registry=[dict(lo=lo,hi=hi)]
                if defect=='image':index.atoms[n]['kind']='image'
                pairs={(j['left'],j['right']) for j in fragmented_native_prose(index,registry,excluded,changed)}
                self.assertNotIn(('opening','tail'),pairs)
                self.assertEqual(explicit_side_note_rows(index,registry,excluded,changed),[])

    def test_compiled_body_is_continuous_and_note_is_separate_searchable_once(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);project=root/'latex'
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Fragmented margin',author=''),load_config(),project,sample_mode=True)
            self.assertTrue(audit['coverage_complete'])
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(project/'main.pdf') as pdf:
                names=pdf.resolve_names();self.assertTrue(set(audit['source_anchors'].values())<=names.keys())
                pages={names['source-'+g]['page'] for g in ('opening','tail','continuation','note')}
                self.assertEqual(len(pages),1)
                page=pdf[pages.pop()];text=page.get_text()
                self.assertEqual(text.count('Notebook 2.4'),1)
                self.assertIn('ends. With activation',text.replace('\n',' '))
                self.assertIn('divided into separate regions.',text.replace('\n',' '))
                note=page.search_for('Notebook 2.4')[0]
                main=page.search_for('With activation')[0]
                self.assertGreater(note.x0,main.x0)
                self.assertGreater(page.search_for('An independent')[0].y0,main.y1)
