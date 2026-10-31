from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.side_notes import explicit_side_note_rows


class SideFormulaIntroductionTests(unittest.TestCase):
    def source(self):
        atoms=[]
        for key,kind,role,text,box in [
                ('intro','text','text','We obtain the following expression for the probability distribution:',[.25,.3,.84,.36]),
                ('note','text','aside_text','Problem 2.4',[.1,.31,.21,.34]),
                ('math','display_math','equation',r'p(x)=\frac{\exp(x)}{1+\exp(x)}\tag{2.4}',[.3,.37,.83,.42]),
                ('body','text','text','A separate explanation follows the equation.',[.25,.44,.84,.48])]:
            atoms.append(dict(id=key,group=key,page=0,kind=kind,role=role,block_type=role,
                content=text,length=len(text),bbox=box,extraction_method='native'))
        return SourceIndex(dict(atoms=atoms,source_page_count=1))

    def test_adjacent_owned_note_is_skipped_without_losing_formula_ownership(self):
        from book_pipeline.publishing.side_notes import include_following_display
        index=self.source();rows=explicit_side_note_rows(index);before=deepcopy(rows)
        self.assertEqual(len(rows),1)
        # The renderer marks both the main paragraph and its relocated notes
        # as skipped before extending the unit. These are its own intervals.
        skips=[dict(lo=lo,hi=hi) for lo,hi in [rows[0]['main_range'],*rows[0]['note_ranges']]]
        result=include_following_display(index,rows,skips)
        self.assertEqual(result[0]['following_range'],index.prefix[2:4])
        self.assertEqual(result[0]['hi'],index.prefix[3])
        self.assertEqual(rows,before)

    def test_cross_page_tall_edited_semantic_and_geometry_barriers_keep_normal_flow(self):
        from book_pipeline.publishing.side_notes import include_following_display
        for defect in ('page','tall','edit','object','column','gap','panel','recovery','no_colon'):
            with self.subTest(defect=defect):
                index=self.source();barriers=[];changed=[]
                if defect=='page':index.atoms[2]['page']=1
                if defect=='tall':index.atoms[2]['bbox'][3]=.6
                if defect=='edit':changed=['math']
                if defect=='object':barriers=[dict(lo=index.prefix[2],hi=index.prefix[3])]
                if defect=='column':index.atoms[2]['bbox']=[.01,.37,.2,.42]
                if defect=='gap':index.atoms[2]['bbox']=[.3,.5,.83,.55]
                if defect=='panel':index.atoms[2]['source_panel']={'id':'panel'}
                if defect=='recovery':index.atoms[2]['recovery_issue']='unconfirmed'
                if defect=='no_colon':index.atoms[0]['content']='An ordinary complete paragraph.'
                rows=explicit_side_note_rows(index)
                self.assertEqual(include_following_display(index,rows,barriers,changed),rows)

    def test_ordinary_text_and_end_of_source_are_not_absorbed(self):
        from book_pipeline.publishing.side_notes import include_following_display
        index=self.source();index.atoms[2].update(kind='text',role='text',block_type='text')
        rows=explicit_side_note_rows(index)
        self.assertEqual(include_following_display(index,rows),rows)
        index=SourceIndex({**index.source,'atoms':index.atoms[:2]});rows=explicit_side_note_rows(index)
        self.assertEqual(include_following_display(index,rows),rows)

    def test_compiled_side_note_introduction_and_equation_move_together(self):
        index=self.source()
        with tempfile.TemporaryDirectory() as temporary:
            project=Path(temporary)/'latex'
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Side equation',author=''),load_config(),project,sample_mode=True)
            row=next(r for r in audit['source_order_layout'] if r['method']=='explicit_geometric_side_note')
            self.assertEqual(row['following_range'],index.prefix[2:4])
            for path in (project/'chapters').glob('*.tex'):
                path.write_text(r'\noindent Preceding paragraph.\par'
                    r'\vspace*{\dimexpr\textheight-3.5\baselineskip\relax}'+'\n'+path.read_text())
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(project/'main.pdf') as pdf:
                names=pdf.resolve_names()
                self.assertTrue(set(audit['source_anchors'].values())<=names.keys())
                self.assertEqual(names['source-intro']['page'],names['source-math']['page'])
                self.assertEqual(names['source-note']['page'],names['source-math']['page'])
                self.assertGreater(names['source-intro']['page'],0)
                self.assertEqual(sum(p.get_text().count('Problem 2.4') for p in pdf),1)
