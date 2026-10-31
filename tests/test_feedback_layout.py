from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unicodedata import normalize

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.native_margin_notes import native_margin_evidence, observed_note_colors
from book_pipeline.publishing.side_notes import explicit_side_note_rows
from book_pipeline.structure.recognized_headings import recognized_heading_registry
from book_pipeline.structure.run_in_titles import mixed_run_in_title_spans


def atom(key, text, role, box, kind='text'):
    return dict(id=key, group=key, page=0, content=text, length=len(text), kind=kind,
                role=role, block_type=role, bbox=box, extraction_method='native')


class FeedbackLayoutTests(unittest.TestCase):
    def mixed(self, root, body_bold=False, below=0):
        path=root/'source.pdf'
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=500,height=700)
            title='Linear and affine functions:'
            page.insert_text((40,100),title,fontname='hebo',fontsize=10)
            x=40+pymupdf.get_text_length(title,fontname='hebo',fontsize=10)+5
            page.insert_text((x,100+below),'An ordinary explanation continues here.',
                             fontname='hebo' if body_bold else 'helv',fontsize=10)
            pdf.save(path)
        box=[.08,.12,.92,.18]
        atoms=[atom('prefix','Linear and affine functions:这是一段介绍线性变换的普通正文，','paragraph_title',box),
               atom('math','f[a+b]=f[a]+f[b]','paragraph_title',box,'math'),
               atom('tail','后面还有关于这个性质的详细说明。','paragraph_title',box)]
        for a in atoms:a['group']='mixed'
        return SourceIndex(dict(source_pdf=str(path),atoms=atoms,source_page_count=1))

    def test_mixed_title_preserves_source_and_only_bolds_label(self):
        for label in ('Linear and affine functions:', '线性与仿射函数：'):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as name:
                index=self.mixed(Path(name))
                index.atoms[0]['content']=label+'这是一段介绍线性变换的普通正文，'
                index=SourceIndex(index.source);before=deepcopy(index.source)
                row=mixed_run_in_title_spans(index)[0]
                self.assertEqual(index.text(row['lo'],row['mid']),label)
                self.assertEqual(recognized_heading_registry(index),[])
                self.assertEqual(index.source,before)

    def test_font_baseline_semantics_edits_numbers_and_atomic_math_guard_split(self):
        for defect in ('bold','below','semantic','edit','number','math','missing'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                index=self.mixed(Path(name),body_bold=defect=='bold',below=16 if defect=='below' else 0)
                barriers=[];changed=[]
                if defect=='semantic':barriers=[dict(lo=0,hi=index.prefix[-1])]
                if defect=='edit':changed=['tail']
                if defect=='number':
                    index.atoms[0]['content']='3.2 '+index.atoms[0]['content']
                    index=SourceIndex(index.source)
                if defect=='math':index.atoms[0]['kind']='math'
                if defect=='missing':Path(index.source['source_pdf']).unlink()
                self.assertEqual(mixed_run_in_title_spans(index,barriers,changed),[])

    def margin(self, root, size=7):
        path=root/'source.pdf'
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=500,height=700)
            page.insert_text((140,154),'The main argument explains a mathematical model.',fontsize=10)
            page.insert_text((140,168),'The second line completes the explanation.',fontsize=10)
            page.insert_text((20,155),'Appendix B.2',fontsize=size,color=(.2,.4,.6))
            page.insert_text((20,165),'Coefficients',fontsize=size,color=(.2,.4,.6))
            pdf.save(path)
        atoms=[atom('main','这里详细介绍数学模型及其性质，并保持原文的论证顺序。'*4,
                    'text',[.28,.20,.90,.25]),
               atom('note','附录B.2\n二项式系数','paragraph_title',[.04,.20,.19,.24])]
        return SourceIndex(dict(source_pdf=str(path),atoms=atoms,source_page_count=1))

    def test_mixed_prose_joins_only_a_live_same_column_continuation(self):
        for defect in ('none','punctuation','changed','missing_glyphs'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                index=self.mixed(Path(name));path=Path(index.source['source_pdf'])
                with pymupdf.open(path) as pdf:
                    page=pdf[0]
                    box=page.search_for('Linear and affine functions:')[0] | page.search_for('An ordinary explanation continues here.')[0]
                    baseline=box.y1+12
                    if defect!='missing_glyphs':page.insert_text((40,baseline),'The next source line continues the argument.',fontsize=10)
                    pdf.saveIncr()
                for a in index.atoms:
                    a['bbox']=[box.x0/500,box.y0/700,box.x1/500,box.y1/700]
                index.atoms[-1]['content']='由这个定义可以得到'+('。' if defect=='punctuation' else '')
                following=atom('next','下一行继续完整说明这一性质。','text',
                    [box.x0/500,(baseline-10.75)/700,box.x1/500,(baseline+3)/700])
                index=SourceIndex({**index.source,'atoms':[*index.atoms,following]})
                rows=mixed_run_in_title_spans(index,changed=['next'] if defect=='changed' else [])
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0].get('right'), 'next' if defect=='none' else None)

    def test_translated_margin_recovers_geometry_color_and_stays_out_of_toc(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.margin(Path(name));before=deepcopy(index.source)
            notes=native_margin_evidence(index)
            self.assertEqual(set(notes),{'note'})
            self.assertEqual(notes['note']['owner'],'main')
            self.assertEqual(notes['note']['color'],[.2,.4,.6])
            self.assertEqual(recognized_heading_registry(index),[])
            rows=explicit_side_note_rows(index)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['inferred_note_groups'],['note'])
            self.assertFalse(rows[0]['main_left'])
            self.assertEqual(index.source,before)

    def test_margin_requires_native_smaller_glyphs_unique_owner_and_no_conflicts(self):
        for defect in ('size','ambiguous','ocr','missing','edit','excluded','semantic'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                index=self.margin(Path(name),size=10 if defect=='size' else 7)
                changed=[];excluded=[];registry=[]
                if defect=='ambiguous':
                    index=SourceIndex({**index.source,'atoms':[dict(index.atoms[0],id='other',group='other'),*index.atoms]})
                if defect=='ocr':index.atoms[-1]['extraction_method']='ocr'
                if defect=='missing':Path(index.source['source_pdf']).unlink()
                if defect=='edit':changed=['note']
                if defect=='excluded':excluded=[dict(lo=index.prefix[-2],hi=index.prefix[-1])]
                if defect=='semantic':registry=[dict(lo=0,hi=index.prefix[-1],kind='theorem')]
                self.assertEqual(explicit_side_note_rows(index,registry,excluded,changed),[])

    def test_owned_ocr_note_uses_live_ink_without_changing_ownership(self):
        for defect in ('none','near_body_size','size','mixed','missing'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                index=self.margin(Path(name),size=10 if defect=='size' else 9 if defect=='near_body_size' else 7)
                index.atoms[-1].update(role='aside_text',block_type='aside_text',extraction_method='ocr')
                rows=explicit_side_note_rows(index)
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['note_colors'],[None])
                if defect=='mixed':
                    with pymupdf.open(index.source['source_pdf']) as pdf:
                        pdf[0].insert_text((20,160),'X',fontsize=7,color=(1,0,0));pdf.saveIncr()
                if defect=='missing':Path(index.source['source_pdf']).unlink()
                before=deepcopy(index.source);original=deepcopy(rows)
                colored=observed_note_colors(index,rows)
                self.assertEqual(colored[0]['main_range'],rows[0]['main_range'])
                self.assertEqual(colored[0]['note_ranges'],rows[0]['note_ranges'])
                self.assertEqual(colored[0]['note_colors'],[[.2,.4,.6]] if defect in {'none','near_body_size'} else [None])
                self.assertEqual(rows,original);self.assertEqual(index.source,before)

    def test_multiple_notes_keep_source_vertical_offsets_without_overlap(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.margin(root)
            index.atoms[0]['bbox']=[.28,.20,.90,.40]
            index.atoms[0]['content']*=2
            index.atoms[0]['length']=len(index.atoms[0]['content'])
            index.atoms[-1].update(role='aside_text',block_type='aside_text')
            with pymupdf.open(index.source['source_pdf']) as pdf:
                pdf[0].insert_text((20,238),'Problem 3.1',fontsize=7,color=(.2,.4,.6));pdf.saveIncr()
            second=atom('later','习题3.1','aside_text',[.04,.32,.19,.35])
            index=SourceIndex({**index.source,'atoms':[*index.atoms,second]})
            rows=explicit_side_note_rows(index)
            self.assertEqual(len(rows),1)
            self.assertAlmostEqual(rows[0]['note_top_fractions'][1],.6)
            project=root/'latex'
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Note positions',author=''),load_config(),project,sample_mode=True)
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(project/'main.pdf') as pdf:
                page=pdf[0];first=page.search_for('附录')[0];second=page.search_for('习题')[0]
                body=page.search_for('这里详细')
                self.assertGreater(second.y0,first.y1+12)
                self.assertLess(second.y1,max(b.y1 for b in body)+30)
                names=pdf.resolve_names()
                self.assertEqual(len({names['source-'+k]['page'] for k in ('main','note','later')}),1)

    def test_compiled_mixed_label_is_body_and_margin_stays_beside_owner(self):
        for fixture in ('mixed','margin'):
            with self.subTest(fixture=fixture), tempfile.TemporaryDirectory() as name:
                root=Path(name);index=getattr(self,fixture)(root);project=root/'latex'
                audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                    recognized_heading_registry(index),[],dict(title='Feedback repair',author=''),
                    load_config(),project,sample_mode=True)
                result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
                self.assertTrue(audit['coverage_complete'])
                with pymupdf.open(project/'main.pdf') as pdf:
                    self.assertTrue(set(audit['source_anchors'].values())<=pdf.resolve_names().keys())
                    self.assertEqual(pdf.get_toc(),[])
                    if fixture=='mixed':
                        self.assertTrue(any(r['method']=='native_mixed_run_in_label' for r in audit['source_order_layout']))
                        spans=[s for p in pdf for b in p.get_text('dict')['blocks']
                               for l in b.get('lines',[]) for s in l['spans']]
                        label=[s for s in spans if 'Linear and affine functions' in normalize('NFKC',s['text'])]
                        prose=[s for s in spans if '这是一段' in s['text']]
                        self.assertTrue(label and prose)
                        self.assertTrue(all('Bold' in s['font'] for s in label))
                        self.assertTrue(all('Bold' not in s['font'] for s in prose))
                    else:
                        main=pdf[0].search_for('这里详细')[0];note=pdf[0].search_for('附录')[0]
                        self.assertLess(note.x1,main.x0)
                        self.assertLess(abs(note.y0-main.y0),14)

    def test_running_head_cannot_overlap_body_even_with_a_long_title(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=SourceIndex(dict(atoms=[atom('body','Introductory text.','text',[.1,.1,.8,.2])]))
            write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],
                          dict(title='Header regression',author=''),load_config(),root,sample_mode=True)
            path=root/'main.tex'
            title='A long mathematical heading with several important definitions and explanations '*4
            path.write_text(path.read_text().replace(r'\end{document}',r'\pagestyle{fancy}\BookHeading{3}{'+title+
                r'}{long-head}\clearpage ORDINARY BODY TEXT.\end{document}'))
            result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                page=pdf[-1];body=page.search_for('ORDINARY BODY')[0]
                head=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']
                      if s['bbox'][1]<body.y0-2]
                self.assertTrue(head)
                self.assertTrue(all(s['bbox'][3]<body.y0-2 for s in head))
                self.assertTrue(all(0<=s['bbox'][0]<s['bbox'][2]<=page.rect.width for s in head))
