from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.cross_figure_prose import cross_figure_prose
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project


class CrossFigureProseTests(unittest.TestCase):
    def source(self, root, *, stop=False, indent=0, font_size=10, plain=False):
        path = root/'original.pdf'
        text = 'The model combines several smaller networks; one'+('.' if stop else '')
        width = pymupdf.get_text_length(text, fontsize=10)
        right_page = 1 if plain else 2
        specs = [(0, 40, 530, text, 10), (right_page, 40+indent, 130 if plain else 260,
                 'interpretation completes this sentence.', font_size)]
        atoms = []
        with pymupdf.open() as pdf:
            for _ in range(3): pdf.new_page(width=500, height=700)
            for n, x, y, content, size in specs:
                pdf[n].insert_text((x,y),content,fontsize=size)
                box = [x/500, (y-1.075*size)/700, (40+width)/500, (y+.3*size)/700]
                atoms.append(dict(id='left' if n==0 else 'right',group=str(n),page=n,
                    kind='text',role='text',block_type='text',content=content,
                    length=len(content),bbox=box,extraction_method='native'))
            following='After the figures, a new paragraph starts.'
            pdf[2].insert_text((40,370),following,fontsize=10)
            pdf.save(path)
        im = root/'diagram.png'
        pix = pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,250,280),False)
        pix.clear_with(180);pix.save(im)
        figure = dict(id='figure',group='figure',page=1,kind='image',role='image_body',
            block_type='image',content='',length=1,bbox=[.1,.2,.8,.6],asset=str(im),extraction_method='native')
        caption = dict(id='caption',group='caption',page=1,kind='text',role='image_caption',
            block_type='image_caption',content='Figure 1. A complete diagram.',length=29,
            bbox=[.15,.62,.75,.65],extraction_method='native')
        return SourceIndex(dict(source_pdf=str(path),source_page_count=3,
            atoms=[atoms[0],*([] if plain else [figure,caption]),atoms[1],dict(id='following',group='following',
                page=2,kind='text',role='text',block_type='text',content=following,
                length=len(following),bbox=[.08,.51,.55,.54],extraction_method='native')]))

    def test_live_native_boundary_moves_only_continuation_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as name:
            index = self.source(Path(name)); before = deepcopy(index.source)
            rows = cross_figure_prose(index)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['figures'],['figure'])
            self.assertEqual(rows[0]['moved_range'],index.prefix[3:5])
            self.assertEqual(index.source,before)
            self.assertEqual(cross_figure_prose(index,pages=[1,3]),[])
            self.assertEqual(cross_figure_prose(index,pages=[]),[])
            self.assertEqual(len(cross_figure_prose(index,pages=[1,2,3])),1)
            for a in (index.atoms[0], index.atoms[-1]):
                a.update(content='译文仍使用原文段落证据',length=12)
            self.assertEqual(len(cross_figure_prose(SourceIndex(index.source))),1)

    def test_translation_punctuation_does_not_override_live_source_boundary(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name))
            index.atoms[0].update(content='译文错误地加上了句号。',length=12)
            self.assertEqual(len(cross_figure_prose(SourceIndex(index.source))),1)

    def test_consecutive_page_continuations_need_no_intervening_figure(self):
        for defect in ('none','period','indent','font','gap','body','heading'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as name:
                index=self.source(Path(name),plain=True,stop=defect=='period',
                    indent=16 if defect=='indent' else 0,font_size=7 if defect=='font' else 10)
                registry=[]
                if defect=='gap':index.atoms[1]['page']=2
                if defect=='body':
                    middle=deepcopy(index.atoms[0]);middle.update(id='middle',group='middle',
                        content='A distinct paragraph.',bbox=[.08,.4,.54,.44])
                    index=SourceIndex(dict(index.source,atoms=[index.atoms[0],middle,*index.atoms[1:]]))
                if defect=='heading':registry=[dict(lo=index.prefix[1],hi=index.prefix[2],kind='section')]
                rows=cross_figure_prose(index,registry,pages=[1,2,3])
                if defect=='none':
                    self.assertEqual(len(rows),1)
                    self.assertEqual(rows[0]['method'],'native_sentence_across_pages')
                    self.assertEqual(rows[0]['figures'],[])
                else:self.assertEqual(rows,[])

    def test_plain_page_join_compiles_as_one_paragraph_without_duplication(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root,plain=True);project=root/'latex'
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[],
                reviewed_cross_figure_pages=[1,2]),[],[],dict(title='Page continuation',author=''),
                load_config(),project,sample_mode=True)
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            self.assertEqual(result['warnings'],[])
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(project/'main.pdf') as pdf:
                names=pdf.resolve_names()
                self.assertEqual(names['source-left']['page'],names['source-right']['page'])
                text=' '.join(p.get_text() for p in pdf)
                self.assertEqual(text.count('interpretation completes'),1)
                blocks=[b[4] for p in pdf for b in p.get_text('blocks') if b[6]==0]
                self.assertTrue(any('networks; one' in b and 'interpretation completes' in b for b in blocks))

    def test_sentence_indent_font_boundaries_and_unproven_content_stop_motion(self):
        for defect in ('period','indent','font','heading','excluded','occupied','edit','no_source',
                       'ocr','display','body','unmatched_note','unselected_page','panel','recovery'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                index = self.source(Path(name),stop=defect=='period',
                                    indent=16 if defect=='indent' else 0,
                                    font_size=7 if defect=='font' else 10)
                registry=[];excluded=[];occupied=[];changed=[]
                span=dict(lo=index.prefix[3],hi=index.prefix[4])
                if defect=='heading':registry=[dict(span,kind='section')]
                if defect=='excluded':excluded=[dict(span,reason='outside_scope')]
                if defect=='occupied':occupied=[span]
                if defect=='edit':changed=['right']
                if defect=='no_source':index.source['source_pdf']=''
                if defect=='ocr':index.atoms[index.ids['right']]['extraction_method']='ocr'
                if defect=='display':index.atoms[2]['kind']='display_math'
                if defect=='body':index.atoms[2].update(role='text',block_type='text')
                if defect=='unmatched_note':index.atoms[2]['role']='page_footnote'
                if defect=='unselected_page':index.atoms[1]['page']=2;index.atoms[2]['page']=2
                if defect=='panel':index.atoms[1]['source_panel']='panel'
                if defect=='recovery':index.atoms[1]['recovery_issue']='unknown'
                self.assertEqual(cross_figure_prose(index,registry,excluded,occupied,changed=changed),[])

    def test_only_matched_footnote_may_be_bypassed(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name));index.atoms[2]['role']='page_footnote'
            note=dict(lo=index.prefix[2],hi=index.prefix[3],status='matched')
            self.assertEqual(len(cross_figure_prose(index,notes=[note])),1)
            note['status']='unmatched'
            self.assertEqual(cross_figure_prose(index,notes=[note]),[])

    def test_independently_proven_running_header_does_not_interrupt_a_sentence(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name));index.atoms[2].update(role='paragraph_title',block_type='paragraph_title')
            self.assertEqual(cross_figure_prose(index),[])
            span=dict(lo=index.prefix[2],hi=index.prefix[3],reason='source_marginal')
            self.assertEqual(cross_figure_prose(index,exclusions=[span]),[])
            proven=dict(span,method='native_repeated_ruled_running_header')
            self.assertEqual(len(cross_figure_prose(index,exclusions=[proven])),1)
            # A real heading elsewhere in the interval remains a barrier.
            index.atoms[1].update(kind='text',role='paragraph_title')
            self.assertEqual(cross_figure_prose(index,exclusions=[proven]),[])

    def test_a_continuation_with_a_footnote_reference_is_not_moved_alone(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name))
            note=dict(lo=index.prefix[2],hi=index.prefix[3],status='matched',
                      reference=dict(lo=index.prefix[3]+2,hi=index.prefix[3]+3))
            self.assertEqual(cross_figure_prose(index,notes=[note]),[])

    def test_compiled_sentence_is_uninterrupted_and_figures_retain_anchors(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);project=root/'latex'
            cfg=load_config();cfg['float_assets']=True
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[],
                reviewed_cross_figure_pages=[1,2,3]),
                [],[],dict(title='Continuation',author=''),cfg,project,sample_mode=True)
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            self.assertTrue(audit['coverage_complete'])
            unreviewed=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Unreviewed',author=''),cfg,root/'unreviewed',sample_mode=True)
            self.assertFalse(any(r['method']=='native_sentence_across_figures'
                                 for r in unreviewed['source_order_layout']))
            self.assertEqual(len([r for r in audit['source_order_layout']
                                  if r['method']=='native_sentence_across_figures']),1)
            with pymupdf.open(project/'main.pdf') as pdf:
                names=pdf.resolve_names()
                self.assertTrue(set(audit['source_anchors'].values())<=names.keys())
                left=names['source-left'];right=names['source-right'];image=names['source-figure']
                self.assertEqual(left['page'],right['page'])
                self.assertLessEqual(right['page'],image['page'])
                following=names['source-following']
                self.assertTrue(following['page']>image['page'] or (
                    following['page']==image['page'] and following['to'][1]<image['to'][1]))
                text=' '.join(p.get_text() for p in pdf)
                self.assertEqual(text.count('interpretation completes'),1)
                blocks=[b[4] for b in pdf[right['page']].get_text('blocks') if b[6]==0]
                self.assertTrue(any('networks; one' in b and 'interpretation completes' in b for b in blocks))
