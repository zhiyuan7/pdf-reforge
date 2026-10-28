"""Source evidence must govern rules, proof marks and actual PDF placement."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config, executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.native_decorations import native_decorations


class NativeDecorationTests(unittest.TestCase):
    def source(self, root, mark_x=430, method='native', rule=True):
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=500, height=700)
            page.insert_text((50, 100), 'A short mathematical example.')
            if rule:
                page.draw_line((40, 110), (460, 110), width=2, color=(0, 0, 0))
            page.insert_text((50, 200), 'The argument is complete.')
            page.insert_text((mark_x, 200), '■', fontname='china-s', fontsize=10, color=(1, .5, 0))
            page.insert_text((50, 250), 'Symbol')
            page.insert_text((100, 250), '■', fontname='china-s', fontsize=10)
            page.insert_text((120, 250), 'means proof finished.')
            boxes = [page.get_text('dict')['blocks'][0]['bbox'], (50, 188, max(440, mark_x+10), 203),
                     (50, 238, 300, 253)]
            pdf.save(root/'source.pdf')
        atoms = []
        for n, text in enumerate(('A short mathematical example.', 'The argument is complete.\n■',
                                  'Symbol ■ means proof finished.')):
            box = [v/(500 if k%2==0 else 700) for k,v in enumerate(boxes[n])]
            atom = dict(id=str(n), group=str(n), page=0, kind='text', role='text', block_type='text',
                        content=text, length=len(text), bbox=box, block_bbox=box, extraction_method=method)
            if n == 1:
                atom['source_text_runs'] = [dict(start=len(text)-1, end=len(text), color=[1,.5,0],
                                                method='native_span_ink_v1')]
            atoms.append(atom)
        return SourceIndex(dict(source_pdf=str(root/'source.pdf'), atoms=atoms, source_page_count=1))

    def test_requires_native_right_edge_final_glyph_and_unedited_group(self):
        for scenario in ('right', 'inline', 'ocr', 'edited', 'missing', 'different_symbol'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as name:
                index = self.source(Path(name), mark_x=200 if scenario=='inline' else 430,
                                    method='ocr' if scenario=='ocr' else 'native')
                if scenario=='missing': index.source['source_pdf'] = str(Path(name)/'missing.pdf')
                if scenario=='different_symbol': index.atoms[1]['content'] = 'Changed ending. □'
                marks, rules = native_decorations(index, {'1'} if scenario=='edited' else ())
                self.assertEqual(set(marks), {'1'} if scenario=='right' else set())
                if scenario=='right':
                    self.assertEqual(set(rules), {('0','after')})
                    self.assertAlmostEqual(rules['0','after']['thickness_pt'], 2)

    def test_source_rule_rejects_color_short_rules_and_crossing_text(self):
        for scenario in ('color', 'short', 'crossing', 'frame', 'table_rules'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as name:
                root=Path(name);index=self.source(root, rule=False)
                with pymupdf.open(root/'source.pdf') as pdf:
                    if scenario=='frame': pdf[0].draw_rect((40,95,460,110),width=2)
                    else: pdf[0].draw_line((40,100 if scenario=='crossing' else 110),
                        (100 if scenario=='short' else 460,100 if scenario=='crossing' else 110),
                        width=2,color=(1,0,0) if scenario=='color' else (0,0,0))
                    if scenario=='table_rules':pdf[0].draw_line((40,130),(460,130),width=2)
                    pdf.save(root/'changed.pdf')
                index.source['source_pdf']=str(root/'changed.pdf')
                self.assertEqual(native_decorations(index)[1],{})

    def test_compiled_mark_keeps_color_and_right_edge_with_searchable_note_and_rule(self):
        for long in (False, True):
            with self.subTest(long=long), tempfile.TemporaryDirectory() as name:
                root=Path(name);index=self.source(root)
                if long:
                    # An almost full final line must wrap its square cleanly.
                    text='A mathematical argument with many intermediate steps. '*16+'\n■'
                    index.atoms[1].update(content=text,length=len(text),source_text_runs=[
                        dict(start=len(text)-1,end=len(text),color=[1,.5,0],method='native_span_ink_v1')])
                    index=SourceIndex(index.source)
                project=root/'project'
                audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                    [],[],dict(title='Decorations',author=''),load_config(),project,sample_mode=True)
                self.assertTrue(audit['coverage_complete'])
                result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
                with pymupdf.open(project/'main.pdf') as pdf:
                    marks=[(n,s) for n,p in enumerate(pdf) for b in p.get_text('dict')['blocks']
                           for line in b.get('lines',[]) for s in line['spans'] if '■' in s['text']]
                    self.assertEqual(len(marks),2)
                    page,mark=next((n,s) for n,s in marks if s['color']==0xff8000)
                    self.assertGreater(mark['bbox'][0],pdf[page].rect.width*.8)
                    # The following source anchor must not introduce a CJK
                    # character's worth of space after a terminal square.
                    self.assertLess(abs(mark['bbox'][2]-(pdf[page].rect.width-36)),1)
                    note=next(s for _,s in marks if s['color']!=0xff8000)
                    self.assertLess(note['bbox'][0],pdf[page].rect.width*.5)
                    strokes=[d for p in pdf for d in p.get_drawings() if d.get('color')==(0,0,0)
                             and d['rect'].width>300 and 1.9<d['width']<2.1]
                    self.assertEqual(len(strokes),1)
                    self.assertEqual(set(pdf.resolve_names()) & {'source-0','source-1','source-2'},
                                     {'source-0','source-1','source-2'})

    def test_gray_upper_rule_and_black_lower_rule_can_share_a_source_group(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root)
            index.atoms[0]['block_type']='paragraph_title'
            with pymupdf.open(root/'source.pdf') as pdf:
                # Outside the repeated-table-rule guard's distance, but still
                # close to the title's live glyph bounding box.
                pdf[0].draw_line((40,78),(460,78),width=1,color=(.5,.5,.5))
                pdf[0].insert_text((50,50),'Context preceding the example.')
                pdf.save(root/'gray.pdf')
            index.source['source_pdf']=str(root/'gray.pdf')
            prefix={**index.atoms[0], 'id':'prefix','group':'prefix','block_type':'text',
                    'content':'Context preceding the example.','length':30,
                    'bbox':[.1,.05,.7,.073],'block_bbox':[.1,.05,.7,.073]}
            index=SourceIndex({**index.source,'atoms':[prefix,*index.atoms]})
            rules=native_decorations(index)[1]
            self.assertEqual(set(rules),{('0','before'),('0','after')})
            self.assertEqual(rules['0','before']['color'],[.5]*3)
            project=root/'project'
            write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Rules',author=''),load_config(),project,sample_mode=True)
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(project/'main.pdf') as pdf:
                lines=[d for d in pdf[0].get_drawings() if d['rect'].width>300 and d['rect'].height<.1]
                gray=next(d for d in lines if d['color']==(.5,.5,.5))
                black=next(d for d in lines if d['color']==(0,0,0))
                title=pdf[0].search_for('A short mathematical example.')[0]
                self.assertLess(gray['rect'].y0,title.y0)
                self.assertGreater(black['rect'].y0,title.y1)
                preceding=pdf[0].search_for('Context preceding the example.')[0]
                self.assertGreater(gray['rect'].y0-preceding.y1,6)


if __name__ == '__main__':
    unittest.main()
