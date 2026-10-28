"""Panel boundaries must be geometric, owned and independent of word counts."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.ingestion.panels import annotate_panels, text_panels, panel_environments, panel_end_marks
from book_pipeline.ingestion.native import merge_text_blocks
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import Renderer, write_project


class SourcePanelTests(unittest.TestCase):
    def marked_source(self):
        source = self.source()
        panel = source.atoms[-1]['source_panel']
        mark = dict(id='end-mark', group='end-mark', page=0, kind='text', role='text',
                    block_type='text', content='♡', length=1, bbox=[.86,.27,.88,.295],
                    source_panel=panel, source_panel_title=False)
        return SourceIndex({**source.source, 'atoms': [*source.atoms, mark]})

    def test_end_mark_requires_isolated_lower_right_owned_geometry(self):
        source = self.marked_source()
        wrappers, _ = panel_environments(source, [])
        self.assertEqual(set(panel_end_marks(source, wrappers)), {'end-mark'})
        for field, value in [('bbox',[.12,.27,.14,.295]),
                             ('bbox',[.86,.14,.88,.165]),
                             ('bbox',[.6,.27,.88,.295]),
                             ('content','A symbol ♡ in prose'), ('content','x'),
                             ('kind','math'), ('source_panel_title',True),
                             ('group','1'), ('source_panel',{})]:
            with self.subTest(field=field,value=value):
                other = self.marked_source()
                other.atoms[-1][field] = value
                other = SourceIndex(other.source)
                self.assertEqual(panel_end_marks(other, wrappers), {})
        self.assertEqual(panel_end_marks(source, wrappers, {'end-mark'}), {})
        self.assertEqual(panel_end_marks(source, []), {})
        # A symbol before the end of a panel is not a closing mark.
        source.atoms.append({**source.atoms[1], 'id':'after', 'group':'after'})
        source = SourceIndex(source.source)
        wrappers, _ = panel_environments(source, [])
        self.assertEqual(panel_end_marks(source, wrappers), {})

    def test_end_mark_keeps_exact_content_coverage_and_destination_inside_box(self):
        with tempfile.TemporaryDirectory() as name:
            index = self.marked_source()
            renderer = Renderer(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],Path(name))
            tex = ''.join(renderer.render())
            start, end = tex.index(r'\BookPanelEndMark'), tex.index(r'\end{BookSourcePanel}')
            self.assertLess(start, tex.index(r'\hypertarget{source-end-mark}'))
            self.assertLess(tex.index(r'\heartsuit'), end)
            self.assertEqual(tex.count(r'\heartsuit'), 1)
            self.assertEqual(set(renderer.source_anchors), {'0','1','end-mark'})
            self.assertEqual(sum(r['end']-r['start'] for r in renderer.ledger), index.prefix[-1])
            evidence = [r for r in renderer.source_order_layout if r['method']=='source_panel_lower_right_end_mark']
            self.assertEqual(len(evidence), 1)

    def test_compiled_end_mark_and_its_link_share_the_right_edge_and_page(self):
        from book_pipeline.config import load_config, executable
        from book_pipeline.publishing.compiler import compile_project
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_project(self.marked_source(),dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                          [],[],dict(title='Panels',author=''),load_config(),root,sample_mode=True)
            result = compile_project(root,{tool:executable(tool) for tool in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                marks = [(n,rect) for n,page in enumerate(pdf) for rect in page.search_for('♡')]
                self.assertEqual(len(marks),1)
                page,rect = marks[0]
                self.assertGreater(rect.x0,pdf[page].rect.width*.8)
                target = pdf.resolve_names()['source-end-mark']
                self.assertEqual(target['page'],page)
                self.assertLess(abs(target['to'][0]-rect.x0),2)
                self.assertIn('The zero vector is unique.',pdf[page].get_text())
                body = pdf[page].search_for('The zero vector is unique.')[0]
                # The source decoration reserves its own ink height, without
                # adding a complete prose baseline below the final sentence.
                self.assertLess(rect.y1-body.y1,12)

    def test_formula_band_keeps_unanimous_panel_ownership_but_rejects_mixed_ownership(self):
        from book_pipeline.ingestion.formula_bands import apply_formula_corrections, METHOD
        for mixed in [False,True]:
            with self.subTest(mixed=mixed):
                source=self.source();source.source.update(input_sha256='pdf',mineru_sha256='middle',pages=[])
                for a in source.atoms:a.update(bbox=[.2,.2,.8,.25],block_bbox=[.2,.2,.8,.25])
                if mixed:source.atoms[-1].pop('source_panel')
                report=dict(method=METHOD,status='complete',source=source.source,usage={},records=[dict(
                    id='band',page=0,groups=['0','1'],preserve=[],bbox=[0,.2,1,.25],
                    answer=dict(items=[dict(kind='formula',latex='x+0=x',number='')]))])
                fixed=apply_formula_corrections(source,report)
                self.assertEqual(bool(fixed.atoms[0].get('source_panel')),not mixed)
                if not mixed:
                    wrappers,_=panel_environments(fixed,[])
                    self.assertEqual(wrappers[0]['hi'],fixed.prefix[-1])
                    with tempfile.TemporaryDirectory() as name:
                        renderer=Renderer(fixed,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],Path(name))
                        tex=''.join(renderer.render())
                        self.assertLess(tex.index(r'\begin{equation*}'),tex.index(r'\end{BookSourcePanel}'))

    def test_nested_fills_and_protruding_title_strip_are_one_panel(self):
        with pymupdf.open() as d:
            p=d.new_page(width=500,height=700)
            p.draw_rect((40,120,460,210),fill=(0,.65,.32))
            p.draw_rect((41,121,459,209),fill=(.95,.98,.96))
            p.draw_rect((50,110,450,130),fill=(0,.65,.32))
            p.insert_text((60,124),'Definition 1.1',fontsize=10)
            p.insert_text((60,160),'The sum is defined coordinatewise.',fontsize=10)
            panels=text_panels(p)
            self.assertEqual(len(panels),1)
            self.assertAlmostEqual(panels[0]['bbox'][1],110/700,places=4)
            blocks=[dict(type='paragraph_title',bbox=[.12,112/700,.3,125/700]),
                    dict(type='text',bbox=[.12,145/700,.7,165/700]),
                    dict(type='text',bbox=[.12,.4,.7,.5])]
            annotate_panels(p,blocks)
            self.assertTrue(blocks[0]['source_panel_title'])
            self.assertFalse(blocks[1]['source_panel_title'])
            self.assertNotIn('source_panel',blocks[2])

    def source(self,intervening=False):
        panel=dict(id='p1-panel-1',bbox=[.1,.1,.9,.3],color=[0,.65,.32])
        atoms=[]
        for n,(text,title) in enumerate([('Definition 1.1',True),('The zero vector is unique.',False)]):
            if intervening and n:
                atoms.append(dict(id='note',group='note',page=0,kind='text',role='text',
                                  block_type='text',content='Outside note.',length=13))
            atoms.append(dict(id=str(n),group=str(n),page=0,kind='text',role='text',
                block_type='paragraph_title' if title else 'text',content=text,length=len(text),
                source_panel=panel,source_panel_title=title))
        return SourceIndex(dict(atoms=atoms,source_page_count=1))

    def test_sparse_declaration_renders_once_with_exact_title_and_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source()
            renderer=Renderer(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],Path(name))
            tex=''.join(renderer.render())
            self.assertEqual(tex.count('Definition 1.1'),1)
            self.assertIn(r'\begin{BookSourcePanel}{',tex)
            self.assertLess(tex.index('The zero vector'),tex.index(r'\end{BookSourcePanel}'))
            self.assertEqual(sum(r['end']-r['start'] for r in renderer.ledger),index.prefix[-1])

    def test_intervening_outside_content_and_existing_semantics_reject_wrapper(self):
        wrappers,evidence=panel_environments(self.source(True),[])
        self.assertEqual(wrappers,[])
        self.assertEqual(evidence[0]['reason'],'noncontiguous_source_ownership')
        wrappers,evidence=panel_environments(self.source(),[dict(kind='definition',lo=0,hi=40)])
        self.assertEqual(wrappers,[])
        self.assertEqual(evidence[0]['reason'],'existing_semantic_environment')

    def test_adjacent_exercise_numbers_and_panel_boundaries_prevent_prose_merge(self):
        def b(text,y,panel=None):
            block=dict(type='text',bbox=[.1,y,.5,y+.02],content=text)
            if panel:block['source_panel']=panel
            return block
        self.assertEqual(len(merge_text_blocks([b('One paragraph continues',.1),b('on the next line.',.123)])),1)
        self.assertEqual(len(merge_text_blocks([b('1.1 First exercise.',.1),b('1.2 Second exercise.',.123)])),2)
        self.assertEqual(len(merge_text_blocks([b('Ordinary prose.',.1),b('Panel text.',.123,dict(id='panel'))])),2)
        title=b('Source strip title.',.1,dict(id='panel'));title['source_panel_title']=True
        self.assertEqual(len(merge_text_blocks([title,b('Panel text.',.123,dict(id='panel'))])),2)
