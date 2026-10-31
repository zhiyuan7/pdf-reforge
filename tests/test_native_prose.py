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


class NativeProseTests(unittest.TestCase):
    def physical_lines(self, root, *, indent=0, leading=12, stop=False,
                       continuation='and finishes.', short=False, size=10):
        path=root/'lines.pdf'
        first='The paragraph explains the mathematical model and its input'
        if short:first='Only a fragment'
        if stop:first+='.'
        width=pymupdf.get_text_length(first,fontsize=10)
        specs=[(40,100,first,10),(40+indent,100+leading,continuation,size),
               (55,160,'An independent paragraph.',10)]
        atoms=[]
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=500,height=700)
            for n,(x,y,text,point) in enumerate(specs):
                page.insert_text((x,y),text,fontsize=point)
                atoms.append(dict(id=str(n),group=str(n),page=0,kind='text',role='text',
                    block_type='text',content=text,length=len(text),extraction_method='native',
                    bbox=[x/500,(y-1.075*point)/700,(x+width)/500,(y+.3*point)/700]))
            pdf.save(path)
        return SourceIndex(dict(source_pdf=str(path),atoms=atoms))

    def test_independent_physical_line_blocks_can_seed_a_source_paragraph(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.physical_lines(Path(name));before=deepcopy(index.source)
            rows=fragmented_native_prose(index)
            self.assertEqual([(r['left'],r['right']) for r in rows],[('0','1')])
            self.assertEqual(rows[0]['boundary'],'native_full_column_wrap')
            self.assertEqual(index.source,before)
            for atom in index.atoms:atom.update(content='已校订的中文文本',length=8)
            self.assertEqual(len(fragmented_native_prose(SourceIndex(index.source))),1)

    def test_physical_line_join_keeps_indents_lists_stops_fonts_and_short_columns(self):
        for options in (dict(indent=15),dict(leading=24),dict(stop=True),dict(size=7),
                        dict(short=True),dict(continuation='b) A distinct list item.'),
                        dict(continuation='A new paragraph begins here.')):
            with self.subTest(options=options),tempfile.TemporaryDirectory() as name:
                index=self.physical_lines(Path(name),**options)
                self.assertEqual(fragmented_native_prose(index),[])

    def test_physical_line_join_obeys_semantic_and_source_boundaries(self):
        for defect in ('heading','page','ocr','edit','excluded','semantic','panel'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as name:
                index=self.physical_lines(Path(name));registry=[];excluded=[];changed=[]
                if defect=='heading':index.atoms[1]['role']='paragraph_title'
                if defect=='page':index.atoms[1]['page']=1
                if defect=='ocr':index.atoms[1]['extraction_method']='ocr'
                if defect=='edit':changed=['1']
                if defect=='excluded':excluded=[dict(lo=index.prefix[1],hi=index.prefix[2])]
                if defect=='semantic':registry=[dict(lo=index.prefix[1],hi=index.prefix[2])]
                if defect=='panel':index.atoms[1]['source_panel']={'id':'panel'}
                self.assertEqual(fragmented_native_prose(index,registry,excluded,changed),[])

    def test_final_short_line_can_finish_a_filled_source_column(self):
        with tempfile.TemporaryDirectory() as name:
            index = self.physical_lines(Path(name))
            final = index.atoms[1]
            final['bbox'][2] = final['bbox'][0]+pymupdf.get_text_length(final['content'],fontsize=10)/500
            rows = fragmented_native_prose(index)
            self.assertEqual([(r['left'],r['right']) for r in rows],[('0','1')])
            self.assertEqual(rows[0]['boundary'],'native_full_column_wrap')

    def test_list_label_stops_an_already_seeded_fragment_chain(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name),continuation='b) A new item.')
            self.assertEqual([(r['left'],r['right']) for r in fragmented_native_prose(index)],[('0','1')])

    def test_compiled_physical_lines_have_no_false_paragraph_indent(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.physical_lines(root);project=root/'latex'
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Physical lines',author=''),load_config(),project,sample_mode=True)
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(project/'main.pdf') as pdf:
                self.assertIn('input and finishes.',pdf[0].get_text())
                start=pdf[0].search_for('The paragraph')[0];tail=pdf[0].search_for('and finishes.')[0]
                self.assertAlmostEqual(start.y0,tail.y0,delta=1)
                self.assertTrue(set(audit['source_anchors'].values())<=pdf.resolve_names().keys())

    def source(self, root, *, gap=4, leading=12, indent=0, stop=False, continuation='and ends.', bold_label=False):
        path = root/'source.pdf'
        first = 'This paragraph'
        second = ' continues here.' if stop else ' continues here'
        width = pymupdf.get_text_length(first, fontsize=10, fontname='hebo' if bold_label else 'helv')
        specs = [(40, 100, first), (40+width+gap, 100, second),
                 (40+indent, 100+leading, continuation), (40, 150, 'A separate paragraph.')]
        atoms = []
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=500, height=700)
            for i, (x, y, text) in enumerate(specs):
                font = 'hebo' if i==0 and bold_label else 'helv'
                page.insert_text((x, y), text, fontsize=10, fontname=font)
                # Some spans on the same baseline are merged by PyMuPDF.
                box = pymupdf.Rect(x, y-10.75, x+pymupdf.get_text_length(text, fontsize=10, fontname=font), y+2.99)
                atoms.append(dict(id=str(i), group=str(i), page=0, kind='text', role='text',
                    block_type='text', content=text.strip(), length=len(text.strip()),
                    bbox=[v/(500 if k%2==0 else 700) for k,v in enumerate(box)],
                    extraction_method='native'))
            pdf.save(path)
        return SourceIndex(dict(source_pdf=str(path), atoms=atoms))

    def test_fragment_chain_wraps_but_preserves_regular_paragraphs_and_source(self):
        with tempfile.TemporaryDirectory() as name:
            index = self.source(Path(name))
            before = deepcopy(index.source)
            rows = fragmented_native_prose(index)
            self.assertEqual([(r['left'], r['right']) for r in rows], [('0', '1'), ('1', '2')])
            self.assertEqual([r['boundary'] for r in rows], ['same_source_baseline', 'source_column_wrap'])
            self.assertEqual(index.source, before)

    def test_indented_spaced_or_complete_sentences_do_not_wrap(self):
        for options in (dict(indent=15), dict(leading=24), dict(stop=True)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as name:
                index = self.source(Path(name), **options)
                self.assertEqual([(r['left'], r['right']) for r in fragmented_native_prose(index)], [('0', '1')])

    def test_separate_columns_and_unseeded_narrow_fragments_are_retained(self):
        with tempfile.TemporaryDirectory() as name:
            index = self.source(Path(name), gap=40)
            self.assertEqual(fragmented_native_prose(index), [])
            index = SourceIndex({**index.source, 'atoms':index.atoms[1:]})
            self.assertEqual(fragmented_native_prose(index), [])

    def test_structural_edited_scan_and_unavailable_source_are_barriers(self):
        for defect in ('semantic', 'excluded', 'edited', 'ocr', 'heading', 'recovery', 'missing', 'page'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                index = self.source(Path(name)); registry=[]; excluded=[]; changed=[]
                interval = dict(lo=index.prefix[1], hi=index.prefix[2])
                if defect=='semantic': registry=[interval]
                if defect=='excluded': excluded=[interval]
                if defect=='edited': changed=['1']
                if defect=='ocr': index.atoms[1]['extraction_method']='ocr'
                if defect=='heading': index.atoms[1]['role']='paragraph_title'
                if defect=='recovery': index.atoms[1]['recovery_issue']='unconfirmed'
                if defect=='missing': Path(index.source['source_pdf']).unlink()
                if defect=='page': index.atoms[1]['page']=1
                self.assertEqual(fragmented_native_prose(index, registry, excluded, changed), [])

    def test_translation_keeps_original_geometric_paragraph_relation(self):
        with tempfile.TemporaryDirectory() as name:
            index = self.source(Path(name))
            for atom in index.atoms:
                atom.update(content='审校后的译文', length=6)
            self.assertEqual(len(fragmented_native_prose(SourceIndex(index.source))), 2)

    def test_parenthetical_continuation_keeps_the_fragment_chain(self):
        with tempfile.TemporaryDirectory() as name:
            index = self.source(Path(name), continuation='(figure 2).')
            self.assertEqual(len(fragmented_native_prose(index)), 2)

    def test_bold_source_label_does_not_become_ordinary_prose(self):
        with tempfile.TemporaryDirectory() as name:
            index = self.source(Path(name), bold_label=True)
            self.assertEqual(fragmented_native_prose(index), [])

    def test_multiline_union_does_not_steal_its_separately_owned_tail(self):
        with tempfile.TemporaryDirectory() as name:
            index = self.source(Path(name))
            first, tail, last = index.atoms[:3]
            # First group also owns a preceding line. Its union contains the tail.
            first['bbox'] = [first['bbox'][0], first['bbox'][1]-.02,
                             tail['bbox'][2], first['bbox'][3]]
            self.assertEqual([(r['left'],r['right']) for r in fragmented_native_prose(index)], [('0','1'),('1','2')])

    def test_compiled_render_joins_fragments_and_resolves_every_source_anchor(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); index = self.source(root)
            audit = write_project(index, dict(corrections=[], joins=[], exclusions=[], bold_spans=[]),
                [], [], dict(title='Source flow', author=''), load_config(), root/'latex', sample_mode=True)
            result = compile_project(root/'latex', {t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'], result['log_tail'])
            self.assertEqual(result['warnings'], [])
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                body = pdf[0].get_text()
                self.assertIn('This paragraph continues here and ends.', body)
                names = pdf.resolve_names()
                self.assertTrue(all('source-'+a['id'] in names for a in index.atoms))
                first = pdf[0].search_for('This paragraph')[0]
                tail = pdf[0].search_for('and ends.')[0]
                separate = pdf[0].search_for('A separate paragraph.')[0]
                self.assertAlmostEqual(first.y0, tail.y0, delta=1)
                self.assertGreater(separate.y0, tail.y1)
