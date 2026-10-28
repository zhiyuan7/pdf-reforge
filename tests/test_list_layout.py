"""Lists retain their source text, references, nesting and prose ownership."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from book_pipeline.ingestion.native import merge_text_blocks
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import Renderer
from book_pipeline.publishing.list_layout import list_paragraphs, multiline_list_rows


def index_for(rows):
    atoms = []
    for n, (text, kind, x, block_type) in enumerate(rows):
        atoms.append(dict(id=f'a{n}', group=f'g{n}', page=0, kind=kind,
                          role=block_type, block_type=block_type, content=text, length=len(text),
                          bbox=[x,.1+n*.05,.85,.14+n*.05]))
    return SourceIndex(dict(atoms=atoms, source_page_count=1))


class ListLayoutTests(unittest.TestCase):
    def test_deleted_terminal_atom_still_closes_list_and_keeps_source_anchor(self):
        index=index_for([('1. First item.\n2. Second item', 'text', .15, 'text'),
                         ('x=1.', 'math', .15, 'text'), ('.', 'text', .15, 'text')])
        for atom in index.atoms:
            atom.update(group='g0',bbox=[.15,.1,.85,.2])
        index=SourceIndex(index.source)
        correction=dict(atom='a2',start_offset=0,end_offset=1,expected='.',replacement='')
        with tempfile.TemporaryDirectory() as name:
            renderer=Renderer(index,dict(corrections=[correction],joins=[],exclusions=[],bold_spans=[]),
                              [],[],Path(name))
            tex=''.join(renderer.render())
        self.assertEqual(tex.count(r'\begin{samepage}'),1)
        self.assertEqual(tex.count(r'\end{samepage}'),1)
        self.assertEqual(tex.count(r'\begingroup'),tex.count(r'\endgroup'))
        self.assertIn('a2',renderer.source_anchors)
        self.assertEqual(sum(r['end']-r['start'] for r in renderer.ledger),index.prefix[-1])

    def test_reviewed_body_punctuation_keeps_list_but_label_and_newline_edits_do_not(self):
        index=index_for([('1. First item.\n2. Second item..', 'text', .15, 'text')])
        text=index.atoms[0]['content']
        correction=dict(atom='a0',start_offset=len(text)-1,end_offset=len(text),expected='.',replacement='')
        with tempfile.TemporaryDirectory() as name:
            renderer=Renderer(index,dict(corrections=[correction],joins=[],exclusions=[],bold_spans=[]),
                              [],[],Path(name))
            tex=''.join(renderer.render())
        self.assertEqual(tex.count(r'\hangafter=1'),2)
        self.assertIn('Second item.',tex)
        self.assertNotIn('Second item..',tex)
        self.assertEqual(sum(r['end']-r['start'] for r in renderer.ledger),index.prefix[-1])
        for start,end,replacement in ((0,1,'3'), (2,3,''), (14,15,''), (16,17,')'),
                                      (20,21,'x\n3. New row')):
            row=dict(atom='a0',start_offset=start,end_offset=end,expected=text[start:end],replacement=replacement)
            self.assertFalse(multiline_list_rows(index,corrections=[row]),row)

    def test_explicit_multiline_list_retains_math_references_and_coverage(self):
        index=index_for([('1. First item refers to result. ', 'text', .15, 'text'),
                         ('x^2+1', 'math', .15, 'text'),
                         ('\n2. Second item. ', 'text', .15, 'text'),
                         ('y=2', 'math', .15, 'text'),
                         ('Ordinary prose.', 'text', .12, 'text')])
        for atom in index.atoms[:4]: atom['group']='g0'
        index=SourceIndex(index.source)
        rows=multiline_list_rows(index)
        self.assertEqual([r['label'] for r in rows],['1.','2.'])
        start=index.atoms[0]['content'].index('result')
        refs=[dict(atom='a0',start_offset=start,end_offset=start+6,text='result',target='result',kind='remark')]
        registry=[dict(kind='remark',lo=index.prefix[4],hi=index.prefix[-1],anchor='result')]
        with tempfile.TemporaryDirectory() as name:
            renderer=Renderer(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),registry,refs,Path(name))
            tex=''.join(renderer.render())
        self.assertEqual(tex.count(r'\hangafter=1'),2)
        self.assertEqual(tex.count(r'\begin{samepage}'),1)
        self.assertEqual(tex.count(r'\end{samepage}'),1)
        self.assertEqual(tex.count('1. First'),1)
        self.assertEqual(tex.count('2. Second'),1)
        self.assertIn(r'\BookLink{result}{result}',tex)
        self.assertIn('x^2+1',tex)
        self.assertEqual(sum(v['end']-v['start'] for v in renderer.ledger),index.prefix[-1])
        self.assertEqual(set(renderer.source_anchors),set(index.ids))
        self.assertFalse(multiline_list_rows(index,barriers=[rows[1]['lo']]))
        self.assertFalse(multiline_list_rows(index,changed_atoms=['a0']))

    def test_short_multiline_list_keeps_its_source_introduction(self):
        index=index_for([('Properties:', 'text', .15, 'text'),
                         ('1. First\n2. Second\n3. Third', 'text', .15, 'text')])
        rows=multiline_list_rows(index)
        self.assertEqual(len(rows),3)
        self.assertEqual({r['keep_lo'] for r in rows},{0})
        self.assertEqual({r['list_hi'] for r in rows},{index.prefix[-1]})
        index.atoms[0]['role']='image_caption'
        self.assertEqual({r['keep_lo'] for r in multiline_list_rows(index)},{index.prefix[1]})

    def test_math_newlines_and_nonconsecutive_prose_do_not_invent_list_rows(self):
        for text,kind in [('1. First\n3. A nonconsecutive reference.', 'text'),
                          ('1. A label\n2. inside a formula', 'math'),
                          ('Prose precedes\n1. First\n2. Second', 'text')]:
            index=index_for([(text,kind,.1,'text')])
            self.assertFalse(multiline_list_rows(index))

    def test_checked_exercises_and_bullets_are_not_joined_to_previous_prose(self):
        first = dict(index=0, type='text', bbox=[.2,.1,.85,.12], content='A paragraph.')
        for text in ['✓ 2.17 Another exercise.', '✔ 3.2 A question.', '• A list item.', '(b) Another part.']:
            second = dict(index=1, type='text', bbox=[.2,.122,.85,.142], content=text)
            with self.subTest(text=text):
                self.assertEqual(len(merge_text_blocks([deepcopy(first), second])), 2)
        second['content'] = 'continues on the next line.'
        self.assertEqual(len(merge_text_blocks([deepcopy(first), second])), 1)

    def test_ordered_items_need_peers_and_section_numbers_are_not_list_labels(self):
        index = index_for([('1. A first step.', 'text', .15, 'text'),
                           ('2. A second step.', 'text', .15, 'text'),
                           ('2.5 A section.', 'text', .15, 'paragraph_title'),
                           ('9. An isolated number.', 'text', .15, 'text')])
        self.assertEqual(set(list_paragraphs(index)), {'g0','g1'})

    def test_nested_bullets_and_indented_continuation_after_math(self):
        index = index_for([('• Parent item.', 'text', .15, 'text'),
                           ('◦ Child item.', 'text', .19, 'text'),
                           ('• Another parent.', 'text', .15, 'text'),
                           ('x_1+x_2=3', 'display_math', .35, 'equation'),
                           ('This still belongs to that item.', 'text', .17, 'text'),
                           ('Ordinary prose resumes.', 'text', .12, 'text')])
        styles = list_paragraphs(index)
        self.assertGreater(styles['g1']['indent_em'], styles['g0']['indent_em'])
        self.assertTrue(styles['g4']['continuation'])
        self.assertNotIn('g3', styles)  # displays do not acquire empty paragraphs
        self.assertNotIn('g5', styles)

    def test_list_rendering_keeps_inline_math_links_and_complete_coverage(self):
        index = index_for([('• First item refers to result.', 'text', .15, 'text'),
                           ('x^2+1', 'math', .15, 'text'),
                           ('• Second item.', 'text', .15, 'text'),
                           ('Ordinary prose.', 'text', .12, 'text')])
        index.atoms[1]['group'] = 'g0'
        index = SourceIndex(index.source)
        registry = [dict(kind='remark', lo=index.prefix[3], hi=index.prefix[-1], anchor='result')]
        start = index.atoms[0]['content'].index('result')
        refs = [dict(atom='a0', start_offset=start, end_offset=start+6, text='result', target='result', kind='remark')]
        with tempfile.TemporaryDirectory() as name:
            renderer = Renderer(index, dict(corrections=[],joins=[],exclusions=[],bold_spans=[]), registry, refs, Path(name))
            tex = ''.join(renderer.render())
        self.assertEqual(tex.count(r'\BookListBegin'), 2)
        self.assertEqual(tex.count(r'\BookListEnd'), 2)
        self.assertEqual(tex.count('•'), 2)  # each source label is displayed once
        self.assertIn(r'\hypertarget{source-a0}{}First item', tex)
        self.assertIn(r'\BookLink{result}{result}', tex)
        self.assertIn('x^2+1', tex)
        self.assertEqual(set(renderer.source_anchors), set(index.ids))
        self.assertEqual(sum(v['end']-v['start'] for v in renderer.ledger), index.prefix[-1])

    def test_caption_and_page_transition_do_not_inherit_a_list(self):
        index = index_for([('• An item.', 'text', .15, 'text'),
                           ('Figure 3. An illustration.', 'text', .17, 'image_caption'),
                           ('Unrelated prose.', 'text', .17, 'text'),
                           ('• Another item.', 'text', .15, 'text'),
                           ('Next page prose.', 'text', .17, 'text')])
        index.atoms[-1]['page'] = 1
        self.assertEqual(set(list_paragraphs(index)), {'g0','g3'})

    def test_separate_columns_do_not_become_nested_list_levels(self):
        index = index_for([('• Left column.', 'text', .12, 'text'),
                           ('◦ A nested item.', 'text', .16, 'text'),
                           ('• Right column.', 'text', .55, 'text')])
        styles = list_paragraphs(index)
        self.assertEqual(styles['g0']['indent_em'], styles['g2']['indent_em'])
        self.assertGreater(styles['g1']['indent_em'], styles['g0']['indent_em'])
