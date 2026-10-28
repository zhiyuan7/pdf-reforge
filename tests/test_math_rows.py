"""Formula grids retain source row membership, labels and navigable math."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import Renderer, write_project
from book_pipeline.publishing.math_rows import horizontal_math_rows


DECISIONS = dict(corrections=[], joins=[], exclusions=[], bold_spans=[])


def grid_source(count=4):
    text = 'Solve the following systems.'
    atoms = [dict(id='prompt',group='prompt',page=0,kind='text',block_type='text',role='text',
                  content=text,length=len(text),bbox=[.08,.1,.92,.14])]
    for n in range(count):
        value = r'\text{(' + chr(97+n) + r')}\quad\begin{aligned}x+y&=2\\x-y&=0\end{aligned}'
        left = .1+n*.16
        atoms.append(dict(id=f'cell-{n}',group=f'cell-{n}',page=0,kind='display_math',
                          role='equation',block_type='equation',content=value,length=len(value),
                          bbox=[left,.16,left+.14,.22]))
    text = 'After the grid.'
    atoms.append(dict(id='after',group='after',page=0,kind='text',block_type='text',role='text',
                      content=text,length=len(text),bbox=[.08,.25,.92,.29]))
    return SourceIndex(dict(atoms=atoms,source_page_count=1))


class MathRowTests(unittest.TestCase):
    def test_source_row_retains_column_spacing_and_all_cells(self):
        index = grid_source()
        rows = horizontal_math_rows(index,DECISIONS)
        self.assertEqual(len(rows),1)
        row = rows[0]
        self.assertEqual([c['atom'] for c in row['cells']], [f'cell-{n}' for n in range(4)])
        self.assertEqual((row['lo'],row['hi']), (index.prefix[1],index.prefix[5]))
        self.assertAlmostEqual(sum(c['gap']+c['width'] for c in row['cells']),(.72-.08)/(.92-.08))
        for macro, labels in [('mathrm',['a','b']), ('text',['1','2'])]:
            with self.subTest(macro=macro,labels=labels):
                other = grid_source(2)
                for n,label in enumerate(labels):
                    atom = other.atoms[n+1]
                    atom['content'] = atom['content'].replace('text',macro).replace('('+chr(97+n)+')','('+label+')')
                self.assertEqual(len(horizontal_math_rows(SourceIndex(other.source),DECISIONS)),1)

    def test_unproven_or_interrupted_ownership_never_groups(self):
        for boundary in ['label','number','overlap','vertical','page','panel','shared_group',
                         'tag','tall','no_column','semantic','exclusion','inside_cell','five_cells']:
            with self.subTest(boundary=boundary):
                index = grid_source(5 if boundary=='five_cells' else 2)
                left, right = index.atoms[1:3]
                barriers, skipped = [], []
                if boundary=='label': right['content']=r'x+y=2'
                if boundary=='number': right['content']=right['content'].replace('(b)','(d)')
                if boundary=='overlap': right['bbox']=[.18,.16,.32,.22]
                if boundary=='vertical': right['bbox']=[.26,.3,.4,.36]
                if boundary=='page': right['page']=1
                if boundary=='panel': right['source_panel']={'id':'different'}
                if boundary=='shared_group': right['group']=left['group']
                if boundary=='tag': right['content']+=r'\tag{1}'
                if boundary=='tall': right['bbox']=[.26,.16,.4,.5]
                if boundary=='no_column': index.atoms[0]['bbox']=[.08,.1,.15,.14];index.atoms[-1]['bbox']=None
                if boundary=='semantic': barriers=[index.prefix[2]]
                if boundary=='exclusion': skipped=[(index.prefix[2],index.prefix[3],'excluded')]
                if boundary=='inside_cell': barriers=[index.prefix[1]+1]
                index = SourceIndex(index.source)
                self.assertEqual(horizontal_math_rows(index,DECISIONS,barriers,skipped),[])

    def test_renderer_keeps_each_math_label_anchor_and_source_unit_once(self):
        index = grid_source()
        registry = [dict(kind='equation',lo=index.prefix[2],hi=index.prefix[3],anchor='equation-b')]
        with tempfile.TemporaryDirectory() as name:
            renderer = Renderer(index,DECISIONS,registry,[],Path(name))
            tex = ''.join(renderer.render())
        self.assertNotIn(r'\begin{equation*}',tex)
        self.assertNotIn(r'\begin{gather*}',tex)
        for atom in index.atoms[1:-1]:
            self.assertIn(atom['content'],tex)
            self.assertEqual(tex.count(r'\hypertarget{source-'+atom['id']+'}'),1)
        self.assertEqual(tex.count(r'\BookAnchor{equation-b}'),1)
        self.assertEqual(set(renderer.source_anchors),set(index.ids))
        self.assertEqual(sum(r['end']-r['start'] for r in renderer.ledger),index.prefix[-1])

    def test_actual_pdf_cells_share_a_row_without_clipping_or_displaced_links(self):
        from book_pipeline.config import load_config, executable
        from book_pipeline.publishing.compiler import compile_project
        index = grid_source()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_project(index,DECISIONS,[],[],dict(title='Grid',author=''),load_config(),root,sample_mode=True)
            result = compile_project(root,{tool:executable(tool) for tool in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                labels = [pdf[0].search_for('('+chr(97+n)+')') for n in range(4)]
                self.assertTrue(all(len(found)==1 for found in labels))
                rects = [found[0] for found in labels]
                self.assertLess(max(r.y0 for r in rects)-min(r.y0 for r in rects),2)
                self.assertEqual(sorted(r.x0 for r in rects),[r.x0 for r in rects])
                baselines = sorted({round(s['origin'][1],2)
                    for b in pdf[0].get_text('dict')['blocks'] if b['type']==0
                    for line in b['lines'] for s in line['spans']
                    if '𝑥' in s['text'] and rects[0].x0 <= s['bbox'][0] < rects[1].x0})
                self.assertEqual(len(baselines),2)
                self.assertGreater(baselines[1]-baselines[0],12)
                self.assertLess(baselines[1]-baselines[0],18)
                names = pdf.resolve_names()
                for n,rect in enumerate(rects):
                    target = names[f'source-cell-{n}']
                    self.assertEqual(target['page'],0)
                    self.assertLess(abs(target['to'][0]-rect.x0),2)
                after = pdf[0].search_for('After the grid.')[0]
                self.assertGreater(after.y0,max(r.y1 for r in rects))
