"""Displays and excerpt navigation preserve prose boundaries and source anchors."""
from pathlib import Path
import tempfile
import unittest

from book_pipeline.config import load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import Renderer, write_project
from book_pipeline.publishing.display_layout import display_runs, prompt_space, list_display_scope, formula_image_prompts, bounded_display_units


def display_index(rows):
    atoms = []
    for n, (kind, text) in enumerate(rows):
        atoms.append(dict(id=f'a{n}', group=f'g{n}', page=0, kind=kind,
                          role='text' if kind == 'text' else 'equation',
                          block_type='text' if kind == 'text' else 'equation',
                          bbox=[.1, .1+n*.03, .9, .12+n*.03],
                          content=text, length=len(text)))
    return SourceIndex(dict(atoms=atoms, source_page_count=1))


DECISIONS = dict(corrections=[], joins=[], exclusions=[], bold_spans=[])


class DisplayLayoutTests(unittest.TestCase):
    def unit_source(self):
        return display_index([('text','2.3 In this system:'),
                              ('display_math',r'\begin{aligned}x+y&=2\\x-y&=0\end{aligned}'),
                              ('text','The two lines have a unique intersection.')])

    def test_bounded_numbered_units_require_complete_source_ownership(self):
        index=self.unit_source()
        self.assertEqual(len(bounded_display_units(index)),1)
        for boundary in ['page','panel','semantic','exclusion','long','caption','following_formula','new_item']:
            with self.subTest(boundary=boundary):
                index=self.unit_source();barriers=[];skipped=[]
                if boundary=='page':index.atoms[-1]['page']=1
                if boundary=='panel':index.atoms[-1]['source_panel']={'id':'other'}
                if boundary=='semantic':barriers=[index.prefix[1]]
                if boundary=='exclusion':skipped=[(index.prefix[1],index.prefix[2],'excluded')]
                if boundary=='long':index.atoms[0].update(content='2.3 A long narrative. '*30,length=630)
                if boundary=='caption':index.atoms[-1]['role']='image_caption'
                if boundary=='new_item':index.atoms[-1]['content']='2.4 Another question.'
                if boundary=='following_formula':
                    index.atoms.append({**index.atoms[1],'id':'more','group':'more'})
                index=SourceIndex(index.source)
                self.assertEqual(bounded_display_units(index,barriers,skipped),[])
        index=display_index([('text','3.4 Explain these transformations.'),
                             ('text','(a) Find the result:'),('display_math','x=y')])
        self.assertEqual(bounded_display_units(index)[0]['method'],'numbered_intro_first_subquestion_formula')
        index.atoms.append({**index.atoms[-1],'id':'next','group':'next','content':'y=z'})
        self.assertEqual(bounded_display_units(SourceIndex(index.source)),[])

    def test_short_prompt_ink_is_not_mistaken_for_the_prose_column(self):
        index=self.unit_source()
        index.atoms[0]['bbox']=[.1,.1,.3,.12]
        index.atoms[1]['bbox']=[.45,.13,.55,.17]
        index.atoms[2]['bbox']=[.1,.18,.9,.22]
        self.assertEqual(len(bounded_display_units(index)),1)
        index.atoms[2]['bbox']=[.15,.18,.35,.22]
        self.assertEqual(bounded_display_units(index),[])

    def test_compiled_numbered_prompt_formula_and_closing_prose_stay_together(self):
        import pymupdf
        from book_pipeline.config import executable
        from book_pipeline.publishing.compiler import compile_project
        for index in [self.unit_source(), display_index([
                ('text','3.4 Explain these transformations.'),
                ('text','(a) Find the result:'),
                ('display_math',r'\begin{aligned}x+y&=2\\x-y&=0\end{aligned}')]), display_index([
                ('text','3.4 Explain these transformations.'),
                ('text','(a) Find the result:'), ('display_math','x+y=2')])]:
            with self.subTest(prompt=index.atoms[0]['content']),tempfile.TemporaryDirectory() as name:
                root=Path(name)
                audit=write_project(index,DECISIONS,[],[],dict(title='Units',author=''),load_config(),root,sample_mode=True)
                p=root/'main.tex';tex=p.read_text()
                p.write_text(tex.replace(r'\input{chapters/generated-001.tex}',
                             '\\noindent\\rule{1pt}{580pt}\\par\n'+r'\input{chapters/generated-001.tex}'))
                result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(result['success'],result['log_tail'])
                self.assertEqual(result['warnings'],[])
                with pymupdf.open(root/'main.pdf') as pdf:
                    self.assertEqual(len(pdf),2)
                    self.assertIn(index.atoms[0]['content'],pdf[1].get_text())
                    if index.atoms[-1]['kind']=='text':
                        self.assertIn(index.atoms[-1]['content'],pdf[1].get_text())
                    self.assertEqual({pdf.resolve_names()['source-'+a['id']]['page'] for a in index.atoms},{1})
                self.assertEqual(audit['accounted_units'],index.prefix[-1])

    def test_page_pressure_cannot_separate_inline_math_prompt_from_its_display(self):
        import pymupdf
        from book_pipeline.config import executable
        from book_pipeline.publishing.compiler import compile_project
        index=display_index([
            ('text','2.3 This system says sin '), ('math',r'\alpha'),
            ('text',' instead of '), ('math',r'\alpha'),
            ('display_math',r'\begin{aligned}2x-y&=3\\4x+2y&=10\\6x-3y&=9\end{aligned}'),
            ('text','Apply the method. Does the system have a solution?'),
            ('text','Following paragraph outside the bounded unit.'),
            ('text','Another ordinary paragraph establishes the normal leading.')])
        for atom in index.atoms[:4]:atom['group']='prompt'
        index=SourceIndex(index.source)
        for pressure in (548, 580):
            with self.subTest(pressure=pressure),tempfile.TemporaryDirectory() as name:
                root=Path(name)
                write_project(index,DECISIONS,[],[],dict(title='Units',author=''),load_config(),root,sample_mode=True)
                p=root/'main.tex'
                p.write_text(p.read_text().replace(r'\input{chapters/generated-001.tex}',
                    '\\noindent\\rule{1pt}{'+str(pressure)+'pt}\\par\n'+r'\input{chapters/generated-001.tex}'))
                result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(result['success'],result['log_tail'])
                self.assertEqual(result['warnings'],[])
                with pymupdf.open(root/'main.pdf') as pdf:
                    names=pdf.resolve_names()
                    self.assertEqual({names['source-'+a['id']]['page'] for a in index.atoms},{1})
                    prompt=pdf[1].search_for('2.3 This system says sin')[0]
                    closing=pdf[1].search_for('Apply the method.')[0]
                    following=pdf[1].search_for('Following paragraph outside')[0]
                    ordinary=pdf[1].search_for('Another ordinary paragraph')[0]
                    # The box retains ordinary prose indent, size and leading.
                    self.assertAlmostEqual(prompt.x0,closing.x0,delta=1)
                    self.assertAlmostEqual(prompt.x0,56.92,delta=1)
                    self.assertAlmostEqual(following.y0-closing.y0,ordinary.y0-following.y0,delta=.2)
                    spans=[s for b in pdf[1].get_text('dict')['blocks'] if b['type']==0
                           for line in b['lines'] for s in line['spans'] if '2.3 This' in s['text']]
                    self.assertEqual(len(spans),1)
                    self.assertAlmostEqual(spans[0]['size'],10.4608,delta=.05)

    def test_formula_image_prompt_requires_same_source_owner_and_short_prose(self):
        index=display_index([('text','Therefore the result is:'),('image','')])
        image=index.atoms[1];image.update(block_type='image',length=1,recovery_issue='formula_source_crop_requires_transcription')
        index=SourceIndex(index.source)
        self.assertEqual(len(formula_image_prompts(index)),1)
        self.assertEqual(formula_image_prompts(index,barriers=[index.prefix[1]]),[])
        self.assertEqual(formula_image_prompts(index,skipped=[(0,index.prefix[1],'excluded')]),[])
        for field,value in [('page',1),('bbox',[.95,.2,1,.3]),('recovery_issue','ordinary_illustration')]:
            old=image.get(field);image[field]=value
            self.assertEqual(formula_image_prompts(index),[]);image[field]=old
        index.atoms[0].update(content='A long narrative paragraph. '*20,length=540)
        index=SourceIndex(index.source)
        self.assertEqual(formula_image_prompts(index),[])

    def test_continued_vertical_prompt_uses_final_source_column_for_formula_ownership(self):
        index=display_index([('text','A short continued prompt'),('image','')])
        prompt,image=index.atoms
        prompt.update(bbox=[.3,.1,.9,.8],writing_direction='vertical_rl',source_prose_end_bbox=[.3,.1,.6,.2])
        image.update(bbox=[.1,.21,.4,.25],block_type='image',length=1,
                     recovery_issue='formula_source_crop_requires_transcription')
        index=SourceIndex(index.source)
        self.assertFalse(formula_image_prompts(index))
        prompt['source_prose_end_bbox']=[.3,.1,.38,.2]
        self.assertEqual(len(formula_image_prompts(index)),1)
        self.assertFalse(formula_image_prompts(index,barriers=[index.prefix[1]]))
        for box in ([.8,.1,.9,.2],[.2,.1,.38,.2],[.3,.1,float('nan'),.2]):
            prompt['source_prose_end_bbox']=box
            self.assertFalse(formula_image_prompts(index))
        prompt['source_prose_end_bbox']=[.3,.1,.38,.2]
        prompt['writing_direction']='horizontal_ltr'
        self.assertFalse(formula_image_prompts(index))

    def test_formula_image_pair_has_natural_height_without_duplicate_content(self):
        from PIL import Image
        index=display_index([('text','Therefore the result is:'),('image','')])
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);asset=path/'crop.png';Image.new('RGB',(200,100),'white').save(asset)
            index.atoms[1].update(block_type='image',length=1,asset=str(asset),recovery_issue='formula_source_crop_requires_transcription')
            index=SourceIndex(index.source)
            renderer=Renderer(index,DECISIONS,[],[],path)
            tex=''.join(renderer.render())
        self.assertLess(tex.index(r'\begingroup\interlinepenalty=10000'),tex.index('Therefore'))
        self.assertGreater(tex.index(r'\endgroup'),tex.index(r'\BookFormulaImage['))
        self.assertIn('\\par\\nobreak\n'+r'\BookFormulaImage[',tex)
        self.assertEqual(tex.count('Therefore the result is:'),1)
        self.assertEqual(tex.count(r'\BookFormulaImage['),1)
        self.assertGreater(tex.index(r'\hypertarget{source-a1}'),tex.index(r'\BookFormulaImage['))
        self.assertEqual(set(renderer.source_anchors),{'a0','a1'})
        self.assertEqual(sum(r['end']-r['start'] for r in renderer.ledger),index.prefix[-1])

    def test_adjacent_derivation_rows_keep_math_numbers_destinations_and_coverage(self):
        index = display_index([('text', 'Given the hypothesis:'),
                               ('display_math', r'x=y\tag{A}'),
                               ('display_math', r'y=z\tag{B}'),
                               ('display_math', r'\therefore x=z'),
                               ('text', 'This proves the claim.')])
        with tempfile.TemporaryDirectory() as name:
            registry = [dict(kind='equation', lo=index.prefix[2], hi=index.prefix[3],
                             anchor='eq-b', label='B')]
            renderer = Renderer(index, DECISIONS, registry, [], Path(name))
            tex = ''.join(renderer.render())
        start, end = tex.index(r'\begin{gather*}'), tex.index(r'\end{gather*}')
        body = tex[start:end]
        self.assertEqual(tex.count(r'\begin{gather*}'), 1)
        self.assertNotIn(r'\begin{equation*}', tex)
        self.assertEqual(body.count('\\\\\n'), 2)
        self.assertNotRegex(body, r'\n[ \t]*\n')  # A blank line is an illegal paragraph in amsmath.
        for atom in index.atoms[1:4]:
            self.assertIn(atom['content'].split('\\tag')[0], body)
            self.assertEqual(body.count(r'\hypertarget{source-'+atom['id']+'}'), 1)
        self.assertIn(r'\tag{A}', body)
        self.assertIn(r'\tag{B}', body)
        self.assertIn(r'\BookAnchor{eq-b}', body)
        self.assertLess(tex.index('Given the hypothesis:'), start)
        self.assertGreater(tex.index('This proves the claim.'), end)
        self.assertEqual(sum(r['end']-r['start'] for r in renderer.ledger), index.prefix[-1])

    def test_runs_stop_at_prose_semantic_edges_exclusions_pages_and_columns(self):
        for barrier in ['prose', 'semantic', 'excluded', 'page', 'column', 'panel']:
            with self.subTest(barrier=barrier):
                index = display_index([('display_math', 'x=y'), ('display_math', 'y=z')])
                barriers, skipped = [], []
                if barrier == 'prose':
                    index = display_index([('display_math','x=y'), ('text','Unless x=0.'), ('display_math','y=z')])
                if barrier == 'semantic': barriers = [index.prefix[1]]
                if barrier == 'excluded': skipped = [(0, index.prefix[1], 'excluded')]
                if barrier == 'page': index.atoms[1]['page'] = 1
                if barrier == 'column': index.atoms[0]['bbox'] = [.05,.1,.45,.2]; index.atoms[1]['bbox'] = [.55,.1,.95,.2]
                if barrier == 'panel': index.atoms[1]['source_panel'] = 'other'
                self.assertEqual(display_runs(index, DECISIONS, barriers, skipped), [])

    def test_tall_or_ambiguous_formulas_use_standalone_rendering_even_after_repair(self):
        for content in [r'\begin{pmatrix}a\\b\end{pmatrix}', r'x=y\tag{1}\tag{2}', r'x=y\tag{1']:
            with self.subTest(content=content):
                index = display_index([('display_math','x=y'), ('display_math',content)])
                self.assertEqual(display_runs(index, DECISIONS), [])
                index.atoms[1]['content'] = 'y=z'
                self.assertEqual(display_runs(index, {**DECISIONS, 'formula_repairs': {'a1': content}}), [])

    def test_long_derivations_have_bounded_chunks_and_preserve_all_rows(self):
        index = display_index([('display_math', f'x_{n}=y_{n}') for n in range(14)])
        runs = display_runs(index, DECISIONS)
        self.assertEqual([r['rows'] for r in runs], [6, 6, 2])
        self.assertEqual([a for r in runs for a in r['atoms']], [a['id'] for a in index.atoms])

    def test_short_list_prompt_reserves_space_for_the_following_derivation(self):
        index = display_index([('text', '(a) Assuming x=y:'), ('display_math','x=y'), ('display_math','y=z')])
        style = dict(group='g0', lo=0, hi=index.prefix[1], page=0, continuation=False)
        self.assertIsNotNone(prompt_space(index, style, display_runs(index, DECISIONS)))
        self.assertIsNone(prompt_space(index, {**style, 'continuation': True}, []))
        with tempfile.TemporaryDirectory() as name:
            renderer = Renderer(index, DECISIONS, [], [], Path(name))
            tex = ''.join(renderer.render())
        self.assertLess(tex.index(r'\Needspace'), tex.index(r'\BookListBegin'))
        self.assertGreater(tex.index(r'\BookListEnd'), tex.index(r'\end{gather*}'))

    def test_list_derivation_preserves_formula_links_and_closes_before_following_prose(self):
        index = display_index([('text', '(b) Under this assumption:'),
                               ('display_math', r'x=y\tag{I}'),
                               ('display_math', 'y=z'),
                               ('text', 'An unrelated paragraph.')])
        with tempfile.TemporaryDirectory() as name:
            registry = [dict(kind='equation', lo=index.prefix[1], hi=index.prefix[2], anchor='eq-i', label='I')]
            renderer = Renderer(index, DECISIONS, registry, [], Path(name))
            tex = ''.join(renderer.render())
        self.assertEqual(tex.count('(b)'), 1)
        self.assertEqual(tex.count(r'\BookListBegin'), 1)
        self.assertEqual(tex.count(r'\BookListEnd'), 1)
        self.assertLess(tex.index(r'\end{gather*}'), tex.index(r'\BookListEnd'))
        self.assertLess(tex.index(r'\BookListEnd'), tex.index('An unrelated paragraph.'))
        self.assertIn(r'\tag{I}', tex)
        self.assertIn(r'\BookAnchor{eq-i}', tex)
        self.assertEqual(set(renderer.source_anchors), set(index.ids))
        self.assertEqual(sum(v['end']-v['start'] for v in renderer.ledger), index.prefix[-1])

    def test_list_derivation_scope_requires_source_ownership_and_a_complete_short_run(self):
        for boundary in ['semantic', 'page', 'column', 'panel', 'long']:
            with self.subTest(boundary=boundary):
                index = display_index([('text', '(c) A short prompt:'),
                                       *[('display_math', f'x_{n}=y') for n in range(7 if boundary=='long' else 2)]])
                style = dict(group='g0', lo=0, hi=index.prefix[1], page=0)
                barriers = [index.prefix[1]] if boundary == 'semantic' else []
                if boundary == 'page': index.atoms[0]['page'] = 1; style['page'] = 1
                if boundary == 'column':
                    index.atoms[0]['bbox'] = [.01,.1,.3,.12]
                    for atom in index.atoms[1:]: atom['bbox'] = [.55,.2,.95,.22]
                if boundary == 'panel': index.atoms[0]['source_panel'] = 'other'
                runs = display_runs(index, DECISIONS)
                self.assertIsNone(list_display_scope(index, style, runs, barriers))

    def test_single_short_display_belongs_to_prompt_but_tall_or_repaired_math_does_not(self):
        index = display_index([('text', '(a) A single identity:'),
                               ('display_math', r'x=y\tag{II}'),
                               ('text', 'Prose resumes.')])
        with tempfile.TemporaryDirectory() as name:
            renderer = Renderer(index, DECISIONS, [], [], Path(name))
            tex = ''.join(renderer.render())
        self.assertLess(tex.index(r'\end{equation*}'), tex.index(r'\BookListEnd'))
        self.assertLess(tex.index(r'\BookListEnd'), tex.index('Prose resumes.'))
        self.assertIn(r'x=y', tex)
        self.assertIn(r'\tag{II}', tex)
        self.assertEqual(set(renderer.source_anchors), set(index.ids))
        style = dict(group='g0', lo=0, hi=index.prefix[1], page=0)
        self.assertIsNone(list_display_scope(index, style, [], skipped=[
            (index.prefix[1],index.prefix[2],'excluded')]))
        for content in [r'\begin{pmatrix}a\\b\end{pmatrix}', r'x=y\tag{1}\tag{2}']:
            self.assertIsNone(list_display_scope(index, style, [], decisions={
                **DECISIONS, 'formula_repairs': {'a1': content}}))

    def test_long_list_prose_keeps_its_paragraph_boundary_and_pagination_behavior(self):
        index = display_index([('text', '(a) ' + 'A sentence explaining the construction. '*9),
                               ('display_math', 'x+y=z')])
        with tempfile.TemporaryDirectory() as name:
            renderer = Renderer(index, DECISIONS, [], [], Path(name))
            tex = ''.join(renderer.render())
        self.assertLess(tex.index(r'\BookListEnd'), tex.index(r'\begin{equation*}'))
        self.assertIn('x+y=z', tex)
        self.assertFalse(any(r['method']=='source_list_includes_derivation' for r in renderer.source_order_layout))

    def test_display_follows_intro_without_an_empty_paragraph_but_next_prose_stays_separate(self):
        for kind,formula,ending in [(k,f,e) for k,f in [('display_math','x+y=y+x'),('math',r'x+y=y+x\tag{1}')]
                                   for e in ['', '\n', '\n\n', '\r\n']]:
            with self.subTest(kind=kind,ending=ending),tempfile.TemporaryDirectory() as name:
                atoms=[]
                for sid,k,text in [('intro','text','Then we have'+ending),('eq',kind,formula),('next','text','The next paragraph.')]:
                    atoms.append(dict(id=sid,group=sid,page=0,kind=k,role='text' if k=='text' else 'equation',
                                      block_type='text' if k=='text' else 'equation',content=text,length=len(text)))
                index=SourceIndex(dict(atoms=atoms,source_page_count=1))
                renderer=Renderer(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],Path(name))
                tex=''.join(renderer.render())
                self.assertIn('Then we have\n'+r'\hypertarget{source-eq}',tex)
                self.assertIn('\n\n'+r'\leavevmode\hypertarget{source-next}',tex)
                self.assertIn('x+y=y+x',tex)
                if kind=='math': self.assertIn(r'\tag{1}',tex)
                self.assertEqual(set(renderer.source_anchors),{'intro','eq','next'})
                self.assertEqual(sum(r['end']-r['start'] for r in renderer.ledger),index.prefix[-1])

    def test_excerpt_omits_automatic_toc_while_preserving_heading_and_navigation(self):
        for scope in ['book','excerpt']:
            with self.subTest(scope=scope),tempfile.TemporaryDirectory() as name:
                atoms=[]
                for sid,text in [('title','1.2 Learning'),('body','A supervised learning problem.')]:
                    atoms.append(dict(id=sid,group=sid,page=0,kind='text',role='text',block_type='text',content=text,length=len(text)))
                index=SourceIndex(dict(atoms=atoms,source_page_count=1))
                registry=[dict(kind='section',lo=0,hi=12,anchor='learning',label='1.2 Learning',heading_level=2,number='1.2')]
                path=Path(name)
                audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),registry,[],
                                    dict(title='Learning',author='Author'),{**load_config(),'publication_scope':scope},path)
                self.assertEqual(r'\tableofcontents' in (path/'main.tex').read_text(),scope=='book')
                chapter=(path/'chapters/generated-001.tex').read_text()
                self.assertIn(r'\BookHeading{2}',chapter)
                self.assertIn('Learning',chapter)
                self.assertIn('learning',audit['anchors'])
