"""A figure cannot float away from its separately extracted source caption."""
from pathlib import Path
import tempfile
import unittest

import pymupdf
from book_pipeline.config import digest
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import Renderer
from book_pipeline.publishing.figure_captions import split_caption_rows
from book_pipeline.config import load_config,executable
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project


class FigureCaptionTests(unittest.TestCase):
    def adjacent_display_source(self, root, intervening=False, next_page=False, inline=False):
        _, renderer = self.render(root)
        atoms = renderer.index.atoms
        if intervening:
            atoms.append(dict(id='prose', group='prose', page=0, kind='text', role='text',
                block_type='text', content='An independent discussion.', length=26))
        formula = r'f(x)=x^2\tag{2.1}'
        atoms.append(dict(id='formula',group='formula',page=int(next_page),
            kind='math' if inline else 'display_math',role='equation',block_type='equation',
            content=formula,length=len(formula),bbox=[.1,.6,.9,.65]))
        prose='Following prose continues after the mathematical construction. '
        atoms.append(dict(id='following',group='following',page=int(next_page),kind='text',
            role='text',block_type='text',content=prose,length=len(prose),bbox=[.1,.7,.9,.8]))
        return SourceIndex({**renderer.index.source,'atoms':atoms})

    def test_only_source_adjacent_display_bounds_the_figure_float(self):
        for options in ({},dict(intervening=True),dict(next_page=True),dict(inline=True)):
            with self.subTest(options=options),tempfile.TemporaryDirectory() as name:
                root=Path(name); index=self.adjacent_display_source(root,**options)
                renderer=Renderer(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                    [],[],root/'render',dict(float_assets=True))
                tex=''.join(renderer.render())
                self.assertEqual(r'\FloatBarrier' in tex,not options)
                self.assertEqual(any(r['method']=='source_adjacent_figure_before_display'
                    for r in renderer.source_order_layout),not options)

    def test_compiled_figure_precedes_adjacent_display_under_page_pressure(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); index=self.adjacent_display_source(root)
            project=root/'latex'
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Figure order',author=''),{**load_config(),'float_assets':True},
                project,sample_mode=True)
            main=project/'main.tex'
            main.write_text(main.read_text().replace(r'\input{chapters/generated-001.tex}',
                '\\noindent\\rule{1pt}{490pt}\\par\n'+r'\input{chapters/generated-001.tex}'))
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            self.assertEqual(result['warnings'],[])
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(project/'main.pdf') as pdf:
                names=pdf.resolve_names()
                image_page=names['source-image']['page']
                formula_page=names['source-formula']['page']
                self.assertEqual(image_page,1)
                self.assertEqual(names['source-caption']['page'],image_page)
                self.assertGreaterEqual(formula_page,image_page)
                if formula_page==image_page:
                    self.assertGreater(pdf[formula_page].search_for('(2.1)')[0].y0,
                        pymupdf.Rect(pdf[image_page].get_image_info()[0]['bbox']).y1)
                self.assertEqual(sum(p.get_text().count('Following prose') for p in pdf),1)

    def test_native_image_row_keeps_a_long_separate_caption_on_one_page(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source=root/'source.pdf'
            with pymupdf.open() as pdf:
                page=pdf.new_page(width=500,height=700)
                for x in (60,190,320):page.draw_rect((x,250,x+100,450),color=(0,0,1))
                pdf.save(source)
            image=root/'plot.png'
            pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,200),False)
            pix.clear_with(200);pix.save(image)
            text='A complete explanatory paragraph before the figure. '*15
            atoms=[dict(id='body',group='body',page=0,kind='text',role='text',block_type='text',
                        content=text,length=len(text),bbox=[.1,.1,.9,.3],extraction_method='native')]
            for n,x in enumerate((.12,.38,.64)):
                atoms.append(dict(id=f'image{n}',group=f'image{n}',page=0,kind='image',role='image_body',
                    block_type='image',content='',length=1,bbox=[x,250/700,x+.2,450/700],
                    extraction_method='native',asset=str(image),asset_sha256=digest(image)))
            caption='Figure 3.1. '+'The plots display three different probability distributions. '*9+'Caption ends here.'
            atoms.append(dict(id='caption',group='caption',page=0,kind='text',role='image_caption',
                block_type='image_caption',content=caption,length=len(caption),bbox=[.1,.66,.9,.8],extraction_method='native'))
            index=SourceIndex(dict(source_pdf=str(source),atoms=atoms))
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Image row',author=''),{**load_config(),'float_assets':True},root/'latex',sample_mode=True)
            self.assertTrue(any(r['method']=='adjacent_source_caption_kept_with_image_row' for r in audit['source_order_layout']))
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                names=pdf.resolve_names()
                page=names['source-image0']['page']
                self.assertEqual(names['source-caption']['page'],page)
                self.assertIn('Caption ends here.',pdf[page].get_text())
                self.assertEqual(sum('Caption ends here.' in p.get_text() for p in pdf),1)

    def test_separate_ocr_title_fragments_form_one_caption_with_original_anchors(self):
        for problem in (None,'distant','another_page','edited','semantic_heading'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as name:
                root=Path(name)
                _, renderer=self.render(root)
                index=renderer.index
                label=index.atoms[-1]
                label.update(content='Fig. 2.1',length=8,role='paragraph_title',block_type='paragraph_title',
                             bbox=[.1,.52,.22,.55])
                title={**label,'id':'title','group':'title','content':'A probability diagram',
                       'length':21,'bbox':[.24,.52,.5,.55]}
                if problem=='distant':title['bbox']=[.24,.6,.5,.63]
                elif problem=='another_page':title['page']=1
                index=SourceIndex({**index.source,'atoms':[*index.atoms,title]})
                registry=[dict(kind='section',anchor='heading',lo=index.prefix[2],hi=index.prefix[3])]
                rows=split_caption_rows(index,registry if problem=='semantic_heading' else [],
                    changed={'title'} if problem=='edited' else ())
                self.assertEqual(bool(rows),problem is None)
                if problem is not None:continue
                renderer=Renderer(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],
                    root/'render',dict(float_assets=True))
                tex=''.join(renderer.render())
                self.assertLess(tex.index('A probability diagram'),tex.index(r'\end{figure}'))
                self.assertEqual(tex.count('Fig. 2.1'),1)
                self.assertIn(r'\BookCaption{',tex)
                self.assertNotIn(r'\Needspace{',tex)
                self.assertEqual(set(renderer.source_anchors),{'image','caption','title'})

    def test_nonstructural_source_title_reserves_following_body_lines(self):
        atoms = []
        for sid, text, kind in [('title', 'Definition: scalar multiplication', 'paragraph_title'),
                                ('body', 'The product of a scalar and a vector is defined coordinatewise.', 'text')]:
            atoms.append(dict(id=sid, group=sid, page=0, kind='text', role=kind, block_type=kind,
                              content=text, length=len(text)))
        with tempfile.TemporaryDirectory() as name:
            renderer = Renderer(SourceIndex(dict(atoms=atoms, source_page_count=1)),
                dict(corrections=[], joins=[], exclusions=[], bold_spans=[]), [], [], Path(name))
            tex = ''.join(renderer.render())
            self.assertLess(tex.index(r'\Needspace{4\baselineskip}'), tex.index('Definition:'))
            self.assertIn('coordinatewise.', tex)

    def render(self, root, caption_page=0, caption_role='image_caption', intervening=False):
        image = root / 'source.png'
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0,0,100,50), False)
        pix.clear_with(128)
        pix.save(image)
        atoms = [dict(id='image', group='figure', page=0, kind='image', role='image_body',
                      block_type='image', content='', length=1, asset=str(image), asset_sha256=digest(image),
                      bbox=[.2,.3,.8,.5])]
        if intervening:
            atoms.append(dict(id='body', group='body', page=0, kind='text', role='text', block_type='text',
                              content='Intervening prose.', length=18, bbox=[.1,.51,.9,.53]))
        text='Figure 2.1: supervised learning'
        atoms.append(dict(id='caption', group='caption', page=caption_page, kind='text', role=caption_role,
                          block_type='image_caption', content=text, length=len(text), bbox=[.25,.52,.75,.55]))
        index = SourceIndex(dict(atoms=atoms, source_page_count=2))
        (root/'render').mkdir()
        renderer = Renderer(index, dict(corrections=[], joins=[], exclusions=[], bold_spans=[]), [], [],
                            root/'render', dict(float_assets=True))
        return ''.join(renderer.render()), renderer

    def test_separate_caption_is_inside_figure_float(self):
        with tempfile.TemporaryDirectory() as name:
            tex, renderer = self.render(Path(name))
            self.assertLess(tex.index('Figure 2.1:'), tex.index(r'\end{figure}'))
            self.assertTrue(any(r['method']=='adjacent_source_caption_kept_with_figure' for r in renderer.source_order_layout))

    def test_another_page_or_ordinary_prose_is_not_adopted_as_caption(self):
        for options in [dict(caption_page=1), dict(caption_role='text'), dict(intervening=True)]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as name:
                tex, _ = self.render(Path(name), **options)
                self.assertLess(tex.index(r'\end{figure}'), tex.index('Figure 2.1:'))
