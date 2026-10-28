"""A figure cannot float away from its separately extracted source caption."""
from pathlib import Path
import tempfile
import unittest

import pymupdf
from book_pipeline.config import digest
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import Renderer
from book_pipeline.publishing.figure_captions import split_caption_rows


class FigureCaptionTests(unittest.TestCase):
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
