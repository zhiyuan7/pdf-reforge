import unittest

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import Renderer


class ImageColumnTests(unittest.TestCase):
    def renderer(self, rows):
        atoms = [dict(id=str(n), group=str(n), page=0, kind=kind,
                      block_type='text' if kind=='text' else 'equation',
                      role='text', content='Example', length=7, bbox=box)
                 for n,(kind,box) in enumerate(rows)]
        renderer = object.__new__(Renderer)
        renderer.index = SourceIndex(dict(atoms=atoms))
        return renderer

    def width(self, rows):
        return self.renderer(rows).image_width(dict(page=0, bbox=[.3,.4,.7,.6]))

    def test_horizontal_row_and_single_image_share_column_geometry(self):
        rows=[('text',[.2,.1,.8,.2]), ('text',[.25,.7,.8,.8]),
              ('text',[0,.3,1,.5]), ('display_math',[0,.3,1,.5])]
        renderer=self.renderer(rows)
        renderer.image_rows={'image':dict(bbox=[.3,.4,.7,.6],members=[],targets=[],asset='row.png')}
        renderer.asset=lambda atom,**kwargs:'row.png'
        tex=renderer.content_fragment(dict(id='image',page=0),0,1,False)
        self.assertIn(r'\BookImage[0.667\linewidth]',tex)
        self.assertAlmostEqual(renderer.image_width(dict(page=0,bbox=[.3,.4,.7,.6])),2/3)

    def test_formula_windows_do_not_shrink_figures(self):
        body = [('text',[.2,.1,.8,.2]), ('text',[.25,.7,.8,.8])]
        window = [('text',[0,.3,1,.5]), ('display_math',[0,.3,1,.5])]
        self.assertAlmostEqual(self.width(body+window), 2/3)
        self.assertEqual(self.width(body+window), self.width(body))

    def test_full_width_text_without_formula_witness_is_retained(self):
        self.assertAlmostEqual(self.width([('text',[.2,.1,.8,.2]),
            ('text',[.25,.7,.8,.8]), ('text',[0,.3,1,.5])]), .4)

    def test_insufficient_independent_boxes_do_not_invent_column(self):
        self.assertAlmostEqual(self.width([('text',[.2,.1,.8,.2]),
            ('text',[0,.3,1,.5]), ('display_math',[0,.3,1,.5])]), .4)
