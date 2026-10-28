from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.caption_layout import caption_separator, caption_sides, observed_caption_boxes, reviewed_caption_continuations
from book_pipeline.config import fingerprint
from book_pipeline.publishing.latex import Renderer


class CaptionLayoutTests(unittest.TestCase):
    def atoms(self, root):
        image = root/'image.png'
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 60), False)
        pix.clear_with(128); pix.save(image)
        base = dict(group='g', page=0, block_type='image', bbox=[.1,.2,.8,.5])
        return [dict(base, id='label', kind='text', role='image_caption', content='FIGURE 4.2', length=10),
                dict(base, id='title', kind='text', role='image_caption', content='A classifier?', length=13),
                dict(base, id='image', kind='image', role='image_body', content='', length=1, asset=str(image))]

    def test_geometry_changes_only_render_order_and_preserves_every_anchor(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); atoms=self.atoms(root); before=deepcopy(atoms)
            (root/'latex').mkdir()
            boxes=dict(label=[.1,.53,.3,.55], title=[.2,.15,.6,.18])
            observations=[dict(text='FIGURE4.2', score=.99, bbox=[100,530,300,550]),
                          dict(text='A classifier?', score=.99, bbox=[200,150,600,180]),
                          dict(text='A classifier?', score=.98, bbox=[201,151,601,181])]
            self.assertEqual(set(observed_caption_boxes(atoms, observations, [1000,1000])), {'label','title'})
            renderer=Renderer(SourceIndex(dict(atoms=atoms,source_page_count=1)),
                dict(corrections=[],joins=[],exclusions=[],bold_spans=[],caption_boxes=boxes),
                [],[],root/'latex',dict(float_assets=True))
            tex=''.join(renderer.render())
            self.assertLess(tex.index('A classifier?'), tex.index(r'\BookImage'))
            self.assertLess(tex.index(r'\BookImage'), tex.index('FIGURE 4.2'))
            for text in ('A classifier?', 'FIGURE 4.2'):
                self.assertEqual(tex.count(text),1)
            self.assertEqual(set(renderer.source_anchors),{'title','image','label'})
            self.assertEqual(atoms,before)

    def test_repeated_text_missing_or_overlapping_geometry_abstains(self):
        with tempfile.TemporaryDirectory() as name:
            atoms=self.atoms(Path(name))
            boxes=dict(label=[.1,.53,.3,.55],title=[.2,.15,.6,.18])
            for change in ({'label':None}, {'label':[.1,.3,.3,.4]}, {'label':[.85,.53,.95,.55]}):
                self.assertIsNone(caption_sides(atoms,{**boxes,**change}))
            obs=[dict(text='FIGURE 4.2',score=.99,bbox=[100,530,300,550]),
                 dict(text='FIGURE 4.2',score=.99,bbox=[100,800,300,820])]
            self.assertEqual(observed_caption_boxes(atoms,obs,[1000,1000]),{})

    def test_label_boundary_is_separated_but_inline_math_and_split_words_are_exact(self):
        left=dict(group='g',kind='text',content='TABLE 2.1')
        right=dict(group='g',kind='text',content='Training samples')
        self.assertEqual(caption_separator(left,right),' ')
        self.assertEqual(caption_separator({**left,'content':'super'},right),'')
        self.assertEqual(caption_separator(left,{**right,'kind':'math'}),'')
        self.assertEqual(caption_separator(left,{**right,'content':' already spaced'}),'')

    def test_reviewed_mixed_caption_continuation_stays_in_float_and_rejects_stale_or_distant_groups(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); atoms=self.atoms(root); (root/'latex').mkdir()
            atoms += [dict(id='prose',group='caption-body',page=0,kind='text',role='text',block_type='text',
                           content='Its entropy is ',length=15,bbox=[.1,.55,.8,.65]),
                      dict(id='math',group='caption-body',page=0,kind='math',role='text',block_type='text',
                           content='H(S)',length=4,bbox=[.1,.55,.8,.65])]
            index=SourceIndex(dict(atoms=atoms,source_page_count=1))
            boxes=dict(label=[.1,.53,.3,.55],title=[.2,.15,.6,.18])
            review=dict(source_sha256=fingerprint(index.source),evidence='source.png',
                        caption_continuations=[dict(asset='image',group='caption-body',quote='Its entropy is H(S)')])
            additions=reviewed_caption_continuations(index,review,boxes)
            renderer=Renderer(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[],
                caption_boxes=boxes,caption_continuations=additions),[],[],root/'latex',dict(float_assets=True))
            tex=''.join(renderer.render())
            self.assertLess(tex.index('Its entropy'),tex.index(r'\end{figure}'))
            self.assertEqual(tex.count('Its entropy'),1)
            self.assertEqual(set(renderer.source_anchors),{a['id'] for a in atoms})
            for patch in (dict(source_sha256='stale'),dict(evidence=''),
                          dict(caption_continuations=[dict(asset='image',group='caption-body',quote='Wrong')])):
                with self.subTest(patch=patch),self.assertRaises(ValueError):
                    reviewed_caption_continuations(index,{**review,**patch},boxes)
            with self.assertRaises(ValueError):
                reviewed_caption_continuations(index,review,boxes,[dict(lo=index.prefix[3],hi=index.prefix[-1])])
            atoms[3]['bbox']=[.1,.8,.8,.9]
            review['source_sha256']=fingerprint(index.source)
            with self.assertRaises(ValueError):
                reviewed_caption_continuations(index,review,boxes)
