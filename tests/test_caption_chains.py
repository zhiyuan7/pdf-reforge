from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.caption_chains import continued_figure_captions
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project


class CaptionChainTests(unittest.TestCase):
    def source(self):
        def atom(key, text, role, box, kind='text'):
            return dict(id=key, group=key, page=0, kind=kind, role=role,
                block_type=role, content=text, length=max(1,len(text)), bbox=box)
        return SourceIndex(dict(atoms=[
            atom('image','','image',[.1,.2,.8,.5],'image'),
            atom('caption','Figure 1.1 A complete caption','image_caption',[.15,.52,.75,.55]),
            atom('tail','across source groups.','image_caption',[.15,.552,.70,.565]),
            atom('last','Its final sentence.','image_caption',[.15,.567,.7,.58]),
            atom('body','Ordinary prose.','text',[.1,.62,.8,.65])],source_page_count=1))

    def test_multiple_continuations_stay_with_figure_without_source_mutation(self):
        index=self.source();before=deepcopy(index.source)
        floats={'image':(0,index.prefix[2],'figure')}
        result,rows=continued_figure_captions(index,floats)
        self.assertEqual(result['image'],(0,index.prefix[4],'figure'))
        self.assertEqual(rows[0]['caption_groups'],['caption','tail','last'])
        self.assertEqual(rows[0]['joins'],[('caption','tail'),('tail','last')])
        self.assertEqual(index.source,before)
        self.assertEqual(floats['image'][1],index.prefix[2])

    def test_geometry_semantics_new_caption_edits_and_page_boundaries_stop_chain(self):
        for defect in ('gap','column','overlap','page','label','body','math_display','footnote',
                       'semantic','skip','edit','panel','recovery','missing','other_float'):
            with self.subTest(defect=defect):
                index=self.source();tail=index.atoms[2];barriers=[];skipped=[];changed=[]
                if defect=='gap':tail['bbox']=[.15,.59,.7,.60]
                if defect=='column':tail['bbox']=[.80,.552,.95,.565]
                if defect=='overlap':tail['bbox']=[.15,.53,.7,.565]
                if defect=='page':tail['page']=1
                if defect=='label':tail['content']='Figure 1.2 Another caption.'
                if defect=='body':tail['role']='text'
                if defect=='math_display':tail['kind']='display_math'
                if defect=='footnote':tail['role']='image_footnote'
                if defect=='semantic':barriers=[index.prefix[2]]
                if defect=='skip':skipped=[(index.prefix[2],index.prefix[3],'excluded')]
                if defect=='edit':changed=['tail']
                if defect=='panel':tail['source_panel']='panel'
                if defect=='recovery':tail['recovery_issue']='unreviewed'
                if defect=='missing':tail['bbox']=None
                floats={'image':(0,index.prefix[2],'figure')}
                if defect=='other_float':floats['other']=(index.prefix[2],index.prefix[4],'figure')
                result,rows=continued_figure_captions(index,floats,barriers,skipped,changed)
                self.assertEqual(result,floats);self.assertEqual(rows,[])

    def test_compiled_caption_is_one_paragraph_with_every_anchor_in_figure(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source()
            image=root/'image.png'
            pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,300,320),False)
            pix.clear_with(180);pix.save(image)
            index.atoms[0]['asset']=str(image)
            cfg=load_config();cfg['float_assets']=True
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Caption continuity',author=''),cfg,root/'latex',sample_mode=True)
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                names=pdf.resolve_names()
                self.assertEqual(len({names['source-'+k]['page'] for k in ('image','caption','tail','last')}),1)
                page=pdf[names['source-caption']['page']]
                text=page.get_text()
                self.assertEqual(text.count('across source groups.'),1)
                first=page.search_for('complete caption')[0];tail=page.search_for('across source groups.')[0]
                self.assertAlmostEqual(first.y0,tail.y0,delta=1)
                blocks=[b[4] for b in page.get_text('blocks') if b[6]==0]
                self.assertTrue(any('complete caption' in b and 'Its final sentence.' in b for b in blocks))
