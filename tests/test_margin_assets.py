"""Numbered marginal codes must stay small and on their citation's page."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import cv2
import pymupdf

from book_pipeline.config import load_config, executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.margin_assets import margin_assets
from book_pipeline.publishing.validation import audit_pdf


class MarginAssetTests(unittest.TestCase):
    def source(self, root, left=True):
        qr = root/'code.png'
        matrix = cv2.QRCodeEncoder_create().encode('https://example.org/mathematics')
        cv2.imwrite(str(qr), cv2.resize(matrix, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        label_box = [.14,.46,.16,.48] if left else [.86,.46,.88,.48]
        qr_box = [.17,.45,.22,.49] if left else [.8,.45,.85,.49]
        specs = [('label','label','text','aside_text','37',label_box),
                 ('context','context','text','text','Unrelated source material before the citation. ',[.23,.1,.78,.2]),
                 ('main','main','text','text','The mathematical classification is explained by an author ',[.23,.38,.78,.49]),
                 ('ref','main','math','text','^{37}',[.23,.38,.78,.49]),
                 ('end','main','text','text',', whose argument continues in this paragraph.',[.23,.38,.78,.49]),
                 ('qr','qr','image','image_body','',qr_box),
                 ('after','after','text','text','The following paragraph completes this discussion.',[.23,.5,.78,.6])]
        atoms = [dict(id=i,group=g,page=0,kind=k,role=r,content=t,length=max(1,len(t)),
                      bbox=b,block_bbox=b,block_type='image' if k=='image' else 'text')
                 for i,g,k,r,t,b in specs]
        atoms[-2]['asset']=str(qr)
        return SourceIndex(dict(atoms=atoms,source_page_count=1))

    def test_exact_number_geometry_and_detected_code_preserve_source(self):
        for left in (True, False):
            with self.subTest(left=left), tempfile.TemporaryDirectory() as name:
                index=self.source(Path(name),left);before=deepcopy(index.source)
                rows=margin_assets(index,[])
                self.assertEqual([(r['label'],r['reference'],r['image']) for r in rows],[('label','ref','qr')])
                self.assertEqual(index.source,before)

    def test_ambiguous_edited_boxed_and_non_qr_content_keep_original_flow(self):
        for problem in ('wrong_number','another_page','distant','duplicate_ref','duplicate_label',
                        'duplicate_image','edited','edited_prose','excluded','boxed','not_superscript','photo','asset_caption'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as name:
                index=self.source(Path(name));changed=set();exclusions=[];registry=[]
                if problem=='wrong_number':index.atoms[3]['content']='^{38}'
                elif problem=='not_superscript':index.atoms[3]['content']='37'
                elif problem=='another_page':index.atoms[5]['page']=1
                elif problem=='distant':index.atoms[5]['bbox']=[.17,.1,.22,.14]
                elif problem in {'duplicate_ref','duplicate_label','duplicate_image'}:
                    a=deepcopy(index.atoms[{'duplicate_ref':3,'duplicate_label':0,'duplicate_image':5}[problem]])
                    a['id']='duplicate'
                    if problem!='duplicate_ref':a['group']='duplicate'
                    index=SourceIndex({**index.source,'atoms':[*index.atoms,a]})
                elif problem=='edited':changed={'ref'}
                elif problem=='edited_prose':changed={'main'}
                elif problem=='excluded':exclusions=[dict(lo=index.prefix[5],hi=index.prefix[6])]
                elif problem=='boxed':registry=[dict(kind='theorem',lo=index.prefix[2],hi=index.prefix[5])]
                elif problem=='photo':cv2.imwrite(index.atoms[5]['asset'],cv2.imread(index.atoms[5]['asset'])*0+128)
                elif problem=='asset_caption':
                    a=deepcopy(index.atoms[0]);a.update(id='caption',group='qr',content='A caption',length=9)
                    index=SourceIndex({**index.source,'atoms':[*index.atoms,a]})
                self.assertEqual(margin_assets(index,registry,exclusions,changed),[])

    def test_compiled_note_tracks_reference_across_page_breaks_on_both_margins(self):
        for long in (False, True):
            with self.subTest(long=long), tempfile.TemporaryDirectory() as name:
                root=Path(name);index=self.source(root)
                if long:
                    index.atoms[1]['content']*=70
                    index.atoms[1]['length']=len(index.atoms[1]['content'])
                    index=SourceIndex(index.source)
                a,z=index.prefix[5:7]
                registry=[dict(kind='figure',anchor='figure-code',lo=a,hi=z)]
                project=root/'latex'
                audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                    registry,[],dict(title='Margins',author=''),load_config(),project,sample_mode=True)
                self.assertTrue(audit['coverage_complete'])
                result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(result['success'],result['log_tail'])
                self.assertEqual(result['warnings'],[])
                self.assertEqual(audit_pdf(project/'main.pdf',audit)['outside_page'],[])
                with pymupdf.open(project/'main.pdf') as pdf:
                    dest=pdf.resolve_names();page=dest['source-ref']['page']
                    self.assertEqual(page>0,long)
                    self.assertEqual(dest['source-label']['page'],page)
                    self.assertEqual(dest['source-qr']['page'],page)
                    self.assertEqual(dest['figure-code']['page'],page)
                    images=pdf[page].get_image_info()
                    self.assertEqual(len(images),1)
                    rect=pymupdf.Rect(images[0]['bbox'])
                    self.assertAlmostEqual(rect.width,26,delta=.1)
                    self.assertGreaterEqual(rect.x0,0)
                    self.assertLessEqual(rect.x1,pdf[page].rect.width)
                    text=pdf[page].search_for('The mathematical')[0]
                    self.assertTrue(rect.x1<text.x0 or rect.x0>pdf[page].rect.width-36)
                    reference_point=pymupdf.Point(dest['source-ref']['to'])*pdf[page].transformation_matrix
                    self.assertLess(abs(rect.y0-reference_point.y),30)
                    self.assertEqual(sum(p.get_text().count('37') for p in pdf),2)
                    tex=(project/'chapters/generated-001.tex').read_text()
                    self.assertNotIn(r'\begin{figure}',tex)
