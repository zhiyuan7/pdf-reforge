from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.structure.recognized_headings import recognized_heading_registry
from book_pipeline.structure.run_in_titles import run_in_title_rows


class RunInTitleTests(unittest.TestCase):
    def source(self, root, *, bold=True, body_bold=False, gap=6, below=0, union=False, title_size=10):
        path = root/'source.pdf'
        label, body = 'Method:', 'The estimate is computed from the observations.'
        label_font = 'hebo' if bold else 'helv'
        width = pymupdf.get_text_length(label, fontsize=title_size, fontname=label_font)
        atoms = []
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=500, height=700)
            for n,(x,y,text,font,role) in enumerate([
                    (40,100,label,label_font,'paragraph_title'),
                    (40+width+gap,100+below,body,'hebo' if body_bold else 'helv','text'),
                    (40,150,'A separate paragraph.','helv','text')]):
                size=title_size if n==0 else 10
                page.insert_text((x,y),text,fontsize=size,fontname=font)
                box = [x/500,(y-1.075*size)/700,
                       (x+pymupdf.get_text_length(text,fontsize=size,fontname=font))/500,(y+.3*size)/700]
                if n==1 and union: box[0]=40/500
                atoms.append(dict(id=str(n),group=str(n),page=0,kind='text',role=role,
                    block_type=role,content=text,length=len(text),bbox=box,extraction_method='native'))
            pdf.save(path)
        return SourceIndex(dict(source_pdf=str(path),atoms=atoms))

    def test_glyph_backed_label_stays_bold_body_instead_of_heading(self):
        for union in (False,True):
            with self.subTest(union=union), tempfile.TemporaryDirectory() as name:
                index=self.source(Path(name),union=union); before=deepcopy(index.source)
                self.assertEqual([(r['left'],r['right']) for r in run_in_title_rows(index)],[('0','1')])
                self.assertEqual(recognized_heading_registry(index),[])
                self.assertEqual(index.source,before)

    def test_translation_and_ocr_geometry_still_require_live_native_glyphs(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name),union=True)
            for n,a in enumerate(index.atoms):
                a.update(content='方法：' if n==0 else '这是审校后的正文。',extraction_method='ocr')
                a['length']=len(a['content'])
            index=SourceIndex(index.source)
            self.assertEqual(len(run_in_title_rows(index)),1)
            Path(index.source['source_pdf']).unlink()
            self.assertEqual(run_in_title_rows(index),[])
            self.assertEqual(len(recognized_heading_registry(index)),1)

    def test_slightly_larger_bold_label_uses_shared_baseline(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name),title_size=11.12,union=True)
            self.assertEqual(len(run_in_title_rows(index)),1)

    def test_independent_headings_fonts_and_column_gaps_are_barriers(self):
        for options in (dict(bold=False),dict(body_bold=True),dict(gap=35),dict(below=16)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as name:
                index=self.source(Path(name),**options)
                self.assertEqual(run_in_title_rows(index),[])
                self.assertEqual(len(recognized_heading_registry(index)),1)

    def test_semantics_edits_missing_geometry_and_page_breaks_are_barriers(self):
        for defect in ('object','edit','recovery','panel','page','box','kind','number'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                index=self.source(Path(name));barriers=[];changed=[]
                if defect=='object':barriers=[dict(lo=0,hi=index.prefix[2])]
                if defect=='edit':changed=['1']
                if defect=='recovery':index.atoms[1]['recovery_issue']='unconfirmed'
                if defect=='panel':index.atoms[1]['source_panel']={"id":"x"}
                if defect=='page':index.atoms[1]['page']=1
                if defect=='box':index.atoms[1]['bbox']=None
                if defect=='kind':index.atoms[1]['kind']='display_math'
                if defect=='number':index.atoms[0]['content']='1.1 Method:'
                self.assertEqual(run_in_title_rows(index,barriers,changed),[])

    def test_compiled_label_and_body_share_baseline_with_all_anchors(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root,union=True);project=root/'latex'
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                recognized_heading_registry(index),[],dict(title='Run-in regression',author=''),
                load_config(),project,sample_mode=True)
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(project/'main.pdf') as pdf:
                names=pdf.resolve_names()
                self.assertTrue(set(audit['source_anchors'].values())<=names.keys())
                self.assertFalse(any('Method' in r[1] for r in pdf.get_toc()))
                page=pdf[names['source-0']['page']]
                label=page.search_for('Method:')[0];body=page.search_for('The estimate')[0]
                self.assertAlmostEqual(label.y1,body.y1,delta=1)
                self.assertLess(label.x1,body.x0)
                separate=page.search_for('A separate paragraph.')[0]
                self.assertGreater(separate.y0,body.y1)
