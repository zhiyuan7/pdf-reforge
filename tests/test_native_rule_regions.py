"""Thin paired rules need exclusive native text and actual inset geometry."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable,load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.native_rule_regions import native_rule_regions


class NativeRuleRegionsTests(unittest.TestCase):
    def source(self,root,scenario='normal'):
        atoms=[]
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=500,height=700)
            for n,(x,y,text) in enumerate([(60,100,'Context for the mathematical example.'),
                    (80,160,'Example 2.1'),(80,185,'1. A complete example with two source rules.'),
                    (60,240,'Following main paragraph.')]):
                page.insert_text((x,y),text)
                box=page.get_text('dict')['blocks'][-1]['bbox']
                # Main prose has its full observed column bounding box.
                box=(60,box[1],460,box[3]) if n in {0,3} else box
                box=[v/(500 if k%2==0 else 700) for k,v in enumerate(box)]
                atoms.append(dict(id=str(n),group=str(n),page=0,kind='text',role='text',
                    content=text,length=len(text),bbox=box,extraction_method='native',
                    block_type='paragraph_title' if n==1 else 'text'))
            for y in (145,205):
                page.draw_line((80,y),(460,y),width=.4,color=(.13,.12,.125))
            if scenario=='table':page.draw_line((80,175),(460,175),width=.4,color=(.13,.12,.125))
            if scenario=='crossing':atoms[2]['bbox'][0]=.1
            if scenario=='scan':atoms[1]['extraction_method']='ocr'
            pdf.save(root/'source.pdf')
        return SourceIndex(dict(source_pdf=str(root/'source.pdf'),atoms=atoms))

    def test_pair_abstains_for_tables_crossing_edits_exclusions_and_semantic_boxes(self):
        for scenario in ('normal','table','crossing','scan','edited','excluded','semantic'):
            with self.subTest(scenario=scenario),tempfile.TemporaryDirectory() as name:
                index=self.source(Path(name),scenario)
                lo,hi=index.prefix[1],index.prefix[3]
                regions=native_rule_regions(index,
                    [dict(lo=lo,hi=hi,kind='example')] if scenario=='semantic' else (),
                    [dict(lo=lo,hi=hi)] if scenario=='excluded' else (),
                    {'2'} if scenario=='edited' else ())
                self.assertEqual(len(regions),int(scenario=='normal'))
                if regions:self.assertAlmostEqual(regions[0]['left_fraction'],.05)

    def test_compiled_inset_keeps_rules_searchable_text_and_all_anchors(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root)
            project=root/'latex'
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Rule evidence',author=''),load_config(),project,sample_mode=True)
            self.assertTrue(audit['coverage_complete'])
            result=compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(project/'main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                page=pdf[0]
                lines=[d for d in page.get_drawings() if d['rect'].width>300 and d['rect'].height<.1
                       and .35<d['width']<.45]
                self.assertEqual(len(lines),2)
                first=page.search_for('Example 2.1')[0];last=page.search_for('A complete example')[0]
                self.assertLess(lines[0]['rect'].y0,first.y0)
                self.assertGreater(lines[1]['rect'].y0,last.y1)
                self.assertGreater(first.x0,36+10)
                self.assertEqual(set(pdf.resolve_names()) & {f'source-{n}' for n in range(4)},
                                 {f'source-{n}' for n in range(4)})


if __name__=='__main__':
    unittest.main()
