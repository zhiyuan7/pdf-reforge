from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.stacked_figures import stacked_figure_floats


class StackedFigureTests(unittest.TestCase):
    def source(self, root):
        image = root/'image.png'
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0,0,200,80), False)
        pix.clear_with(160); pix.save(image)
        atoms = [dict(id=f'img{i}', group=f'g{i}', page=0, kind='image', role='image_body',
                      block_type='image', content='',length=1, extraction_method='native',
                      bbox=[.15,.2+i*.14,.8,.33+i*.14],asset=str(image)) for i in range(3)]
        atoms.append(dict(id='caption',group='cap',page=0,kind='text',role='image_caption',
                          block_type='image_caption',content='Figure 2.8 Three shared panels.',length=31,
                          extraction_method='native',bbox=[.2,.62,.75,.66]))
        return SourceIndex(dict(atoms=atoms,source_page_count=1))

    def floats(self, index):
        return {f'g{i}': (index.prefix[i],index.prefix[i+1] if i < 2 else index.prefix[-1],'figure') for i in range(3)}

    def test_shared_caption_keeps_all_panels_without_mutating_atoms(self):
        with tempfile.TemporaryDirectory() as name:
            index=self.source(Path(name)); before=deepcopy(index.source)
            rows=stacked_figure_floats(index,self.floats(index))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['groups'],['g0','g1','g2'])
            self.assertEqual(rows[0]['hi'],index.prefix[-1])
            self.assertEqual(index.source,before)

    def test_separate_captions_pages_gaps_and_semantic_boundaries_are_barriers(self):
        for defect in ('caption','page','gap','column','scan','recovery','tall','semantic','excluded','above'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as name:
                index=self.source(Path(name)); barriers=[]; skipped=[]; floats=self.floats(index)
                if defect=='caption': floats['g2']=(2,3,'figure')
                if defect=='page':index.atoms[-1]['page']=1
                if defect=='gap':index.atoms[2]['bbox'][1]+=.1
                if defect=='column':index.atoms[2]['bbox']=[.85,.48,.99,.61]
                if defect=='scan':index.atoms[-1]['extraction_method']='ocr'
                if defect=='recovery':index.atoms[-1]['recovery_issue']='formula_source_crop_requires_transcription'
                if defect=='tall':
                    index.atoms[0]['bbox'][1]=.01; index.atoms[-1]['bbox'][3]=.9
                if defect=='semantic':barriers=[index.prefix[1]]
                if defect=='excluded':skipped=[(index.prefix[1],index.prefix[2],'excluded')]
                if defect=='above':index.atoms[-1]['bbox']=[.2,.1,.75,.15]
                self.assertEqual(stacked_figure_floats(index,floats,barriers,skipped),[])

    def test_compiled_stack_is_one_float_with_all_image_and_caption_anchors(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); index=self.source(root); config=load_config();config['float_assets']=True
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Shared panels',author=''),config,root/'latex',sample_mode=True)
            tex=''.join(p.read_text() for p in (root/'latex/chapters').glob('*.tex'))
            self.assertEqual(tex.count(r'\begin{figure}'),1)
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            self.assertEqual(result['warnings'],[])
            self.assertTrue(audit['coverage_complete'])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                names=pdf.resolve_names()
                self.assertTrue(all('source-'+a['id'] in names for a in index.atoms))
                self.assertEqual(len({names['source-'+a['id']]['page'] for a in index.atoms}),1)
                self.assertEqual(pdf[0].get_text().count('Three shared panels.'),1)
