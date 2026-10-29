"""Short source lines with several fractions retain readable mathematical size."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable,load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.display_layout import fraction_prose_rows,fraction_prose_units
from book_pipeline.publishing.latex import write_project


def source():
    values=[('text','Evaluate '),('math',r'\frac{u}{v}=\frac{r}{s}'),
            ('text',' and compare '),('math',r'\frac{p}{q}'),('text','.')]
    atoms=[dict(id=f'a{n}',group='row',page=0,kind=kind,block_type='text',role='text',
                bbox=[.2,.2,.8,.238],content=text,length=len(text)) for n,(kind,text) in enumerate(values)]
    return SourceIndex(dict(atoms=atoms,source_page_count=1))


class FractionProseRowTests(unittest.TestCase):
    def test_fraction_chain_and_short_source_explanation_stay_together_under_pressure(self):
        atoms=source().atoms
        atoms += [dict(a,id=a['id']+'b',group='row2',bbox=[.2,.25,.8,.288]) for a in source().atoms]
        for n,quote in enumerate(('The fraction chain has this consequence for the original expression. '*2,
                                  'The final paragraph states the interpretation of this calculation. '*2)):
            atoms.append(dict(id=f'closing{n}',group=f'closing{n}',page=0,kind='text',role='text',
                block_type='text',bbox=[.2,.3+n*.04,.8,.33+n*.04],content=quote,length=len(quote)))
        index=SourceIndex(dict(atoms=atoms,source_page_count=1))
        rows=fraction_prose_rows(index)
        self.assertEqual(len(fraction_prose_units(index,rows)),1)
        for box in ([.7,.25,1,.288],[.2,.1,.8,.138],[.2,.5,.8,.538]):
            altered=SourceIndex(dict(atoms=[dict(a,bbox=box) if a['group']=='row2' else dict(a) for a in atoms]))
            self.assertEqual(fraction_prose_units(altered,fraction_prose_rows(altered)),[])
        self.assertEqual(fraction_prose_units(index,rows,barriers=[index.prefix[-1]-1]),
                         [dict(method='bounded_fraction_chain_closing_prose',lo=0,hi=index.prefix[-2],page=0)])
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Bounded chain',author=''),load_config(),root,sample_mode=True)
            main=root/'main.tex'
            main.write_text(main.read_text().replace(r'\input{chapters/generated-001.tex}',
                r'\noindent\rule{1pt}{\dimexpr\textheight-5\baselineskip\relax}\par'+'\n'
                +r'\input{chapters/generated-001.tex}'))
            result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                self.assertEqual(len(pdf),2)
                self.assertEqual({pdf.resolve_names()['source-'+a['id']]['page'] for a in atoms},{1})

    def test_source_geometry_complete_group_and_semantic_edges_are_required(self):
        self.assertEqual(len(fraction_prose_rows(source())),1)
        for defect in ('long','flat','tall','caption','panel','page','tag','matrix','box','boundary','excluded'):
            index=source();barriers=[];skipped=[]
            if defect=='long':index.atoms[0]['content']='An ordinary explanatory paragraph. '*20
            if defect=='flat':
                for a in index.atoms:a['bbox']=[.2,.2,.8,.218]
            if defect=='tall':
                for a in index.atoms:a['bbox']=[.2,.2,.8,.3]
            if defect=='caption':index.atoms[0]['role']='image_caption'
            if defect=='panel':index.atoms[0]['source_panel']={'id':'note'}
            if defect=='page':index.atoms[-1]['page']=1
            if defect=='tag':index.atoms[1]['content']+=r'\tag{3}'
            if defect=='matrix':index.atoms[1]['content']+=r'\begin{matrix}a\\b\end{matrix}'
            if defect=='box':index.atoms[-1]['bbox']=[.1,.2,.8,.238]
            if defect=='boundary':barriers=[index.prefix[1]]
            if defect=='excluded':skipped=[(index.prefix[1],index.prefix[2],'test')]
            with self.subTest(defect=defect):
                self.assertEqual(fraction_prose_rows(index,barriers,skipped),[])

    def test_compiled_fraction_numerators_retain_body_size_and_source_anchors(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=source()
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Fraction rows',author=''),load_config(),root,sample_mode=True)
            result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'main.pdf') as pdf:
                text=pdf[0].get_text()
                for value in ('Evaluate','and compare'):self.assertIn(value,text)
                spans=[s for b in pdf[0].get_text('dict')['blocks'] if b['type']==0
                       for line in b['lines'] for s in line['spans']]
                math=[s for s in spans if any(c in s['text'] for c in '𝑢𝑣𝑟𝑠𝑝𝑞')]
                self.assertTrue(math,[(s['font'],s['text']) for s in spans])
                self.assertTrue(all(s['size']>10.4 for s in math),math)
                self.assertTrue(all('source-'+a['id'] in pdf.resolve_names() for a in index.atoms))
            self.assertEqual(audit['accounted_units'],index.prefix[-1])
