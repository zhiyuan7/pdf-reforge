"""A formula-introducing paragraph must not leave its title behind."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project


class HeadingIntroductionTests(unittest.TestCase):
    def test_title_moves_with_its_first_formula_introduction(self):
        atoms = []
        for n,(key,kind,role,text) in enumerate([('heading','text','paragraph_title','2.1.3 激活函数'),
                                  ('intro','text','text','考虑以下两种激活函数：'),
                                  ('math','display_math','equation',r'f(x)=\begin{cases}0&x<0\\1&x\geq0\end{cases}'),
                                  ('body','text','text','这里讨论这些函数的导数。')]):
            atoms.append(dict(id=key,group=key,page=0,kind=kind,role=role,
                block_type=role,content=text,length=len(text),bbox=[.1,.2+n*.06,.8,.24+n*.06]))
        index = SourceIndex(dict(atoms=atoms))
        registry = [dict(kind='subsection',heading_level=3,lo=0,hi=index.prefix[1],
                         anchor='section',label=atoms[0]['content'],name=atoms[0]['content'])]
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)/'latex'
            write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                registry,[],dict(title='标题分页回归',author=''),load_config(),project,
                heading_layout=dict(standalone_levels=[],page_break_after_levels=[]))
            # Put the section at the same page-foot boundary as the real failure.
            # Reserve space before the renderer's heading guards, not inside them.
            for path in (project/'chapters').glob('*.tex'):
                text = path.read_text()
                if r'\BookHeading{3}' in text:
                    path.write_text(r'\noindent A preceding paragraph.\par'
                        r'\vspace*{\dimexpr\textheight-6.25\baselineskip\relax}'+'\n'+text)
            result = compile_project(project,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            with pymupdf.open(project/'main.pdf') as pdf:
                names = pdf.resolve_names()
                self.assertEqual(names['source-heading']['page'],names['source-intro']['page'])
