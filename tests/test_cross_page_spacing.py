"""Merged paragraphs preserve word boundaries at original physical page joins."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import load_config, executable
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project


class CrossPageSpacingTests(unittest.TestCase):
    def test_saved_merged_latin_words_remain_separate_and_cjk_stays_contiguous(self):
        specs=[('latin',0,'The columns moved one'),('latin',1,'to the right.'),
               ('spaced',1,'This already has '),('spaced',2,'a space.'),
               ('chinese',2,'跨页段落'),('chinese',3,'继续。'),
               ('hyphen',3,'A source hyphen-'),('hyphen',4,'ated word.')]
        atoms=[dict(id=str(n),group=g,page=p,kind='text',role='text',block_type='text',
                    content=t,length=len(t),bbox=[.1,.2,.8,.3]) for n,(g,p,t) in enumerate(specs)]
        index=SourceIndex(dict(atoms=atoms))
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Page joins',author=''),load_config(),root,sample_mode=True)
            result=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail'])
            with pymupdf.open(root/'main.pdf') as pdf:
                text=' '.join(p.get_text() for p in pdf)
                self.assertIn('moved one to the right.',text)
                self.assertIn('already has a space.',text)
                self.assertIn('跨页段落继续。',text)
                self.assertIn('hyphen-ated word.',text)
