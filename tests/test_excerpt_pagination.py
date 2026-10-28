from pathlib import Path
import tempfile
import unittest
import pymupdf
from book_pipeline.config import executable,load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.processed_book import saved_heading_policy
from book_pipeline.structure.headings import validate_metadata


class ExcerptPaginationTests(unittest.TestCase):
    def test_headingless_and_section_only_body_excerpts_use_arabic_pages(self):
        metadata=validate_metadata(dict(book=dict(title='Body excerpt',author='Test author')))
        for category,number,expected in [('body','2.1','mainmatter'),('preface','','frontmatter'),
                                          (None,None,'mainmatter')]:
            with self.subTest(category=category),tempfile.TemporaryDirectory() as name:
                text='A body fragment with no chapter title.'
                index=SourceIndex(dict(atoms=[dict(id='a',group='g',page=0,kind='text',role='text',
                    block_type='text',content=text,length=len(text))],source_page_count=1))
                registry=[] if category is None else [dict(kind='section',heading_level=2,
                    anchor='h',lo=0,hi=len(text),label=text,number=number,toc_category=category)]
                config={**load_config(),'publication_scope':'excerpt'}
                settings,config=saved_heading_policy(metadata,config)
                root=Path(name)
                write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),registry,[],
                    metadata['book'],config,root,heading_layout=settings)
                tex=(root/'main.tex').read_text()
                self.assertIn('\\'+expected,tex)
                compiled=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(compiled['success'],compiled['log_tail'])
                with pymupdf.open(root/'main.pdf') as pdf:
                    self.assertEqual(len(pdf),1)
                    # Physical printed footer is also checked, not just TeX.
                    margins=pdf[0].get_text(clip=pymupdf.Rect(0,0,pdf[0].rect.width,70))
                    margins+=pdf[0].get_text(clip=pymupdf.Rect(0,pdf[0].rect.height-70,
                                                           pdf[0].rect.width,pdf[0].rect.height))
                    self.assertIn('1' if expected=='mainmatter' else 'i',margins.splitlines())

    def test_short_excerpt_sections_flow_while_book_and_explicit_breaks_are_retained(self):
        texts=['2.1 First section','A short opening paragraph.','2.2 Second section','The next section continues here.']
        atoms=[dict(id=f'a{n}',group=f'g{n}',page=0,kind='text',role='text',block_type='text',
                    content=t,length=len(t)) for n,t in enumerate(texts)]
        index=SourceIndex(dict(atoms=atoms,source_page_count=1))
        registry=[dict(kind='section',heading_level=2,anchor=f'h{n}',label=texts[n],number=f'2.{n//2+1}',
                       lo=index.prefix[n],hi=index.prefix[n+1]) for n in (0,2)]
        metadata=validate_metadata(dict(book=dict(title='Excerpt flow',author='Test author')))
        for scope,forced,pages in [('excerpt',False,1),('book',False,2),('excerpt',True,2)]:
            with self.subTest(scope=scope,forced=forced),tempfile.TemporaryDirectory() as name:
                config={**load_config(),'publication_scope':scope}
                if forced:config['page_break_before_anchors']=['h2']
                settings,config=saved_heading_policy(metadata,config)
                root=Path(name)
                write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),registry,[],
                              metadata['book'],config,root,heading_layout=settings,sample_mode=True)
                compiled=compile_project(root,{t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(compiled['success'],compiled['log_tail'])
                with pymupdf.open(root/'main.pdf') as pdf:
                    self.assertEqual(len(pdf),pages)
                    self.assertIn('First section',pdf[0].get_text())
                    self.assertIn('Second section',pdf[-1].get_text())
