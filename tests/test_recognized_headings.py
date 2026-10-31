"""Translated labels and split chapter titles keep their source hierarchy."""
import unittest
from pathlib import Path
import tempfile

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.structure.recognized_headings import recognized_heading_registry


class RecognizedHeadingTests(unittest.TestCase):
    def index(self, rows):
        return SourceIndex(dict(atoms=[dict(id=str(n), group=str(n), page=page,
            kind='text', role=role, block_type=role, content=text, length=len(text), bbox=box)
            for n,(text,role,page,box) in enumerate(rows)]))

    def test_translated_chapters_and_appendix_subsections_keep_levels(self):
        labels = ['第七章 梯度', '7.1 求导', '章节 8 性能', 'Chapter 9 Regularization',
                  '附录B 数学', 'B. 1 函数', 'B. 1.2 连续性', '10 Networks',
                  '附录B. 3.2 向量范数', 'Appendix C.1 Probability']
        index = self.index([(t,'paragraph_title',n,[.1,.2,.7,.25]) for n,t in enumerate(labels)])
        rows = recognized_heading_registry(index)
        self.assertEqual([r['heading_level'] for r in rows], [1,2,1,1,1,2,3,1,3,2])
        self.assertEqual([r['kind'] for r in rows],
                         ['chapter','section','chapter','chapter','chapter','section','subsection','chapter',
                          'subsection','section'])
        self.assertEqual([r['label'] for r in rows], labels)

    def test_separate_number_and_title_are_one_heading_with_all_atoms(self):
        index = self.index([('附录C','paragraph_title',0,[.15,.25,.3,.28]),
                            ('概率','paragraph_title',0,[.15,.31,.6,.34]),
                            ('C. 1 随机变量','paragraph_title',0,[.1,.5,.6,.52])])
        rows = recognized_heading_registry(index)
        self.assertEqual([r['heading_level'] for r in rows], [1,2])
        self.assertEqual(rows[0]['source_groups'], ['0','1'])
        self.assertEqual(index.text(rows[0]['lo'], rows[0]['hi']), '附录C概率')
        self.assertEqual(rows[0]['label'], '附录C 概率')

    def test_no_merge_across_body_page_or_geometry_barriers(self):
        for role, page, box in [('text',0,[.1,.31,.7,.35]),
                                ('paragraph_title',1,[.1,.31,.7,.35]),
                                ('paragraph_title',0,[.5,.31,.9,.35]),
                                ('paragraph_title',0,[.1,.6,.7,.65])]:
            with self.subTest(role=role, page=page, box=box):
                index = self.index([('第12章','paragraph_title',0,[.1,.25,.3,.28]),
                                    ('标题',role,page,box)])
                self.assertEqual(recognized_heading_registry(index)[0]['source_groups'], ['0'])

    def test_numbered_next_heading_and_prose_are_not_swallowed(self):
        index = self.index([('第12章','paragraph_title',0,[.1,.25,.3,.28]),
                            ('12.1 主题','paragraph_title',0,[.1,.31,.7,.35]),
                            ('第13章在正文中出现。','text',0,[.1,.4,.7,.5])])
        self.assertEqual(len(recognized_heading_registry(index)), 2)
        self.assertEqual(recognized_heading_registry(index, [1]), [])

    def test_compiled_chapter_starts_new_page_and_split_title_keeps_anchors(self):
        import pymupdf
        from book_pipeline.config import executable, load_config
        from book_pipeline.publishing.compiler import compile_project
        from book_pipeline.publishing.latex import write_project
        index = self.index([('上一章的习题。','text',0,[.1,.2,.8,.25]),
                            ('第七章','paragraph_title',1,[.1,.25,.3,.28]),
                            ('梯度与初始化','paragraph_title',1,[.1,.31,.7,.34]),
                            ('本章讨论如何计算梯度。','text',1,[.1,.4,.8,.45]),
                            ('7.1 问题定义','paragraph_title',1,[.1,.5,.7,.52]),
                            ('以下给出一个网络。','text',1,[.1,.55,.8,.6])])
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)/'latex'
            cfg = load_config(); cfg['start_page_levels'] = [1]
            audit = write_project(index, dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                recognized_heading_registry(index), [], dict(title='标题回归',author=''), cfg,
                project, heading_layout=dict(standalone_levels=[],centered_levels=[1],page_break_after_levels=[]))
            result = compile_project(project, {t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'], result['log_tail'])
            self.assertEqual(result['warnings'], [])
            with pymupdf.open(project/'main.pdf') as pdf:
                names = pdf.resolve_names()
                self.assertEqual(set(audit['source_anchors'].values())-names.keys(), set())
                self.assertLess(names['source-0']['page'], names['source-1']['page'])
                self.assertEqual(names['source-1']['page'], names['source-2']['page'])
                page = pdf[names['source-1']['page']]
                title_text = ''.join(s['text'] for b in page.get_text('dict')['blocks']
                    for line in b.get('lines', []) for s in line['spans'] if s['size'] > 16)
                self.assertEqual(title_text.count('梯度与初始化'), 1)
                toc = pdf.get_toc()
                chapter = next(r for r in toc if '梯度与初始化' in r[1])
                section = next(r for r in toc if '问题定义' in r[1])
                self.assertEqual(chapter[0]+1, section[0])
