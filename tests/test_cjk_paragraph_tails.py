"""CJK paragraph endings stay readable at narrow inline-formula boundaries."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import Renderer, write_project
from book_pipeline.publishing.raster_styles import METHOD


DECISIONS = dict(corrections=[], joins=[], exclusions=[], bold_spans=[])


def source():
    rows = [('body', 'text', '甲乙丙丁戊己'), ('body', 'math', r'\frac{x}{y}'),
            ('body', 'text', '推得'), ('after', 'text', '后续文字。')]
    return SourceIndex(dict(source_page_count=1, atoms=[
        dict(id=f'a{n}', group=group, page=0, kind=kind, role='text', block_type='text',
             content=text, length=len(text), bbox=[.1, .1+n*.04, .9, .13+n*.04])
        for n, (group, kind, text) in enumerate(rows)]))


class CJKParagraphTailTests(unittest.TestCase):
    def test_only_complete_terminal_pairs_are_protected(self):
        for variant in ('terminal', 'continued', 'joined', 'corrected', 'caption', 'partial'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as name:
                index = source()
                atom = index.atoms[2]
                decisions = {**DECISIONS}
                if variant == 'continued': index.atoms[3]['group'] = 'body'
                if variant == 'caption': atom['role'] = 'figure_caption'
                if variant == 'corrected':
                    decisions['corrections'] = [dict(atom='a2', start_offset=0, end_offset=2,
                                                     replacement='得到')]
                renderer = Renderer(index, decisions, [], [], Path(name))
                if variant == 'joined': renderer.join_left.add('a2')
                lo = 1 if variant == 'partial' else 0
                tex = renderer.fragment(atom, lo, 2, False)
                self.assertEqual(r'\nobreak{}' in tex, variant == 'terminal')

    def test_actual_pdf_keeps_styled_linked_tail_and_destination_with_text(self):
        index = source()
        decisions = {**DECISIONS, 'source_styles': {'a2': [dict(start=0, end=2, quote='推得',
                        italic=False, bold=True, cjk_family='sans', method=METHOD)]}}
        refs = [dict(atom='a2', start_offset=0, end_offset=2, text='推得', target='source-a3')]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_project(index, decisions, [], refs, dict(title='Tail', author=''),
                          load_config(), root, sample_mode=True)
            chapter = root/'chapters/generated-001.tex'
            body = chapter.read_text()
            # Find a real line-break opportunity, rather than assume that a
            # synthetic paragraph happens to exercise the failure at one width.
            blocks = []
            for width in range(62, 103, 4):
                value = body.replace('source-', f'w{width}-source-')
                blocks.append(r'\begin{minipage}{'+str(width)+r'pt}'+value+r'\end{minipage}\par')
            chapter.write_text('\n'.join(blocks))
            tools = {t: executable(t) for t in ('xelatex', 'latexmk')}
            result = compile_project(root, tools)
            self.assertTrue(result['success'], result['log_tail'])
            self.assertEqual(result['warnings'], [])
            with pymupdf.open(root/'main.pdf') as pdf:
                protected = [[(c['c'], c['origin']) for b in page.get_text('rawdict')['blocks']
                              if b['type'] == 0 for line in b['lines'] for span in line['spans']
                              for c in span['chars'] if c['c'] in '推得'] for page in pdf]
                chars = [c for page in protected for c in page]
                self.assertEqual(len(chars), 22)
                self.assertTrue(all(abs(a[1][1]-z[1][1]) < 1 for a, z in zip(chars[::2], chars[1::2])))
                names = pdf.resolve_names()
                for width in range(62, 103, 4):
                    target = names[f'w{width}-source-a2']
                    page = pdf[target['page']]
                    rects = page.search_for('推得')
                    self.assertTrue(any(abs(r.x0-target['to'][0]) < 2
                                        and abs(r.y0-(page.rect.height-target['to'][1])) < 15 for r in rects), (width, target, rects))
                self.assertTrue(all('NotoSansCJK' in s['font'] for page in pdf
                    for b in page.get_text('dict')['blocks'] if b['type'] == 0
                    for line in b['lines'] for s in line['spans'] if '推' in s['text'] or '得' in s['text']))
            chapter.write_text(chapter.read_text().replace(r'\nobreak{}', ''))
            baseline = compile_project(root, tools)
            self.assertTrue(baseline['success'], baseline['log_tail'])
            with pymupdf.open(root/'main.pdf') as pdf:
                chars = [(c['c'], c['origin']) for page in pdf for b in page.get_text('rawdict')['blocks']
                         if b['type'] == 0 for line in b['lines'] for span in line['spans']
                         for c in span['chars'] if c['c'] in '推得']
                self.assertTrue(any(abs(a[1][1]-z[1][1]) > 1 for a, z in zip(chars[::2], chars[1::2])))
