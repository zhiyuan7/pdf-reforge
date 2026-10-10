import tempfile
import unittest
from pathlib import Path

from book_pipeline.config import save
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.processed_book import EditedBook
from book_pipeline.structure.headings import heading_tail_end
from book_pipeline.structure.page_content import subheading_excerpt
from book_pipeline.structure.toc_matching import TocMatcher


def atom(identifier, text, kind='text'):
    return dict(id=identifier, group=identifier, page=0, kind=kind, role='text',
                block_type='text', content=text, length=max(1, len(text)))


class HeadingTailTests(unittest.TestCase):
    def test_catalog_match_keeps_footnote_and_closing_punctuation(self):
        source = '3. 极限点引理 (波尔查诺-魏尔斯特拉斯原理①). 我们记得, 含有点 '
        entry = dict(id='toc-0044', title='3. 极限点引理 (波尔查诺-魏尔斯特拉斯原理)',
                     printed_page='59')
        index = SourceIndex(dict(atoms=[atom('source', source)]))
        matcher = TocMatcher(index, [entry], [])
        matcher.automatic(False)
        match = matcher.matches[entry['id']]
        self.assertEqual(match['source_text'], source.split(' 我们')[0])
        self.assertTrue(source[match['hi']:].lstrip().startswith('我们记得'))

    def test_exact_catalog_match_also_keeps_sentence_punctuation(self):
        source = '3. 极限点引理（原理）！正文'
        entry = dict(id='toc-0044', title='3. 极限点引理（原理）', printed_page='59')
        index = SourceIndex(dict(atoms=[atom('source', source)]))
        matcher = TocMatcher(index, [entry], [])
        matcher.automatic(False)
        self.assertEqual(matcher.matches[entry['id']]['source_text'], '3. 极限点引理（原理）！')

    def test_subheading_excerpt_keeps_closing_marks_after_sentence_end(self):
        text = '3. 小标题（说明。）① 正文。'
        block = dict(text=text, parts=[dict(kind='text', text=text, start_offset=0)])
        result = subheading_excerpt(block, '数字.')
        self.assertEqual(result['text'], '3. 小标题（说明。）①')
        self.assertEqual(result['parts'][0]['end_offset'], len(result['text']))

    def test_tail_does_not_take_body_numbers_opening_marks_or_next_paragraph(self):
        for tail in (' 正文', ' 1. 后文', ' （正文）', '\n① 新段落'):
            text = '3. 小标题。' + tail
            with self.subTest(tail=tail):
                self.assertEqual(heading_tail_end(text, len('3. 小标题。')), len('3. 小标题。'))

    def test_subheading_never_splits_following_formula(self):
        text = '3. 小标题。'
        formula = ')x + 1'
        block = dict(text=text+formula, parts=[dict(kind='text', text=text),
                                              dict(kind='math', text=formula)])
        result = subheading_excerpt(block, '数字.')
        self.assertEqual(result['text'], text)
        self.assertEqual(len(result['parts']), 1)

    def test_saved_book_moves_tail_once_and_keeps_notes_and_images(self):
        title = '3. 极限点引理 (波尔查诺-魏尔斯特拉斯原理'
        text = title + '①). 我们记得, 含有点 '
        note = '① 波尔查诺是数学家。'
        source = SourceIndex(dict(atoms=[atom('source', text), atom('image', '', 'image'),
                                        atom('note', note)]))
        old_end = len(title)
        note_lo = len(text)+1
        heading = dict(start=source.address(0), end=source.address(old_end, True),
                       kind='subsection', heading_level=3, toc_category='body', toc_id='toc-0044')
        body = dict(b='body', section='toc-0044', source_blocks=['body'], text=text[old_end:],
                    pieces=[dict(lo=0, hi=len(text)-old_end, source_lo=old_end,
                                 source_hi=len(text), corrected=False)],
                    parts=[dict(source_atom='source', start_offset=old_end, end_offset=len(text),
                                kind='text', text=text[old_end:]),
                           dict(source_atom='image', start_offset=0, end_offset=1,
                                kind='image', text='')])
        note_block = dict(b='note', section='toc-0044', source_blocks=['note'], text=note,
                          pieces=[dict(lo=0, hi=len(note), source_lo=note_lo,
                                       source_hi=note_lo+len(note), corrected=False)],
                          parts=[dict(source_atom='note', start_offset=0, end_offset=len(note),
                                      kind='text', text=note)])
        book = dict(source={}, sections=[dict(id='toc-0044', heading=heading)],
                    blocks=[body, note_block], targets=[dict(id='H0044', kind='heading', lo=0,
                        hi=old_end, context=dict(heading='toc-0044'), label=title)],
                    footnotes=[dict(id='footnote-note', block='note', lo=note_lo,
                                    hi=note_lo+len(note), marker=dict(text='①', start=0, end=1),
                                    reference=dict(lo=old_end, hi=old_end+1))])
        with tempfile.TemporaryDirectory() as directory:
            save(Path(directory)/'sections/toc-0044/processed.json', dict(modules=[]))
            edited = EditedBook(source, book, directory)
        item = edited.registry[0]
        self.assertEqual(edited.index.text(item['lo'], item['hi']), title+'①).')
        self.assertTrue(edited.index.text(item['hi'], item['hi']+8).lstrip().startswith('我们记得'))
        self.assertEqual(edited.index.prefix[-1], source.prefix[-1])
        self.assertEqual(sum(a['kind']=='image' for a in edited.index.atoms), 1)
        self.assertEqual(len(edited.footnotes), 1)
        self.assertTrue(all(a['source_hi'] <= b['source_lo']
                            for a, b in zip(edited.segments, edited.segments[1:])))


if __name__ == '__main__':
    unittest.main()
