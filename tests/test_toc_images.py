"""Image-owned TOC parsing must not depend on garbled multi-column OCR."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.structure.toc_catalog import image_entries, read_catalog
from book_pipeline.structure.toc_matching import TocMatcher


def entry(title, page=None, level=1, category='non_body', box=None):
    return dict(title=title,printed_page=page,category=category,level=level,
                bbox=box or [.1,.1,.4,.13])


class ImageTocTests(unittest.TestCase):
    def test_blank_page_labels_and_column_order_survive_without_ocr_spans(self):
        rows=[entry('关于作者'),entry('第1章 复数','1',category='body',box=[.1,.2,.4,.23]),
              entry('28 指数函数','65',2,'body',[.1,.8,.4,.83]),
              entry('29 对数函数','67',2,'body',[.55,.1,.9,.13])]
        result=image_entries(dict(entries=rows),9,dict(path='image.png',sha256='hash'))
        self.assertEqual([r['title'] for r in result],[r['title'] for r in rows])
        self.assertIsNone(result[0]['printed_page'])
        self.assertEqual(result[-1]['toc_page'],9)
        self.assertNotIn('toc_start',result[-1])

    def test_bad_page_label_geometry_and_skipped_parent_are_rejected(self):
        for bad in [entry('前言','无'),entry('前言',''),entry('前言',box=[0,0,2,1]),
                    entry('1.2 节','2',2,'body')]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                image_entries(dict(entries=[bad]),1,{})

    def test_duplicate_region_is_rejected_but_repeated_exercise_titles_are_kept(self):
        first=entry('习题','12',category='body')
        with self.assertRaises(ValueError):image_entries(dict(entries=[first,deepcopy(first)]),1,{})
        second={**first,'bbox':[.1,.4,.4,.43]}
        self.assertEqual(len(image_entries(dict(entries=[first,second]),1,{})),2)

    def test_cross_page_hierarchy_keeps_previous_parent(self):
        previous=image_entries(dict(entries=[entry('第1章 复数','1',category='body')]),1,{})
        rows=image_entries(dict(entries=[entry('2 基本性质','2',2,'body')]),2,{},previous)
        self.assertEqual(rows[0]['id'],'toc-0002')

    def test_actual_image_request_contains_no_ocr_text_and_records_image_hash(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name);pdf=directory/'source.pdf'
            with pymupdf.open() as doc:doc.new_page();doc.save(pdf)
            index=SourceIndex(dict(source_pdf=str(pdf),source_page_count=1,selected_pages=[0],
                input_sha256='pdf-sha',mineru_sha256='ocr-sha',atoms=[]))
            requests=[]
            class Client:
                def ask(self,task,system,payload,images,validator):
                    requests.append((task,payload,images))
                    answer=dict(entries=[entry('前言')]);validator(answer);return answer
            with patch('book_pipeline.structure.toc_catalog.analyze_toc',return_value=dict(pages=[1])):
                rows,_,_=read_catalog(index,{},Client(),dict(toc_pages=[],image_dpi=72),directory/'headings')
            self.assertEqual(set(requests[0][1]),{'physical_page','previous_entries'})
            self.assertTrue(requests[0][2][0].is_file())
            self.assertEqual(len(rows[0]['toc_image']['sha256']),64)
            self.assertEqual(rows[0]['title'],'前言')

    def test_image_title_matches_body_ocr_even_when_toc_ocr_is_wrong(self):
        atoms=[]
        for n,(text,kind) in enumerate([('29错字 1复敷','index'),('前言','paragraph_title'),('第1章 复数','paragraph_title')]):
            atoms.append(dict(id=f'a{n}',group=f'g{n}',page=n,kind='text',role=kind,block_type=kind,
                              content=text,length=len(text),bbox=[.1,.1,.8,.2]))
        index=SourceIndex(dict(atoms=atoms,source_page_count=3))
        rows=image_entries(dict(entries=[entry('前言'),entry('第1章 复数','1',category='body',box=[.1,.3,.4,.33])]),1,{})
        matcher=TocMatcher(index,rows,[1]);matcher.automatic(True)
        self.assertEqual({k:v['page'] for k,v in matcher.matches.items()},{'toc-0001':2,'toc-0002':3})
        self.assertIsNone(matcher.expected_page(rows[0]))


if __name__=='__main__':unittest.main()
