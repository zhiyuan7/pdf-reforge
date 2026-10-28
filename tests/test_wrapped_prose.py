"""Source obstacles, semantic barriers, and the PDF actually shipped."""
from copy import deepcopy
from pathlib import Path
import re
import tempfile
import unittest

import pymupdf

from book_pipeline.config import executable, load_config
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.compiler import compile_project
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.wrapped_prose import wrapped_prose


def atom(name, box, text, kind='text', role='text'):
    return dict(id=name,group=name,page=0,kind=kind,role=role,block_type='image' if kind=='image' else 'text',
                content=text,length=max(1,len(text)),bbox=box,block_bbox=box,extraction_method='ocr')


class WrappedProseTests(unittest.TestCase):
    def figure(self, root, left=False):
        path=root/'figure.png'
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=150,height=65)
            page.draw_line((10,50),(130,12),color=(.1,.1,.7),width=2)
            page.insert_text((45,30),'diagram',fontsize=9)
            page.get_pixmap().save(path)
        main=([.48,.2,.94,.31] if left else [.06,.2,.6,.31])
        image=([.06,.2,.45,.31] if left else [.63,.2,.94,.31])
        aa=[atom('opening',main, 'We move the arrow parallel to itself and keep its direction. '*3+
                 'The mathematical argument continues with the geometric construction and'),
            {**atom('picture',image,'','image','image_body'),'asset':str(path)},
            atom('continuation',[.06,.32,.94,.4],
                 'just thinking of the vector completes the construction. '*6),
            atom('after',[.06,.43,.94,.46],'Following paragraph begins after both the image and its explanation.')]
        index=SourceIndex(dict(atoms=aa))
        fig=dict(kind='figure',lo=index.prefix[1],hi=index.prefix[2],anchor='figure-target')
        return index,[fig]

    def test_split_figure_both_sides_preserves_source_intervals(self):
        with tempfile.TemporaryDirectory() as name:
            for left in (False,True):
                index,registry=self.figure(Path(name),left);before=deepcopy(index.source)
                joins,rows=wrapped_prose(index,registry)
                self.assertEqual(joins,[]);self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['note_left'],left)
                self.assertEqual(rows[0]['main_ranges'],[[index.prefix[0],index.prefix[1]],
                                                       [index.prefix[2],index.prefix[3]]])
                self.assertEqual(rows[0]['targets'],['figure-target'])
                self.assertEqual(index.source,before)

    def test_rejects_sentence_end_indented_restart_columns_edits_and_semantics(self):
        with tempfile.TemporaryDirectory() as name:
            base,registry=self.figure(Path(name))
            for reason in ('sentence','uppercase','indented','distant','unrelated_image','table',
                           'page','edited','formula_crop','semantic','footnote','noncontiguous','caption_page'):
                with self.subTest(reason=reason):
                    index=SourceIndex(deepcopy(base.source));objects=deepcopy(registry);exclusions=[];changed=set()
                    if reason=='sentence':index.atoms[0]['content']+='.'
                    elif reason=='uppercase':index.atoms[2]['content']='Another independent paragraph.'
                    elif reason=='indented':index.atoms[2]['block_bbox']=[.14,.32,.94,.4]
                    elif reason=='distant':index.atoms[2]['block_bbox']=[.06,.4,.94,.5]
                    elif reason=='unrelated_image':index.atoms[1]['block_bbox']=[.63,.45,.94,.56]
                    elif reason=='table':index.atoms[1]['kind']='table'
                    elif reason=='page':index.atoms[2]['page']=1
                    elif reason=='edited':changed={'opening'}
                    elif reason=='formula_crop':index.atoms[1]['recovery_issue']='formula_source_crop_requires_transcription'
                    elif reason=='semantic':objects.append(dict(kind='proof',lo=0,hi=index.prefix[3],anchor='proof'))
                    elif reason=='footnote':exclusions=[dict(lo=index.prefix[2],hi=index.prefix[3])]
                    elif reason=='noncontiguous':index.atoms.append({**index.atoms[0],'id':'repeat'})
                    elif reason=='caption_page':index.atoms.insert(2,{**index.atoms[1],
                        'id':'caption','kind':'text','role':'image_caption','page':1})
                    index=SourceIndex(index.source)
                    self.assertEqual(wrapped_prose(index,objects,exclusions,changed),([],[]))

    def test_note_shadow_rejoins_chinese_word_but_width_alone_is_insufficient(self):
        aa=[atom('narrow',[.10,.779,.48,.798],'回忆一下，我们将两个元素之和定'),
            atom('full',[.06,.801,.93,.844],'义为由对应的坐标相加得到的元素。')]
        index=SourceIndex(dict(atoms=aa));before=deepcopy(index.source)
        note=dict(page=1,note_bbox=[.50,.677,.925,.791])
        joins,rows=wrapped_prose(index,[],side_notes=[note])
        self.assertEqual(rows,[]);self.assertEqual([(j['left'],j['right']) for j in joins],[('narrow','full')])
        self.assertEqual(wrapped_prose(index,[]),([],[]))
        self.assertEqual(wrapped_prose(index,[],changed={'full'},side_notes=[note]),([],[]))
        self.assertEqual(index.source,before)

    def test_compiled_figure_wrap_has_one_paragraph_no_overlap_and_owned_destinations(self):
        for left in (False,True):
            with self.subTest(left=left), tempfile.TemporaryDirectory() as name:
                root=Path(name);index,registry=self.figure(root,left);before=deepcopy(index.source)
                directory=root/'latex'
                audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                    registry,[],dict(title='Source wrapping',author=''),load_config(),directory,sample_mode=True)
                self.assertEqual(index.source,before);self.assertTrue(audit['coverage_complete'])
                result=compile_project(directory,{t:executable(t) for t in ('xelatex','latexmk')})
                self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
                with pymupdf.open(directory/'main.pdf') as pdf:
                    self.assertEqual(len(pdf),1);page=pdf[0];names=pdf.resolve_names()
                    for a in index.atoms:self.assertIn('source-'+a['id'],names)
                    self.assertIn('figure-target',names)
                    self.assertEqual(names['source-picture']['page'],names['figure-target']['page'])
                    image=pymupdf.Rect(page.get_image_info()[0]['bbox'])
                    self.assertLess(image.width,page.rect.width*.5)
                    lines=[l for b in page.get_text('dict')['blocks'] for l in b.get('lines',[])]
                    prose=[l for l in lines if any('arrow' in s['text'] or 'thinking' in s['text'] for s in l['spans'])]
                    self.assertTrue(any(l['bbox'][1]<image.y1 for l in prose))
                    self.assertTrue(any(l['bbox'][2]-l['bbox'][0]>page.rect.width*.6
                                        and l['bbox'][1]>image.y1 for l in prose))
                    for word in page.get_text('words'):
                        if word[4] in {'arrow','parallel','thinking','vector'}:
                            self.assertTrue((pymupdf.Rect(word[:4]) & image).is_empty)
                    text=page.get_text(flags=pymupdf.TEXT_DEHYPHENATE)
                    body=' '.join(re.sub(r'(?<=[a-z])-\s+(?=[a-z])','',text).split())
                    self.assertIn('construction and just thinking',body)
                    self.assertEqual(len(re.findall(r'just thinking of the vector',body)),6)
                    self.assertGreater(page.search_for('Following paragraph')[0].y0,image.y1)
                    opening=page.search_for('We move')[0]
                    to=pymupdf.Point(names['source-opening']['to'])*page.transformation_matrix
                    self.assertLess(abs(to.x-opening.x0),2)
