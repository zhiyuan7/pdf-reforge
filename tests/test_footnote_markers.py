import unittest
from pathlib import Path
import tempfile
import pymupdf
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.structure.section_review import split_source
from book_pipeline.structure.footnotes import extract_marker,marker_occurrences,classify_footnotes

class FootnoteMarkerTests(unittest.TestCase):
    def test_native_joined_numeric_note_label_and_exact_raised_callout(self):
        for defect in ('none','flat_callout','flat_note_label','scan'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as name:
                root=Path(name)
                with pymupdf.open() as pdf:
                    page=pdf.new_page(width=500,height=700)
                    page.insert_text((50,100),'A sentence.',fontname='tiro',fontsize=11)
                    x=50+pymupdf.get_text_length('A sentence.',fontname='tiro',fontsize=11)
                    page.insert_text((x,100 if defect=='flat_callout' else 96),'4',fontname='tiro',fontsize=7)
                    page.insert_text((x+4,100),' A control 4 remains normal.',fontname='tiro',fontsize=11)
                    page.insert_text((50,600 if defect=='flat_note_label' else 596),'4',fontname='tiro',
                                     fontsize=9 if defect=='flat_note_label' else 6)
                    page.insert_text((55,600),'5-choosability example.',fontname='tiro',fontsize=9)
                    raw=page.get_text('dict')['blocks']
                    spans=[s for b in raw for l in b.get('lines',[]) for s in l['spans']]
                    body=[s for s in spans if s['bbox'][1]<200];note=[s for s in spans if s['bbox'][1]>500]
                    def box(ss):
                        return [min(s['bbox'][0] for s in ss)/500,min(s['bbox'][1] for s in ss)/700,
                                max(s['bbox'][2] for s in ss)/500,max(s['bbox'][3] for s in ss)/700]
                    text=''.join(s['text'] for s in body)
                    atoms=[dict(id='body',group='body',page=0,kind='text',role='text',block_type='text',
                        content=text,length=len(text),bbox=box(body),extraction_method='native')]
                    for k,t in enumerate(('4','5-choosability example.')):
                        atoms.append(dict(id=f'n{k}',group='note',page=0,kind='text',role='page_footnote',
                            block_type='page_footnote',content=t,length=len(t),bbox=box(note),
                            extraction_method='ocr' if defect=='scan' else 'native'))
                    pdf.save(root/'source.pdf')
                index=SourceIndex(dict(source_pdf=str(root/'source.pdf'),source_page_count=1,
                    input_sha256='pdf',mineru_sha256='native',atoms=atoms))
                result=classify_footnotes(index,split_source(index,dict(headings=[])))['footnotes']['items'][0]
                self.assertEqual(result['status'],'matched' if defect=='none' else 'unmatched',defect)
                if defect=='none':
                    self.assertEqual(result['marker']['text'],'4')
                    self.assertEqual(len(result['candidates']),1)
                    self.assertEqual(result['reference']['start_offset'],text.index('4'))

    def test_prose_numerals_do_not_masquerade_as_missing_callouts(self):
        marker=extract_marker('^{1} A source note.')
        for text in ('Axiom 1: positive','Step 1 and 2','There is 1 event.','Definition 1.2'):
            self.assertEqual(list(marker_occurrences(text,marker,'text')),[],text)
        for text in ('After a sentence.¹ Next','After a sentence.^{1} Next',r'After.\textsuperscript{1} Next'):
            self.assertEqual(len(list(marker_occurrences(text,marker,'text'))),1,text)
        self.assertEqual(list(marker_occurrences('x^{1}',marker,'math')),[])
        self.assertEqual(len(list(marker_occurrences('^{1}',marker,'math'))),1)
        self.assertEqual(len(list(marker_occurrences('After① next',extract_marker('① note'),'text'))),1)
