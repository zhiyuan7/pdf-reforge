from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import Renderer
from book_pipeline.publishing.reviewed_callouts import missing_reference

class ReviewedCalloutTests(unittest.TestCase):
    def source(self):
        return SourceIndex(dict(source_page_count=1,atoms=[dict(id='a0',group='g0',page=0,kind='text',role='text',
            block_type='text',content='End. Following prose continues.',length=len('End. Following prose continues.'),bbox=[.1,.1,.9,.2])]))

    def test_missing_reference_requires_exact_context_evidence_page_and_bounds(self):
        index=self.source();note=dict(page=1,status='unmatched',reason='reference_missing',marker=dict(text='1'))
        row=dict(atom='a0',offset=4,before='End.',after=' Following prose',bbox=[.2,.12,.22,.14])
        result=missing_reference(index,note,row,'review.png')
        self.assertEqual(result['lo'],result['hi'])
        self.assertTrue(result['reviewed_missing_callout'])
        for key,value in [('offset',3),('offset',True),('before','Wrong'),('after','wrong'),('bbox',[0,0,1,1])]:
            bad={**row,key:value}
            with self.assertRaises(ValueError):missing_reference(index,note,bad,'review.png')
        for bad in ({**note,'page':2},{**note,'status':'matched'}):
            with self.assertRaises(ValueError):missing_reference(index,bad,row,'review.png')
        with self.assertRaises(ValueError):missing_reference(index,note,row,'')

    def test_rendered_insertion_preserves_source_bytes_and_is_emitted_once_at_boundary(self):
        index=self.source();atom=index.atoms[0]
        with tempfile.TemporaryDirectory() as name:
            r=Renderer(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),[],[],Path(name))
            note=dict(id='note-1',reference_anchor='ref-note-1',reference=dict(reviewed_missing_callout=True))
            r.add_footnote_marker(4,4,note,True)
            whole=r.text_fragment(atom,0,len(atom['content']))
            split=r.text_fragment(atom,0,4)+r.text_fragment(atom,4,len(atom['content']))
            self.assertEqual(split,whole)
            self.assertEqual(whole,r'End.\BookFootnoteReference{note-1} Following prose continues.')
            self.assertEqual(atom['content'],'End. Following prose continues.')
            with self.assertRaises(ValueError):r.add_footnote_marker(4,4,note,True)
            with self.assertRaises(ValueError):r.add_footnote_marker(0,0,note,True)
