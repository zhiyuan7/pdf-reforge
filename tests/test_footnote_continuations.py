from copy import deepcopy
import unittest

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.processed_book import merge_footnote_continuations


class FootnoteContinuationTests(unittest.TestCase):
    def test_only_adjacent_fragments_within_the_parent_note_are_merged(self):
        atoms=[]
        for i,(text,box) in enumerate([('1 note',[.1,.8,.9,.9]),('x',[.5,.81,.7,.86]),
                                     ('sidebar',[.1,.4,.3,.45])]):
            atoms.append(dict(id=f'a{i}',group=f'b{i}',kind='text',role='text',block_type='page_footnote',
                              content=text,length=len(text),page=0,bbox=box))
        source=SourceIndex(dict(source_pdf='test.pdf',input_sha256='x',mineru_sha256='y',atoms=atoms))
        notes=[]
        for i,atom in enumerate(atoms):
            lo,hi=source.prefix[i:i+2]
            notes.append(dict(id=f'n{i}',source_lo=lo,source_hi=hi,lo=lo,hi=hi,page=1,
                              marker={'text':'1'} if i==0 else None,text=atom['content'],end={'atom':atom['id']}))
        before=deepcopy(source.source)
        merge_footnote_continuations(source,notes)
        self.assertEqual([n['id'] for n in notes],['n0','n2'])
        self.assertEqual(notes[0]['text'],'1 notex')
        self.assertEqual(notes[0]['continuation_note_ids'],['n1'])
        self.assertEqual(source.source,before)


if __name__=='__main__':unittest.main()
