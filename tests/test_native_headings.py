from copy import deepcopy
import unittest

from book_pipeline.ingestion.native_headings import split_run_in_headings


class NativeHeadingBoundaryTests(unittest.TestCase):
    def fixture(self):
        lookup = dict(a=dict(text='Two models:', font='Arial-Bold', bbox=[.1,.2,.3,.22]),
                      b=dict(text=' The model ', font='Arial', bbox=[.31,.2,.5,.22]),
                      c=dict(text='y', font='MathItalic', bbox=[.5,.2,.51,.22]),
                      d=dict(text=' predicts a measurement.', font='Arial', bbox=[.51,.2,.85,.22]))
        row = dict(type='paragraph_title',bbox=[.1,.2,.85,.22],parts=[
            dict(kind='text',spans=['a','b']),dict(kind='math',spans=['c'],latex='y'),
            dict(kind='text',spans=['d'])])
        return [row], lookup

    def test_splits_only_bold_title_and_retains_all_spans_and_math(self):
        rows,lookup=self.fixture();before=deepcopy(rows)
        result=split_run_in_headings(rows,lookup)
        self.assertEqual(rows,before)
        self.assertEqual([r['type'] for r in result],['paragraph_title','text'])
        self.assertEqual([s for r in result for p in r['parts'] for s in p['spans']],list('abcd'))
        self.assertEqual(result[1]['parts'][1]['latex'],'y')
        self.assertEqual(result[0]['bbox'],lookup['a']['bbox'])
        self.assertEqual(result[1]['bbox'],[.31,.2,.85,.22])

    def test_ambiguous_or_numbered_or_non_title_rows_keep_ownership(self):
        for defect in ('no_colon','no_bold','all_bold','numbered','not_title','atomic_math'):
            with self.subTest(defect=defect):
                rows,lookup=self.fixture()
                if defect=='no_colon':lookup['a']['text']='Two models'
                if defect=='no_bold':lookup['a']['font']='Arial'
                if defect=='all_bold':
                    for s in lookup.values():s['font']='Arial-Bold'
                if defect=='numbered':lookup['a']['text']='3.2 Two models:'
                if defect=='not_title':rows[0]['type']='text'
                if defect=='atomic_math':rows[0]['parts'][0].update(kind='math',latex='x')
                self.assertEqual(split_run_in_headings(rows,lookup),rows)

    def test_split_word_reaches_existing_hyphen_review_with_its_continuation(self):
        from book_pipeline.ingestion.native_hyphens import candidates_for
        for defect in ('none','different_column','too_far','uppercase'):
            with self.subTest(defect=defect):
                rows,lookup=self.fixture()
                lookup['d']['text']=' describes a trans-'
                for sid,s in lookup.items():s.update(line=0,id=sid)
                lookup['e']=dict(id='e',text='formation in this model.',font='Arial',line=1,
                                 bbox=[.1,.22,.85,.24])
                if defect=='different_column':lookup['e']['bbox'][0]=.2
                if defect=='too_far':lookup['e']['bbox']=[.1,.4,.85,.42]
                if defect=='uppercase':lookup['e']['text']='Formation starts a new paragraph.'
                rows.append(dict(type='text',bbox=lookup['e']['bbox'],parts=[dict(kind='text',spans=['e'])]))
                result=split_run_in_headings(rows,lookup)
                self.assertEqual(len(result),2 if defect=='none' else 3)
                candidates=candidates_for(result,lookup)
                self.assertEqual(len(candidates),int(defect=='none'))
                if candidates:self.assertEqual(candidates[0]['fragments'],['trans','formation'])
