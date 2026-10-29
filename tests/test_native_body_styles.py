"""Native fonts require complete ownership, including repeated prose near math."""
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.native_body_styles import font_style, group_ranges, recover_native_body_styles, bounded_line_ranges, native_minor_headings, native_math_evidence


class NativeBodyStylesTests(unittest.TestCase):
    def test_chinese_font_ownership_preserves_kai_bold_and_original_punctuation(self):
        atoms=[dict(id='t',kind='text',content='定理 3.1 考虑函数 '),
               dict(id='m',kind='math',content='f(x)'),
               dict(id='z',kind='text',content='，其最优值存在. 若步长固定。')]
        spans=[dict(text='定理3.1',font='FandolSong-Bold'),
               dict(text='考虑函数',font='FandolKai-Regular'),
               dict(text='f(x)，',font='Math-Regular'),
               dict(text='其最优值存在．若步长固定．',font='FandolKai-Regular')]
        ranges=group_ranges(atoms,spans)
        self.assertEqual(ranges['t'][0]['quote'],'定理 3.1')
        self.assertTrue(ranges['t'][0]['bold'])
        self.assertEqual([r['quote'] for r in ranges['z']],['其最优值存在','若步长固定'])
        for r in ranges['z']:
            self.assertEqual(r['cjk_family'],'calligraphic')
            self.assertFalse(r['italic'])
            self.assertFalse(r['bold'])
        self.assertNotIn('m',ranges)
        self.assertEqual(atoms[-1]['content'],'，其最优值存在. 若步长固定。')
        self.assertEqual(group_ranges(atoms,spans[:-1]),{})
        atoms[-1]['content']='，其最小值存在. 若步长固定。'
        self.assertEqual(group_ranges(atoms,spans),{})

    def test_wide_chinese_formula_band_requires_a_unique_complete_line(self):
        atoms=[dict(id='a',kind='text',content='其最优值点为 '),
               dict(id='m',kind='math',content='x^*'),
               dict(id='z',kind='text',content='. 若步长固定')]
        line=[dict(text='其最优值点为',font='FandolKai-Regular',line=2),
              dict(text='x∗',font='Math-Regular',line=2),
              dict(text='．若步长固定',font='FandolKai-Regular',line=2)]
        spans=[dict(text='f(x)=x²',font='Math-Regular',line=1),*line,
               dict(text='α=1',font='Math-Regular',line=3)]
        ranges=bounded_line_ranges(atoms,spans)
        self.assertEqual(ranges['a'][0]['quote'],'其最优值点为')
        repeated=spans+[{**s,'line':4} for s in line]
        self.assertEqual(bounded_line_ranges(atoms,repeated),{})
        self.assertEqual(bounded_line_ranges(atoms,spans[:-2]),{})

    def test_type1_names_distinguish_prose_from_math(self):
        for font,expected in [('ABCDEF+CMTI10',(True,False)),('CMBX10',(False,True)),
                ('CMBXTI10',(True,True)),('CMSSBX10',(False,True)),('CMMI10',(False,False)),
                ('CMR10',(False,False)),('Times-BoldItalic',(True,True)),
                ('NimbusRomNo9L-ReguItal',(True,False)),('NimbusRomNo9L-Medi',(False,True)),
                ('NimbusRomNo9L-MediItal',(True,True)),('Other-Medi',(False,False))]:
            self.assertEqual(font_style(font),expected,font)

    def test_complete_mixed_group_preserves_repeated_words_and_multiline_phrase(self):
        texts = [('text','The optimal set is '),('math','X'),('text',', not an optimal\npoint.')]
        atoms = [dict(id=str(n),kind=k,content=t) for n,(k,t) in enumerate(texts)]
        spans = [dict(text=t,font=f) for t,f in [('The ','CMR10'),('optimal set','CMTI10'),
            (' is X, not an ','CMR10'),('optimal point','CMTI10'),('.','CMR10')]]
        ranges = group_ranges(atoms,spans)
        self.assertEqual([r['quote'] for r in ranges['0']],['optimal set'])
        self.assertEqual([r['quote'] for r in ranges['2']],['optimal\npoint'])
        self.assertNotIn('1',ranges)
        # Neither partial matching nor rewritten text may supply font evidence.
        self.assertEqual(group_ranges(atoms,spans[:-1]),{})
        atoms[0]['content']='The feasible set is '
        self.assertEqual(group_ranges(atoms,spans),{})

    def test_two_possible_math_owners_abstain(self):
        atoms=[dict(id=str(n),kind=k,content=t) for n,(k,t) in enumerate(
            [('math','x'),('text','same'),('math','y')])]
        self.assertEqual(group_ranges(atoms,[dict(text='xsamesamey',font='CMTI10')]),{})

    def test_live_prose_quotes_and_variable_weight_require_unique_complete_ownership(self):
        atoms=[dict(id=str(n),kind=k,content=t) for n,(k,t) in enumerate([
            ('text','A grade '),('math',r'{\bf A}^{\flat}'),('text',' means '),
            ('math',r'\boldsymbol { Q }'),('text',' is true.')])]
        spans=[dict(text='A grade “A” means ',font='TeXGyreSchola-Regular'),
               dict(text='Q',font='CenturySchL-Ital'),
               dict(text=' is true.',font='TeXGyreSchola-Regular')]
        literals,weights=native_math_evidence(atoms,spans)
        self.assertEqual(literals['1']['text'],'“A”')
        self.assertEqual(weights['3']['latex'],'Q')
        for wrong in [spans[:-1],spans+[dict(text='more',font='CMR10')]]:
            self.assertEqual(native_math_evidence(atoms,wrong),({},{}))
        self.assertEqual(native_math_evidence(atoms,[{**s,'font':'Math-Regular'} for s in spans])[0],{})
        # Separate glyph runs may be scripts; never flatten their quotation.
        split=[dict(text='A grade “',font='CMR10'),dict(text='A',font='CMR10'),
               dict(text='” means ',font='CMR10'),*spans[1:]]
        self.assertEqual(native_math_evidence(atoms,split)[0],{})
        self.assertEqual(native_math_evidence(atoms,[spans[0],{**spans[1],'font':'CenturySchL-BoldItal'},spans[2]])[1],{})

    def test_flat_native_math_restores_only_observed_punctuation_and_roman_words(self):
        atoms=[dict(id='a',kind='text',content='Use '),dict(id='m',kind='math',
               content=r'\Rightarrow \mathbf { a n d } \Leftrightarrow'),
               dict(id='z',kind='text',content=' here.')]
        spans=[dict(text='Use ⇒',font='Math-Regular',line=1),
               dict(text='and',font='TeXGyreSchola-Regular',line=1),
               dict(text='⇔ here.',font='Math-Regular',line=1)]
        result=native_math_evidence(atoms,spans)[1]['m']['latex']
        self.assertEqual(result,r'\Rightarrow \text{ and } \Leftrightarrow')
        for raw,live,expected in [('y ;','y:','y:'),('x=0,','x=0','x=0'),
                                  (r'(P\lor Q)\land R', '(P∨Q)∧R.',r'(P\lor Q)\land R.'),
                                  ('x=0 or y=0','x=0ory=0',r'x=0 \text{ or } y=0')]:
            atoms[1]['content']=raw
            source=[dict(text='Use '+live+' here.',font='TeXGyreSchola-Regular',line=1)]
            if ' or ' in raw:
                source=[dict(text='Use x=0',font='Math-Regular',line=1),
                        dict(text='or',font='TeXGyreSchola-Regular',line=1),
                        dict(text='y=0 here.',font='Math-Regular',line=1)]
            self.assertEqual(native_math_evidence(atoms,source)[1]['m']['latex'],expected)
        for raw,live in [(r'x_2','x2'),(r'\frac{x}{y}','xy'),('x=y','x=z'),
                          (r'\alpha','α')]:
            atoms[1]['content']=raw
            self.assertEqual(native_math_evidence(atoms,[dict(text='Use '+live+' here.',font='CMR10')])[1],{})
        atoms[1]['content']='xy'
        source=[dict(text='Use x',font='CMR10',line=1,size=10,bbox=[.1,.1,.2,.12]),
                dict(text='y',font='CMR10',line=1,size=7,bbox=[.2,.09,.21,.104]),
                dict(text='. here.',font='CMR10',line=1,size=10,bbox=[.21,.1,.3,.12])]
        self.assertEqual(native_math_evidence(atoms,source)[1],{})
        source[1]['size']=10  # A same-size glyph raised off the baseline also abstains.
        self.assertEqual(native_math_evidence(atoms,source)[1],{})
        for raw in [r'\mathrm{y.}',r'\mathrm{y,}']:
            atoms[1]['content']=raw
            self.assertEqual(native_math_evidence(atoms,[dict(text='Use y. here.',font='Math-Regular')])[1],{})
        atoms[1]['content']=r'x=0\text{ or }y=0'
        source=[dict(text='Use x=0',font='Math-Regular'),dict(text='or',font='CMR10'),
                dict(text='y=0 here.',font='Math-Regular')]
        self.assertEqual(native_math_evidence(atoms,source)[1],{})

    def test_formula_band_style_requires_unique_complete_line_window(self):
        text='The following rows are cyclic shifts.'
        atoms=[dict(id='a',kind='text',content=text)]
        spans=[dict(text='H = [1 0 1]',font='CMR10',line=0),
               dict(text='The following rows are ',font='CMR10',line=1),
               dict(text='cyclic shifts.',font='CMTI10',line=1),
               dict(text='The next paragraph.',font='CMR10',line=2)]
        self.assertEqual(bounded_line_ranges(atoms,spans)['a'][0]['quote'],'cyclic shifts.')
        # A duplicate plain-font match must veto the emphasized match too.
        self.assertEqual(bounded_line_ranges(atoms,spans+[dict(text=text,font='CMR10',line=3)]),{})
        self.assertEqual(bounded_line_ranges(atoms,spans[:3]+[dict(text=' extra',font='CMR10',line=1)]+spans[3:]),{})

    def test_ligature_equivalence_retains_literal_byte_offsets(self):
        for source,live in [('first ﬂoor','ﬁrst floor'),('ﬁrst floor','first ﬂoor')]:
            atoms=[dict(id='a',kind='text',content=source)]
            ranges=group_ranges(atoms,[dict(text=live,font='CMTI10')])
            self.assertEqual(ranges['a'][0]['quote'],source)
            self.assertEqual(ranges['a'][0]['end'],len(source))

    def test_live_terminal_period_splits_a_cached_cross_page_merge(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'source.pdf'
            with pymupdf.open() as pdf:
                for text in ['End with x.','A new paragraph.']:
                    page=pdf.new_page(width=500,height=700)
                    page.insert_text((50,100),text,fontname='tiro',fontsize=10)
                boxes=[p.get_text('dict')['blocks'][0]['bbox'] for p in pdf]
                pdf.save(path)
            atoms=[]
            for n,(kind,text,page) in enumerate([('text','End with ',0),('math','x',0),
                                                ('text','A new paragraph.',1)]):
                box=[v/(500 if k%2==0 else 700) for k,v in enumerate(boxes[page])]
                atoms.append(dict(id=str(n),group='merged',kind=kind,content=text,length=len(text),
                    page=page,bbox=box,role='text',block_type='text',extraction_method='ocr',
                    original_atom=str(n),original_start_offset=0,original_end_offset=len(text)))
            index=SourceIndex(dict(source_pdf=str(path),atoms=atoms))
            _,report=recover_native_body_styles(index,index)
            self.assertEqual(report['math_styles']['1']['latex'],'x.')
            self.assertIn('2',report['paragraph_starts'])
            self.assertEqual(report['status'],'requires_visual_review')

    def test_minor_heading_requires_complete_native_body_size_italic_evidence(self):
        for defect in ('none','slightly_larger','large','bold','numbered','ocr','wrong_text','chapter'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as name:
                path=Path(name)/'source.pdf'
                title='3.2 Graph construction' if defect=='numbered' else 'Graph construction'
                with pymupdf.open() as pdf:
                    page=pdf.new_page(width=500,height=700)
                    page.insert_text((50,150),title,fontname='tibi' if defect=='bold' else 'tiit',fontsize=15 if defect=='large' else 11 if defect=='slightly_larger' else 10)
                    page.insert_text((50,180),'An ordinary paragraph explains this construction precisely.',fontname='tiro',fontsize=10)
                    blocks=page.get_text('dict')['blocks'];pdf.save(path)
                atoms=[]
                for n,(text,role) in enumerate([(title,'paragraph_title'),('An ordinary paragraph explains this construction precisely.','text')]):
                    box=[v/(500 if k%2==0 else 700) for k,v in enumerate(blocks[n]['bbox'])]
                    atoms.append(dict(id=str(n),group=str(n),kind='text',role=role,block_type=role,
                        page=0,content=text,length=len(text),bbox=box,extraction_method='native'))
                if defect=='ocr':atoms[0]['extraction_method']='ocr'
                if defect=='wrong_text':atoms[0]['content']='Different heading'
                index=SourceIndex(dict(source_pdf=str(path),atoms=atoms))
                registry=[dict(kind='chapter' if defect=='chapter' else 'section',anchor='h',lo=0,hi=index.prefix[1])]
                self.assertEqual(bool(native_minor_headings(index,registry)),defect in {'none','slightly_larger'})

    def test_live_pdf_requires_native_unchanged_group_and_unique_geometry(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'source.pdf'
            with pymupdf.open() as pdf:
                page=pdf.new_page(width=500,height=700)
                page.insert_text((50,100),'The ',fontname='tiro')
                page.insert_text((70,100),'optimal set',fontname='tiit')
                page.insert_text((130,100),' is closed.',fontname='tiro')
                box=page.get_text('dict')['blocks'][0]['bbox']
                text='The optimal set is closed.'
                pdf.save(path)
            box=[v/(500 if n%2==0 else 700) for n,v in enumerate(box)]
            a=dict(id='a',group='g',kind='text',content=text,length=len(text),page=0,
                extraction_method='native',block_type='text',role='text',bbox=box,
                original_atom='a',original_start_offset=0,original_end_offset=len(text))
            original=SourceIndex(dict(source_pdf=str(path),atoms=[a]))
            styles,_=recover_native_body_styles(original,original)
            self.assertEqual([r['quote'] for r in styles['a']],['optimal set'])
            punctuation=dict(atom='a',start_offset=len(text)-1,end_offset=len(text),
                             expected='.',replacement='!')
            self.assertEqual(recover_native_body_styles(original,original,corrections=[punctuation])[0],styles)
            changed_word=dict(atom='a',start_offset=4,end_offset=11,expected='optimal',replacement='feasible')
            self.assertEqual(recover_native_body_styles(original,original,corrections=[changed_word])[0],{})
            caption=SourceIndex(dict(source_pdf=str(path),atoms=[{**a,'role':'image_caption','block_type':'image_caption'}]))
            self.assertEqual(recover_native_body_styles(caption,caption)[0],styles)
            self.assertEqual(recover_native_body_styles(original,original,{'a'})[0],{})
            edited=SourceIndex(dict(source_pdf=str(path),atoms=[{**a,'content':'An '+text[4:]}]))
            self.assertEqual(recover_native_body_styles(edited,original)[0],{})
            scanned=SourceIndex(dict(source_pdf=str(path),atoms=[{**a,'extraction_method':'ocr'}]))
            self.assertEqual(recover_native_body_styles(scanned,scanned)[0],styles)
            # The same exact hidden text behind a page scan is not evidence.
            with pymupdf.open(path) as pdf:
                pix=pdf[0].get_pixmap()
                pdf[0].insert_image(pdf[0].rect,stream=pix.tobytes('png'))
                pdf.save(Path(name)/'scan.pdf')
            hidden=SourceIndex(dict(source_pdf=str(Path(name)/'scan.pdf'),atoms=scanned.atoms))
            self.assertEqual(recover_native_body_styles(hidden,hidden)[0],{})


if __name__=='__main__':
    unittest.main()
