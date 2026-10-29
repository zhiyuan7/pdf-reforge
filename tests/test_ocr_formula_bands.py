"""Proof reasons omitted by OCR need full-width review even without text hits."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import pymupdf
from PIL import Image

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.ingestion.formula_bands import (
    detect_bands, band_image, band_details, apply_formula_corrections, normalized_inline_text, validate_answer, METHOD)


class OCRFormulaBandTests(unittest.TestCase):
    def test_numbered_multiline_continuation_cannot_be_split_off(self):
        text=r'\begin{array}{c}x=y+z\\+w,\quad n\geq0\end{array}\tag{7.3}'
        atom=dict(id='eq',group='eq',page=0,kind='display_math',role='equation',block_type='equation',
            bbox=[.2,.2,.8,.26],content=text,length=len(text),extraction_method='ocr')
        band=detect_bands(SourceIndex(dict(atoms=[atom])))[0]
        self.assertEqual(band['multiline_equation_numbers'],['7.3'])
        bad=dict(items=[dict(kind='formula',latex='x=y+z',number='7.3'),
                        dict(kind='formula',latex=r'+w,\quad n\geq0',number='')])
        with self.assertRaisesRegex(ValueError,'continuation'):validate_answer(bad,band)
        validate_answer(dict(items=[dict(kind='formula',
            latex=r'\begin{aligned}x&=y+z\\&\quad+w,\quad n\geq0\end{aligned}',number='7.3')]),band)
        validate_answer(bad,{**band,'multiline_equation_numbers':[]})

    def test_reread_paragraph_owner_is_complete_unique_and_allows_inline_resegmentation(self):
        from book_pipeline.ingestion.formula_bands import unique_prose_owner
        prose='Consider the numerical approximation with stepsize '
        tail=' and compare its convergence with the reference solution at t.'
        group=dict(group='g',type='text',bbox=[.1,.3,.8,.4],atoms=[
            dict(kind='text',content=prose),dict(kind='math',content='h=.1'),dict(kind='text',content=tail)])
        parts=[('text',prose),('math','h=0.1'),('text',tail[:-2]),('math','t'),('text','.')]
        self.assertIs(unique_prose_owner(parts,[group]),group)
        self.assertIsNone(unique_prose_owner(parts,[group,{**group,'group':'duplicate'}]))
        self.assertIsNone(unique_prose_owner(parts[:-3],[group]))
        self.assertIsNone(unique_prose_owner([('text','A short label')],[group]))

    def test_native_annotation_scope_and_live_colors_require_independent_review(self):
        with tempfile.TemporaryDirectory() as name,pymupdf.open() as pdf:
            path=Path(name)/'source.pdf';page=pdf.new_page(width=400,height=400)
            page.insert_text((100,100),'x + y',color=(1,.3,0))
            box=page.get_text('dict')['blocks'][0]['bbox'];pdf.save(path)
            atom=dict(id='c',group='c',page=0,kind='display_math',block_type='equation',
                      content='x+y',length=3,bbox=[v/400 for v in box],extraction_method='native')
            source=SourceIndex(dict(source_pdf=str(path),atoms=[atom]))
            bands=detect_bands(source)
            self.assertEqual(bands[0]['native_features'],{'c':['colored_math']})
            validate_answer(dict(items=[dict(kind='formula',latex=r'\textcolor[rgb]{1,0.3,0}{x+y}',number='')]),bands[0])
        atom.update(content=r'\underbrace{x+y}_{L}',bbox=[.2,.2,.8,.3])
        bands=detect_bands(SourceIndex(dict(atoms=[atom])))
        self.assertEqual(bands[0]['native_features'],{'c':['brace_scope']})

    def test_tiny_native_descender_overlap_does_not_erase_following_side_note_geometry(self):
        atoms=[dict(id='f',group='f',kind='display_math',block_type='equation',page=0,
                    bbox=[.3,.2,.8,.2305],content='x=y',length=3,extraction_method='native'),
               dict(id='p',group='p',kind='text',block_type='text',page=0,
                    bbox=[.2,.23,.8,.30],content='following prose',length=15,extraction_method='native'),
               dict(id='n',group='n',kind='text',block_type='text',page=0,
                    bbox=[.05,.235,.15,.27],content='margin note',length=11,extraction_method='native')]
        self.assertEqual(detect_bands(SourceIndex(dict(atoms=atoms))),[])
        # The same uncertain geometry from OCR still requires visual reading.
        for a in atoms:a['extraction_method']='ocr'
        self.assertEqual(len(detect_bands(SourceIndex(dict(atoms=atoms)))),1)

    def test_plain_scan_prose_uses_two_blind_readings_and_caches_the_consensus_policy(self):
        from copy import deepcopy
        from book_pipeline.config import read
        from book_pipeline.ingestion.formula_bands import correct_formula_bands
        text = 'This printed program has a long history in the study of learning.'
        answer = dict(items=[dict(kind='text',text=text),dict(kind='formula',latex='a/h=1',number='9')])
        witness = deepcopy(answer)
        witness['items'][0]['text'] = text.replace('program','programm')
        calls=[]
        class Client:
            _api=None
            def __init__(self,*args,**kwargs):self.ledger=self
            def summary(self):return {}
            def ask(self,task,prompt,payload,**kwargs):
                calls.append((task,payload))
                if task.startswith('literal_scan_band'):
                    output=dict(edits=[],unresolved=[])
                elif task.startswith('independent_scan_prose') or '不提供任何已有转录' in prompt or '后续图片是同一横带' in prompt:
                    output=deepcopy(witness)
                    self_outer.assertNotIn('items',payload)
                else:output=deepcopy(answer)
                kwargs['validator'](output)
                return output
        self_outer=self
        with tempfile.TemporaryDirectory() as name,pymupdf.open() as pdf:
            path=Path(name)/'source.pdf'
            pdf.new_page(width=400,height=300).insert_text((40,100),text)
            pdf.save(path)
            atom=dict(id='a',group='a',page=0,kind='display_math',content='a/h=1',length=5,extraction_method='ocr')
            index=SourceIndex(dict(atoms=[atom],input_sha256='p',mineru_sha256='m',source_pdf=str(path)))
            band=dict(id='b',page=0,bbox=[0,.2,1,.5],groups=['a'],preserve=[])
            config=dict(model='fake',base_url='https://example.invalid',thinking='disabled',concurrency=1)
            module='book_pipeline.ingestion.formula_bands.'
            with patch(module+'detect_bands',side_effect=lambda *args:[dict(band)]),patch(module+'DeepSeekClient',Client):
                directory=Path(name)/'corrections'
                result=correct_formula_bands(index,config,directory)
                self.assertEqual(len(calls),4)
                record=result['records'][0]
                self.assertEqual(record['answer'],witness)
                self.assertEqual(record['answer']['items'][1],answer['items'][1])
                self.assertEqual(len(record['prose_consensus']['accepted']),1)
                self.assertFalse(record['applied_edits'])
                correct_formula_bands(index,config,directory)
                self.assertEqual(len(calls),4)
                with patch('book_pipeline.ingestion.scan_prose_consensus.METHOD','new-prose-policy'):
                    correct_formula_bands(index,config,directory)
                self.assertEqual(len(calls),8)
                self.assertEqual(read(directory/'regions/b.json')['answer'],witness)

    def test_local_edits_need_complete_independent_item_agreement(self):
        from book_pipeline.ingestion.formula_bands import agreed_detail_edits, comparable_item
        band = dict(preserve=[])
        answer = dict(items=[dict(kind='formula', latex=r'h/h=0,\ a/bh=0 \text{by (5).}', number=''),
                             dict(kind='formula', latex=r'a/bh=1 \text{by (1.4).}', number='')])
        good = dict(item=0, field='latex', before='h/h=0', after='b/h=0', reason='read glyph')
        bad = dict(item=1, field='latex', before='a/bh=1', after='a/bb=1', reason='wrong suggestion')
        review = dict(edits=[good, bad], unresolved=[])
        witness = dict(items=[dict(kind='formula', latex=r'b/h=0,\quad a/bh=0\qquad\text{by (5).}', number=''),
                              dict(kind='formula', latex=answer['items'][1]['latex'], number='')])
        corrected, accepted, rejected = agreed_detail_edits(answer, review, witness, band)
        self.assertEqual(accepted, [good]); self.assertEqual(rejected, [bad])
        self.assertEqual(corrected['items'][1], answer['items'][1])
        self.assertEqual(corrected['items'][0]['number'], '')
        self.assertEqual(comparable_item(r'\bar{a} / h'), comparable_item(r'\bar a/h'))
        for unrelated in (r'b/h=0,\ a/bh=0\text{by (6).}', r'b/h=0'):
            witness['items'][0]['latex'] = unrelated
            self.assertEqual(agreed_detail_edits(answer, review, witness, band)[1], [])
        witness['items'] = [dict(kind='formula', latex=r'b/h=0,\ a/bh=0 \text{by (5).}', number='')]*2
        self.assertEqual(agreed_detail_edits(answer, review, witness, band)[1], [])

    def test_detail_edits_preserve_items_and_reject_ambiguous_or_structural_changes(self):
        from copy import deepcopy
        from book_pipeline.ingestion.formula_bands import apply_detail_edits
        answer = dict(items=[dict(kind='formula', latex=r'h/h=0 \text{by (11)}', number='7'),
                             dict(kind='text', text='old words words')])
        original = deepcopy(answer)
        edit = dict(item=0, field='latex', before='h/h', after='b/h', reason='source glyph')
        review = dict(edits=[edit], unresolved=['left label ownership'])
        output = apply_detail_edits(answer, review, dict(preserve=[]))
        self.assertEqual(answer, original)
        self.assertEqual(output['items'][0], {**answer['items'][0], 'latex':r'b/h=0 \text{by (11)}'})
        self.assertEqual(output['items'][1], answer['items'][1])
        for invalid in ([dict(edit, item=True)], [dict(edit, item=2)],
                        [dict(edit, field='number', before='7', after='8')],
                        [dict(edit, item=1, field='text', before='words', after='word')],
                        [dict(edit, before='missing')], [dict(edit, before='')],
                        [dict(edit, after='x'*65)], [edit, edit],
                        [dict(edit, after='$unbalanced')]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                apply_detail_edits(answer, dict(edits=invalid, unresolved=[]), dict(preserve=[]))
        self.assertEqual(apply_detail_edits(answer, dict(edits=[], unresolved=[]), dict(preserve=[])), answer)

    def test_scan_detail_views_keep_pixels_overlap_and_full_context(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'band.png'
            full = Image.linear_gradient('L').resize((1400, 210))
            full.save(path)
            original = path.read_bytes()
            views = band_details(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(views), 2)
            self.assertEqual(views[0]['pixel_bbox'], [0, 0, 840, 210])
            self.assertEqual(views[1]['pixel_bbox'], [560, 0, 1400, 210])
            for view in views:
                with Image.open(view['image']) as detail:
                    source = full.crop(view['pixel_bbox'])
                    self.assertEqual(detail.size, (source.width*2, source.height*2))
                    self.assertEqual(detail.resize(source.size, Image.Resampling.NEAREST).tobytes(), source.tobytes())
            full.resize((1024, 210)).save(path)
            self.assertEqual(band_details(path), [])

    def test_scan_detail_policy_invalidates_record_without_changing_native_requests(self):
        from book_pipeline.ingestion.formula_bands import correct_formula_bands
        calls = []
        class Client:
            _api = None
            def __init__(self, *args, **kwargs):
                self.ledger = self
            def summary(self):
                return {}
            def ask(self, task, prompt, payload, **kwargs):
                calls.append((prompt, kwargs['images']))
                answer = (dict(edits=[], unresolved=[]) if task.startswith('literal_scan_band')
                          else dict(items=[dict(kind='text', text='(3) If $x=1$, $y=1$.')]))
                kwargs['validator'](answer)
                return answer
        with tempfile.TemporaryDirectory() as name, pymupdf.open() as pdf:
            path = Path(name)/'source.pdf'
            pdf.new_page(width=400, height=300).insert_text((40,100), 'x = 1')
            pdf.save(path)
            atom = dict(id='a', group='a', page=0, kind='display_math',
                        content='x=1', length=3, extraction_method='native')
            index = SourceIndex(dict(atoms=[atom], input_sha256='p', mineru_sha256='m', source_pdf=str(path)))
            band = dict(id='b', page=0, bbox=[0,.2,1,.5], groups=['a'], preserve=[])
            config = dict(model='fake', base_url='https://example.invalid', thinking='disabled', concurrency=1)
            module = 'book_pipeline.ingestion.formula_bands.'
            with patch(module+'detect_bands', side_effect=lambda *args: [dict(band)]), patch(module+'DeepSeekClient', Client):
                directory = Path(name)/'corrections'
                correct_formula_bands(index, config, directory)
                self.assertEqual(len(calls[-1][1]), 1)
                native_prompt = calls[-1][0]
                index.atoms[0]['extraction_method'] = 'ocr'
                correct_formula_bands(index, config, directory)
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(calls[-1][1]), 3)
                self.assertTrue(calls[-2][0].startswith(native_prompt))
                self.assertNotEqual(native_prompt, calls[-1][0])
                correct_formula_bands(index, config, directory)
                self.assertEqual(len(calls), 3)
                with patch(module+'DETAIL_METHOD', 'new-detail-policy'):
                    correct_formula_bands(index, config, directory)
                self.assertEqual(len(calls), 5)

    def marked_formula(self, method='native', reason=False):
        panel = dict(id='box', bbox=[.1,.1,.9,.3], color=[0,.65,.32])
        atoms = [dict(id='formula',group='formula',kind='display_math',role='equation',
            block_type='equation',page=0,bbox=[.3,.26,.7,.285],content='x+y',length=3,
            source_panel=panel,extraction_method=method)]
        if reason:
            atoms.append(dict(id='reason',group='reason',kind='text',role='text',block_type='text',
                page=0,bbox=[.72,.26,.82,.28],content='by (i)',length=6,source_panel=panel,
                extraction_method=method))
        atoms.append(dict(id='mark',group='mark',kind='text',role='text',block_type='text',
            page=0,bbox=[.86,.275,.88,.295],content='♣',length=1,source_panel=panel,
            extraction_method=method))
        return SourceIndex(dict(atoms=atoms,pages=[dict(page=0)],input_sha256='p',mineru_sha256='m'))

    def test_owned_panel_end_mark_does_not_trigger_native_formula_retranscription(self):
        source = self.marked_formula()
        self.assertEqual(detect_bands(source), [])
        # An unowned symbol remains eligible: geometry alone is insufficient.
        source.atoms[-1].pop('source_panel')
        self.assertEqual(len(detect_bands(source)), 1)

    def test_necessary_formula_review_preserves_exact_end_mark_and_geometry(self):
        for method, reason in [('ocr',False),('native',True)]:
            with self.subTest(method=method):
                source = self.marked_formula(method,reason)
                band = detect_bands(source)[0]
                self.assertEqual([p['group'] for p in band['preserve']], ['mark'])
                items = [dict(kind='formula',latex='x+y',number=''),dict(kind='preserve',group='mark')]
                output = apply_formula_corrections(source,dict(method=METHOD,status='complete',usage={},
                    source=source.source,records=[dict(**band,answer=dict(items=items))]))
                self.assertEqual(output.atoms[-1],source.atoms[-1])

    def test_recognition_padding_stops_in_gap_before_neighboring_prose(self):
        with tempfile.TemporaryDirectory() as name, pymupdf.open() as document:
            page=document.new_page(width=300,height=200)
            page.insert_text((20,95),'previous words',fontsize=10)
            page.insert_text((120,109),'x = 0',fontsize=10)
            blocks=page.get_text('blocks')
            atoms=[dict(id=str(n),group=str(n),kind='text' if n==0 else 'display_math',
                block_type='text' if n==0 else 'equation',page=0,content=b[4],length=len(b[4]),
                bbox=[b[0]/300,b[1]/200,b[2]/300,b[3]/200],extraction_method='ocr') for n,b in enumerate(blocks)]
            index=SourceIndex(dict(atoms=atoms,pages=[dict(page=0,size_pt=[300,200])]))
            band=detect_bands(index)[0]
            self.assertEqual(band['groups'],['1'])
            self.assertGreater(band['crop_bbox'][1],atoms[0]['bbox'][3])
            self.assertLess(band['crop_bbox'][1],atoms[1]['bbox'][1])
            self.assertEqual(band['bbox'][1],atoms[1]['bbox'][1])
            image=Path(name)/'bounded.png';band_image(page,band,image,False)
            box=band['crop_bbox'];clip=pymupdf.Rect(0,box[1]*200,300,box[3]*200)
            self.assertEqual(image.read_bytes(),page.get_pixmap(dpi=220,clip=clip,alpha=False).tobytes('png'))

    def test_isolated_ocr_equation_is_reviewed_native_equation_is_retained(self):
        for method,count in [('ocr',1),('native',0)]:
            with self.subTest(method=method):
                atom=dict(id='eq',group='eq',kind='display_math',block_type='equation',
                          page=0,bbox=[.3,.2,.7,.23],content='P+Q>P',length=5,extraction_method=method)
                bands=detect_bands(SourceIndex(dict(atoms=[atom])))
                self.assertEqual(len(bands),count)
                if bands:self.assertEqual(bands[0]['bbox'],[0.,.2,1.,.23])

    def test_existing_overlapping_band_keeps_its_identifier_before_new_band(self):
        atoms=[]
        for sid,kind,box,method in [('isolated','equation',[.3,.1,.7,.12],'ocr'),
                                   ('overlap','equation',[.3,.4,.7,.42],'ocr'),
                                   ('reason','text',[.8,.4,.9,.42],'ocr')]:
            atoms.append(dict(id=sid,group=sid,kind='text' if kind=='text' else 'display_math',
                              block_type=kind,page=0,bbox=box,content='x',length=1,extraction_method=method))
        bands=detect_bands(SourceIndex(dict(atoms=atoms)))
        self.assertEqual(bands[0]['id'],'p000001-band-0001')
        self.assertIn('overlap',bands[0]['groups'])
        self.assertEqual(bands[1]['groups'],['isolated'])

    def test_context_recovers_missing_row_gap_and_owns_intervening_prose_once(self):
        atoms=[]
        for sid,kind,box in [('eq1','equation',[.3,.2,.7,.22]),
                             ('intro','text',[.1,.23,.8,.25]),
                             # Two omitted OCR rows lie between .25 and .29.
                             ('eq2','equation',[.3,.29,.7,.31])]:
            atoms.append(dict(id=sid,group=sid,kind='text' if kind=='text' else 'display_math',
                              block_type=kind,page=0,bbox=box,content='x',length=1,extraction_method='ocr'))
        index=SourceIndex(dict(atoms=atoms,pages=[dict(page=0)],input_sha256='p',mineru_sha256='m'))
        bands=detect_bands(index)
        self.assertEqual(len(bands),1)
        self.assertEqual(bands[0]['bbox'],[0.,.2,1.,.31])
        self.assertEqual(bands[0]['groups'],['eq1','intro','eq2'])
        report=dict(method=METHOD,status='complete',source=dict(input_sha256='p',mineru_sha256='m'),usage={},
                    records=[dict(**bands[0],answer=dict(items=[dict(kind='formula',latex='x=y',number=''),
                        dict(kind='text',text='For this reason'),dict(kind='formula',latex='y=z',number='')]))])
        output=apply_formula_corrections(index,report)
        self.assertEqual(len(output.groups),3)
        self.assertEqual(sum(a['content']=='For this reason' for a in output.atoms),1)

    def test_context_has_height_limit_and_does_not_cross_titles_or_protected_text(self):
        def source(types, tops):
            return SourceIndex(dict(atoms=[dict(id=str(n),group=str(n),kind='display_math' if t=='equation' else 'text',
                block_type=t,page=0,bbox=[.2,y,.8,y+.02],content='x',length=1,extraction_method='ocr')
                for n,(t,y) in enumerate(zip(types,tops))]))
        self.assertEqual(len(detect_bands(source(['equation']*10,[.1+n*.03 for n in range(10)]))),2)
        self.assertEqual(len(detect_bands(source(['equation','paragraph_title','equation'],[.1,.13,.16]))),2)
        self.assertEqual(len(detect_bands(source(['equation','text','equation'],[.1,.13,.16]),{'1'})),2)

    def test_faint_derivative_is_high_contrast_and_source_and_clear_render_are_unchanged(self):
        with tempfile.TemporaryDirectory() as name, pymupdf.open() as document:
            page=document.new_page(width=300,height=200)
            page.insert_text((40,100),'a / h = b / ah',fontsize=16,color=(.72,.72,.72))
            target=Path(name)/'faint.png'
            metadata=band_image(page,dict(bbox=[0,.3,1,.6]),target)
            self.assertEqual(metadata['dpi'],300)
            self.assertTrue(target.with_name('faint.raw.png').exists())
            self.assertLess(Image.open(target).getextrema()[0],30)
            plain=Path(name)/'native.png'
            self.assertIsNone(band_image(page,dict(bbox=[0,.3,1,.6]),plain,False))
            clip=pymupdf.Rect(0,58,300,122)
            self.assertEqual(plain.read_bytes(),page.get_pixmap(dpi=220,clip=clip,alpha=False).tobytes('png'))

    def test_window_edge_gap_is_read_without_consuming_following_paragraph(self):
        atoms=[]
        for n,(kind,y,z) in enumerate([('equation',.2,.22),('text',.26,.3)]):
            atoms.append(dict(id=str(n),group=str(n),kind='display_math' if kind=='equation' else 'text',
                block_type=kind,page=0,bbox=[.2,y,.8,z],content='x',length=1,extraction_method='ocr'))
        band=detect_bands(SourceIndex(dict(atoms=atoms)))[0]
        self.assertEqual(band['bbox'],[0.,.2,1.,.26])
        self.assertEqual(band['groups'],['0'])
        self.assertEqual(band['owned_ocr_gap'],[.22,.26])

    def test_tex_inline_delimiters_become_math_and_unbalanced_delimiters_fail(self):
        for text in [r'Since \(a/h=0\),',r'Since \\(a/h=0\\),']:
            self.assertEqual(normalized_inline_text(text),'Since $a/h=0$,')
            band=dict(id='b',page=0,bbox=[0,.2,1,.3],groups=['0'],preserve=[])
            answer=dict(items=[dict(kind='text',text=text)])
            validate_answer(answer,band)
            atom=dict(id='0',group='0',page=0,bbox=band['bbox'],block_type='text',role='text',
                      kind='text',content='old',length=3)
            source=dict(atoms=[atom],pages=[dict(page=0)],input_sha256='p',mineru_sha256='m')
            output=apply_formula_corrections(SourceIndex(source),dict(method=METHOD,status='complete',usage={},
                source=dict(input_sha256='p',mineru_sha256='m'),records=[dict(**band,answer=answer)]))
            self.assertEqual([(a['kind'],a['content']) for a in output.atoms],
                             [('text','Since '),('math','a/h=0'),('text',',')])
        for text in [r'Unfinished \(a/h',r'Display \[a/h\]']:
            with self.assertRaises(ValueError):
                validate_answer(dict(items=[dict(kind='text',text=text)]),dict(preserve=[]))
