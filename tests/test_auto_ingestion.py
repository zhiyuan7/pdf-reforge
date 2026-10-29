"""Mixed PDF routing, lossless native spans and translation source invariants."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf
from book_pipeline.config import digest, load_config, read, save
from book_pipeline.ingestion.native import prepare_source, classify_page, spans_for, validate_layout, normalize_marginals
from book_pipeline.ingestion.source import SourceIndex, load_source, materialize_source
from book_pipeline.ingestion.translation import Translator, protect, translate_source, translate_catalog
from book_pipeline.ai.client import APIFailure, DecisionUncertain


class Client:
    def __init__(self, answer=None):
        self.answer = answer
        self.calls = []
        self.ledger = self
    def summary(self): return {}
    def ask(self, task, system, payload, images=(), validator=None):
        self.calls.append((task, payload))
        value = self.answer(payload) if callable(self.answer) else deepcopy(self.answer)
        if value is None:
            value = dict(blocks=[dict(type='text', bbox=[.1,.1,.9,.9],
                                     parts=[dict(kind='text', spans=[s['id'] for s in payload['spans']])])])
        validator(value)
        return value


def middle_source(root, contents, image=False):
    pdf = root / 'book.pdf'
    with pymupdf.open() as d:
        d.new_page().insert_text((60,80), 'Supervised learning')
        d.save(pdf)
    source = root / 'raw'; source.mkdir()
    blocks = [dict(index=i, type=kind, bbox=[.1,.1,.9,.2], content=value)
              for i, (kind, value) in enumerate(contents)]
    if image:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0,0,20,20), False)
        pix.clear_with(128); pix.save(source/'figure.png')
        blocks.append(dict(index=len(blocks),type='image',bbox=[.1,.2,.9,.8],
                           content=[dict(type='image_body',image_path='figure.png',content='')]))
    save(source/'middle_json.json',dict(schema='docvortex.middle',schema_version='2.0',
         pages=[dict(page_idx=0,blocks=blocks)]))
    save(source/'manifest.json',dict(input_pdf=str(pdf),input_sha256=digest(pdf)))
    return pdf, source


class AutoIngestionTests(unittest.TestCase):
    def test_span_coverage_rejects_missing_duplicate_and_foreign_ids(self):
        spans=[dict(id='s0'),dict(id='s1')]
        def value(ids):return dict(blocks=[dict(type='text',bbox=[0,0,1,1],parts=[dict(kind='text',spans=ids)])])
        validate_layout(value(['s0','s1']),spans)
        for ids in [['s0'],['s0','s0','s1'],['s0','s1','s2']]:
            with self.subTest(ids=ids),self.assertRaises(ValueError):validate_layout(value(ids),spans)

    def test_native_scan_blank_mix_and_original_page_numbers(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf=root/'mixed.pdf'
            with pymupdf.open() as d:
                d.new_page().insert_text((50,80),'A native English paragraph with selectable text.')
                page=d.new_page()
                pix=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,100,100),False)
                pix.clear_with(100);page.insert_image(page.rect,stream=pix.tobytes('png'))
                d.new_page();d.save(pdf)
            called=[]
            def ocr(pdf, directory, pages, config, progress):
                called.extend(pages);directory.mkdir()
                save(directory/'middle_json.json',dict(pages=[dict(page_idx=1,blocks=[
                    dict(index=0,type='text',content='识别到的扫描正文',bbox=[0,0,1,1])])]))
            output=prepare_source(pdf,root/'source',load_config(),client=Client(),ocr_runner=ocr)
            self.assertEqual(called,[1])
            manifest=read(output/'manifest.json')
            self.assertEqual([p['method'] for p in manifest['pages']],['native','ocr','blank'])
            source=load_source(pdf,output)
            self.assertIn('selectable text',source['atoms'][0]['content'])
            self.assertEqual(source['atoms'][1]['page'],1)
            # Resume uses the complete cache without loading OCR or an API.
            prepare_source(pdf,root/'source',load_config(),client=Client(),ocr_runner=lambda *a:self.fail('OCR rerun'))

    def test_layout_uncertainty_falls_back_but_service_failure_does_not(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf=root/'text.pdf'
            with pymupdf.open() as d:
                d.new_page().insert_text((50,80),'A native paragraph.');d.save(pdf)
            with patch('book_pipeline.ingestion.native.native_page',side_effect=APIFailure('bad key')):
                with self.assertRaises(APIFailure):prepare_source(pdf,root/'a',load_config(),client=Client())
            with patch('book_pipeline.ingestion.native.native_page',side_effect=DecisionUncertain('bad layout')):
                def ocr(pdf,directory,pages,*args):
                    self.assertEqual(pages,[0]);directory.mkdir()
                    save(directory/'middle_json.json',dict(pages=[dict(page_idx=0,blocks=[])]))
                prepare_source(pdf,root/'b',load_config(),client=Client(),ocr_runner=ocr)
                self.assertEqual(read(root/'b'/'manifest.json')['pages'][0]['method'],'ocr')

    def test_terms_first_occurrence_is_source_order_and_numbers_math_are_immutable(self):
        client=Client(dict(items=[dict(id='b',text='⟪TERM:supervised learning⟫。'),
                                 dict(id='a',text='⟪TERM:Supervised learning⟫：⟪KEEP_0⟫，见图⟪KEEP_1⟫。')],
                           terms={'Supervised learning':'监督学习','supervised learning':'监督学习'}))
        translator=Translator(client)
        result=translator.translate([dict(id='a',text='Supervised learning: $y=f[x]$, see figure 2.1.'),
                                     dict(id='b',text='supervised learning.')])
        self.assertEqual(result['a'],'监督学习（Supervised learning）：$y=f[x]$，见图2.1。')
        self.assertEqual(result['b'],'监督学习。')
        self.assertEqual(translator.glossary,{'supervised learning':'监督学习'})

    def test_term_marker_does_not_duplicate_its_literal_translation(self):
        translator=Translator(Client(),{'training':'训练'})
        self.assertEqual(translator.render(dict(text='A training criterion.'),'一个⟪TERM:training⟫训练准则。'),
                         '一个训练（training）准则。')
        self.assertEqual(translator.render(dict(text='training training'),'⟪TERM:training⟫训练'),'训练训练')

    def test_translation_rejects_lost_numbers_and_changed_glossary(self):
        for answer in [dict(items=[dict(id='a',text='监督学习')],terms={}),
                       dict(items=[dict(id='a',text='⟪TERM:Supervised learning⟫ ⟪KEEP_0⟫')],terms={'Supervised learning':'指导学习'})]:
            with self.subTest(answer=answer),self.assertRaises(ValueError):
                Translator(Client(answer),{'supervised learning':'监督学习'}).translate([
                    dict(id='a',text='Supervised learning 2.1')])

    def test_chinese_book_skips_translation_and_no_api(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf,source=middle_source(root,[('text','这是一本中文教材，不需要翻译。')])
            client=Client()
            self.assertEqual(translate_source(pdf,source,root/'translated',load_config(),client=client),source)
            self.assertEqual(client.calls,[])

    def test_known_term_alias_does_not_invent_an_english_annotation(self):
        answer=dict(items=[dict(id='a',text='应用⟪TERM:linear transformation⟫⟪KEEP_0⟫。')],
                    terms={'linear transformation':'线性变换'})
        translator=Translator(Client(answer),{'linear transformation':'线性变换'})
        result=translator.translate([dict(id='a',text='Apply a linear transform $W$.')])
        self.assertEqual(result['a'],'应用线性变换$W$。')
        self.assertNotIn('linear transformation',translator.seen)
        with self.assertRaises(ValueError):
            Translator(Client(answer)).translate([dict(id='a',text='Apply a linear transform $W$.')])

    def test_unused_dictionary_metadata_does_not_enter_the_glossary(self):
        translator=Translator(Client(dict(items=[dict(id='a',text='使用一个模型。')],terms={'model':'模型'})))
        self.assertEqual(translator.translate([dict(id='a',text='Use a model.')]),{'a':'使用一个模型。'})
        self.assertEqual(translator.glossary,{})

    def test_bibliography_author_fragments_are_preserved_without_translation(self):
        text='Goyal, P., Dollár, P., Girshick, R., Noordhuis,\nP., Wesolowski, L., Kyrola, A., Tulloch, A.,'
        translator=Translator(Client(dict(items=[dict(id='a',text=text)],terms={})))
        self.assertEqual(translator.translate([dict(id='a',text=text)]),{'a':text})
        text='Fawzi, A., Balog, M., Huang, A., Hubert, T.,\nRomera-Paredes, B., Barekatain, M., Novikov,'
        self.assertEqual(Translator(Client(dict(items=[dict(id='a',text=text)],terms={}))).translate([
            dict(id='a',text=text)]),{'a':text})
        text='Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020).'
        result='Kitaev, N., Kaiser, Ł., & Levskaya, A. (⟪KEEP_0⟫)。'
        self.assertEqual(Translator(Client(dict(items=[dict(id='a',text=result)],terms={}))).translate([
            dict(id='a',text=text)]),{'a':result.replace('⟪KEEP_0⟫','2020')})
        from book_pipeline.ingestion.translation import author_names_only
        self.assertTrue(author_names_only('Kynkäänniemi, T., Karras, T., Laine, S., Lehti-'))
        self.assertTrue(author_names_only('Liu, Z., Lin, Y., Cao, Y., & Guo, B. (2021c). Swin'))
        self.assertFalse(author_names_only('Liu, Z., Lin, Y. (2021). This paper proposes a new network.'))
        prose='This paragraph describes a new neural network.'
        with self.assertRaises(ValueError):
            Translator(Client(dict(items=[dict(id='a',text=prose)],terms={}))).translate([dict(id='a',text=prose)])

    def test_wrapped_url_can_rejoin_without_becoming_translatable_prose(self):
        original='https:\n//www.skynettoday.com/overviews/neural-\nnet-history. 37'
        text='https://www.skynettoday.com/overviews/neural-net-history。⟪KEEP_0⟫'
        result=Translator(Client(dict(items=[dict(id='a',text=text)],terms={}))).translate([dict(id='a',text=original)])
        self.assertEqual(result['a'],text.replace('⟪KEEP_0⟫','37'))

    def test_translation_preserves_assets_math_raw_bytes_and_source_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf,source=middle_source(root,[
                ('paragraph_title','Supervised learning'),
                ('text',[dict(type='text',content='Supervised learning'),dict(type='equation_inline',content='x^2')]),
                ('equation','y=f[x]')],image=True)
            before=digest(source/'middle_json.json')
            def answer(payload):
                return dict(items=[dict(id=row['id'],text='⟪TERM:Supervised learning⟫'+''.join(__import__('re').findall(r'⟪KEEP_\d+⟫',row['text']))) for row in payload['items']],
                            terms={'Supervised learning':'监督学习'})
            out=translate_source(pdf,source,root/'translated',load_config(),client=Client(answer))
            self.assertEqual(digest(source/'middle_json.json'),before)
            index=SourceIndex(load_source(pdf,out))
            self.assertEqual([a['content'] for a in index.atoms if a['kind'] in {'math','display_math'}],['x^2','y=f[x]'])
            img=next(a for a in index.atoms if a['kind']=='image')
            self.assertEqual(digest(img['asset']),digest(source/'figure.png'))
            self.assertTrue(index.source.get('translation'))
            entries=translate_catalog([dict(id='e1',title='2 Supervised learning')],index.source,Client())
            self.assertEqual(entries[0]['title'],'监督学习（Supervised learning）')
            self.assertEqual(entries[0]['original_title'],'2 Supervised learning')
            self.assertEqual(translate_source(pdf,source,out,load_config(),client=Client()),out)

    def test_source_toc_does_not_consume_first_terminology_annotation(self):
        translator=Translator(Client(dict(items=[dict(id='a',text='⟪TERM:supervised learning⟫'),
                                                 dict(id='b',text='⟪TERM:supervised learning⟫')],
                                         terms={'supervised learning':'监督学习'})))
        result=translator.translate([dict(id='a',text='supervised learning',annotate=False),
                                     dict(id='b',text='supervised learning')])
        self.assertEqual(result,{'a':'监督学习','b':'监督学习（supervised learning）'})

    def test_wrapped_terms_short_connectors_and_unchanged_acronyms(self):
        translator=Translator(Client(dict(items=[dict(id='a',text='⟪TERM:training set⟫与⟪TERM:ReLU⟫'),
                                                 dict(id='b',text='因此：')],
                                         terms={'training set':'训练集','ReLU':'ReLU'})))
        result=translator.translate([dict(id='a',text='training\nset and ReLU'),dict(id='b',text='so:')])
        self.assertEqual(result['a'],'训练集（training set）与ReLU')
        self.assertEqual(result['b'],'因此：')

    def test_html_table_protects_tags_cells_and_formula_numbers(self):
        raw='<table><tr><td>Loss $x^2$</td><td>2.1</td></tr></table>'
        text,keep=protect(raw)
        answer=text.replace('Loss','损失')
        translator=Translator(Client(dict(items=[dict(id='a',text=answer)],terms={})))
        self.assertEqual(translator.translate([dict(id='a',text=raw)])['a'],raw.replace('Loss','损失'))

    def test_number_inventory_survives_adjacent_chinese_without_releasing_identifiers(self):
        from collections import Counter
        source='In Figure 6.2 we use $p$ for 12 trials.'
        translator=Translator(Client(dict(items=[dict(id='a',text='在图⟪KEEP_0⟫中，我们用⟪KEEP_1⟫进行⟪KEEP_2⟫次试验。')],terms={})))
        output=translator.translate([dict(id='a',text=source)])['a']
        self.assertEqual(Counter(protect(output)[1]),Counter(protect(source)[1]))
        self.assertEqual(protect('图6.2中有12次试验')[1],['6.2','12'])
        self.assertEqual(protect('x2 2t item_12 H2O')[1],[])
        for damaged in (output.replace('12','13'),output+'另有12次',output.replace('6.2','')):
            self.assertNotEqual(Counter(protect(damaged)[1]),Counter(protect(source)[1]))

    def test_same_named_headings_keep_their_numbers_in_catalogue_lookup(self):
        source=dict(translation=dict(titles=[dict(original='1.1 Supervised learning',translated='1.1 监督学习'),
                                            dict(original='Chapter 2 Supervised learning',translated='第2章 监督学习')],glossary={}))
        entries=translate_catalog([dict(id='a',title='1.1 Supervised learning'),dict(id='b',title='2 Supervised learning')],source,Client())
        self.assertEqual([e['title'] for e in entries],['1.1 监督学习','第2章 监督学习'])

    def test_illustrations_cannot_silently_disappear_from_native_layout(self):
        value=dict(blocks=[dict(type='text',bbox=[0,0,1,1],parts=[dict(kind='text',spans=['s0'])])])
        with self.assertRaisesRegex(ValueError,'illustration'):
            validate_layout(value,[dict(id='s0')],[[.1,.1,.8,.8]])


    def test_literal_known_term_is_annotated_at_first_body_occurrence(self):
        translator=Translator(Client(dict(items=[dict(id='a',text='监督学习和监督学习')],terms={})),
                              {'supervised learning':'监督学习'})
        answer=translator.translate([dict(id='a',text='supervised learning and supervised learning')])
        self.assertEqual(answer['a'],'监督学习（supervised learning）和监督学习')

    def test_repeated_footer_requires_two_matching_pages_before_reclassifying_footnote(self):
        def page(n,kind):return dict(page_idx=n,blocks=[dict(type=kind,bbox=[.2,.9,.8,.94],content='Repeated draft footer')])
        pages=[page(0,'footer'),page(1,'page_footnote')]
        normalize_marginals(pages)
        self.assertEqual(pages[1]['blocks'][0]['type'],'page_footnote')
        pages.append(page(2,'footer'));normalize_marginals(pages)
        self.assertEqual(pages[1]['blocks'][0]['type'],'footer')


    def test_math_can_follow_chinese_word_order_without_identity_changes(self):
        client=Client(dict(items=[dict(id='a',text='从输入⟪KEEP_1⟫计算预测⟪KEEP_0⟫。')],terms={}))
        result=Translator(client).translate([dict(id='a',text='Compute prediction $y$ from input $x$.')])
        self.assertEqual(result['a'],'从输入$x$计算预测$y$。')

    def test_inline_math_is_sent_with_its_whole_paragraph_and_restored(self):
        from book_pipeline.ingestion.translation import pack_inline_math,unpack_inline_math,text_leaves
        block=dict(type='text',bbox=[.1,.1,.9,.2],content=[dict(type='text',content='The output '),
             dict(type='equation_inline',content='y'),dict(type='text',content=' depends on input '),
             dict(type='equation_inline',content='x'),dict(type='text',content='.')])
        pack_inline_math(block)
        self.assertEqual(len(list(text_leaves(block))),1)
        self.assertEqual(block['content'][0]['content'],'The output ⟪MATH_0⟫ depends on input ⟪MATH_1⟫.')
        block['content'][0]['content']='输入⟪MATH_1⟫决定输出⟪MATH_0⟫。'
        unpack_inline_math(block)
        self.assertEqual([n['content'] for n in block['content']],['输入','x','决定输出','y','。'])


    def test_running_title_next_to_page_number_is_a_header(self):
        pages=[dict(page_idx=0,blocks=[dict(type='paragraph_title',bbox=[.1,.6,.8,.63],content='2.2 Linear regression')]),
               dict(page_idx=1,blocks=[dict(type='paragraph_title',bbox=[.1,.13,.5,.15],content='2.2 Linear regression'),
                                      dict(type='page_number',bbox=[.8,.13,.85,.15],content='19')])]
        normalize_marginals(pages)
        self.assertEqual(pages[0]['blocks'][0]['type'],'paragraph_title')
        self.assertEqual(pages[1]['blocks'][0]['type'],'header')

    def test_alternating_running_titles_need_peer_band_folio_and_separate_body(self):
        from copy import deepcopy
        pages=[dict(page_idx=0,blocks=[dict(type='header',bbox=[.5,.1,.9,.113],content='Chapter title')]),
               dict(page_idx=1,blocks=[dict(type='paragraph_title',bbox=[.1,.1,.4,.113],content='3.2 A different section'),
                   dict(type='page_number',bbox=[.9,.1,.92,.113],content='18'),
                   dict(type='text',bbox=[.1,.135,.75,.2],content='A paragraph continuing from the previous page.')])]
        for defect in ('none','no_peer','no_folio','different_band','body_too_close','folio_inside'):
            p=deepcopy(pages)
            if defect=='no_peer':p[0]['blocks']=[]
            if defect=='no_folio':p[1]['blocks'][1]['type']='text'
            if defect=='different_band':p[0]['blocks'][0]['bbox']=[.5,.06,.9,.073]
            if defect=='body_too_close':p[1]['blocks'][2]['bbox'][1]=.12
            if defect=='folio_inside':p[1]['blocks'][1]['bbox']=[.2,.1,.22,.113]
            normalize_marginals(p)
            self.assertEqual(p[1]['blocks'][0]['type'],'header' if defect=='none' else 'paragraph_title',defect)


    def test_high_alternating_scan_header_requires_peer_folio_and_body_gap(self):
        pages=[dict(page_idx=0,blocks=[
            dict(type='header',bbox=[.39,.049,.63,.065],content='A running section'),
            dict(type='page_number',bbox=[.88,.043,.91,.061],content='57')]),
            dict(page_idx=1,blocks=[
                dict(type='paragraph_title',bbox=[.43,.052,.54,.068],content='5 A running chapter'),
                dict(type='page_number',bbox=[.085,.048,.114,.066],content='58'),
                dict(type='text',bbox=[.19,.084,.62,.102],content='An example continues here.')])]
        for defect in ('none','no_peer','no_folio','wrong_band','body_too_close','tall_heading'):
            with self.subTest(defect=defect):
                p=deepcopy(pages)
                if defect=='no_peer':p[0]['blocks']=[]
                if defect=='no_folio':p[1]['blocks'][1]['type']='text'
                if defect=='wrong_band':p[0]['blocks'][0]['bbox']=[.39,.1,.63,.116]
                if defect=='body_too_close':p[1]['blocks'][2]['bbox'][1]=.07
                if defect=='tall_heading':p[1]['blocks'][0]['bbox'][3]=.09
                normalize_marginals(p)
                self.assertEqual(p[1]['blocks'][0]['type'],
                    'header' if defect=='none' else 'paragraph_title')

    def test_invalid_translation_batch_splits_without_bypassing_validation(self):
        class SplitClient(Client):
            def ask(self,task,system,payload,images=(),validator=None):
                if len(payload['items'])>1: raise DecisionUncertain('mixed paragraphs')
                answer=dict(items=[dict(id=payload['items'][0]['id'],text='译文')],terms={})
                validator(answer);return answer
        translator=Translator(SplitClient())
        self.assertEqual(translator.translate([dict(id='a',text='Paragraph one'),dict(id='b',text='Paragraph two')]),
                         {'a':'译文','b':'译文'})
        self.assertEqual(set(translator.templates),{'a','b'})


    def test_coordinated_terms_share_the_final_noun_without_being_invented(self):
        from book_pipeline.ingestion.translation import source_phrase
        text='supervised, unsu-\npervised, and reinforcement learning'
        for term in ['supervised learning','unsupervised learning','reinforcement learning']:
            self.assertTrue(source_phrase(term,text),term)
        self.assertFalse(source_phrase('supervised learning','supervised models concern unrelated tasks in machine learning'))
        self.assertTrue(source_phrase('dot product','their dot productgenerally decreases'))
        self.assertTrue(source_phrase('relative position','the relativeposition of two embeddings'))
        self.assertTrue(source_phrase('Jensen-Shannon divergence','Jensen–Shannon divergence'))
        self.assertFalse(source_phrase('linear function','a nonlinear function'))

    def test_indented_paragraphs_are_not_merged_by_their_shared_outer_bbox(self):
        from book_pipeline.ingestion.native import merge_text_blocks
        first=dict(index=0,type='text',bbox=[.1,.1,.9,.15],native_first_line_x=.1,content=[dict(type='text',content='First paragraph.')])
        second=dict(index=1,type='text',bbox=[.1,.153,.9,.20],native_first_line_x=.125,content=[dict(type='text',content='Second paragraph.')])
        self.assertEqual(len(merge_text_blocks([deepcopy(first),deepcopy(second)])),2)
        second['native_first_line_x']=.1
        self.assertEqual(len(merge_text_blocks([deepcopy(first),deepcopy(second)])),1)

    def test_parallel_translation_commits_terms_in_book_order(self):
        import time
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf,source=middle_source(root,[('text','linear model. '*400),('text','linear model. '*400)])
            def answer(payload):
                row=payload['items'][0]
                if row['id']=='t0':time.sleep(.02)
                return dict(items=[dict(id=row['id'],text='⟪TERM:linear model⟫。'*400)],
                            terms={'linear model':'线性模型' if row['id']=='t0' else '线性模式'})
            result=translate_source(pdf,source,root/'translated',load_config(),client=Client(answer))
            data=load_source(pdf,result)
            self.assertEqual(data['translation']['glossary']['linear model'],'线性模型')
            self.assertEqual(data['atoms'][0]['content'].count('（linear model）'),1)
            self.assertNotIn('linear model',data['atoms'][1]['content'])
            self.assertNotIn('线性模式',data['atoms'][1]['content'])

    def test_translation_resumes_committed_batches_and_pending_glossary_snapshots(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf,source=middle_source(root,[('text','linear model. '*400),('text','linear model. '*400)])
            config=load_config();config['concurrency']=2
            def answer(payload):
                row=payload['items'][0]
                return dict(items=[dict(id=row['id'],text='⟪TERM:linear model⟫。'*400)],terms={'linear model':'线性模型'})
            def failing(payload):
                if payload['items'][0]['id']=='t1': raise APIFailure('temporary service outage')
                return answer(payload)
            with self.assertRaises(APIFailure):
                translate_source(pdf,source,root/'translated',config,client=Client(failing))
            saved=read(root/'translated/translation_checkpoint.json')['checkpoint']
            self.assertEqual(saved['next_batch'],1)
            resumed=Client(answer)
            result=translate_source(pdf,source,root/'translated',config,client=resumed)
            self.assertEqual([c[1]['items'][0]['id'] for c in resumed.calls],['t1'])
            self.assertEqual(resumed.calls[0][1]['glossary'],{})
            translated=load_source(pdf,result)
            self.assertEqual(sum(a['content'].count('（linear model）') for a in translated['atoms']),1)


    def test_corrected_source_round_trip_preserves_formula_and_image(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf,source=middle_source(root,[('text','Body text'),('equation',r'x^2\tag{1}')],image=True)
            index=SourceIndex(load_source(pdf,source))
            target=materialize_source(index,root/'corrected')
            restored=load_source(pdf,target)
            self.assertEqual([(a['kind'],a['role'],a['content']) for a in restored['atoms']],
                             [(a['kind'],a['role'],a['content']) for a in index.atoms])


if __name__=='__main__':unittest.main()
