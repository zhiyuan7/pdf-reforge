from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

from book_pipeline.config import load_config,executable,save,read,digest
from book_pipeline.ingestion.source import SourceIndex,load_source
from book_pipeline.publishing.native_code import recover_native_code
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.compiler import compile_project


class NativeCodeTests(unittest.TestCase):
    def source(self,root):
        font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
        if not font.exists():self.skipTest('Native code test needs DejaVuSansMono')
        # Real PDF subsets can lack the post table accepted by PDF readers
        # but required by xdvipdfmx when the font is reused independently.
        from fontTools.ttLib import TTFont
        subset=TTFont(font);subset.getGlyphOrder();del subset['post']
        font=root/'mono-subset.ttf';subset.save(font)
        text='def solve(x):\n    y = x + 1\n\n    return y\n# π is literal'
        caption='Algorithm 4. A literal code example with a separate lateral explanation.'
        panel=dict(id='p',bbox=[.08,.12,.62,.29],color=[.2,.3,.4],background_color=[.95,.95,.95])
        path=root/'source.pdf'
        with pymupdf.open() as pdf:
            page=pdf.new_page(width=500,height=700)
            page.draw_rect(pymupdf.Rect(40,84,310,203),fill=(.95,.95,.95))
            for n,line in enumerate(text.split('\n')):
                if line:page.insert_text((50,100+n*15),line,fontname='code',fontfile=str(font),fontsize=8,color=(0,.3,.5))
            code=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]
            page.insert_text((340,94),'Algorithm 4. A literal',fontsize=8)
            page.insert_text((340,104),'code',fontname='code',fontfile=str(font),fontsize=8)
            page.insert_text((361,104),'example with a separate',fontsize=8)
            page.insert_text((340,114),'lateral expla-',fontsize=8)
            page.insert_text((340,124),'nation.',fontsize=8)
            note=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['bbox'][0]>330]
            pdf.save(path)
        box=lambda ss:[min(s['bbox'][0] for s in ss)/500,min(s['bbox'][1] for s in ss)/700,
                       max(s['bbox'][2] for s in ss)/500,max(s['bbox'][3] for s in ss)/700]
        atoms=[]
        for n,(value,bbox) in enumerate([(text,box(code)),(caption,box(note))]):
            atoms.append(dict(id=str(n),group=str(n),page=0,kind='text',role='text',block_type='text',
                content=value,length=len(value),bbox=bbox,extraction_method='native'))
        atoms[0]['source_panel']=panel
        return SourceIndex(dict(source_pdf=str(path),source_page_count=1,atoms=atoms))

    def test_exact_native_code_owns_fonts_lines_and_caption_without_rewriting_text(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);before=deepcopy(index.source)
            layouts,rows=recover_native_code(index,root/'out')
            self.assertEqual(layouts['0']['lines'],4)
            self.assertEqual(len(rows),1)
            self.assertEqual(index.source,before)
            self.assertTrue(list((root/'out/fonts').glob('*.ttf')))
            self.assertEqual(recover_native_code(index,root/'changed',{'0'})[0],{})
            wrong=SourceIndex(deepcopy(index.source));wrong.atoms[0]['content']=wrong.atoms[0]['content'].replace('x + 1','x + 2')
            self.assertEqual(recover_native_code(wrong,root/'wrong')[0],{})

    def test_translation_preserves_verified_code_and_recovers_its_fonts_afterward(self):
        from book_pipeline.ingestion.translation import translate_source
        from tests.test_auto_ingestion import Client
        import re
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);raw=root/'raw'
            blocks=[]
            for n,atom in enumerate(index.atoms):
                blocks.append(dict(index=n,type='text',bbox=atom['bbox'],
                    content=[dict(type='text',content=atom['content'])],
                    **({'source_panel':atom['source_panel']} if atom.get('source_panel') else {})))
            pdf=Path(index.source['source_pdf'])
            save(raw/'middle_json.json',dict(schema='docvortex.middle',schema_version='2.0',
                pages=[dict(page_idx=0,blocks=blocks)]))
            save(raw/'manifest.json',dict(input_sha256=digest(pdf),pages=[dict(page=1,method='native')]))
            before=digest(raw/'middle_json.json')
            def answer(payload):
                return dict(items=[dict(id=row['id'],text='算法说明 '+''.join(re.findall(r'⟪KEEP_\d+⟫',row['text'])))
                                   for row in payload['items']],terms={})
            client=Client(answer)
            dest=translate_source(pdf,raw,root/'translated',load_config(),client=client)
            translated=SourceIndex(load_source(pdf,dest))
            self.assertEqual(translated.atoms[0]['content'],index.atoms[0]['content'])
            self.assertEqual(digest(raw/'middle_json.json'),before)
            self.assertNotIn('def solve',str(client.calls))
            layouts,_=recover_native_code(translated,root/'fonts-after')
            self.assertIn(translated.atoms[0]['id'],layouts)
            self.assertEqual(read(dest/'manifest.json')['translation']['preserved_native_code_groups'],
                             [translated.atoms[0]['group']])

    def test_code_identity_is_independent_of_publishable_font_format(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root)
            with patch.object(pymupdf.Page,'get_fonts',return_value=[]):
                fallback,_=recover_native_code(index,root/'fonts')
                layouts,_=recover_native_code(index,root/'identity',identify_only=True)
            self.assertEqual(fallback['0']['method'],'verified_native_code_searchable_crop_v1')
            self.assertTrue((root/'fonts'/fallback['0']['asset']).is_file())
            self.assertTrue(layouts['0']['identity_verified'])
            self.assertFalse((root/'identity/fonts').exists())

    def test_unembeddable_font_keeps_original_ink_search_indentation_and_caption(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root);before=deepcopy(index.source)
            with patch.object(pymupdf.Page,'get_fonts',return_value=[]):
                audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                    [],[],dict(title='Fallback code',author=''),load_config(),root/'latex',sample_mode=True)
            self.assertEqual(index.source,before)
            self.assertTrue(audit['coverage_complete'])
            self.assertTrue(any(r['method']=='verified_native_code_searchable_crop_v1'
                                for r in audit['source_order_layout']))
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                page=pdf[0];start=page.search_for('def')[0];ret=page.search_for('return')[0]
                self.assertEqual(page.get_text().count('def solve'),1)
                self.assertIn('π',page.get_text())
                self.assertGreater(ret.x0,start.x0+8)
                self.assertGreater(ret.y0,start.y0+20)
                self.assertGreater(page.search_for('Algorithm 4.')[0].x0,ret.x1)
                self.assertTrue(page.get_images())

    def test_compiled_code_keeps_searchable_unicode_indentation_colors_and_side_caption(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);index=self.source(root)
            audit=write_project(index,dict(corrections=[],joins=[],exclusions=[],bold_spans=[]),
                [],[],dict(title='Native code',author=''),load_config(),root/'latex',sample_mode=True)
            self.assertTrue(audit['coverage_complete'])
            result=compile_project(root/'latex',{t:executable(t) for t in ('xelatex','latexmk')})
            self.assertTrue(result['success'],result['log_tail']);self.assertEqual(result['warnings'],[])
            with pymupdf.open(root/'latex/main.pdf') as pdf:
                self.assertEqual(len(pdf),1)
                page=pdf[0];start=page.search_for('def')[0];ret=page.search_for('return')[0]
                self.assertGreater(ret.x0,start.x0+8)
                self.assertGreater(ret.y0,start.y0+20)
                self.assertGreater(page.search_for('Algorithm 4.')[0].x0,ret.x1)
                self.assertIn('π',page.get_text())
                spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]
                code=next(s for s in spans if 'def solve' in s['text'])
                self.assertIn('DejaVuSansMono',code['font'])
                with pymupdf.open(index.source['source_pdf']) as source:
                    ink=next(s['color'] for b in source[0].get_text('dict')['blocks']
                        for l in b.get('lines',[]) for s in l['spans'] if 'def solve' in s['text'])
                self.assertEqual(code['color'],ink)


if __name__=='__main__':unittest.main()
