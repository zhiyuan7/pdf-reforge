"""Ambiguous algorithms require visual classification and exact source spans."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import pymupdf

from book_pipeline.config import digest, load_config, read, save
from book_pipeline.ingestion.raster_panels import recover_raster_panels
from book_pipeline.ingestion.raster_prose import source_lines, validate, layout_atoms
from book_pipeline.ingestion.source import load_source


class Client:
    def __init__(self, classification='prose_panel'):
        self.classification, self.calls, self.ledger = classification, 0, self

    def summary(self): return {}

    def ask(self, task, system, payload, images, validator, **kwargs):
        self.calls += 1
        answer = dict(classification=self.classification,reason='Image checked',lines=[])
        if self.classification == 'prose_panel':
            answer['lines'] = [dict(id=n,kind=k,bbox=b) for n,(k,b) in enumerate([
                ('title',[.12,.143,.5,.166]), ('text',[.12,.19,.8,.218]),
                ('display',[.36,.25,.53,.28]), ('text',[.12,.3,.8,.326])])]
        validator(answer)
        return answer


def fixture(root, subtype='algorithm'):
    pdf = root/'input.pdf'
    with pymupdf.open() as original, pymupdf.open() as output:
        page=original.new_page(width=500,height=700)
        page.draw_rect((40,90,460,240),color=(.4,.4,.4),fill=(1,.98,.76))
        page.draw_rect((40,90,460,120),color=None,fill=(1,.97,.6))
        page.draw_line((40,120),(460,120),color=(0,0,0),width=.7)
        for x,y,text in [(60,110,'A prose title'),(60,150,'A source paragraph states that'),
                         (180,190,'x + 0 = x.'),(60,225,'A following source paragraph.')]:
            page.insert_text((x,y),text,fontsize=10)
        p=output.new_page(width=500,height=700)
        p.insert_image(p.rect,stream=page.get_pixmap(dpi=150).tobytes('png'))
        output.save(pdf)
    origin=root/'ocr'
    save(origin/'middle_json.json',dict(schema='docvortex.middle',schema_version='2.0',pages=[dict(
        page_idx=0,blocks=[dict(index=2,type='code',sub_type=subtype,bbox=[.09,.13,.91,.34],content=[
            dict(type='algorithm_body',content=[dict(type='text',content='A prose title\n\nA source paragraph states that\n    '),
                dict(type='equation_inline',content='x+0=x'),
                dict(type='text',content='.\n\nA following source paragraph.')])])])]))
    save(origin/'manifest.json',dict(input_pdf=str(pdf),input_sha256=digest(pdf),mineru_version='test'))
    return pdf,origin


class RasterProseTests(unittest.TestCase):
    def test_layout_preserves_all_source_characters_and_only_promotes_math_row(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);pdf,origin=fixture(root)
            before=[digest(pdf),digest(origin/'manifest.json'),digest(origin/'middle_json.json')]
            raw=load_source(pdf,origin)['atoms']
            client=Client();dest=recover_raster_panels(pdf,origin,root/'derived',load_config(),client=client)
            atoms=load_source(pdf,dest)['atoms']
            self.assertEqual(''.join(a['content'] for a in raw), ''.join(a['content'] for a in atoms))
            self.assertEqual([a['content'].strip() for a in atoms if a['kind']=='display_math'],['x+0=x.'])
            self.assertTrue(all(a.get('source_panel') for a in atoms))
            self.assertTrue(atoms[0]['source_panel_title'])
            title_group=atoms[0]['group']
            self.assertTrue(all(a['source_panel_title']==(a['group']==title_group) for a in atoms))
            spans=read(dest/'review.json')['prose_reviews'][0]['source_spans']
            lengths={a['id']:[] for a in raw}
            for row in spans:
                for span in row['original_spans']: lengths[span['atom']].append((span['start'],span['end']))
            for atom in raw:
                intervals=lengths[atom['id']]
                self.assertEqual(intervals[0][0],0)
                self.assertEqual(intervals[-1][1],len(atom['content']))
                self.assertTrue(all(a[1]==b[0] for a,b in zip(intervals,intervals[1:])))
            self.assertEqual(before,[digest(pdf),digest(origin/'manifest.json'),digest(origin/'middle_json.json')])
            self.assertEqual(recover_raster_panels(pdf,origin,dest,load_config(),client=client),dest)
            self.assertEqual(client.calls,1)
            config=load_config();config['image_dpi']+=1
            with self.assertRaisesRegex(ValueError,'cache changed'):
                recover_raster_panels(pdf,origin,dest,config,client=client)

    def test_true_algorithms_uncertainty_and_ordinary_code_retain_original_source(self):
        for classification,subtype in [('algorithm','algorithm'),('uncertain','algorithm'),('prose_panel','code')]:
            with self.subTest(classification=classification,subtype=subtype), tempfile.TemporaryDirectory() as name:
                root=Path(name);pdf,origin=fixture(root,subtype);client=Client(classification)
                self.assertEqual(recover_raster_panels(pdf,origin,root/'derived',load_config(),client=client),origin)
                self.assertEqual(client.calls,int(subtype=='algorithm'))

    def test_missing_duplicate_reordered_outside_or_prose_display_rows_fail_validation(self):
        atoms=[dict(id='s0',kind='text',content='Title\nBody prose\n'),dict(id='s1',kind='math',content='x=0')]
        lines=source_lines(atoms)
        panel=dict(bbox=[.1,.1,.9,.8],title_bbox=[.1,.1,.9,.2])
        answer=dict(classification='prose_panel',reason='image',lines=[dict(id=n,kind=k,bbox=b) for n,(k,b) in enumerate([
            ('title',[.2,.12,.8,.18]),('text',[.2,.3,.8,.4]),('display',[.3,.5,.6,.6])])])
        validate(answer,panel,lines)
        for change in [lambda a:a['lines'].pop(),lambda a:a['lines'][1].update(id=0),
                       lambda a:a['lines'].reverse(),lambda a:a['lines'][2].update(bbox=[0,0,1,1]),
                       lambda a:a['lines'][1].update(kind='display'),
                       lambda a:a['lines'][2].update(bbox=[.3,.31,.6,.6])]:
            value=deepcopy(answer);change(value)
            with self.assertRaises(ValueError):validate(value,panel,lines)
