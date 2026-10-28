"""Typography requires literal ownership and independent shape evidence."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pymupdf

from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.latex import Renderer
from book_pipeline.publishing.raster_body_styles import (analyze_page, context_windows, enough_context, font_scores, is_bold, is_italic,
    anchored_word_span, bracket_ink, complete_line_box, literal_span, mixed_fragments, owned_ranges, prose_ink, recover_body_styles, supported_words, templates,
    scored_scan_word, scored_word, word_boxes, caption_candidates, anchored_captions, caption_line_contexts,
    edge_prose_contexts, anchored_mixed_prose, competing_times_observations)


class RasterBodyStyleTests(unittest.TestCase):
    def test_competing_family_can_veto_but_never_grant_times_emphasis(self):
        rgb=np.full((100,100,3),255,np.uint8)
        obs=self.observation()
        target='book_pipeline.publishing.scan_font_calibration.shape_scores'
        for scores, kept in (([.3,.4,.8,.7],True),([.9,.8,.3,.4],False),([],False)):
            with patch(target,return_value={'computer_modern':scores}):
                result=competing_times_observations(rgb,obs)
                self.assertEqual(bool(result),kept)
                if kept:self.assertEqual(result,obs)
        for defect in ('confidence','outside','bold'):
            altered=deepcopy(obs)
            if defect=='confidence':altered[0]['words'][0]['score']=.98
            if defect=='outside':altered[0]['words'][0]['bbox'][2]=101
            if defect=='bold':altered[0]['words'][0]['scores']=dict(tiro=.3,tibo=.9,tiit=.3,tibi=.4)
            with patch(target,return_value={'computer_modern':[.9,.3,.8,.3]}):
                self.assertEqual(competing_times_observations(rgb,altered),[])
        with patch(target,side_effect=[{'computer_modern':[.3,.4,.8,.7]},
                                       {'computer_modern':[.9,.8,.3,.4]}]):
            self.assertEqual(competing_times_observations(rgb,obs),[])

    def test_wrapped_mixed_prose_keeps_literal_edge_context_without_internal_skips(self):
        source = ' is a probability distribution or a probability measure if it satisfies the following three axioms:'
        line = 'event A is a probability distribution or a probability measure if it'
        contexts = edge_prose_contexts(source, line)
        self.assertEqual(len(contexts), 1)
        a, b = contexts[0]
        self.assertEqual(source[slice(*a)], line[slice(*b)])
        self.assertEqual(source[slice(*a)], 'is a probability distribution or a probability measure if it')
        for bad in ('event A is a probability DIFFERENT or a probability measure if it',
                    'noise probability distribution or a probability measure noise',
                    'probability distribution', line+' '+line):
            self.assertEqual(edge_prose_contexts(source, bad), [], bad)
        # A changed initial word is not a line boundary, and a substring of
        # an independently read word is never literal word evidence.
        self.assertEqual(edge_prose_contexts('A classification task with variable',
            'BA classification task with variable x'), [])

        scores = dict(tiro=.3, tibo=.94, tiit=.3, tibi=.5)
        word = dict(text='distribution',score=.999,scores=scores,bbox=[30,20,50,30])
        obs = [dict(text=line,score=.999,bbox=[10,20,80,30],words=[word])]
        item = dict(atom='a',text=source,bbox=[.05,.15,.9,.35])
        runs = owned_ranges(obs,[item],[100]*4,mixed={'a':['A']})
        self.assertEqual([r['quote'] for r in runs['a']],['distribution'])
        self.assertTrue(runs['a'][0]['bold'])
        self.assertFalse(runs['a'][0]['italic'])
        self.assertEqual(owned_ranges(obs,[item,dict(item,atom='duplicate')],
            [100]*4,mixed={'a':['A'],'duplicate':['A']}), {})
        self.assertEqual(owned_ranges(obs,[item],[100]*4,mixed={'a':['distribution']}), {})

    def mixed_fixture(self, contents):
        base = dict(page=0, group='G', bbox=[.05,.15,.9,.35], extraction_method='ocr',
                    role='paragraph', block_type='paragraph')
        raw = [dict(base,id=f'r{n}',kind=kind,content=text,length=len(text))
               for n,(kind,text) in enumerate(contents)]
        edited = [dict(a,id=f'e{n}',original_atom=a['id'],original_start_offset=0,
                       original_end_offset=len(a['content'])) for n,a in enumerate(raw)]
        source = SourceIndex(dict(atoms=raw))
        index = SourceIndex(dict(atoms=edited))
        items = [dict(atom=a['id'],text=a['content'],bbox=a['bbox']) for a in edited if a['kind']=='text']
        return source,index,items

    def test_complete_mixed_group_anchors_short_prose_across_bracketed_math(self):
        raw,index,items = self.mixed_fixture([
            ('text','2.1 Definition. A mapping '),('math','X'),
            ('text',' that assigns a real number '),('math','Y'),('text',' to each event.')])
        word = dict(text='mapping',score=.999,bbox=[30,20,50,30],
                    scores=dict(tiro=.3,tibo=.3,tiit=.9,tibi=.9))
        line = '2.1 Definition. A mapping X that assigns a real number Y to each'
        obs = [dict(text=line,score=.999,bbox=[10,20,80,30],words=[word])]
        mixed = mixed_fragments(index,raw,items)
        self.assertEqual(owned_ranges(obs,items,[100]*4,mixed=mixed),{})
        prepared = anchored_mixed_prose(obs,index,raw,items,mixed)
        runs = owned_ranges(obs,prepared,[100]*4,mixed=mixed)
        self.assertEqual([r['quote'] for r in runs['e0']],['mapping'])
        for defect in ('missing','reordered','role','moved','shortened','math_changed','excluded','duplicate','gray','changed_line'):
            altered=deepcopy(index.source); newobs=deepcopy(obs); gray=[]; excluded=set()
            if defect=='missing':altered['atoms']=altered['atoms'][:-1]
            if defect=='reordered':altered['atoms'][1:3]=altered['atoms'][1:3][::-1]
            if defect=='role':altered['atoms'][1]['role']='heading'
            if defect=='moved':
                for atom in altered['atoms']:atom['bbox']=[.06,.15,.91,.35]
            if defect=='shortened':
                atom=altered['atoms'][2];atom['content']=atom['content'][:-1];atom['original_end_offset']-=1
            if defect=='math_changed':altered['atoms'][1]['content']='Z'
            if defect=='excluded':excluded={'e1'}
            if defect=='duplicate':altered['atoms'] += [dict(a,id=a['id']+'copy',group='other') for a in altered['atoms']]
            if defect=='gray':gray=[dict(bbox=[.1,.2,.8,.3])]
            if defect=='changed_line':newobs[0]['text']=line.replace('that assigns','that changes')
            other=SourceIndex(altered)
            candidates=[dict(atom=a['id'],text=a['content'],bbox=a['bbox']) for a in other.atoms if a['kind']=='text']
            allowed=mixed_fragments(other,raw,candidates,excluded)
            prepared=anchored_mixed_prose(newobs,other,raw,candidates,allowed)
            self.assertEqual(owned_ranges(newobs,prepared,[100]*4,gray,allowed),{},defect)

    def test_same_group_repeated_word_needs_unique_literal_multiword_context(self):
        raw,index,items=self.mixed_fixture([
            ('text',' is strictly increasing if '),('math','X'),
            ('text',' is strictly decreasing if '),('math','Y')])
        word=dict(text='strictly',score=.999,bbox=[30,20,50,30],
                  scores=dict(tiro=.3,tibo=.94,tiit=.3,tibi=.94))
        obs=[dict(text='... is strictly increasing if X',score=.999,bbox=[10,20,80,30],words=[word])]
        mixed=mixed_fragments(index,raw,items)
        self.assertEqual(owned_ranges(obs,items,[100]*4,mixed=mixed),{})
        prepared=anchored_mixed_prose(obs,index,raw,items,mixed)
        self.assertEqual([r['quote'] for r in owned_ranges(obs,prepared,[100]*4,mixed=mixed)['e0']],['strictly'])
        for text in (' is strictly increasing if ', ' strictly '):
            # An overlapping atom outside the proven group always vetoes.
            extra=dict(atom='other',text=text,bbox=items[0]['bbox'])
            self.assertEqual(owned_ranges(obs,[*prepared,extra],[100]*4,mixed=mixed),{})
        repeated=deepcopy(raw.source)
        repeated['atoms'][2]['content']=repeated['atoms'][0]['content']
        revised=deepcopy(index.source)
        revised['atoms'][2].update(content=repeated['atoms'][0]['content'],
            original_end_offset=len(repeated['atoms'][0]['content']))
        raw2,index2=SourceIndex(repeated),SourceIndex(revised)
        newitems=[dict(i,text=index2.atoms[index2.ids[i['atom']]]['content']) for i in items]
        allowed=mixed_fragments(index2,raw2,newitems)
        self.assertEqual(owned_ranges(obs,anchored_mixed_prose(obs,index2,raw2,newitems,allowed),
                                     [100]*4,mixed=allowed),{})

    def test_short_wrapped_edge_needs_two_touching_styles_and_complete_unchanged_group(self):
        raw,index,items=self.mixed_fixture([
            ('text','The sequence '),('math','X'),('text',' is strictly descending when '),('math','Y')])
        scores=dict(tiro=.3,tibo=.94,tiit=.3,tibi=.4)
        words=[dict(text='strictly',score=.999,bbox=[10,20,40,30],scores=scores),
               dict(text='descending',score=.999,bbox=[40,20,70,30],scores=scores)]
        obs=[dict(text='strictly descending when Y',score=.999,bbox=[10,20,80,30],words=words)]
        def ranges(source,edited,observed):
            candidates=[dict(atom=a['id'],text=a['content'],bbox=a['bbox']) for a in edited.atoms if a['kind']=='text']
            mixed=mixed_fragments(edited,source,candidates)
            return owned_ranges(observed,anchored_mixed_prose(observed,edited,source,candidates,mixed),
                                [100]*4,mixed=mixed)
        self.assertEqual([r['quote'] for r in ranges(raw,index,obs)['e2']],['strictly','descending'])
        # Ordinary four-word context requirements stay unchanged.
        self.assertEqual(edge_prose_contexts(items[1]['text'],obs[0]['text']),[])
        for defect in ('one_word','confidence','line_confidence','detached','normal','different_style',
                       'short','changed_word','internal_match','missing_sibling','reordered','duplicate'):
            altered=deepcopy(index.source);observed=deepcopy(obs)
            if defect=='one_word':observed[0]['words']=observed[0]['words'][:1]
            if defect=='confidence':observed[0]['words'][1]['score']=.98
            if defect=='line_confidence':observed[0]['score']=.98
            if defect=='detached':observed[0]['words'][1]['bbox'][0]+=4
            if defect=='normal':observed[0]['words'][1]['scores']['tiro']=.95
            if defect=='different_style':observed[0]['words'][1]['scores']=dict(tiro=.3,tibo=.3,tiit=.95,tibi=.4)
            if defect=='short':observed[0]['text']='strictly descending Y'
            if defect=='changed_word':observed[0]['text']='strictly descending whenever Y'
            if defect=='internal_match':observed[0]['text']='noise strictly descending when noise'
            if defect=='missing_sibling':altered['atoms']=altered['atoms'][:-1]
            if defect=='reordered':altered['atoms'][2:]=altered['atoms'][2:][::-1]
            if defect=='duplicate':altered['atoms'] += [dict(a,id=a['id']+'copy',group='other') for a in altered['atoms']]
            self.assertEqual(ranges(raw,SourceIndex(altered),observed),{},defect)

    def test_mixed_caption_line_needs_literal_prose_on_both_sides_and_unique_alignment(self):
        group = [dict(atom='left', text='Objects in '), dict(math=True),
                 dict(atom='right', text=' may be represented as arrows or as points.')]
        line = 'Objects in X2 may be represented'
        contexts = caption_line_contexts(line, group)
        self.assertEqual([(c['atom'], c['source'], c['observed']) for c in contexts],
                         [('left', [0, 10], [0, 10]), ('right', [1, 19], [14, 32])])
        for c in contexts:
            atom = next(a for a in group if a.get('atom') == c['atom'])
            self.assertEqual(atom['text'][slice(*c['source'])], line[slice(*c['observed'])])
        for bad in ('Objects in X2 may be different', 'Objects in X2 may',
                    'Objects in X2 may be represente', 'Objects on X2 may be represented',
                    'in X2 may be represented'):
            self.assertEqual(caption_line_contexts(bad, group), [], bad)
        self.assertEqual(caption_line_contexts(line, group+group), [])
        self.assertEqual(caption_line_contexts(line, [group[0], group[2]]), [])

    def test_mixed_caption_styles_only_prose_with_original_group_and_geometry(self):
        raw, edited = self.caption_sources()
        def mixed(index):
            image, text = index.atoms
            atoms = [image, dict(text, content='Objects in ', length=11),
                     dict(text, id='math', kind='math', content='X^2', length=3),
                     dict(text, id='right', content=' may be represented as arrows or as points.', length=43)]
            return SourceIndex(dict(index.source, atoms=atoms))
        raw = mixed(raw)
        edited = SourceIndex(dict(edited.source, atoms=[dict(a, id='e-'+a['id'], original_atom=a['id'],
            original_start_offset=0, original_end_offset=a['length']) for a in raw.atoms]))
        candidates = caption_candidates(edited, raw)[0]
        scores = dict(tiro=.2,tibo=.3,tiit=.85,tibi=.7)
        words = [dict(text=w,score=.999,scores=scores,bbox=b) for w,b in
                 [('Objects',[250,410,350,435]),('in',[350,410,380,435]),
                  ('X2',[380,410,410,435]),('may',[410,410,450,435]),
                  ('be',[450,410,480,435]),('represented',[480,410,650,435])]]
        obs = [dict(text='Objects in X2 may be represented',score=.999,bbox=[250,410,750,435],words=words)]
        items = anchored_captions(obs,candidates,[1000]*4)
        runs = owned_ranges(obs,items,[1000]*4)
        self.assertEqual([r['quote'] for r in runs['e-caption']],['Objects','in'])
        self.assertEqual([r['quote'] for r in runs['e-right']],['may','be','represented'])
        self.assertNotIn('e-math',runs)
        duplicate = owned_ranges(obs,[*items,dict(items[0],atom='duplicate')],[1000]*4)
        self.assertNotIn('e-caption',duplicate)
        self.assertEqual(duplicate['e-right'],runs['e-right'])
        for defect in ('confidence','geometry','changed_prose'):
            altered=deepcopy(obs)
            if defect=='confidence':altered[0]['score']=.98
            if defect=='geometry':altered[0]['bbox'][1:4:2]=[300,325]
            if defect=='changed_prose':altered[0]['text']='Objects in X2 may be different'
            self.assertEqual(anchored_captions(altered,candidates,[1000]*4),[],defect)

        # A repeated preposition on opposite sides of math needs its own
        # touching word and its exact observed prose interval.
        group = [dict(atom='left',text='Pictures of '),dict(math=True),
                 dict(atom='right',text=' may be thought of as arrows.')]
        candidates = [dict(atom=a['atom'],text=a['text'],image_bbox=[.2,.1,.8,.4],prose_group=group)
                      for a in group if not a.get('math')]
        line = 'Pictures of X2 may be thought of'
        words = [dict(text=w,score=.999,scores=scores,bbox=[250+n*50,410,300+n*50,435])
                 for n,w in enumerate(line.split())]
        obs = [dict(text=line,score=.999,bbox=[250,410,750,435],words=words)]
        runs = owned_ranges(obs,anchored_captions(obs,candidates,[1000]*4),[1000]*4)
        self.assertEqual([r['quote'] for r in runs['left']],['Pictures','of'])
        self.assertEqual([r['quote'] for r in runs['right']],['may','be','thought','of'])

    def test_single_letter_still_needs_independent_ocr_shape_and_touching_prose(self):
        scores = dict(tiro=.2,tibo=.3,tiit=.85,tibi=.7)
        words = [dict(text='A',score=.999,scores=scores,bbox=[10,20,20,30]),
                 dict(text='vector',score=.999,scores=scores,bbox=[20,20,60,30])]
        self.assertEqual(supported_words(words,'A vector'),words)
        self.assertEqual(supported_words(words[:1],'A vector'),[])
        for defect in ('confidence','normal','gap','wrong_word'):
            altered=deepcopy(words)
            if defect=='confidence':altered[0]['score']=.98
            if defect=='normal':altered[0]['scores']=dict(scores,tiro=.9)
            if defect=='gap':altered[0]['bbox'][2]=15
            if defect=='wrong_word':altered[0]['text']='I'
            self.assertNotIn(altered[0],supported_words(altered,'A vector'),defect)

    def test_horizontal_completion_owns_clipped_strokes_and_vetoes_other_ink(self):
        rgb=np.full((80,180,3),255,np.uint8)
        rgb[25:50,40:70]=0
        rgb[25:50,90:123]=0
        box=[35,20,120,55]
        completed=complete_line_box(rgb,box)
        self.assertEqual(completed,[35,20,125,55])
        # The transformed operation must also recover the left edge.
        mirrored=rgb[:,::-1].copy()
        self.assertEqual(complete_line_box(mirrored,[60,20,145,55]),[55,20,145,55])
        # A separate component or one reaching the bounded search edge
        # invalidates the proposed extension; do not acquire nearby labels.
        neighbor=rgb.copy();neighbor[23:29,124:126]=0
        self.assertEqual(complete_line_box(neighbor,box),box)
        connected=rgb.copy();connected[25:50,120:130]=0
        self.assertEqual(complete_line_box(connected,box),box)
        # A wide line cannot turn the 20%-of-height search into column padding.
        distant=rgb.copy();distant[25:50,135:150]=0
        self.assertEqual(complete_line_box(distant,box),completed)
        # Faint source pixels can connect an italic tip that Otsu splits off.
        # The actual gray bridge is required; a detached black dot is vetoed.
        tip=rgb.copy();tip[25:50,120:123]=255
        tip[27:30,122:125]=0
        self.assertEqual(complete_line_box(tip,box),box)
        tip[27:30,120:122]=190
        self.assertEqual(complete_line_box(tip,box),[35,20,127,55])

    def caption_sources(self):
        box = [.2, .1, .8, .4]
        atoms = [dict(id='image', group='g', page=0, kind='image', role='image_body',
                      block_type='image', extraction_method='ocr', bbox=box,
                      asset_sha256='source-image', content=''),
                 dict(id='caption', group='g', page=0, kind='text', role='image_caption',
                      block_type='image', extraction_method='ocr', bbox=box,
                      content='Several vectors point in different directions.')]
        for atom in atoms:
            atom['length'] = len(atom['content']) if atom['kind'] == 'text' else 1
        raw = SourceIndex(dict(atoms=atoms))
        edited = SourceIndex(dict(atoms=[dict(a, id='e-'+a['id'], original_atom=a['id'],
            original_start_offset=0, original_end_offset=a['length']) for a in raw.atoms]))
        return raw, edited

    def caption_observation(self):
        return [dict(text='Several vectors point in different directions.', score=.999,
            bbox=[250, 410, 750, 435], words=[dict(text='vectors', score=.999,
                bbox=[340, 410, 430, 435], scores=dict(tiro=.3, tibo=.4, tiit=.85, tibi=.7))])]

    def test_caption_inherited_image_box_is_replaced_by_exact_observed_prose_bounds(self):
        raw, edited = self.caption_sources(); before = deepcopy(edited.source)
        candidates = caption_candidates(edited, raw)[0]
        obs = self.caption_observation()
        self.assertEqual(owned_ranges(obs, [dict(atom='e-caption', text=candidates[0]['text'],
            bbox=edited.atoms[0]['bbox'])], [1000]*4), {})
        items = anchored_captions(obs, candidates, [1000]*4)
        self.assertEqual(items[0]['bbox'], [.25, .41, .75, .435])
        self.assertEqual(owned_ranges(obs, items, [1000]*4)['e-caption'][0]['quote'], 'vectors')
        self.assertEqual(edited.source, before)
        # A duplicate owner remains ambiguous after locating the real line.
        duplicate = [*items, dict(items[0], atom='other')]
        self.assertEqual(owned_ranges(obs, duplicate, [1000]*4), {})

    def test_caption_requires_complete_unchanged_group_and_original_image(self):
        raw, edited = self.caption_sources()
        for change in (dict(content='Rewritten caption.'), dict(extraction_method='native'),
                       dict(recovery_issue='uncertain'), dict(writing_direction='vertical'),
                       dict(role='text'), dict(kind='math'), dict(bbox=[.3,.1,.8,.4]),
                       dict(source_style_runs=[dict(start=0,end=3)])):
            altered = SourceIndex(dict(edited.source, atoms=[edited.atoms[0], dict(edited.atoms[1], **change)]))
            self.assertEqual(caption_candidates(altered, raw), {}, change)
        for change in (dict(asset_sha256='different-image'), dict(page=1)):
            altered = SourceIndex(dict(edited.source, atoms=[dict(edited.atoms[0], **change), edited.atoms[1]]))
            self.assertEqual(caption_candidates(altered, raw), {})
        self.assertEqual(caption_candidates(edited, raw, {'e-image'}), {})
        self.assertEqual(caption_candidates(edited, raw, {'e-caption'}), {})
        translated = SourceIndex(dict(edited.source, translation={'method':'test'}))
        self.assertEqual(caption_candidates(translated, raw), {})
        # Even an unedited remaining caption cannot stand in for an omitted
        # mathematical sibling or a translated original.
        math = dict(raw.atoms[1], id='math', kind='math', content='x', length=1)
        complete = SourceIndex(dict(raw.source, atoms=[*raw.atoms, math]))
        self.assertEqual(caption_candidates(edited, complete), {})
        self.assertEqual(caption_candidates(edited, SourceIndex(dict(raw.source, translation=True))), {})

    def test_caption_geometry_context_and_two_readings_cannot_be_bypassed(self):
        raw, edited = self.caption_sources(); candidates = caption_candidates(edited, raw)[0]
        for box in ([250,300,750,325], [250,470,750,495], [0,410,500,435], [250,410,750,460]):
            obs = self.caption_observation(); obs[0]['bbox'] = box
            self.assertEqual(anchored_captions(obs,candidates,[1000]*4), [], box)
        for text, score in (('Several vectors point in wrong directions.', .999),
                            ('vectors', .999), ('Several vectors point in different directions.', .98)):
            obs=self.caption_observation(); obs[0].update(text=text, score=score)
            self.assertEqual(anchored_captions(obs,candidates,[1000]*4), [])
        for defect in ('word_confidence', 'normal_shape', 'gray_note'):
            obs=self.caption_observation(); gray=[]
            if defect=='word_confidence':obs[0]['words'][0]['score']=.98
            if defect=='normal_shape':obs[0]['words'][0]['scores']['tiro']=.9
            if defect=='gray_note':gray=[dict(bbox=[.2,.4,.8,.45])]
            items=anchored_captions(obs,candidates,[1000]*4)
            self.assertEqual(owned_ranges(obs,items,[1000]*4,gray), {}, defect)

    def test_caption_only_page_uses_observation_cache_and_revalidates_group(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);scan,_=self.sources(root)
            raw,edited=self.caption_sources()
            raw=SourceIndex(dict(raw.source,source_pdf=scan.source['source_pdf']))
            edited=SourceIndex(dict(edited.source,source_pdf=scan.source['source_pdf']))
            observed=self.caption_observation()
            for line in observed:
                line['bbox']=[v*417/1000 for v in line['bbox']]
                for word in line['words']:word['bbox']=[v*417/1000 for v in word['bbox']]
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'model1')), \
                 patch('book_pipeline.publishing.raster_body_styles.analyze_page',return_value=observed) as ocr, \
                 patch('book_pipeline.publishing.scan_font_calibration.calibrate_observations', \
                       return_value=(observed, dict(status='calibrated_metric_typography',family='computer_modern'))):
                styles,report=recover_body_styles(edited,raw,root/'render')
                self.assertEqual(styles['e-caption'][0]['quote'],'vectors')
                self.assertEqual(report['records'][0]['caption_anchors'][0]['atom'],'e-caption')
                self.assertEqual(recover_body_styles(edited,raw,root/'render')[0],styles)
                self.assertEqual(ocr.call_count,1)
                self.assertEqual(recover_body_styles(edited,raw,root/'render',{'e-image'})[0],{})

    def test_font_weight_and_slant_have_independent_negative_controls(self):
        for text in ('vector', 'different', 'Experience', 'classification'):
            for name, mask in templates(text).items():
                sample = cv2.resize(cv2.resize(mask, None, fx=.5, fy=.5), (mask.shape[1], mask.shape[0]))
                scores = font_scores(sample, text)
                with self.subTest(text=text, font=name, scores=scores):
                    # Low-resolution bold italic is too close to ordinary
                    # italic in these fixtures; retain slant and abstain on
                    # weight rather than weakening the weight margin.
                    self.assertEqual(is_bold(scores), name == 'tibo')
                    self.assertEqual(is_italic(scores), name in ('tiit', 'tibi'))
        for scores in ({}, dict(tiro=.4,tibo=.69,tiit=.3,tibi=.5),
                       dict(tiro=.6,tibo=.75,tiit=.4,tibi=.5)):
            self.assertFalse(is_bold(scores))

    def test_gray_retry_keeps_verified_binary_styles_and_requires_both_shape_views(self):
        strong = dict(tiro=.2, tibo=.3, tiit=.81, tibi=.6)
        bold = dict(tiro=.2, tibo=.85, tiit=.4, tibi=.6)
        weak = dict(tiro=.2, tibo=.3, tiit=.59, tibi=.57)
        binary = np.ones((20, 60), np.uint8)
        gray = binary*100
        module = 'book_pipeline.publishing.raster_body_styles.scored_word'
        for accepted in (strong, bold):
            with patch(module, return_value=('sample', accepted)) as score:
                self.assertEqual(scored_scan_word(binary, gray, 'sample'), ('sample', accepted))
                self.assertEqual(score.call_count, 1)
        with patch(module, side_effect=[('sample', weak), ('sample', strong)]):
            self.assertEqual(scored_scan_word(binary, gray, 'sample'), ('sample', strong))
        for base, soft in (({}, strong), (dict(weak, tiro=.58), strong),
                           (weak, dict(strong, tiit=.74)), (weak, dict(strong, tiro=.7))):
            with patch(module, side_effect=[('sample', base), ('sample', soft)]):
                self.assertEqual(scored_scan_word(binary, gray, 'sample'), ('sample', base))
        # A retry cannot substitute a different OCR word or bypass the
        # literal punctuation/letter ownership checked by the renderer.
        with patch(module, side_effect=[('sample', weak), ('other', strong)]):
            self.assertEqual(scored_scan_word(binary, gray, 'sample'), ('sample', weak))

    def test_low_resolution_gray_ink_retains_slant_without_styling_upright_controls(self):
        rescued = 0
        for text, size in ((text, size) for text in ('vector', 'matrix', 'learning', 'different')
                           for size in (10, 13, 16)):
            for name in ('tiro', 'tibo', 'tiit', 'tibi'):
                with pymupdf.open() as pdf:
                    page = pdf.new_page(width=180, height=45)
                    page.insert_text((15,30), text, fontname=name, fontsize=size)
                    pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
                    ink = 255-np.frombuffer(pix.samples, np.uint8).reshape(pix.height,pix.width)
                gray = cv2.resize(ink, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
                _, old = scored_word(binary,text)
                word, scores = scored_scan_word(binary,gray,text)
                with self.subTest(text=text,font=name,size=size):
                    self.assertEqual(word,text)
                    if is_italic(scores):self.assertIn(name, ('tiit','tibi'))
                    if is_italic(old) or is_bold(old):self.assertEqual(scores,old)
                    if name == 'tiro':self.assertFalse(is_bold(scores))
                    rescued += is_italic(scores) and not is_italic(old)
        self.assertGreater(rescued,0)

    def test_bold_requires_both_independent_readings_and_exact_ownership(self):
        obs=self.observation();items=self.items()
        obs[0]['words'][0]['scores']=dict(tiro=.3,tibo=.82,tiit=.4,tibi=.6)
        runs=owned_ranges(obs,items,[100]*4)['a']
        self.assertTrue(runs[0]['bold']);self.assertFalse(runs[0]['italic'])
        obs[0]['score']=.98
        self.assertEqual(owned_ranges(obs,items,[100]*4),{})
        obs[0]['score']=.999;obs[0]['words'][0]['score']=.98
        self.assertEqual(owned_ranges(obs,items,[100]*4),{})

    def test_local_context_cuts_only_blank_columns_and_needs_substantial_prose(self):
        mask=np.zeros((20,500),np.uint8)
        for x in range(5,495,20):mask[3:17,x:x+14]=255
        windows=context_windows(mask,[(200,240)])
        self.assertTrue(windows)
        for a,z in windows:
            self.assertTrue(a==0 or not mask[:,a].any())
            self.assertTrue(z==500 or not mask[:,z].any())
            self.assertLessEqual(a,200);self.assertGreaterEqual(z,240)
        for text in ('the additive inverse of','这些元素被称为向量vector'):
            self.assertTrue(enough_context(text))
        for text in ('vector','a b c d','向量vector','a firm'):
            self.assertFalse(enough_context(text))

    def test_low_confidence_line_needs_a_fresh_high_confidence_context(self):
        # Simulate a mixed math row. The independently read word is known,
        # while each context retry is a separate model call on source pixels.
        rgb=np.full((50,300,3),255,np.uint8)
        rgb[12:32,20:280]=0
        poly=[[10,10],[290,10],[290,35],[10,35]]
        class Model:
            def __init__(self, confidence):self.confidence=confidence;self.calls=0
            def ocr(self, pixels, det=True):
                if det:return [[(poly,('For x, the additive inverse of x',.97))]]
                self.calls+=1
                if self.calls==1:return [[('additive',.999)]]
                return [[('the additive inverse of',self.confidence)]]
        scores=dict(tiro=.2,tibo=.3,tiit=.85,tibi=.65)
        with patch('book_pipeline.publishing.raster_body_styles.word_boxes',return_value=[(80,180)]), \
             patch('book_pipeline.publishing.raster_body_styles.scored_word',return_value=('additive',scores)), \
             patch('book_pipeline.publishing.raster_body_styles.context_windows',return_value=[(20,250)]):
            self.assertEqual(analyze_page(rgb,Model(.98)),[])
            rows=analyze_page(rgb,Model(.999))
            self.assertEqual(len(rows),2)  # Full strip and a bounded local retry.
            self.assertNotEqual(rows[0]['bbox'],rows[1]['bbox'])
            self.assertTrue(all(r['text']=='the additive inverse of' and r['score']==.999 for r in rows))

    def test_print_ligatures_and_tracking_keep_weight_separate_from_slant(self):
        # Independent raster fixture: print Unicode ligatures with tracking,
        # then downsample. Upright and bold controls use identical treatment.
        for word in ('affiliation', 'office', 'flying', 'different'):
            printed = word.replace('ffi', 'ﬃ').replace('ff', 'ﬀ').replace('fi', 'ﬁ').replace('fl', 'ﬂ')
            for name in ('tiro', 'tibo', 'tiit', 'tibi'):
                font = pymupdf.Font(name)
                with pymupdf.open() as pdf:
                    page = pdf.new_page(width=400, height=100)
                    writer = pymupdf.TextWriter(page.rect)
                    x = 20
                    for char in printed:
                        writer.append((x, 70), char, font=font, fontsize=48)
                        x += font.text_length(char, fontsize=48)+1.7
                    writer.write_text(page)
                    pix = page.get_pixmap(colorspace=pymupdf.csGRAY)
                    mask = 255-np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
                mask = cv2.resize(cv2.resize(mask, None, fx=.5, fy=.5), (400, 100))
                scores = font_scores(mask, word)
                with self.subTest(word=word, font=name, scores=scores):
                    self.assertEqual(is_italic(scores), name in ('tiit', 'tibi'))

    def test_clipped_descender_is_completed_without_admitting_neighboring_ink(self):
        rgb = np.full((80, 100, 3), 255, np.uint8)
        rgb[22:48, 20:25] = 0
        rgb[25:43, 40:55] = 0
        box = [10, 20, 80, 45]
        self.assertEqual(complete_line_box(rgb, box), [10, 20, 80, 50])
        # A separate next-line glyph in the added strip vetoes recovery.
        neighbor = rgb.copy(); neighbor[46:55, 60:65] = 0
        self.assertEqual(complete_line_box(neighbor, box), box)
        # Reaching the bounded search edge leaves completion uncertain.
        clipped = rgb.copy(); clipped[45:60, 20:25] = 0
        self.assertEqual(complete_line_box(clipped, box), box)
        self.assertEqual(complete_line_box(np.full_like(rgb, 255), box), box)
        # A word at the image edge must also remain safe.
        self.assertEqual(complete_line_box(rgb[:45], box), box)

    def test_literal_line_matches_only_whitespace_and_unique_occurrences(self):
        self.assertEqual(literal_span('Before. A classification task. After.',
                                      'Aclassification task.'), (8,30))
        for text, quote in [('A task. A task.', 'A task.'), ('A task.', 'A Task.'),
                            ('A task.', ''), ('A task.', 'B task.')]:
            self.assertIsNone(literal_span(text,quote))

    def test_shape_comparison_separates_italic_from_weight_and_normal_words(self):
        # Downsample and blur simulate a 150 dpi source enlarged for CPU OCR.
        for text in ['classification', 'which', 'vector', 'and', 'Experience']:
            for name, image in templates(text).items():
                h,w=image.shape
                reduced=cv2.resize(image,(max(8,w//2),max(8,h//2)),interpolation=cv2.INTER_AREA)
                sample=cv2.resize(reduced,(w,h),interpolation=cv2.INTER_LINEAR)
                scores=font_scores(sample,text)
                with self.subTest(text=text,font=name,scores=scores):
                    if name in ('tiit','tibi'):
                        self.assertTrue(is_italic(scores))
                    else:
                        self.assertFalse(is_italic(scores))
        self.assertFalse(is_italic({}))
        self.assertEqual(font_scores(np.zeros((20,40),np.uint8),'plain'),{})

    def observation(self):
        return [dict(text='A classification task.',score=.999,bbox=[10,20,80,30],words=[
            dict(text='classification',score=.999,bbox=[20,20,65,30],
                 scores=dict(tiro=.3,tibo=.4,tiit=.85,tibi=.8))])]

    def test_detached_punctuation_does_not_override_letters_or_remove_letter_ink(self):
        letters = templates('cross-entropy')['tiit']
        h, w = letters.shape
        mask = np.zeros((h, w+20), np.uint8)
        mask[:, :w] = letters
        mask[-4:, -4:] = 255
        ink, text = prose_ink(mask, 'cross-entropy,')
        self.assertEqual(text, 'cross-entropy')
        self.assertTrue(np.array_equal(ink[:, :w], letters))
        self.assertTrue(is_italic(font_scores(ink, text)))
        for mark in ('.', ',', ';', ':', '!', '?'):
            self.assertEqual(prose_ink(mask, 'cross-entropy'+mark)[1], 'cross-entropy')
        for text in ('cross-entropy', "words'", 'word-'):
            self.assertEqual(prose_ink(mask, text)[1], text)
        # Full-height final components, including the last letter, stay intact.
        self.assertEqual(prose_ink(letters, 'cross-entropy.')[1], 'cross-entropy.')
        mask[:, -4:] = 255
        self.assertEqual(prose_ink(mask, 'cross-entropy,')[1], 'cross-entropy,')

    def test_narrow_blank_split_and_adjacent_shape_support_survive_ownership(self):
        mask = np.zeros((50, 70), np.uint8)
        mask[:, :30] = 255; mask[:, 35:] = 255
        self.assertEqual(len(word_boxes(mask)), 1)
        self.assertEqual(len(word_boxes(mask, .10)), 2)
        # A 4px interword space in a taller row needs the finer partition.
        tall = np.zeros((60, 70), np.uint8)
        tall[:, :30] = 255; tall[:, 34:] = 255
        self.assertEqual(len(word_boxes(tall, .10)), 1)
        self.assertEqual(len(word_boxes(tall, .06)), 2)
        words = [dict(text='which', score=.999, bbox=[10,20,40,30],
                      scores=dict(tiro=.3,tibo=.33,tiit=.615,tibi=.61)),
                 dict(text='one', score=.999, bbox=[40,20,80,30],
                      scores=dict(tiro=.4,tibo=.42,tiit=.84,tibi=.78))]
        self.assertFalse(is_italic(words[0]['scores']))
        self.assertEqual(supported_words(words, 'which one?'), words)
        self.assertEqual(supported_words(supported_words(words, 'which one?'), 'which one?'), words)
        obs = [dict(text='which one?', score=.999, bbox=[10,20,80,30], words=words)]
        items = [dict(atom='a',text='Some kinds of which one? problems.',bbox=[.05,.15,.9,.35])]
        runs = owned_ranges(obs,items,[100]*4)
        self.assertEqual([r['quote'] for r in runs['a']], ['which','one'])
        # Independent confidence, shape, adjacency and literal order remain mandatory.
        for defect in ('normal','confidence','gap','text','unknown','too_weak'):
            modified=deepcopy(words);line='which one?'
            if defect=='normal':modified[0]['scores']['tiro']=.8
            if defect=='confidence':modified[0]['score']=.98
            if defect=='gap':modified[0]['bbox'][2]=35
            if defect=='text':line='one which?'
            if defect=='unknown':modified[0]['scores']={}
            if defect=='too_weak':modified[0]['scores']['tiit']=.54;modified[0]['scores']['tibi']=.54
            self.assertNotIn(modified[0],supported_words(modified,line),defect)

    def test_repeated_short_word_requires_exact_unique_adjacent_italic_context(self):
        scores = dict(tiro=.3, tibo=.4, tiit=.85, tibi=.8)
        words = [dict(text='Golden', score=.999, bbox=[30,20,50,30], scores=scores),
                 dict(text='of', score=.999, bbox=[50,20,60,30], scores=scores),
                 dict(text='Valley', score=.999, bbox=[60,20,80,30], scores=scores)]
        text = 'An image of the Golden of Valley appears.'
        observation = [dict(text=text, score=.999, bbox=[10,20,90,30], words=words)]
        items = [dict(atom='a',text=text,bbox=[.05,.15,.95,.35])]
        runs = owned_ranges(observation,items,[100]*4)['a']
        self.assertEqual([r['quote'] for r in runs], ['Golden','of','Valley'])
        self.assertEqual(runs[1]['start'], text.index('of Valley'))
        for defect in ('upright', 'confidence', 'gap', 'repeated_phrase', 'conflicting_anchors', 'no_anchor'):
            modified=deepcopy(words);source=text
            if defect=='upright':
                for n in (0,2): modified[n]['scores']['tiro']=.9
            if defect=='confidence':
                for n in (0,2): modified[n]['score']=.98
            if defect=='gap':modified[0]['bbox'][2]=45;modified[2]['bbox'][0]=65
            if defect=='repeated_phrase':source += ' Golden of Valley'
            if defect=='conflicting_anchors':source='Golden of something and another of Valley'
            if defect=='no_anchor':modified=modified[1:2]
            self.assertIsNone(anchored_word_span(source,modified[0] if defect=='no_anchor' else modified[1],modified),defect)
        # Context cannot authorize a substring of an ordinary token.
        self.assertIsNone(anchored_word_span('Golden office Valley',words[1],words))

    def test_repeated_reading_survives_collection_and_requires_styled_neighbor_at_render(self):
        text = 'a probability distribution or a probability measure'
        words = [dict(text='probability',score=.999,bbox=[20,10,70,30],scores={}),
                 dict(text='distribution',score=.999,bbox=[70,10,130,30],scores={})]
        self.assertEqual(anchored_word_span(text,words[0],words,require_style=False), (2,13))
        self.assertIsNone(anchored_word_span(text,words[0],words))
        bold=dict(tiro=.3,tibo=.94,tiit=.3,tibi=.4)
        styled=[dict(w,scores=bold) for w in words]
        obs=[dict(text=text,score=.999,bbox=[5,5,190,35],words=styled)]
        item=dict(atom='a',text=text,bbox=[0,0,1,1])
        ranges=owned_ranges(obs,[item],[200,40]*2)['a']
        self.assertEqual([(r['quote'],r['start']) for r in ranges],
                         [('probability',2),('distribution',14)])
        for defect in ('low_confidence','detached','ambiguous','conflicting','short_anchor'):
            bad=deepcopy(words);line=text
            if defect=='low_confidence':bad[1]['score']=.98
            if defect=='detached':bad[1]['bbox'][0]=74
            if defect=='ambiguous':line += ' probability distribution'
            if defect=='conflicting':
                line='measure probability and probability distribution'
                bad.append(dict(text='measure',score=.999,bbox=[5,10,20,30],scores={}))
            if defect=='short_anchor':
                line='probability of or probability of';bad[1]['text']='of'
            self.assertIsNone(anchored_word_span(line,bad[0],bad,require_style=False),defect)

        rgb=np.full((50,200,3),255,np.uint8)
        rgb[12:28,20:130]=0
        poly=[[0,5],[195,5],[195,35],[0,35]]
        class Model:
            def __init__(self):self.n=0
            def ocr(self,pixels,det=True):
                if det:return [[(poly,(text,.999))]]
                token=words[self.n%2]['text'];self.n+=1
                return [[(token,.999)]]
        module='book_pipeline.publishing.raster_body_styles.'
        with patch(module+'word_boxes',return_value=[(20,70),(70,130)]), \
             patch(module+'complete_vertical_box',return_value=[0,3,197,37]), \
             patch(module+'complete_line_box',return_value=[0,3,197,37]), \
             patch(module+'completed_word_observation',return_value=None):
            collected=analyze_page(rgb,Model(),calibration=True)
        self.assertTrue(collected)
        self.assertTrue(all('probability' in [w['text'] for w in row['words']] for row in collected))

    def test_finer_partitions_preserve_independently_confirmed_punctuation(self):
        scores=dict(tiro=.3,tibo=.4,tiit=.85,tibi=.8)
        short=dict(text='which',score=.999,bbox=[20,20,60,30],scores=scores)
        complete=dict(short,text='which?',bbox=[20,20,65,30])
        item=dict(atom='a',text='Choose which?',bbox=[.05,.15,.9,.35])
        for words in ([short,complete],[complete,short]):
            # The candidates may also arrive from separate detector strips.
            observations=[dict(text=item['text'],score=.999,bbox=[10,20,80,30],words=[w]) for w in words]
            runs=owned_ranges(observations,[item],[100]*4)['a']
            self.assertEqual([r['quote'] for r in runs],['which?'])
        # A longer word with insufficient independent evidence cannot expand
        # an accepted range merely because the shorter observation passed.
        weak=deepcopy(complete);weak['scores']={**scores,'tiro':.9}
        observation=[dict(text=item['text'],score=.999,bbox=[10,20,80,30],words=[short,weak])]
        self.assertEqual(owned_ranges(observation,[item],[100]*4)['a'][0]['quote'],'which')

    def test_punctuation_fallback_preserves_an_existing_reliable_match(self):
        good = dict(tiro=.3,tibo=.4,tiit=.8,tibi=.7)
        weak = dict(tiro=.3,tibo=.4,tiit=.55,tibi=.6)
        mask = np.ones((20,80),np.uint8)
        with patch('book_pipeline.publishing.raster_body_styles.font_scores',return_value=good) as score, \
             patch('book_pipeline.publishing.raster_body_styles.prose_ink') as strip:
            self.assertEqual(scored_word(mask,'classification,'),('classification,',good))
            strip.assert_not_called();self.assertEqual(score.call_count,1)
        for fallback, expected in ((good,'cross-entropy'),(weak,'cross-entropy,')):
            with patch('book_pipeline.publishing.raster_body_styles.font_scores',side_effect=[weak,fallback]), \
                 patch('book_pipeline.publishing.raster_body_styles.prose_ink',return_value=(mask,'cross-entropy')):
                text, scores = scored_word(mask,'cross-entropy,')
                self.assertEqual(text,expected)
                self.assertEqual(scores,fallback if expected=='cross-entropy' else weak)

    def test_upright_closing_bracket_has_independent_shape_and_remains_unstyled(self):
        letters = templates('class')['tiit']
        for bracket in (')', ']'):
            mark = templates(bracket)['tiro']
            h=max(letters.shape[0],mark.shape[0]);w=letters.shape[1]
            mask=np.zeros((h,w+8+mark.shape[1]),np.uint8)
            mask[-letters.shape[0]:,:w]=letters
            mask[:mark.shape[0],w+8:]=mark
            ink,text=bracket_ink(mask,'class'+bracket)
            self.assertEqual(text,'class')
            self.assertTrue(is_italic(font_scores(ink,text)))
            self.assertEqual(scored_word(mask,'class'+bracket)[0],'class')
        # Neither a connected mark nor an ordinary last letter can be cut.
        mask[:,w:w+8]=255
        self.assertEqual(bracket_ink(mask,'class]')[1],'class]')
        wrong=templates('class')['tiit']
        self.assertEqual(bracket_ink(wrong,'class)')[1],'class)')
        self.assertEqual(scored_word(wrong,'class)')[1],{})

    def items(self):
        return [dict(atom='a',text='Before. A classification task. After.',bbox=[.05,.15,.9,.35])]

    def test_complete_prose_fragment_can_own_emphasis_across_math_without_styling_math(self):
        obs = self.observation()
        obs[0]['text'] = 'A classification task with variable x and a value in R.'
        items = [dict(atom='a', text='A classification task with variable ', bbox=[.05,.15,.9,.35]),
                 dict(atom='b', text=' and a value in ', bbox=[.05,.15,.9,.35])]
        mixed = dict(a=['x', r'\mathbf{R}'], b=['x', r'\mathbf{R}'])
        self.assertEqual(owned_ranges(obs, items, [100]*4), {})
        runs = owned_ranges(obs, items, [100]*4, mixed=mixed)
        self.assertEqual(runs['a'][0]['quote'], 'classification')
        for defect in ('repeated_line', 'duplicate_owner', 'duplicate_ineligible', 'math_token',
                       'changed_context', 'short_context', 'word_fragment', 'gray', 'ambiguous_word'):
            modified=deepcopy(items); altered=deepcopy(obs); allowed=deepcopy(mixed); gray=[]
            if defect=='repeated_line':altered[0]['text'] += ' classification'
            if defect in ('duplicate_owner', 'duplicate_ineligible'):
                modified.append(dict(modified[0],atom='duplicate'))
                if defect=='duplicate_owner':allowed['duplicate']=allowed['a']
            if defect=='math_token':allowed['a']=[r'\text{classification}']
            if defect=='changed_context':modified[0]['text']='A classification error with variable '
            if defect=='short_context':modified[0]['text']='classification '
            if defect=='word_fragment':altered[0]['text']='B'+altered[0]['text']
            if defect=='gray':gray=[dict(bbox=[.1,.2,.8,.3])]
            if defect=='ambiguous_word':modified[1]['text']=' classification '
            self.assertEqual(owned_ranges(altered,modified,[100]*4,gray,allowed),{},defect)

    def test_mixed_context_requires_unchanged_supported_siblings(self):
        with tempfile.TemporaryDirectory() as name:
            raw, edited = self.sources(Path(name))
            math=dict(edited.atoms[0],id='math',original_atom='rawmath',kind='math',content='x',length=1,
                      original_start_offset=0,original_end_offset=1)
            source_math=dict(math,id='rawmath')
            raw=SourceIndex(dict(raw.source,atoms=[*raw.atoms,source_math]))
            edited=SourceIndex(dict(edited.source,atoms=[*edited.atoms,math]))
            self.assertEqual(mixed_fragments(edited,raw,self.items()),{'a':['x']})
            self.assertEqual(mixed_fragments(edited,raw,self.items(),{'math'}),{})
            for change in [dict(content='y'),dict(kind='image'),dict(recovery_issue='uncertain'),
                           dict(bbox=[.1,.2,.3,.4]),dict(page=1),dict(writing_direction='vertical')]:
                altered=SourceIndex(dict(edited.source,atoms=[edited.atoms[0],dict(math,**change)]))
                self.assertEqual(mixed_fragments(altered,raw,self.items()),{},change)

    def test_line_word_geometry_and_unique_atom_all_must_agree(self):
        items=self.items();obs=self.observation()
        runs=owned_ranges(obs,items,[100]*4)
        self.assertEqual(runs['a'][0]['quote'],'classification')
        self.assertEqual(runs['a'][0]['start'],10)
        # Rounded source boxes may cut a few pixels off the observed first
        # line. Exact unique bytes authorize bounded coordinate tolerance.
        rounded=deepcopy(items);rounded[0]['bbox']=[.1,.212,.8,.3]
        self.assertTrue(owned_ranges(obs,rounded,[100]*4))
        for defect in ('repeated','wrong_line','wrong_word','word_confidence','line_confidence',
                       'gray','duplicate_owner','bbox','normal_font'):
            with self.subTest(defect=defect):
                altered=deepcopy(obs);modified=deepcopy(items);gray=[]
                if defect=='repeated':modified[0]['text']*=2
                if defect=='wrong_line':altered[0]['text']='A Classification task.'
                if defect=='wrong_word':altered[0]['words'][0]['text']='class'
                if defect=='word_confidence':altered[0]['words'][0]['score']=.98
                if defect=='line_confidence':altered[0]['score']=.98
                if defect=='gray':gray=[dict(bbox=[.1,.2,.8,.3])]
                if defect=='duplicate_owner':modified.append({**modified[0],'atom':'b'})
                if defect=='bbox':modified[0]['bbox']=[.8,.2,.9,.3]
                if defect=='normal_font':altered[0]['words'][0]['scores']['tiro']=.9
                self.assertEqual(owned_ranges(altered,modified,[100]*4,gray),{})

    def sources(self,root):
        path=root/'scan.pdf'
        with pymupdf.open() as original,pymupdf.open() as scan:
            p=original.new_page(width=100,height=100)
            p.insert_text((10,28),'A classification task.',fontname='tiit',fontsize=7)
            q=scan.new_page(width=100,height=100)
            q.insert_image(q.rect,stream=p.get_pixmap(dpi=150).tobytes('png'))
            scan.save(path)
        atom=dict(id='raw',kind='text',group='g',page=0,role='text',block_type='text',
                  extraction_method='ocr',content=self.items()[0]['text'],bbox=[.05,.15,.9,.35])
        atom['length']=len(atom['content'])
        original=SourceIndex(dict(source_pdf=str(path),atoms=[atom]))
        edited=SourceIndex(dict(source_pdf=str(path),atoms=[dict(atom,id='a',original_atom='raw',
            original_start_offset=0,original_end_offset=len(atom['content']))]))
        return original,edited

    def test_pixel_model_cache_and_rewritten_native_excluded_text_protection(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);raw,edited=self.sources(root);before=deepcopy(edited.source)
            observed=self.observation()
            # Observed boxes here are normalized to a 100px fixture above.
            for row in observed:
                row['bbox']=[v*417/100 for v in row['bbox']]
                for word in row['words']:word['bbox']=[v*417/100 for v in word['bbox']]
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'model1')), \
                 patch('book_pipeline.publishing.raster_body_styles.analyze_page',return_value=observed) as ocr, \
                 patch('book_pipeline.publishing.scan_font_calibration.calibrate_observations', \
                       return_value=(observed, dict(status='calibrated_metric_typography',family='computer_modern'))):
                a,report=recover_body_styles(edited,raw,root/'render')
                b,_=recover_body_styles(edited,raw,root/'render')
                self.assertEqual(a,b);self.assertTrue(a);self.assertEqual(ocr.call_count,1)
                self.assertEqual(report['status'],'requires_visual_review')
                renderer=Renderer(edited,dict(source_styles=a,corrections=[],joins=[],
                    exclusions=[],bold_spans=[]),[],[],root)
                tex=renderer.source_text_fragment(edited.atoms[0],0,len(edited.atoms[0]['content']))
                self.assertIn(r'\itshape classification',tex)
                self.assertEqual(renderer.source_order_layout[0]['observed_bbox'],a['a'][0]['observed_bbox'])
                with patch('book_pipeline.publishing.raster_body_styles.OBSERVATION_METHOD','new-observations'):
                    refreshed,_=recover_body_styles(edited,raw,root/'render')
                    self.assertEqual(refreshed,a)
                    self.assertEqual(ocr.call_count,1)
                with patch('book_pipeline.publishing.raster_body_styles.CALIBRATION_OBSERVATION_METHOD','new-calibration-observations'):
                    refreshed,_=recover_body_styles(edited,raw,root/'render')
                    self.assertEqual(refreshed,a)
                    self.assertEqual(ocr.call_count,2)
                self.assertEqual(recover_body_styles(edited,raw,root/'render',{'a'})[0],{})
                for change in [dict(content='Rewritten.'),dict(extraction_method='native'),dict(kind='math')]:
                    changed=SourceIndex({**edited.source,'atoms':[{**edited.atoms[0],**change}]})
                    self.assertEqual(recover_body_styles(changed,raw,root/'render')[0],{})
            self.assertEqual(edited.source,before)
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'model2')), \
                 patch('book_pipeline.publishing.raster_body_styles.analyze_page',return_value=observed) as ocr, \
                 patch('book_pipeline.publishing.scan_font_calibration.calibrate_observations', \
                       return_value=(observed, dict(status='calibrated_metric_typography',family='computer_modern'))):
                recover_body_styles(edited,raw,root/'render');self.assertEqual(ocr.call_count,1)

    def test_times_cache_survives_calibration_only_observation_changes(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);raw,edited=self.sources(root)
            observed=self.observation()
            for line in observed:
                line['bbox']=[v*417/100 for v in line['bbox']]
                for word in line['words']:word['bbox']=[v*417/100 for v in word['bbox']]
            module='book_pipeline.publishing.raster_body_styles.'
            with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'model1')), \
                 patch(module+'analyze_page',return_value=observed) as ocr, \
                 patch(module+'competing_times_observations',return_value=observed), \
                 patch('book_pipeline.publishing.scan_font_calibration.calibrate_observations',
                       return_value=([],dict(status='calibrated_font_family',family='times'))):
                baseline,_=recover_body_styles(edited,raw,root/'render')
                self.assertTrue(baseline);self.assertEqual(ocr.call_count,2)
                with patch(module+'CALIBRATION_OBSERVATION_METHOD','new-calibration'):
                    self.assertEqual(recover_body_styles(edited,raw,root/'render')[0],baseline)
                    self.assertEqual(ocr.call_count,3)
                    self.assertTrue(ocr.call_args.kwargs['calibration'])
                with patch(module+'OBSERVATION_METHOD','new-times'):
                    self.assertEqual(recover_body_styles(edited,raw,root/'render')[0],baseline)
                    self.assertEqual(ocr.call_count,4)
                    self.assertFalse(ocr.call_args.kwargs.get('calibration',False))

    def test_scan_calibration_is_independent_of_hidden_ocr_and_visible_text_stays_native(self):
        for render_mode in (None,3,0):
            with self.subTest(render_mode=render_mode),tempfile.TemporaryDirectory() as name:
                root=Path(name);raw,edited=self.sources(root)
                with pymupdf.open(raw.source['source_pdf']) as pdf:
                    if render_mode is not None:
                        pdf[0].insert_text((10,70),'Archive text',render_mode=render_mode,fontsize=7)
                        pdf.saveIncr()
                observed=self.observation()
                for row in observed:
                    row['bbox']=[v*417/100 for v in row['bbox']]
                    for word in row['words']:word['bbox']=[v*417/100 for v in word['bbox']]
                with patch('book_pipeline.publishing.image_search.local_ocr',return_value=(object(),'model1')), \
                     patch('book_pipeline.publishing.raster_body_styles.analyze_page',return_value=observed) as ocr:
                    result,report=recover_body_styles(edited,raw,root/'render')
                    self.assertEqual(result,{})
                    self.assertEqual(ocr.call_count,0 if render_mode==0 else 2)
                    self.assertEqual(report['status'],'not_needed' if render_mode==0 else 'requires_scan_font_calibration')
                    if render_mode!=0:
                        self.assertTrue(ocr.call_args_list[0].kwargs['calibration'])
                        self.assertEqual(report['records'][0]['font_calibration']['status'],
                                         'abstained_insufficient_font_evidence')
