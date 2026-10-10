import unittest
from copy import deepcopy
from pathlib import Path
import tempfile

from book_pipeline.ingestion.formula_bands import METHOD, apply_formula_corrections, detect_bands, normalized_math
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.publishing.image_rows import detect_image_rows


def atom(identifier, kind, box, text='', block_type=None):
    return dict(id=identifier, group=identifier, page=0, kind=kind,
                role='text', block_type=block_type or ('equation' if kind == 'display_math' else kind),
                bbox=box, block_bbox=box, content=text, length=max(1, len(text)))


def source(atoms):
    return SourceIndex(dict(atoms=atoms, pages=[dict(page=0, atoms=[a['id'] for a in atoms], size_pt=[500, 700])],
                            input_sha256='pdf', mineru_sha256='ocr', source_pdf='source.pdf'))


class FormulaBandTests(unittest.TestCase):
    def test_preserved_caption_is_not_duplicated_by_model_transcription(self):
        picture = atom('picture', 'image', [.1, .2, .3, .4])
        caption = atom('caption', 'text', [.1, .4, .3, .42], '图16')
        caption.update(group='picture', role='image_caption')
        index = source([picture, caption, atom('eq', 'display_math', [.4, .3, .7, .5], 'x=1'),
                        atom('word', 'text', [.8, .35, .9, .4], '其中')])
        band = detect_bands(index)[0]
        report = dict(method=METHOD, status='complete', source=index.source, usage={},
                      records=[dict(**band, answer=dict(items=[dict(kind='preserve', group='picture'),
                          dict(kind='text', text='图 16'), dict(kind='formula', latex='x=1', number='')]))])
        corrected = apply_formula_corrections(index, report)
        self.assertEqual([a['content'] for a in corrected.atoms if a['kind']=='text'], ['图16'])
        self.assertEqual(len(corrected.source['formula_corrections']['duplicate_preserved_captions']), 1)

    def test_condition_is_not_accepted_as_an_equation_number(self):
        from book_pipeline.ingestion.formula_bands import validate_answer
        answer = dict(items=[dict(kind='formula', latex='x=a', number='0<θ_i<1')])
        with self.assertRaisesRegex(ValueError, 'condition is not an equation number'):
            validate_answer(answer, dict(preserve=[]))

    def test_source_operator_spelling_survives_tex_compatibility(self):
        from book_pipeline.ingestion.math_syntax import math_compatibility
        tex, evidence = math_compatibility(r'\sh x + \ch x + \arctg x')
        self.assertEqual(tex, r'\operatorname{sh} x + \operatorname{ch} x + \operatorname{arctg} x')
        self.assertEqual(len(evidence), 3)

    def test_json_overescaping_keeps_tex_row_breaks(self):
        tex = r'\begin{aligned}\alpha &= 1 \\ x &= 2\end{aligned}'
        self.assertEqual(normalized_math(tex), tex)
        self.assertEqual(normalized_math(tex.replace('\\', '\\\\')), tex)

    def test_full_width_band_catches_left_words_and_right_number(self):
        index = source([atom('eq', 'display_math', [.4, .2, .7, .6], 'x=1'),
                        atom('left', 'text', [.1, .35, .2, .4], '解得'),
                        atom('right', 'text', [.85, .4, .94, .43], '(3.32)'),
                        atom('above', 'text', [.4, .1, .7, .2], '上文')])
        bands = detect_bands(index)
        self.assertEqual(len(bands), 1)
        self.assertEqual(bands[0]['bbox'], [0, .2, 1, .6])
        self.assertEqual(set(bands[0]['groups']), {'eq', 'left', 'right'})

    def test_equal_x_ranges_on_different_rows_do_not_trigger(self):
        index = source([atom('eq', 'display_math', [.2, .2, .8, .4], 'x=1'),
                        atom('below', 'text', [.2, .5, .8, .6], '下文')])
        self.assertEqual(detect_bands(index), [])

    def test_intersecting_bands_merge_and_preserve_heading_and_image(self):
        index = source([atom('heading', 'text', [.1, .1, .3, .25], '标题'),
                        atom('eq1', 'display_math', [.4, .2, .7, .4], 'x=1'),
                        atom('word', 'text', [.1, .3, .3, .55], '解得'),
                        atom('eq2', 'display_math', [.4, .5, .7, .6], 'y=2'),
                        atom('picture', 'image', [.8, .3, .9, .4])])
        bands = detect_bands(index, ['heading'])
        self.assertEqual(len(bands), 1)
        self.assertEqual(set(bands[0]['triggers']), {'eq1', 'eq2'})
        self.assertEqual({p['group'] for p in bands[0]['preserve']}, {'heading', 'picture'})

    def test_replacement_keeps_heading_address_and_creates_three_equations(self):
        index = source([atom('heading', 'text', [.1, .1, .3, .2], '标题'),
                        atom('eq', 'display_math', [.4, .3, .8, .7], r'x=1\tag{3.31}'),
                        atom('word', 'text', [.1, .4, .2, .45], '解得'),
                        atom('number', 'text', [.85, .6, .95, .65], '(3.32)')])
        before = deepcopy(index.source)
        band = detect_bands(index)[0]
        answer = dict(items=[dict(kind='formula', latex='x=1', number='3.31'),
                             dict(kind='text', text='解得'),
                             dict(kind='formula', latex='y=2', number='3.32'),
                             dict(kind='text', text='或'),
                             dict(kind='formula', latex='z=3', number='3.33')])
        report = dict(method=METHOD, status='complete', source=index.source,
                      records=[dict(**band, answer=answer)], usage={})
        corrected = apply_formula_corrections(index, report)
        self.assertEqual(index.source, before)
        self.assertEqual(corrected.atoms[0]['id'], 'heading')
        self.assertEqual(sum(a['kind'] == 'display_math' for a in corrected.atoms), 3)
        self.assertNotEqual(index.source['mineru_sha256'], corrected.source['mineru_sha256'])
        self.assertEqual(len(corrected.ids), len(corrected.atoms))
        self.assertEqual(corrected.source['pages'][0]['atoms'], [a['id'] for a in corrected.atoms])
        with self.assertRaisesRegex(ValueError, 'different PDF'):
            apply_formula_corrections(corrected, report)


class ImageRowTests(unittest.TestCase):
    def test_shared_caption_below_inherited_boxes_is_included(self):
        import pymupdf
        from book_pipeline.publishing.image_rows import recover_horizontal_images
        images = [atom('left', 'image', [.1, .2, .3, .3]), atom('right', 'image', [.5, .2, .7, .3])]
        label = atom('label', 'text', images[1]['bbox'], '图35')
        label.update(group='right', role='image_caption')
        prose = atom('prose', 'text', [.1, .4, .9, .5], '这段正文在图注的下面，必须留在图片区域以外。')
        index = source([*images, label, prose])
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / 'source.pdf'
            with pymupdf.open() as document:
                document.new_page(width=500, height=700)
                document.save(pdf)
            index.source['source_pdf'] = str(pdf)
            rows = list(recover_horizontal_images(index, [], directory).values())
        self.assertGreater(rows[0]['bbox'][3], .35)
        self.assertLess(rows[0]['bbox'][3], .4)
        self.assertNotIn('prose', rows[0]['members'])

    def test_image_labels_do_not_become_subheadings_or_example_boundaries(self):
        from book_pipeline.structure.page_content import discover_keywords, keyword_blocks
        from book_pipeline.structure.keyword_matching import validate_candidates
        block = dict(b='figure', text='b.图35', parts=[], candidate_eligible=False)
        candidates = validate_candidates()
        with tempfile.TemporaryDirectory() as directory:
            report = discover_keywords([(dict(id='section'), [block])], directory, candidates)
        self.assertEqual(report['match_count'], 0)
        result = keyword_blocks(None, [block], candidates, ['小写字母.'])
        self.assertIsNone(result[0]['matched_keyword'])
        self.assertEqual(result[0]['text'], block['text'])

    def test_row_renderer_keeps_all_image_and_caption_destinations(self):
        import pymupdf
        from book_pipeline.publishing.latex import Renderer
        images = [atom('left', 'image', [.1, .2, .2, .3]),
                  atom('right', 'image', [.4, .2, .5, .3])]
        captions = [atom('left-label', 'text', [.1, .30, .2, .32], '1'),
                    atom('right-label', 'text', [.4, .30, .5, .32], '2')]
        for image, caption in zip(images, captions):
            caption.update(group=image['group'], role='image_caption')
        index = source([images[0], captions[0], images[1], captions[1]])
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / 'source.pdf'
            with pymupdf.open() as document:
                document.new_page(width=500, height=700)
                document.save(pdf)
            index.source['source_pdf'] = str(pdf)
            registry = [dict(id=str(n), kind='figure', number='', label='', anchor='figure-'+str(n),
                             source_atom=image['id'], lo=index.prefix[index.ids[image['id']]],
                             hi=index.prefix[index.ids[image['id']]+1]) for n, image in enumerate(images)]
            renderer = Renderer(index, dict(corrections=[], exclusions=[], joins=[], objects=[]),
                                registry, [], directory, dict(float_assets=True))
            tex = ''.join(renderer.render())
            self.assertEqual(tex.count(r'\BookImage['), 1)
            self.assertEqual(tex.count(r'\begin{figure}[htp]'), 1)
            self.assertEqual(set(renderer.source_anchors), set(index.ids))
            self.assertEqual(sorted(renderer.anchors), ['figure-0', 'figure-1'])
            self.assertEqual(sum(item['end']-item['start'] for item in renderer.ledger), index.prefix[-1])

    def test_nine_side_by_side_images_form_one_left_to_right_row(self):
        images = [atom(str(n), 'image', [n / 10, .2, n / 10 + .05, .3]) for n in range(9)]
        index = source(list(reversed(images)) + [atom('below', 'image', [.1, .4, .4, .6])])
        rows = detect_image_rows(index)
        self.assertEqual(len(rows), 1)
        self.assertEqual([a['id'] for a in rows[0]['images']], [str(n) for n in range(9)])

    def test_stacked_images_with_same_x_range_remain_separate(self):
        index = source([atom('a', 'image', [.1, .2, .4, .3]), atom('b', 'image', [.1, .3, .4, .4])])
        self.assertEqual(detect_image_rows(index), [])


if __name__ == '__main__':
    unittest.main()
