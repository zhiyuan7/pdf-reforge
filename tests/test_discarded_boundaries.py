"""Discarded special-block roles cannot constrain otherwise valid examples."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from book_pipeline.ai.client import DecisionUncertain, DeepSeekClient
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.structure.book_mapping import classified_modules
from book_pipeline.structure.page_content import empty_edits, resolve_special_blocks, run_page_content

CONFIG = dict(model='test', base_url='https://example.invalid', api_key_env='TEST_KEY',
              attempts=1, max_tokens=1000, thinking='disabled', shared_cache_dirs=[],
              pricing={}, concurrency=1,
              keyword_candidates=dict(statements=[], examples=['例'], subheadings=['数字', '数字.']))


def fixture(titles=False):
    atoms = []
    def add(aid, text):
        atoms.append(dict(id=aid, group=aid, kind='text', role='text', block_type='text',
                          content=text, length=len(text), page=0, bbox=[.1, .2, .9, .3]))
    add('heading', '示例章节')
    heading = dict(toc_id='section', title='示例章节', number='', toc_parent_id=None,
                   heading_level=1, kind='chapter', toc_category='body',
                   start=dict(atom='heading', offset=0), end=dict(atom='heading', offset=4))
    for n in range(1, 12):
        add(f'example-{n}', f'例{n} 求极限。')
        add(f'solution-{n}', '解：先化简。')
        add(f'candidate-{n}', f'{n}. 另一主题' if titles else str(195+n))
        add(f'continuation-{n}', '继续推导，结果为零。')
        add(f'ordinary-{n}', '这段是普通讨论。')
    source = SourceIndex(dict(source_pdf='unused.pdf', input_sha256='source', mineru_sha256='ocr',
                              source_page_count=1, atoms=atoms))
    return source, dict(headings=[heading], metadata=dict(title='测试书', author='测试'))


class DiscardedBoundaryTests(unittest.TestCase):
    def run_content(self, titles=False, invalid_initial=False):
        source, outline = fixture(titles)
        original = deepcopy(source.source)
        payloads = []
        def transport(task, messages, config):
            payload = json.loads(messages[1]['content'][0]['text'])
            payloads.append(payload)
            if task.endswith('-subheading'):
                value = {'y': int(titles or payload['block'] == '196')}
            elif task.endswith('-example'):
                if invalid_initial and payload['blocks'][0]['t'].startswith('例1 '):
                    value = {}
                else:
                    stop = next((b['i'] for b in payload['blocks'] if b['t'] == '这段是普通讨论。'), len(payload['blocks'])+1)
                    # A retained title still ends the example. For the false margin
                    # candidate reproduce the model's original, valid continuation.
                    if titles:
                        stop = min(stop, payload.get('next_declaration', stop))
                    value = dict(y=1, q=[1], a=list(range(2, stop)))
            else:
                raise AssertionError(task)
            return dict(content=json.dumps(value), finish_reason='stop', usage={})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            client = DeepSeekClient(CONFIG, root/'cache', transport=transport)
            report = run_page_content(source, outline, client, root/'content')
            processed = json.loads((root/'content/sections/section/processed.json').read_text())
            judgments = json.loads((root/'content/special_block_results.json').read_text())
            resolution = json.loads((root/'content/special_block_resolution.json').read_text())
            plan = json.loads((root/'content/split_source.json').read_text())
            groups = classified_modules(source, {'section': processed}, plan, report, root/'content')
            if not titles:
                self.assertFalse(any(m['type'] == 'subheading' for m in groups['section']))
        self.assertEqual(source.source, original)
        return source, report, processed, judgments, resolution, payloads

    def test_discarded_margin_does_not_split_example_or_create_unresolved_error(self):
        source, report, processed, judgments, _, payloads = self.run_content()
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['statistics']['examples'], 11)
        self.assertEqual(report['sections'][0]['resolved']['u'], [])
        self.assertFalse(any('next_declaration' in p for p in payloads))
        discarded = [r for r in judgments if r.get('discard_reason')]
        self.assertEqual(len(discarded), 1)
        self.assertFalse(discarded[0]['adopted'])
        self.assertEqual(report['statistics']['inferred_subheadings'], 0)
        text = ''.join(p['text'] for u in processed['units'] for p in u['parts'])
        self.assertEqual(text, ''.join(a['content'] for a in source.source['atoms'][1:]))
        examples = [m for m in processed['modules'] if m['type'] == 'example']
        first = examples[0]
        self.assertIn('继续推导，结果为零。', ''.join(source.text(r['lo'], r['hi']) for r in first['answer_ranges']))

    def test_retained_titles_still_limit_examples(self):
        _, report, _, _, _, payloads = self.run_content(titles=True)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['statistics']['examples'], 11)
        self.assertEqual(report['statistics']['inferred_subheadings'], 11)
        self.assertTrue(any('next_declaration' in p for p in payloads))
        modules = report['sections'][0]['resolved']['s']
        for example, title in zip(modules[::2], modules[1::2]):
            self.assertEqual((example['type'], title['type']), ('example', 'subheading'))
            self.assertTrue(all(r['hi'] <= title['lo'] for r in example['selected_ranges']))

    def test_independent_invalid_decision_is_not_silently_discarded(self):
        _, report, _, _, _, _ = self.run_content(invalid_initial=True)
        self.assertEqual(report['status'], 'unresolved')
        self.assertEqual(len(report['sections'][0]['resolved']['u']), 1)

    def test_later_discard_revokes_only_errors_from_its_boundary(self):
        def row(bid, keyword, lo, hi, kind='statement'):
            module = dict(lo=lo, hi=hi, type=kind, keyword=keyword, selected_ranges=[dict(lo=lo, hi=hi)])
            return dict(section='section', b=bid, keyword=keyword, present=True, module=module)
        initial = [row('example-0', '例', 0, 50, 'example'), row('dropped-declaration', '定理', 10, 101)]
        initial += [row(f'example-{n}', '例', n*100, n*100+50, 'example') for n in range(1, 12)]
        initial += [row(f'statement-{n}', '定理', 2000+n*100, 2050+n*100) for n in range(10)]
        candidates = [dict(b=r['b'], keyword=r['keyword'], section='section', lo=r['module']['lo'], category='examples' if r['keyword']=='例' else 'statements') for r in initial]
        original = deepcopy(initial)
        def judge(client, candidate, directory, next_declaration, **kwargs):
            if candidate['b'] == 'dropped-declaration':
                return dict(keyword='定理', present=False, blocks=[])
            self.assertEqual((candidate['b'], next_declaration), ('example-0', 'dropped-declaration'))
            raise DecisionUncertain('An example cannot cross a confirmed next declaration')
        with tempfile.TemporaryDirectory() as folder, patch('book_pipeline.structure.page_content.judge_candidate', side_effect=judge):
            result = resolve_special_blocks(None, candidates, initial, {'section': empty_edits()},
                                           SimpleNamespace(config=CONFIG), Path(folder), None)
            evidence = json.loads((Path(folder)/'special_block_resolution.json').read_text())
        judgments, edits, _, _, _, retained, _ = result
        self.assertEqual(initial, original)
        self.assertEqual(retained, {'例'})
        self.assertEqual(edits['section']['u'], [])
        self.assertEqual(edits['section']['rejected'], [])
        self.assertEqual(len(edits['section']['s']), 12)
        self.assertEqual(judgments[0]['module']['hi'], 50)
        self.assertNotIn('error', judgments[0])
        self.assertEqual(evidence['passes'][0]['removed_keywords'], ['定理'])
        self.assertEqual(evidence['passes'][1]['boundary_reviews'], [])


if __name__ == '__main__':
    unittest.main()
