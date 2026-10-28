"""Protected formulas remain legible to the translator and immutable in output."""
import json
from pathlib import Path
import tempfile
import unittest

from book_pipeline.ai.client import DeepSeekClient, DecisionUncertain
from book_pipeline.config import fingerprint, load_config, save
from book_pipeline.ingestion.translation import PROMPT, Translator


class TranslationContextTests(unittest.TestCase):
    def test_per_item_formula_context_invalidates_opaque_cache_and_preserves_values(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {**load_config(), 'budget': None, 'attempts': 1}
            items = [dict(id='a', text='If $x<2$, then $y=3$.'),
                     dict(id='b', text='If $u>4$, then $v=5$.')]
            old_payload = dict(items=[dict(id=i['id'], text='If ⟪KEEP_0⟫, then ⟪KEEP_1⟫.')
                                      for i in items], glossary={})
            settings = dict(task='translate-book-zh', system=PROMPT, payload=old_payload,
                            images=[], model=config['model'], url=config['base_url'],
                            thinking=config['thinking'], max_tokens=config['max_tokens'])
            key = fingerprint(settings)
            wrong = dict(items=[dict(id=i['id'], text='无语义上下文的旧译文⟪KEEP_0⟫⟪KEEP_1⟫。')
                                for i in items], terms={})
            save(Path(directory)/(key+'.json'), dict(key=key, result=wrong, result_sha256=fingerprint(wrong)))
            calls = []
            def transport(task, messages, settings):
                payload = json.loads(messages[1]['content'][0]['text']); calls.append(payload)
                return dict(content=json.dumps(dict(items=[dict(id=i['id'], text='若⟪KEEP_0⟫，则⟪KEEP_1⟫。')
                                                            for i in items], terms={})), finish_reason='stop',
                            usage=dict(prompt_tokens=20, completion_tokens=20))
            client = DeepSeekClient(config, directory, transport=transport)
            result = Translator(client).translate(items)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['items'][0]['protected_values'], {'⟪KEEP_0⟫': '$x<2$', '⟪KEEP_1⟫': '$y=3$'})
            self.assertEqual(calls[0]['items'][1]['protected_values'], {'⟪KEEP_0⟫': '$u>4$', '⟪KEEP_1⟫': '$v=5$'})
            self.assertEqual(result, {'a': '若$x<2$，则$y=3$。', 'b': '若$u>4$，则$v=5$。'})
            # The same signature can resume without another request.
            self.assertEqual(Translator(client).translate(items), result)
            self.assertEqual(len(calls), 1)

    def test_readable_context_does_not_allow_literal_formula_instead_of_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {**load_config(), 'budget': None, 'attempts': 1}
            def transport(*args):
                return dict(content=json.dumps(dict(items=[dict(id='a', text='若$x<2$，则$y=3$。')], terms={})),
                            finish_reason='stop', usage=dict(prompt_tokens=20, completion_tokens=20))
            client = DeepSeekClient(config, directory, transport=transport)
            with self.assertRaises(DecisionUncertain):
                Translator(client).translate([dict(id='a', text='If $x<2$, then $y=3$.')])

    def test_context_cannot_be_duplicated_beside_valid_tokens(self):
        for leaked in ('$x<2$', '2', '<em>', 'https://example.org'):
            with self.subTest(leaked=leaked), tempfile.TemporaryDirectory() as directory:
                config = {**load_config(), 'budget': None, 'attempts': 1}
                def transport(*args):
                    return dict(content=json.dumps(dict(items=[dict(id='a', text='若⟪KEEP_0⟫，则⟪KEEP_1⟫。 '+leaked)], terms={})),
                                finish_reason='stop', usage=dict(prompt_tokens=20, completion_tokens=20))
                client = DeepSeekClient(config, directory, transport=transport)
                with self.assertRaises(DecisionUncertain):
                    Translator(client).translate([dict(id='a', text='If $x<2$, then $y=3$.')])


if __name__ == '__main__':
    unittest.main()
