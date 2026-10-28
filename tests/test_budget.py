"""Budget limits cover concurrency, retries, resumes, and unknown billing."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from book_pipeline.ai.budget import BudgetExceeded, BudgetGuard
from book_pipeline.ai.client import DeepSeekClient
from book_pipeline.config import load_config


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = load_config()
        self.config['budget'] = {**self.config['budget'], 'ledger': str(self.root / 'budget.db')}
        self.config['_budget_job_id'] = 'book'

    def test_parallel_reservations_cannot_exceed_book_limit(self):
        def reserve(_):
            try:
                return BudgetGuard(self.config, 'cache').reserve('layout', 32768)
            except BudgetExceeded:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            tokens = list(pool.map(reserve, range(8)))
        self.assertEqual(sum(t is not None for t in tokens), 2)
        with sqlite3.connect(self.root / 'budget.db') as db:
            self.assertLessEqual(db.execute('SELECT SUM(cny) FROM charges').fetchone()[0], 8)

    def test_usage_releases_unused_reservation_and_resume_remembers_charge(self):
        guard = BudgetGuard(self.config, 'one-phase')
        token = guard.reserve('layout', 32768)
        guard.settle(token, dict(prompt_tokens=100, completion_tokens=100))
        resumed = BudgetGuard(self.config, 'other-phase')
        self.assertIsNotNone(resumed.reserve('formula', 32768))
        with sqlite3.connect(self.root / 'budget.db') as db:
            self.assertAlmostEqual(db.execute("SELECT cny FROM charges WHERE state='settled'").fetchone()[0], .0012)

    def test_daily_budget_is_shared_between_books(self):
        self.config['budget']['daily_cny'] = 3
        BudgetGuard(self.config, '').reserve('layout', 32768)
        other = {**self.config, '_budget_job_id': 'other-book'}
        with self.assertRaises(BudgetExceeded):
            BudgetGuard(other, '').reserve('layout', 32768)

    def test_unknown_usage_keeps_full_reservation(self):
        guard = BudgetGuard(self.config, '')
        for _ in range(2):
            guard.settle(guard.reserve('layout', 32768), {})
        with self.assertRaises(BudgetExceeded):
            BudgetGuard(self.config, '').reserve('retry', 32768)

    def test_feedback_loop_stops_before_another_billable_call(self):
        calls = []
        def transport(*args):
            calls.append(1)
            return dict(content='{}', finish_reason='stop')
        client = DeepSeekClient(self.config, self.root / 'cache', transport=transport, sleep=lambda _: None)
        with self.assertRaises(BudgetExceeded):
            client.ask('repair', 'system', {}, validator=lambda _: (_ for _ in ()).throw(ValueError('bad')),
                       retry_until_valid=True)
        self.assertEqual(len(calls), 2)

    def test_valid_cache_does_not_reserve_or_spend(self):
        client = DeepSeekClient(self.config, self.root / 'cache', transport=lambda *args:
                               dict(content='{"ok":1}', finish_reason='stop'), sleep=lambda _: None)
        client.ask('layout', 'system', {})
        client.ask('layout', 'system', {})
        with sqlite3.connect(self.root / 'budget.db') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM charges').fetchone()[0], 1)

    def test_bounded_reasoning_fallback_keeps_billing_and_attempt_limit(self):
        requests=[]
        def transport(task, messages, config):
            requests.append(config['thinking'])
            return dict(content='', finish_reason='length',
                        usage=dict(prompt_tokens=100,completion_tokens=100))
        config={**self.config, 'thinking':'enabled', 'attempts':2}
        client=DeepSeekClient(config,self.root/'fallback',transport=transport,sleep=lambda _:None)
        with self.assertRaises(RuntimeError):
            client.ask('joint_geometry','system',{})
        self.assertEqual(requests,['enabled','disabled'])
        with sqlite3.connect(self.root/'budget.db') as db:
            rows=db.execute('SELECT cny,state FROM charges').fetchall()
        self.assertEqual(rows,[(.0012,'settled'),(.0012,'settled')])

    def test_parallel_clients_wait_for_billing_instead_of_falsely_exhausting_budget(self):
        self.config['budget']['per_book_cny'] = 3
        def ask(number):
            def transport(*args):
                time.sleep(.04)
                return dict(content='{"ok":1}', finish_reason='stop',
                            usage=dict(prompt_tokens=100, completion_tokens=100))
            client = DeepSeekClient(self.config, self.root/str(number), transport=transport)
            return client.ask('layout', 'system', dict(number=number))
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(ask, range(4))), [dict(ok=1)]*4)
