"""Process-safe, persistent reservations made before any billable request.

Reserve the entire model context at peak prices, including unknown vision input.
Missing usage and interrupted requests keep their reservation (fail closed).
"""
from contextlib import contextmanager
from datetime import datetime
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid
from zoneinfo import ZoneInfo

from book_pipeline.config import ROOT


class BudgetExceeded(RuntimeError):
    pass


class BudgetGuard:
    def __init__(self, config, job):
        self.config = config
        self.settings = config.get('budget')
        self.job = str(config.get('_budget_job_id', job))
        self.path = Path(os.environ.get('PDF_REFORGE_BUDGET_DB',
                                      (self.settings or {}).get('ledger', 'output/budget.sqlite3')))
        if not self.path.is_absolute():
            self.path = ROOT / self.path

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=60) as db:
            db.execute('CREATE TABLE IF NOT EXISTS charges '
                       '(id TEXT PRIMARY KEY, day TEXT, job TEXT, task TEXT, cny REAL, state TEXT)')
            db.execute('BEGIN IMMEDIATE')
            yield db

    def upper_bound(self, max_tokens):
        pricing = self.config.get('pricing', {})
        if pricing.get('model') != self.config['model']:
            raise BudgetExceeded('预算保护：模型没有匹配的价格，拒绝付费调用')
        rates = pricing['peak']
        # A conservative conversion ceiling, configurable if the exchange rate rises.
        conversion = self.settings['usd_to_cny_ceiling'] if pricing['currency'] == 'USD' else 1
        amount = (self.settings['context_tokens'] * max(rates['input_uncached'], rates['input_cached'])
                  + max_tokens * rates['output']) / 1_000_000 * conversion
        if not math.isfinite(amount) or amount <= 0:
            raise BudgetExceeded('预算保护：无效的价格上界')
        return amount

    def reserve(self, task, max_tokens, wait_for_inflight=False):
        if not self.settings:
            return None
        amount = self.upper_bound(max_tokens)
        return self.reserve_amount(task, amount, wait_for_inflight)

    def reserve_amount(self, task, amount, wait_for_inflight=False):
        """Reserve a known CNY amount for services billed in other units."""
        if not self.settings:
            return None
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
            raise BudgetExceeded('预算保护：无效的费用上界')
        day = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
        daily = min(self.settings['daily_cny'], float(os.environ.get('PDF_REFORGE_DAILY_BUDGET_CNY', self.settings['daily_cny'])))
        if not math.isfinite(daily) or daily <= 0:
            raise BudgetExceeded('预算保护：无效的每日限额')
        deadline = time.monotonic() + 2*self.config.get('timeout_seconds', 180)
        while True:
            day = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
            with self.transaction() as db:
                spent_day = db.execute("SELECT COALESCE(SUM(cny),0) FROM charges WHERE day=? OR state IN ('reserved','unknown')", (day,)).fetchone()[0]
                spent_job = db.execute('SELECT COALESCE(SUM(cny),0) FROM charges WHERE job=?', (self.job,)).fetchone()[0]
                pending_job = db.execute("SELECT COALESCE(SUM(cny),0) FROM charges WHERE job=? AND state='reserved'", (self.job,)).fetchone()[0]
                pending_day = db.execute("SELECT COALESCE(SUM(cny),0) FROM charges WHERE state='reserved'").fetchone()[0]
                blocked = [(name, spent, limit, pending) for name, spent, limit, pending in
                           [('单次', 0, self.settings['per_call_cny'], 0),
                            ('本书累计', spent_job, self.settings['per_book_cny'], pending_job),
                            ('当日累计', spent_day, daily, pending_day)] if spent + amount > limit + 1e-9]
                if not blocked:
                    token = uuid.uuid4().hex
                    db.execute('INSERT INTO charges VALUES (?,?,?,?,?,?)',
                               (token, day, self.job, task, amount, 'reserved'))
                    return token
                # Release the database lock while another live request settles.
                # Unknown billing never enables a wait/release cycle.
                can_wait = (wait_for_inflight and time.monotonic() < deadline and
                            all(pending > 0 and spent-pending+amount <= limit+1e-9
                                for _,spent,limit,pending in blocked))
                if not can_wait:
                    name, spent, limit, _ = blocked[0]
                    raise BudgetExceeded(f'预算保护：{name}已用/预留 {spent:.4f} 元，'
                                         f'下次最坏费用 {amount:.4f} 元，限额 {limit:.2f} 元；停止并保留检查点')
            time.sleep(.05)

    def settle(self, token, usage=None, rejected=False):
        if token is None:
            return
        if rejected:
            amount = 0
        elif not usage or not all(k in usage for k in ('prompt_tokens', 'completion_tokens')):
            with self.transaction() as db:
                db.execute("UPDATE charges SET state='unknown' WHERE id=?", (token,))
            return  # Unknown billing remains charged at the full reservation.
        else:
            pricing = self.config['pricing']
            rates = pricing['peak']
            hit = usage.get('prompt_cache_hit_tokens', 0) or 0
            prompt = usage['prompt_tokens'] or 0
            miss = usage.get('prompt_cache_miss_tokens', max(0, prompt-hit)) or 0
            # Keep the limit conservative throughout discounted hours.
            amount = (hit*rates['input_cached'] + miss*rates['input_uncached']
                      + (usage['completion_tokens'] or 0)*rates['output']) / 1_000_000
            if pricing['currency'] == 'USD':
                amount *= self.settings['usd_to_cny_ceiling']
        self.settle_amount(token, amount)

    def settle_amount(self, token, amount=None):
        """Keep unknown requests reserved; settle confirmed non-token charges."""
        if token is None:
            return
        if amount is None:
            with self.transaction() as db:
                db.execute("UPDATE charges SET state='unknown' WHERE id=?", (token,))
            return
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
            raise BudgetExceeded('预算保护：无效的结算费用')
        with self.transaction() as db:
            db.execute('UPDATE charges SET cny=?, state=? WHERE id=?', (amount, 'settled', token))
