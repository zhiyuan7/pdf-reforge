"""Auditable token-based estimates. Provider billing remains the settlement truth."""
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing/"


def estimate(usage, config, timestamp):
    pricing = config.get("pricing", {})
    if not usage or pricing.get("model") != config["model"]:
        return None
    dt = datetime.fromisoformat(timestamp).astimezone(timezone.utc)
    peak = dt.weekday() < 5 and (1 <= dt.hour < 4 or 6 <= dt.hour < 10)
    rates = pricing["peak" if peak else "off_peak"]
    hit = usage.get("prompt_cache_hit_tokens", 0) or 0
    prompt = usage.get("prompt_tokens", 0) or 0
    miss = usage.get("prompt_cache_miss_tokens", max(0, prompt - hit)) or 0
    output = usage.get("completion_tokens", 0) or 0
    value = sum(Decimal(str(tokens)) * Decimal(str(rates[name])) for tokens, name in
                [(hit, "input_cached"), (miss, "input_uncached"), (output, "output")]) / Decimal(1000000)
    return {"estimated_cost": float(value), "currency": pricing.get("currency", "USD"),
            "period": "peak" if peak else "off_peak", "rates_per_million": rates,
            "pricing_checked_on": pricing.get("checked_on"), "source": PRICING_URL}


class UsageLedger:
    def __init__(self, path, config, start_line=0):
        self.path, self.config, self.start_line = Path(path), config, start_line

    def summary(self):
        rows = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        def totals(current):
            paid = [r for r in current if not r.get('cached') and r.get('usage')]
            estimates = [r['cost'] for r in paid if r.get('cost')]
            return {'requests':len(paid),'cache_hits':sum(bool(r.get('cached')) for r in current),
                    'tokens':sum(r['usage'].get('total_tokens',0) or 0 for r in paid),
                    'estimated_cost':round(sum(r['estimated_cost'] for r in estimates),8),
                    'unpriced_requests':len(paid)-len(estimates),'error_events':sum('error' in r for r in current)}
        paid = [r for r in rows if not r.get("cached") and r.get("usage")]
        estimates = [r["cost"] for r in paid if r.get("cost")]
        return {"requests": len(paid), "cache_hits": sum(bool(r.get("cached")) for r in rows),
                "tokens": sum(r["usage"].get("total_tokens", 0) or 0 for r in paid),
                "estimated_cost": round(sum(r["estimated_cost"] for r in estimates), 8),
                "currency": self.config.get("pricing", {}).get("currency", "USD"),
                "unpriced_requests": len(paid) - len(estimates), "error_events": sum("error" in r for r in rows),
                "scope": "current_job_including_retries_and_resumes", "historical_events": self.start_line,
                "current_run": totals(rows[self.start_line:]),
                "is_estimate": True, "pricing_source": PRICING_URL,
                "pricing_checked_on": self.config.get("pricing", {}).get("checked_on")}
