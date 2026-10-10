"""Configuration, atomic checkpoints and reproducible fingerprints."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def load_environment(path=None):
    """Read plain KEY=value settings without executing shell code or replacing env."""
    path = Path(path or ROOT / ".env")
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip().replace("_", "a").isalnum():
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Separate worker threads must never share a temporary file.
    import uuid
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_config(path=None):
    load_environment()
    defaults = read(ROOT / "book_config.json")
    custom = read(path) if path else {}
    unknown = set(custom) - set(defaults)
    if unknown:
        raise ValueError("Unknown book configuration: " + ", ".join(sorted(unknown)))
    config = {**defaults, **custom}
    toc_pages = config.get('toc_pages', [])
    if not isinstance(toc_pages,list) or any(isinstance(p,bool) or not isinstance(p,int) or p<1 for p in toc_pages) or len(set(toc_pages))!=len(toc_pages) or toc_pages!=sorted(toc_pages):
        raise ValueError('toc_pages must be increasing unique positive physical PDF page numbers')
    rules = custom.get("layout_rules", {})
    if not isinstance(rules, dict) or set(rules) - set(defaults["layout_rules"]):
        raise ValueError("Unknown or invalid layout_rules setting")
    config["layout_rules"] = {**defaults["layout_rules"], **rules}
    for name in ("top_margin", "bottom_margin", "position_tolerance"):
        value = config["layout_rules"][name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= .2:
            raise ValueError("layout_rules." + name + " must be > 0 and <= 0.2")
    for name in ("repeat_page_gap", "min_repeat_pages"):
        value = config["layout_rules"][name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError("layout_rules." + name + " must be an integer >= 2")
    for name in ("concurrency", "attempts", "max_tokens", "image_dpi", "timeout_seconds"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(name + " must be a positive integer")
    if config["attempts"] > 10:
        raise ValueError("attempts <= 10 is required")
    if not 72 <= config["image_dpi"] <= 300:
        raise ValueError("image_dpi must be 72..300")
    for name in ("model", "base_url", "api_key_env", "main_font", "heading_font"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError("Invalid " + name)
    if not config["base_url"].startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
        raise ValueError("base_url requires HTTPS (except a local test server)")
    if config["thinking"] not in ("enabled", "disabled"):
        raise ValueError("thinking must be enabled or disabled")
    for name in ("shared_cache_dirs",):
        if not isinstance(config[name], list) or not all(isinstance(p,str) and p for p in config[name]):
            raise ValueError(name + " must be a list of paths")
    pricing=config["pricing"]
    if not isinstance(pricing,dict) or pricing.get('currency') not in ('USD','CNY'):
        raise ValueError('pricing requires a supported currency')
    for period in ('peak','off_peak'):
        rates=pricing.get(period,{})
        if set(rates)!={'input_cached','input_uncached','output'} or any(isinstance(x,bool) or not isinstance(x,(int,float)) or x<0 for x in rates.values()):
            raise ValueError('pricing.'+period+' requires three nonnegative rates per million tokens')
    from .structure.numbered import BOX_KINDS
    labels = config["numbered_object_labels"]
    if not isinstance(labels, dict) or any(not isinstance(label, str) or not label.strip() or
            not isinstance(kind, str) or kind not in BOX_KINDS for label, kind in labels.items()):
        raise ValueError("numbered_object_labels must map type names to supported box kinds")
    from .structure.keyword_matching import validate_candidates
    config['keyword_candidates'] = validate_candidates(config.get('keyword_candidates'), labels)
    return config


def executable(name, configured="auto"):
    if configured != "auto":
        path = Path(configured).expanduser()
        return str(path.absolute()) if path.is_file() and os.access(path, os.X_OK) else shutil.which(configured)
    found = shutil.which(name)
    if found:
        return found
    for path in sorted((ROOT / ".tools").glob("**/bin/*/" + name)):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None
