"""DeepSeek JSON/vision requests with validated caching and bounded retries."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import threading
import time
from urllib.parse import urlsplit
from urllib.request import getproxies, proxy_bypass

from book_pipeline.config import ROOT, digest, fingerprint, read, save, load_environment
from book_pipeline.ai.costs import estimate, UsageLedger


class APIFailure(RuntimeError):
    """Configuration/service failures cannot be reported as conversion success."""


class DecisionUncertain(RuntimeError):
    """Invalid semantic responses may fall back to source-preserving output."""


def api_proxy(base_url):
    """Select only this API's proxy; an unrelated ALL_PROXY may be invalid."""
    url = urlsplit(base_url)
    if proxy_bypass(url.netloc):
        return None
    proxies = getproxies()
    proxy = proxies.get(url.scheme) or proxies.get('all')
    if proxy and '://' not in proxy:
        proxy = 'http://' + proxy
    if proxy and proxy.startswith('socks://'):
        proxy = 'socks5://' + proxy[len('socks://'):]
    return proxy


class DeepSeekClient:
    def __init__(self, config, cache_dir, transport=None, sleep=time.sleep):
        self.config = config
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.sleep = sleep
        self.lock = threading.Lock()
        self.transport = transport
        self.calls = self.cache / "usage.jsonl"
        self.cancel = threading.Event()
        self._api = None
        self.start_line = len(self.calls.read_text().splitlines()) if self.calls.exists() else 0
        self.ledger = UsageLedger(self.calls, config, self.start_line)
        self.shared_caches = [ROOT / p for p in config.get("shared_cache_dirs", [])]

    def _connect(self):
        if self._api is None:
            load_environment()
            key = os.environ.get(self.config["api_key_env"])
            if not key:
                raise APIFailure("Missing environment variable " + self.config["api_key_env"])
            try:
                from httpx import HTTPTransport
                from openai import OpenAI, DefaultHttpxClient
                # An explicit transport avoids parsing unused environment proxies
                # while keeping environment CA certificates available for TLS.
                transport = HTTPTransport(proxy=api_proxy(self.config["base_url"]))
                http = DefaultHttpxClient(transport=transport, timeout=self.config["timeout_seconds"])
                try:
                    self._api = OpenAI(api_key=key, base_url=self.config["base_url"],
                                       timeout=self.config["timeout_seconds"], max_retries=0, http_client=http)
                except Exception:
                    http.close()
                    raise
            except ImportError:
                raise APIFailure('DeepSeek 客户端依赖缺失；请运行 pip install -r requirements.txt（包含 SOCKS 代理支持）。') from None
            except ValueError:
                # Proxy exceptions can contain credentials; never expose them.
                raise APIFailure('DeepSeek 客户端初始化失败；请检查接口地址和 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY，'
                                 '代理协议支持 http、https、socks5、socks5h。') from None

    @property
    def api(self):
        self._connect()
        return self._api

    def record(self, task, attempt, usage=None, cached=False, error=None):
        row = {"time": datetime.now(timezone.utc).isoformat(), "task": task,
               "attempt": attempt, "cached": cached, "usage": usage or {}}
        if error:
            row["error"] = error  # Error type only; never API exceptions containing credentials.
        if usage and not cached:
            row["cost"] = estimate(usage, self.config, row["time"])
        with self.lock:
            with self.calls.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def ask(self, task, system, payload, images=(), validator=None, *, max_tokens=None, thinking=None):
        request_config = {**self.config,
                          "max_tokens": self.config["max_tokens"] if max_tokens is None else max_tokens,
                          "thinking": self.config["thinking"] if thinking is None else thinking}
        if request_config["thinking"] not in {"enabled", "disabled"} or not isinstance(request_config["max_tokens"], int) or request_config["max_tokens"] < 1:
            raise ValueError("Invalid request output budget or thinking mode")
        images = [Path(p) for p in images]
        key = fingerprint({"task": task, "system": system, "payload": payload,
                           "images": [digest(p) for p in images], "model": self.config["model"],
                           "url": self.config["base_url"], "thinking": request_config["thinking"],
                           "max_tokens": request_config["max_tokens"]})
        path = self.cache / (key + ".json")
        for candidate in [path] + [directory / path.name for directory in self.shared_caches]:
            if not candidate.is_file():
                continue
            try:
                cached = read(candidate)
                if cached["key"] != key or fingerprint(cached["result"]) != cached["result_sha256"]:
                    raise ValueError("Changed cached response")
                if validator:
                    validator(cached["result"])
                if candidate != path:
                    save(path, cached)
                self.record(task, 0, cached=True)
                return cached["result"]
            except (ValueError, KeyError, TypeError, AttributeError, IndexError):
                if candidate == path:
                    path.unlink()
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        content = [{"type": "text", "text": text}]
        for image in images:
            mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
            data = base64.b64encode(image.read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + data, "detail": "high"}})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        last_type = "invalid_response"
        last_service_failure = False
        for attempt in range(1, self.config["attempts"] + 1):
            if self.cancel.is_set():
                raise APIFailure("Conversion cancelled")
            raw = None
            try:
                if self.transport:
                    raw = self.transport(task, messages, request_config)
                else:
                    self._connect()
                    answer = self.api.chat.completions.create(
                        model=self.config["model"], messages=messages,
                        response_format={"type": "json_object"}, max_tokens=request_config["max_tokens"],
                        extra_body={"thinking": {"type": request_config["thinking"]}})
                    raw = {"content": answer.choices[0].message.content,
                           "finish_reason": answer.choices[0].finish_reason,
                           "usage": answer.usage.model_dump() if answer.usage else {}}
            except APIFailure:
                self.record(task, attempt, error="APIFailure")
                raise
            except Exception as error:
                last_service_failure = True
                code = getattr(error, "status_code", None)
                last_type = type(error).__name__
                self.record(task, attempt, error=last_type)
                if code in (400, 401, 402, 403, 404, 422):
                    raise APIFailure(f"DeepSeek request rejected (HTTP {code}); check key, model and vision/JSON support") from None
            else:
                last_service_failure = False
                try:
                    self.record(task, attempt, raw.get("usage"))
                    if raw.get("finish_reason") != "stop" or not raw.get("content"):
                        raise ValueError("Empty or truncated JSON response")
                    result = json.loads(raw["content"])
                    if not isinstance(result, dict):
                        raise ValueError("JSON result must be an object")
                    if validator:
                        validator(result)
                    save(path, {"key": key, "result": result, "result_sha256": fingerprint(result)})
                    return result
                except (ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
                    last_type = type(error).__name__
                    save(self.cache / "invalid" / (key + f".{attempt}.json"),
                         {"task": task, "attempt": attempt, "validation_error": str(error)[:500],
                          "response": raw.get("content", "") if isinstance(raw, dict) else ""})
                    # Only actual response validation errors are sent to the model.
                    messages.append({"role": "user", "content": "Return corrected JSON only. Validation error: " + str(error)[:500]})
                    self.record(task, attempt, error=last_type)
            if attempt < self.config["attempts"]:
                self.sleep(min(30, 2 ** attempt + random.random()))
        if last_service_failure:
            raise APIFailure("DeepSeek service unavailable after retries: " + last_type)
        raise DecisionUncertain("Invalid model decision after retries: " + last_type)
