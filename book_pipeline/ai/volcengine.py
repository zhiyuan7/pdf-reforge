"""Volcengine TranslateText: signed requests, character billing and durable cache.

API: https://docs.volcengine.com/docs/MachineTranslation/TextTranslationAPI?lang=zh
Pricing verified 2026-10-29: 49 CNY/million source characters after the monthly
free allowance. An allowance is used only for a user-attested account/month.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid
from zoneinfo import ZoneInfo

from book_pipeline.ai.budget import BudgetGuard
from book_pipeline.ai.client import APIFailure, DecisionUncertain, api_proxy
from book_pipeline.config import fingerprint, load_environment, read, save

HOST = 'translate.volcengineapi.com'
QUERY = 'Action=TranslateText&Version=2020-06-01'
URL = 'https://' + HOST + '/?' + QUERY
METHOD = 'volcengine_translate_text_v6'
# Raw service responses remain reusable when local validation improves.
REQUEST_METHOD = 'volcengine_translate_text_v2'
PRICE_PER_MILLION = 49
PRICING_URL = 'https://docs.volcengine.com/docs/MachineTranslation/Productbilling?lang=zh'


def signed_headers(body, access_key, secret_key, date=None):
    """Sign the exact UTF-8 body using the provider's documented V4 algorithm."""
    date = date or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    sha = lambda value: hashlib.sha256(value).hexdigest()
    mac = lambda key, value: hmac.new(key, value.encode(), hashlib.sha256).digest()
    headers = {'Content-Type': 'application/json', 'Host': HOST, 'X-Date': date,
               'X-Content-Sha256': sha(body)}
    ordered = sorted((key.lower(), value) for key, value in headers.items())
    canonical = ''.join(key + ':' + value + '\n' for key, value in ordered)
    names = ';'.join(key for key, _ in ordered)
    scope = date[:8] + '/cn-north-1/translate/request'
    request = '\n'.join(['POST', '/', QUERY, canonical, names, sha(body)])
    key = mac(mac(mac(mac(secret_key.encode(), date[:8]), 'cn-north-1'), 'translate'), 'request')
    signature = mac(key, '\n'.join(['HMAC-SHA256', date, scope, sha(request.encode())])).hex()
    headers['Authorization'] = ('HMAC-SHA256 Credential=' + access_key + '/' + scope +
                                ', SignedHeaders=' + names + ', Signature=' + signature)
    return headers


def chunks(text, limit=4500):
    """Split long prose on whitespace, preserving every source character."""
    result = []
    while len(text) > limit:
        stops = [m.end() for m in re.finditer(r'\s+', text[:limit])]
        if not stops:
            raise DecisionUncertain('翻译文本含超过接口长度的连续标识符')
        cut = stops[-1]
        result.append(text[:cut])
        text = text[cut:]
    if text:
        result.append(text)
    return result


def normalize_prose_wraps(text):
    """PDF line wraps are not sentence boundaries; keep blank paragraphs."""
    parts=re.split(r'(\n[ \t]*\n)',text.replace('\r\n','\n'))
    for number in range(0,len(parts),2):
        paragraph=re.sub(r'-[ \t]*\n[ \t]*(?=[A-Za-z])','-',parts[number])
        parts[number]=re.sub(r'[ \t]*\n[ \t]*',' ',paragraph)
    return ''.join(parts)


_SMALL = dict(zip(('zero one two three four five six seven eight nine ten eleven twelve thirteen '
                  'fourteen fifteen sixteen seventeen eighteen nineteen').split(), range(20)))
_TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
_NUMBER_WORDS = '|'.join([*_SMALL, *_TENS, 'hundred', 'thousand', 'million', 'billion'])
_WRITTEN_NUMBER = re.compile(r'(?<!\w)(?:' + _NUMBER_WORDS + r')'
    r'(?:(?:[-\s]+(?:and\s+)?)(?:' + _NUMBER_WORDS + r'))*(?!\w)', re.I)


def _small_number(words):
    if len(words) == 1:
        return {**_SMALL, **_TENS}.get(words[0])
    if len(words) == 2 and words[0] in _TENS and 0 < _SMALL.get(words[1], 0) < 10:
        return _TENS[words[0]] + _SMALL[words[1]]
    if len(words) >= 2 and 0 < _SMALL.get(words[0], 0) < 10 and words[1] == 'hundred':
        tail = words[2:]
        if tail[:1] == ['and']:
            tail = tail[1:]
        rest = _small_number(tail) if tail else 0
        if rest is not None and rest < 100:
            return _SMALL[words[0]] * 100 + rest
    return None


def written_number(text):
    words = re.sub(r'-', ' ', text.casefold()).split()
    total, start = 0, 0
    for name, scale in [('billion', 1_000_000_000), ('million', 1_000_000), ('thousand', 1000)]:
        if name not in words[start:]:
            continue
        stop = words.index(name, start)
        value = _small_number(words[start:stop])
        if value is None or value <= 0:
            return None
        total += value*scale
        start = stop+1
    tail = words[start:]
    if tail[:1] == ['and'] and total:
        tail = tail[1:]
    value = _small_number(tail) if tail else 0
    return total+value if value is not None else None


def normalize_written_numbers(source, translated):
    """MT may write 'twenty-five' as 25. Render only source-attested values
    with Chinese digits, keeping the existing ban on invented numeric literals.
    Existing numeric/formula placeholders cannot be changed by this operation.
    """
    known = Counter()
    for match in _WRITTEN_NUMBER.finditer(source):
        words=re.sub(r'-',' ',match[0].casefold()).split()
        while words:
            if words[0]=='and':
                words=words[1:]
                continue
            for stop in range(len(words),0,-1):
                value=written_number(' '.join(words[:stop]))
                if value is not None:
                    known[value]+=1
                    words=words[stop:]
                    break
            else:
                words=words[1:]
    decades=dict(zip('twenties thirties forties fifties sixties seventies eighties nineties'.split(),range(20,100,10)))
    known.update(decades[m[0].casefold()] for m in re.finditer(r'\b(?:'+'|'.join(decades)+r')\b',source,re.I))
    # A scalar attached to a standard unit is not a free-standing numeral in
    # the original protector (unlike variables such as 2t). MT may spell its
    # unit out; only its exact source value/count can become Chinese digits.
    units=r'ns|us|µs|μs|ms|s|mm|cm|km|m|kg|g|Hz|kHz|MHz|GHz|MB|GB|TB|bits?|fps'
    known.update(int(m[1]) for m in re.finditer(
        r'(?<![A-Za-z0-9_])(\d+)(?:'+units+r')(?![A-Za-z0-9_])',source))
    def replace(match):
        if not re.fullmatch(r'\d+|\d{1,3}(?:,\d{3})+', match[0]):
            return match[0]
        value = int(match[0].replace(',', ''))
        if known[value]:
            known[value] -= 1
            return chinese_number(value)
        return match[0]
    translated = re.sub(r'(?<![A-Za-z0-9_])\d+(?:[.,]\d+)*(?![A-Za-z0-9_])', replace, translated)
    # English month names become Chinese numeric months. Accept only the
    # source-attested month and occurrence count; other new digits still fail.
    months = {name:number for number,name in enumerate(
        'January February March April May June July August September October November December'.split(),1)}
    known_months = Counter(months[m[0]] for m in re.finditer(
        r'\b(?:'+'|'.join(months)+r')\b', source))
    def month(match):
        value=int(match[0])
        if known_months[value]:
            known_months[value]-=1
            return chinese_number(value)
        return match[0]
    return re.sub(r'(?<![A-Za-z0-9_.])(1[0-2]|[1-9])(?=\s*月)', month, translated)


def chinese_number(number):
    if number == 0:
        return '零'
    for scale, unit in [(100_000_000, '亿'), (10_000, '万')]:
        if number >= scale:
            high, low = divmod(number, scale)
            return chinese_number(high)+unit+('零' if low and low < scale//10 else '')+(chinese_number(low) if low else '')
    result, pending_zero = '', False
    for scale, unit in [(1000, '千'), (100, '百'), (10, '十'), (1, '')]:
        digit, number = divmod(number, scale)
        if digit:
            result += ('零' if pending_zero else '')+'零一二三四五六七八九'[digit]+unit
            pending_zero = False
        elif result and number:
            pending_zero = True
    return result[1:] if result.startswith('一十') else result


class CharacterQuota:
    """Shared account/month allowance; unknown calls continue consuming quota."""
    def __init__(self, guard, account, allowance):
        self.guard, self.account, self.allowance = guard, account, allowance or {}

    def reserve(self, count):
        month = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m')
        available = self.allowance.get('characters', 0) if self.allowance.get('month') == month else 0
        token = uuid.uuid4().hex
        with self.guard.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS translation_allowances '
                       '(account TEXT, month TEXT, characters INTEGER, evidence TEXT, PRIMARY KEY(account,month))')
            db.execute('CREATE TABLE IF NOT EXISTS translation_characters '
                       '(id TEXT PRIMARY KEY, account TEXT, month TEXT, characters INTEGER, free_characters INTEGER)')
            db.execute('INSERT OR IGNORE INTO translation_allowances VALUES (?,?,?,?)',
                       (self.account, month, available, self.allowance.get('evidence', 'no verified allowance')))
            limit = db.execute('SELECT characters FROM translation_allowances WHERE account=? AND month=?',
                               (self.account, month)).fetchone()[0]
            used = db.execute('SELECT COALESCE(SUM(free_characters),0) FROM translation_characters '
                              'WHERE account=? AND month=?', (self.account, month)).fetchone()[0]
            free = min(count, max(0, limit-used))
            db.execute('INSERT INTO translation_characters VALUES (?,?,?,?,?)',
                       (token, self.account, month, count, free))
        return token, free

    def release_unsubmitted(self, token):
        # Only a request known not to have reached the service may release quota.
        with self.guard.transaction() as db:
            db.execute('DELETE FROM translation_characters WHERE id=?', (token,))


class TranslationUsage:
    def __init__(self, path):
        self.path = path

    def summary(self):
        rows = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        return dict(provider='volcengine', requests=sum(not r.get('cached') for r in rows),
                    cache_hits=sum(bool(r.get('cached')) for r in rows),
                    characters=sum(r.get('characters', 0) for r in rows),
                    free_characters=sum(r.get('free_characters', 0) for r in rows),
                    estimated_cost=sum(r.get('conservative_cny', 0) for r in rows), currency='CNY',
                    error_events=sum('error' in r for r in rows), is_estimate=True,
                    integrity_fallbacks=len(list((self.path.parent/'integrity_fallbacks').glob('*.json'))),
                    pricing_source=PRICING_URL, pricing_checked_on='2026-10-29')


class VolcengineTranslationClient:
    def __init__(self, config, cache_dir, transport=None):
        self.config, self.transport = config, transport
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.calls = self.cache / 'volcengine_usage.jsonl'
        self.ledger = TranslationUsage(self.calls)
        self.budget = BudgetGuard(config, self.cache.resolve())
        self.lock, self.request_lock = threading.Lock(), threading.Lock()
        self.cancel = threading.Event()
        self.last_request = 0

    def record(self, **row):
        row['time'] = datetime.now(timezone.utc).isoformat()
        with self.lock, self.calls.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    def _translate_segments(self, text, protected, visible_terms, item, rejected):
        """Translate original prose again with protected spans kept locally.

        This preserves exact source literals and their order. Sentence context
        is reduced, so every such fallback remains a semantic review item.
        """
        literals=sorted({*protected,*visible_terms},key=len,reverse=True)
        if not literals:
            raise DecisionUncertain('火山译文没有可安全分离的保护内容')
        pattern=re.compile('('+'|'.join(re.escape(t) for t in literals)+')')
        parts=[]
        term_queues={value:iter(names) for value,names in visible_terms.items()}
        source_parts=pattern.split(text)
        for number,part in enumerate(source_parts):
            if part in visible_terms:
                parts.append('⟪TERM:'+next(term_queues[part])+'⟫')
                continue
            if part in protected or not re.search(r'[A-Za-z]',part):
                parts.append(part)
                continue
            leading=part[:len(part)-len(part.lstrip())]
            trailing=part[len(part.rstrip()):]
            prose=part.strip()
            if (number==len(source_parts)-1 and number>0
                    and source_parts[number-1] in protected and prose in {'times?','times.'}):
                # A terminal count after a preserved number/symbol is a
                # repetition suffix, not a standalone query to MT.
                translated='次'+{'?':'？','.':'。'}[prose[-1]]
            else:
                translated=' '.join(self._request([piece])[0] for piece in chunks(prose))
            parts.append(leading+normalize_written_numbers(prose,translated)+trailing)
        result=''.join(parts)
        save(self.cache/'integrity_fallbacks'/(fingerprint(dict(item=item,text=text))+'.json'),
             dict(method='original_prose_segments_v1',item=item,prepared_text=text,
                  rejected_result=rejected,result=result,semantic_review_required=True))
        return result

    def _request(self, texts):
        if not texts or len(texts) > 16 or sum(map(len, texts)) > 5000:
            raise ValueError('Volcengine request exceeds text limits')
        body = dict(SourceLanguage='en', TargetLanguage='zh', TextList=texts)
        key = fingerprint(dict(method=REQUEST_METHOD, url=URL, body=body))
        path = self.cache / (key + '.json')
        # Serialize duplicate calls and throttle this client below the 10 QPS limit.
        with self.request_lock:
            if path.exists():
                cached = read(path)
                result = cached.get('translations')
                if (cached.get('key') == key and cached.get('sha256') == fingerprint(result) and
                        isinstance(result, list) and len(result) == len(texts) and
                        all(isinstance(t, str) and t.strip() for t in result)):
                    self.record(cached=True)
                    return result
                raise APIFailure('火山翻译缓存已改变，请复核缓存文件')
            if self.cancel.is_set():
                raise APIFailure('Conversion cancelled')
            load_environment()
            ak, sk = os.environ.get('VOLCENGINE_ACCESS_KEY_ID'), os.environ.get('VOLCENGINE_SECRET_ACCESS_KEY')
            if not ak or not sk:
                raise APIFailure('火山翻译需要 VOLCENGINE_ACCESS_KEY_ID 和 VOLCENGINE_SECRET_ACCESS_KEY')
            quota = CharacterQuota(self.budget, hashlib.sha256(ak.encode()).hexdigest(),
                                   self.config.get('translation_service', {}).get('free_quota'))
            count = sum(map(len, texts))
            quota_token, free = quota.reserve(count)
            amount = (count-free) * PRICE_PER_MILLION / 1_000_000
            try:
                token = self.budget.reserve_amount('volcengine-translate-text', amount, wait_for_inflight=True)
            except BaseException:
                quota.release_unsubmitted(quota_token)
                raise
            encoded = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
            # No exception text, request headers or credential values enter logs.
            try:
                delay = .15 - (time.monotonic()-self.last_request)
                if delay > 0:
                    time.sleep(delay)
                self.last_request = time.monotonic()
                if self.transport:
                    data = self.transport(body)
                else:
                    import httpx
                    with httpx.Client(transport=httpx.HTTPTransport(proxy=api_proxy(URL)),
                                      timeout=self.config['timeout_seconds']) as http:
                        response = http.post(URL, headers=signed_headers(encoded, ak, sk), content=encoded)
                        response.raise_for_status()
                        data = response.json()
                if data.get('ResponseMetadata', {}).get('Error'):
                    raise APIFailure('火山翻译请求被拒绝，请检查服务开通及翻译权限')
            except BaseException as error:
                self.budget.settle_amount(token)
                self.record(characters=count, free_characters=free, conservative_cny=amount,
                            error=type(error).__name__)
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise APIFailure('火山翻译调用失败（' + type(error).__name__ + '），已保留检查点和费用预留') from None
            self.budget.settle_amount(token, amount)
            self.record(characters=count, free_characters=free, conservative_cny=amount,
                        request_id=data.get('ResponseMetadata', {}).get('RequestId'))
            translations = data.get('TranslationList')
            if (not isinstance(translations, list) or len(translations) != len(texts) or
                    any(not isinstance(r, dict) or not isinstance(r.get('Translation'), str) or
                        not r['Translation'].strip() for r in translations)):
                raise DecisionUncertain('火山翻译结果条数或正文不完整')
            result = [row['Translation'] for row in translations]
            save(path, dict(key=key, translations=result, sha256=fingerprint(result)))
            return result

    def ask(self, task, system, payload, images=(), validator=None, **kwargs):
        """Adapt only translation tasks; never route layout/vision to TranslateText."""
        if task != 'translate-book-zh' or images:
            raise ValueError('Volcengine client supports text translation only')
        output, terms, prepared, pending = [], {}, [], []
        for item in payload['items']:
            text = item['text']
            if 'ZXQREF' in text:
                raise DecisionUncertain('原文与翻译保护标记冲突')
            protected = {}
            def opaque(value):
                token = f'ZXQREF{len(protected):06d}QXZ'
                protected[token] = value
                return token
            text = re.sub(r'⟪KEEP_\d+⟫', lambda m: opaque(m[0]), text)
            text=normalize_prose_wraps(text)
            glossary = payload.get('glossary', {})
            visible_terms = {}
            if glossary:
                alternatives = '|'.join(re.escape(t) for t in sorted(glossary, key=len, reverse=True))
                pattern = re.compile(r'(?<!\w)(?:' + alternatives + r')(?!\w)', re.I)
                lookup = {term.casefold():term for term in glossary}
                def term(match):
                    english = lookup[match[0].casefold()]
                    terms[english] = glossary[english]
                    # MT can drop opaque noun phrases in short titles. Supply
                    # readable canonical terminology and verify every instance
                    # before restoring the same terminology contract locally.
                    visible_terms.setdefault(glossary[english], []).append(english)
                    return glossary[english]
                text = pattern.sub(term, text)
            pieces = chunks(text)
            prepared.append((item, protected, visible_terms, len(pieces), text))
            remainder=re.sub(r'ZXQREF\d{6}QXZ','',text)
            for value in visible_terms:remainder=remainder.replace(value,'')
            needs_translation=bool(re.search(r'[A-Za-z]',remainder))
            pending.extend((piece,needs_translation) for piece in pieces)
        translated, group = [], []
        for piece,needs_translation in pending:
            if not needs_translation:
                if group:translated.extend(self._request(group));group=[]
                translated.append(piece)
                continue
            if group and (len(group) == 16 or sum(map(len, group))+len(piece) > 5000):
                translated.extend(self._request(group)); group = []
            group.append(piece)
        if group:
            translated.extend(self._request(group))
        offset = 0
        for item, protected, visible_terms, count, prepared_text in prepared:
            result = ' '.join(translated[offset:offset+count])
            offset += count
            # MT may insert spaces inside opaque ASCII words. Canonicalize
            # whitespace only when every non-space character matches a known
            # token, then enforce its exact identity and occurrence count.
            original_result = result
            token_pattern = r'Z\s*X\s*Q\s*R\s*E\s*F\s*'+r'\d\s*'*6+r'Q\s*X\s*Z'
            def canonical_token(match):
                token=re.sub(r'\s+', '',match[0])
                return token if token in protected else match[0]
            result=re.sub(token_pattern,canonical_token,result)
            if result != original_result:
                save(self.cache/'normalizations'/(fingerprint(dict(item=item,result=original_result))+'.json'),
                     dict(method='opaque_token_whitespace_only_v1', item=item,
                          original_result=original_result, normalized_result=result))
            pattern = re.compile('|'.join(re.escape(t) for t in sorted(visible_terms,key=len,reverse=True))) if visible_terms else None
            expected_terms=Counter({text:len(names) for text,names in visible_terms.items()})
            original=item['text']
            for token,value in item.get('protected_values',{}).items():original=original.replace(token,value)
            from book_pipeline.ingestion.translation import author_names_only
            untranslated=(bool(protected or visible_terms) and not re.search(r'[\u3400-\u9fff]',result)
                          and len(re.findall(r'[A-Za-z]+',item['text']))>4 and not author_names_only(original))
            segmented=False
            if (Counter(re.findall(r'ZXQREF\d{6}QXZ', result)) != Counter(protected.keys())
                    or (pattern and Counter(m[0] for m in pattern.finditer(result)) != expected_terms)
                    or untranslated):
                result=self._translate_segments(prepared_text,protected,visible_terms,item,result)
                segmented=True
            if Counter(re.findall(r'ZXQREF\d{6}QXZ', result)) != Counter(protected.keys()):
                save(self.cache/'invalid'/(fingerprint(dict(item=item,result=result))+'.json'),
                     dict(validation_error='opaque_placeholder_identity_or_count', item=item,
                          protected=protected, result=result))
                raise DecisionUncertain('火山翻译改变了公式或术语保护标记')
            for token, value in protected.items():
                result = result.replace(token, value)
            if visible_terms:
                if segmented:
                    actual=Counter(re.findall(r'⟪TERM:([^⟫]+)⟫',result))
                    if actual != Counter(name for names in visible_terms.values() for name in names):
                        raise DecisionUncertain('分段重译改变了本地保留的术语标记')
                else:
                    actual=Counter(m[0] for m in pattern.finditer(result))
                    if actual != expected_terms:
                        raise DecisionUncertain('火山翻译改变了规范术语或其出现次数')
                    queues={text:iter(names) for text,names in visible_terms.items()}
                    result=pattern.sub(lambda m:'⟪TERM:'+next(queues[m[0]])+'⟫',result)
            result = normalize_written_numbers(item['text'], result)
            output.append(dict(id=item['id'], text=result))
        answer = dict(items=output, terms=terms)
        if validator:
            try:
                validator(answer)
            except (ValueError, KeyError, TypeError) as error:
                save(self.cache/'invalid'/(fingerprint(payload)+'.json'),
                     dict(validation_error=str(error), payload=payload, result=answer))
                raise DecisionUncertain('火山译文未通过原文完整性检查：' + str(error)[:240]) from None
        return answer
