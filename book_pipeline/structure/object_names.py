"""Find explicit object names at declaration starts, retaining source offsets."""
from __future__ import annotations

import re
import unicodedata


# Other Unicode paired brackets are discovered by their character names below.
PAIRS = {'(': ')）', '（': '）)', '[': ']］', '［': '］]', '{': '}｝', '｛': '｝}',
         '“': '”', '‘': '’', '「': '」', '『': '』', '《': '》', '〈': '〉',
         '«': '»', '‹': '›', '„': '“', '‚': '‘', '〝': '〞〟',
         '"': '"', "'": "'", '`': '`', '**': '**', '*': '*',
         '<': '>', '＜': '＞', '——': '——', '—': '—', '–': '–'}
NAME_TERMS = re.compile(
    r'(?:定理|引理|命题|推论|原理|原则|公理|准则|法则|规则|定律|公式|恒等式|不等式|等式|'
    r'检验法|判别法|算法|方法|变换|条件|性质|定义|问题|方程|表示|分解|展开|'
    r'\b(?:theorem|lemma|proposition|corollary|principle|axiom|criterion|rule|law|formula|'
    r'identity|inequality|algorithm|method|transform|condition|definition|problem|equation))$', re.I)
STATEMENT = re.compile(
    r'^(?:设|假设|假定|若|如果|对于|对任意|任意|存在|令|已知|证明|的|显然|'
    r'我们|由此|因此|根据|可见|即|则|当.+时|'
    r'let\b|suppose\b|assume\b|given\b|if\b|for\s+(?:all|any)\b)|'
    r'(?:是|称为|称作|等于|属于|满足|意味着|有界|成立|不可能|充要条件为|推出)|'
    r'\b(?:is|are|has|have|must|implies|there\s+exists)\b', re.I)
FOOTNOTE = re.compile(r'\s*(?:\^\s*(?:\{\s*[①-⑳0-9*†‡]+\s*\}|[①-⑳0-9*†‡]+)|[①-⑳†‡]+)\s*$')


def closing_marks(opener):
    if opener in PAIRS:
        return PAIRS[opener]
    if len(opener) != 1 or unicodedata.category(opener) != 'Ps':
        return None
    name = unicodedata.name(opener, '')
    for left, right in [('LEFT', 'RIGHT'), ('OPENING', 'CLOSING')]:
        if left in name:
            try:
                return unicodedata.lookup(name.replace(left, right))
            except KeyError:
                pass
    return None


def opening_at(text, position):
    for mark in ('**', '——', text[position:position+1]):
        if text.startswith(mark, position) and closing_marks(mark):
            return mark
    return None


def enclosed(text, start):
    """Balanced brackets/quotes, including mixed CJK/ASCII bracket pairs."""
    opening = opening_at(text, start)
    if not opening:
        return None
    stack = [(opening, closing_marks(opening))]
    cursor = start + len(opening)
    while cursor < min(len(text), start+256):
        left, closers = stack[-1]
        apostrophe = (text[cursor] in "'’" and cursor > 0 and cursor+1 < len(text)
                      and text[cursor-1].isascii() and text[cursor+1].isascii()
                      and text[cursor-1].isalnum() and text[cursor+1].isalnum())
        # Multi-character markup closes as a unit; other closers are alternatives.
        close = left if len(left)>1 and closers==left else text[cursor] if text[cursor] in closers else None
        if close and not apostrophe and text.startswith(close, cursor):
            stack.pop()
            if not stack:
                return start+len(opening), cursor, cursor+len(close)
            cursor += len(close)
            continue
        nested = opening_at(text, cursor)
        word_dash = text[cursor] in '—–' and cursor>0 and cursor+1<len(text) and text[cursor-1].isalnum() and text[cursor+1].isalnum()
        if nested and not apostrophe and not word_dash:
            stack.append((nested, closing_marks(nested)))
            cursor += len(nested)
        else:
            cursor += 1
    return None


def trim_name(text, start, end):
    """Unwrap nested name punctuation and exclude a trailing footnote marker."""
    while True:
        while start < end and text[start].isspace(): start += 1
        while start < end and text[end-1].isspace(): end -= 1
        inner = enclosed(text[:end], start) if start < end else None
        if not inner or inner[2] != end:
            break
        start, end = inner[:2]
    marker = FOOTNOTE.search(text[start:end])
    if marker:
        end = start+marker.start()
    while start < end and text[end-1].isspace(): end -= 1
    return start, end


def plausible_name(text, parts, start, end, keyword):
    name = text[start:end]
    compact = ''.join(name.split())
    strong = bool(NAME_TERMS.search(name) or compact.endswith(keyword))
    if not compact or len(compact) > 120 or not re.search(r'[\u3400-\u9fffA-Za-z]', name):
        return False
    if re.match(r'^(?:的|显然|我们|由此|因此|根据|可见|即|则)',name):
        return False
    if (STATEMENT.search(name) and not strong) or re.search(r'[。！？!?；;，,:：=<>≤≥⇒⇔]', name):
        return False
    if re.match(r'^(?:若|如果|对于|对任意|任意|设)',name) and re.search(r'则|就|那么|成立',name):
        return False
    if re.fullmatch(r'[A-Za-z]', compact):
        return False
    # A formula condition cannot become a title merely because it is bracketed.
    cursor = 0
    math = False
    for part in parts:
        hi = cursor+len(part['text'])
        if cursor < end and hi > start and part['kind'] != 'text':
            if part['kind'] not in {'math', 'display_math'}:
                return False
            math = True
        cursor = hi
    if (math or re.search(r'\\|[_^{}]', name)) and not strong:
        return False
    return True


def explicit_name(text, label_end, parts, keyword, block_ends=()):
    """Return literal name/title/body offsets, or None for ordinary statements.

    Paired punctuation permits a short noun-like title. Unwrapped titles need
    a naming term and an explicit separator or a separate source block.
    """
    cursor = label_end
    while cursor < len(text) and (text[cursor].isspace() or text[cursor] in '.．。:：、'):
        cursor += 1
    if cursor >= len(text):
        return None
    prefix = cursor
    wrapped = enclosed(text,cursor)
    if not wrapped and text[cursor] in '—–-':
        cursor += 1
        while cursor < len(text) and text[cursor].isspace(): cursor += 1
        wrapped = enclosed(text,cursor)
    if not wrapped and opening_at(text,cursor):
        return None
    if wrapped:
        start, end, title_end = wrapped
        start, end = trim_name(text, start, end)
    else:
        boundary = re.search(r'[。．.:：\n—–]', text[cursor:cursor+160])
        ends = [cursor+boundary.start()] if boundary else []
        ends += [n for n in block_ends if cursor < n <= cursor+160 and n < len(text)]
        if not ends:
            return None
        title_end = min(ends)
        start, end = trim_name(text, cursor, title_end)
        if not NAME_TERMS.search(text[start:end]):
            return None
    if not plausible_name(text, parts, start, end, keyword):
        return None
    body = title_end
    while body < len(text) and (text[body].isspace() or text[body] in '.．。:：、—–'):
        body += 1
    # A quoted complete statement, with no following body, stays in the body.
    if body == len(text) and not NAME_TERMS.search(text[start:end]):
        return None
    return dict(name=text[start:end], start=start, end=end,
                title_start=prefix, title_end=title_end, body_offset=body)
