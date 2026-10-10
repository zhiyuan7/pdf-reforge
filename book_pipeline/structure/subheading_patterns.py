"""Numbering forms shared by discovery, prefix matching and presentation."""
from functools import lru_cache
from itertools import product
import hashlib
import re
import unicodedata


NUMERALS = '零〇一二三四五六七八九十百千万亿两壹贰叁肆伍陆柒捌玖拾佰仟'
GREEK = 'Α-ΡΣ-Ωα-ωϐ-ϖϰ-ϵ'
TOKENS = {
    '数字': r'\d+',
    '汉字': '[' + NUMERALS + ']+',
    '小写字母': r'[a-z]',
    '大写字母': r'[A-Z]',
    # Read old candidate lists and style settings; matching lists expand this alias.
    '字母': r'[A-Za-z]',
    '罗马数字': r'(?:[IVXLCDM]+|[ivxlcdm]+)',
    '希腊字母': '[' + GREEK + ']',
}
TOKEN = '|'.join(sorted(TOKENS, key=len, reverse=True))
DEFAULT_TOKENS = tuple(token for token in TOKENS if token != '字母')
ROMAN = re.compile(r'M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})')
FORM = re.compile(r'(?P<open>\()?((?:' + TOKEN + r')(?:\.(?:' + TOKEN + r')){0,5})(?P<end>[.、):]?)')
GREEK_COMMANDS = dict(zip(
    ('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega').split(),
    'αβγδεζηθικλμνξοπρστυφχψω'))
GREEK_COMMANDS.update(dict(Gamma='Γ', Delta='Δ', Theta='Θ', Lambda='Λ', Xi='Ξ', Pi='Π', Sigma='Σ', Upsilon='Υ', Phi='Φ', Psi='Ψ', Omega='Ω',
                          varepsilon='ϵ', vartheta='ϑ', varpi='ϖ', varrho='ϱ', varsigma='ς', varphi='ϕ'))


def canonical_pattern(pattern):
    """Use ASCII parentheses/dots; retain the ideographic list separator."""
    return unicodedata.normalize('NFKC', pattern) if isinstance(pattern, str) else None


@lru_cache(maxsize=2048)
def _parts(pattern):
    pattern = canonical_pattern(pattern)
    if not pattern:
        return None
    match = FORM.fullmatch(pattern)
    if not match or (match['open'] and match['end'] != ')'):
        return None
    return match.group(2).split('.'), bool(match['open']), match['end']


def pattern_depth(pattern):
    parts = _parts(pattern) if isinstance(pattern, str) else None
    return len(parts[0]) if parts else None


def expand_letter_patterns(patterns):
    """Upgrade the old case-insensitive alias, including mixed nested forms."""
    result = set()
    for pattern in patterns:
        tokens, wrapped, suffix = _parts(pattern)
        choices = [('小写字母', '大写字母') if token == '字母' else (token,) for token in tokens]
        result.update(('(' if wrapped else '') + '.'.join(items) + suffix for items in product(*choices))
    return result


def _prefix_view(text, with_offsets=False):
    """Normalize a matching view, optionally retaining raw character addresses."""
    if not with_offsets:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\\(?:text|mathrm|operatorname|mathit|mathbf)\s*\{\s*([^{}]*?)\s*\}', r'\1', text)
        text = re.sub(r'\\([A-Za-z]+)', lambda m: GREEK_COMMANDS.get(m[1], m[0]), text)
        return re.sub(r'\\[()[\]]|\$', '', text).lstrip()
    chars, offsets = [], []
    for n, char in enumerate(text):
        normalized = unicodedata.normalize('NFKC', char)
        chars.extend(normalized)
        offsets.extend([(n, n + 1)] * len(normalized))
    view = ''.join(chars)

    def replace(expression, replacement):
        nonlocal view, offsets
        chunks, positions, cursor = [], [], 0
        for match in re.finditer(expression, view):
            chunks.append(view[cursor:match.start()])
            positions.extend(offsets[cursor:match.start()])
            value, spans = replacement(match)
            chunks.append(value)
            positions.extend(spans)
            cursor = match.end()
        chunks.append(view[cursor:])
        positions.extend(offsets[cursor:])
        view, offsets = ''.join(chunks), positions

    replace(r'\\(?:text|mathrm|operatorname|mathit|mathbf)\s*\{\s*([^{}]*?)\s*\}',
            lambda m: (m[1], offsets[m.start(1):m.end(1)]))
    replace(r'\\([A-Za-z]+)', lambda m: (
        GREEK_COMMANDS[m[1]], [(offsets[m.start()][0], offsets[m.end() - 1][1])]
    ) if m[1] in GREEK_COMMANDS else (m[0], offsets[m.start():m.end()]))
    replace(r'\\[()[\]]|\$', lambda m: ('', []))
    leading = len(view) - len(view.lstrip())
    view, offsets = view[leading:], offsets[leading:]
    return view, offsets


@lru_cache(maxsize=2048)
def _prefix_expression(pattern):
    parts = _parts(pattern) if isinstance(pattern, str) else None
    if not parts:
        return None
    tokens, wrapped, suffix = parts
    expression = (r'\(\s*' if wrapped else '') + r'\s*\.\s*'.join(
        '(?P<n' + str(n) + '>' + TOKENS[t] + ')' for n, t in enumerate(tokens))
    if suffix:
        expression += r'\s*' + re.escape(suffix)
        if suffix == '.':
            # A one-level numeric label must not consume the start of 3.1.
            expression += r'(?!\s*\d)'
    elif len(tokens) == 1:
        expression += r'(?=\s|$)'
    else:
        last_char = TOKENS[tokens[-1]]
        next_char = r'\dA-Za-z' + NUMERALS + GREEK
        expression += r'(?!' + last_char + r'|\s*\.\s*[' + next_char + '])'
    return re.compile(expression)


def match_prefix(text, pattern):
    return _match_prefix_view(_prefix_view(text), pattern)


def _match_prefix_view(view, pattern):
    expression = _prefix_expression(pattern) if isinstance(pattern, str) else None
    match = expression.match(view) if expression else None
    if match:
        for n, token in enumerate(_parts(pattern)[0]):
            if token == '罗马数字':
                numeral = match['n' + str(n)]
                # Common single labels I/V/X remain Roman; C/D/L/M belong
                # to alphabetic lists. Multi-letter values must be canonical.
                if numeral.upper() in {'C', 'D', 'L', 'M'} or not ROMAN.fullmatch(numeral.upper()):
                    return None
    return match


def prefix_source_end(text, pattern):
    """End of the matched numbering prefix in the unmodified source text."""
    expression = _prefix_expression(pattern) if isinstance(pattern, str) else None
    if not expression:
        return None
    view, offsets = _prefix_view(text, with_offsets=True)
    match = _match_prefix_view(view, pattern)
    return offsets[match.end() - 1][1] if match else None


def first_prefix_pattern(text, patterns):
    view = _prefix_view(text)
    return next((pattern for pattern in patterns if _match_prefix_view(view, pattern)), None)


def numbering_values(text, pattern):
    """Positive ordinals with token/case identity for sibling sequences."""
    match = match_prefix(text, pattern)
    if not match:
        return None
    chinese_digits = dict(zip('零〇一二三四五六七八九两壹贰叁肆伍陆柒捌玖',
                              (0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9)))
    chinese_units = dict(十=10, 拾=10, 百=100, 佰=100, 千=1000, 仟=1000, 万=10000, 亿=100000000)
    greek = 'αβγδεζηθικλμνξοπρστυφχψω'
    variants = str.maketrans('ϐϵϑϰϖϱςϕ', 'βεθκπρσφ')
    values = []
    for n, token in enumerate(_parts(pattern)[0]):
        value = match['n' + str(n)]
        case = ''
        if token == '数字':
            ordinal = int(value)
        elif token == '汉字':
            if all(c in chinese_digits for c in value):
                ordinal = int(''.join(str(chinese_digits[c]) for c in value))
            else:
                total = section = digit = 0
                for char in value:
                    if char in chinese_digits:
                        digit = chinese_digits[char]
                    else:
                        unit = chinese_units[char]
                        if unit < 10000:
                            section += (digit or 1) * unit
                        elif unit == 10000:
                            total += (section + digit) * unit
                            section = 0
                        else:
                            total = (total + section + digit) * unit
                            section = 0
                        digit = 0
                ordinal = total + section + digit
        elif token == '罗马数字':
            weights = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)
            digits = [weights[c] for c in value.upper()]
            ordinal = sum(-v if i + 1 < len(digits) and v < digits[i + 1] else v
                          for i, v in enumerate(digits))
            case = 'upper' if value.isupper() else 'lower'
        elif token == '希腊字母':
            letter = value.lower().translate(variants)
            if letter not in greek:
                return None
            ordinal = greek.index(letter) + 1
            case = 'upper' if value.isupper() else 'lower'
        else:
            ordinal = ord(value.lower()) - ord('a') + 1
            case = 'upper' if value.isupper() else 'lower'
        if ordinal <= 0:
            return None
        values.append((token, ordinal, case))
    return tuple(values)


def pattern_sort_key(pattern):
    return pattern_depth(pattern) or 99, canonical_pattern(pattern) or ''


def pattern_kind(pattern):
    """Stable URL-safe sample/filter ids, keeping existing numeric sample URLs."""
    pattern = canonical_pattern(pattern)
    if pattern == '数字.' or re.fullmatch(r'数字(?:\.数字){1,5}', pattern or ''):
        return 'subheading-' + str(pattern_depth(pattern))
    return 'subheading-' + hashlib.sha256(pattern.encode('utf-8')).hexdigest()[:12]
