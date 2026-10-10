"""User-editable candidates matched locally at source block and line starts."""
import re

from book_pipeline.structure.numbered import DEFAULT_LABELS
from book_pipeline.structure.section_review import formal_view
from book_pipeline.structure.subheading_patterns import (
    DEFAULT_TOKENS, canonical_pattern, expand_letter_patterns, first_prefix_pattern, pattern_depth,
)


MINIMUM_OCCURRENCES = 10
EXCLUDED_KEYWORDS = frozenset({'习题', '练习', '问题'})
FORMULA_LABELS = {keyword: 'formula' for keyword in ('公式', '恒等式', '不等式', '等式', '式')}


def default_candidates(labels=None):
    labels = {**DEFAULT_LABELS, **FORMULA_LABELS, **(labels or {})}
    forms = [token + suffix for token in DEFAULT_TOKENS for suffix in ('', '.', '、', ')', ':')]
    forms += ['(' + token + ')' for token in DEFAULT_TOKENS]
    for depth in range(2, 7):
        bodies = ['.'.join([token] * depth) for token in DEFAULT_TOKENS]
        bodies += [token + '.' + '.'.join(['数字'] * (depth - 1)) for token in DEFAULT_TOKENS if token != '数字']
        for body in bodies:
            forms += [body, body + '.', '(' + body + ')']
    return dict(statements=ordered_keywords(k for k, kind in labels.items() if kind != 'example' and k not in EXCLUDED_KEYWORDS),
                examples=ordered_keywords(k for k, kind in labels.items() if kind == 'example' and k not in EXCLUDED_KEYWORDS),
                subheadings=ordered_patterns(forms))


def validate_candidates(value=None, labels=None):
    if value is None:
        value = default_candidates(labels)
    if not isinstance(value, dict) or set(value) != {'statements', 'examples', 'subheadings'}:
        raise ValueError('候选集合须包含 statements、examples、subheadings 三组。')
    result, seen = {}, set()
    for group in ('statements', 'examples', 'subheadings'):
        entries = value[group]
        if not isinstance(entries, list) or len(entries) > 1000:
            raise ValueError('每组候选集合须为列表，最多 1000 项。')
        cleaned = []
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip() or len(entry) > 80 or any(c in entry for c in '\n\r\t'):
                raise ValueError('每个候选关键词须为 1–80 个字符的单行文字。')
            keyword = entry.strip()
            # Also remove these labels from saved or explicitly supplied lists.
            if keyword in EXCLUDED_KEYWORDS:
                continue
            if group == 'subheadings':
                keyword = canonical_pattern(keyword)
                if not pattern_depth(keyword):
                    raise ValueError('小标题请使用通用编号形式，例如“数字.数字”“(汉字)”“小写字母.”：' + entry)
            if keyword in seen:
                raise ValueError('候选关键词不能重复或跨组重复：' + keyword)
            seen.add(keyword)
            cleaned.append(keyword)
        if group == 'subheadings':
            cleaned = expand_letter_patterns(cleaned)
            if len(cleaned) > 1000:
                raise ValueError('展开大小写后的编号形式最多 1000 项。')
        result[group] = ordered_patterns(cleaned) if group == 'subheadings' else ordered_keywords(cleaned)
    return result


def candidate_labels(candidates, custom=None):
    labels = {**DEFAULT_LABELS, **FORMULA_LABELS, **(custom or {})}
    return {**{k: labels.get(k, 'other') if labels.get(k) != 'example' else 'other'
               for k in candidates['statements']}, **{k: 'example' for k in candidates['examples']}}


def ordered_keywords(keywords):
    # Prefer a containing label (例题/注释) to its shorter prefix (例/注).
    return sorted(set(keywords), key=lambda keyword: (-len(keyword), keyword))


def ordered_patterns(patterns):
    # Roman numerals are more specific than Latin letters at an identical start.
    return sorted(set(patterns), key=lambda p: (-pattern_depth(p), -len(p), -p.count('罗马数字'), p))


def block_matches(block, candidates):
    """Yield at most one match per start, including starts of merged source blocks."""
    keywords = ordered_keywords(candidates['statements'] + candidates['examples'])
    views = [(k, formal_view(k)[0]) for k in keywords]
    text = block['text']
    source_starts = sorted({0, *block.get('block_starts', []), len(text)})
    starts = {0, *source_starts[:-1], *(m.end() for m in re.finditer('\n', text))}
    for start in sorted(starts):
        end = text.find('\n', start)
        end = len(text) if end < 0 else end
        end = min(end, next((offset for offset in source_starts if offset > start), len(text)))
        line = text[start:end]
        view, _ = formal_view(line)
        keyword = next((k for k, prefix in views if view.startswith(prefix)), None)
        if keyword is None:
            keyword = first_prefix_pattern(line, candidates['subheadings'])
        if keyword is not None:
            yield dict(keyword=keyword, offset=start, line_end=end)
