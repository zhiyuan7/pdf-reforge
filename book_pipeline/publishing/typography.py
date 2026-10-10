"""Point sizes shared by the editor, block preview and complete book rendering."""
import math
from book_pipeline.structure.subheading_patterns import canonical_pattern, expand_letter_patterns, pattern_depth

DEFAULT_TYPOGRAPHY = dict(body_size=10.5, subheading_sizes={'数字.': 15, '数字.数字': 13, '数字.数字.数字': 12})


def trim_title_punctuation(text):
    """Remove trailing separators, keeping inner dots, primes and paired marks."""
    end = len(text)
    while end and (text[end - 1].isspace() or text[end - 1] in '.,;:!?，。．、；：！？…⋯'):
        end -= 1
    return text[:end]


def heading_pattern(item):
    if item.get('subheading_pattern'):
        return canonical_pattern(item['subheading_pattern'])
    depth = max(1, min(6, len(str(item.get('number', '')).rstrip('.').split('.'))))
    return '数字.' if depth == 1 else '.'.join(['数字'] * depth)


def recommended_size(body, pattern):
    return round(body + {1: 4.5, 2: 2.5, 3: 1.5}.get(pattern_depth(pattern), 1), 1)


def validate_typography(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict) or set(value) - {'body_size', 'subheading_sizes'}:
        raise ValueError('字号设置仅支持正文和各编号形式的小标题。')
    body = value.get('body_size', 10.5)
    if isinstance(body, bool) or not isinstance(body, (int, float)) or not math.isfinite(body) or not 9 <= body <= 18:
        raise ValueError('正文字号应在 9–18 pt 之间。')
    sizes = value.get('subheading_sizes', {})
    if not isinstance(sizes, dict):
        raise ValueError('请按小标题的编号形式设置字号。')
    for pattern, size in sizes.items():
        if not pattern_depth(pattern) or pattern != canonical_pattern(pattern) or isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or not body < size <= 36:
            raise ValueError('小标题字号须大于正文字号，且不超过 36 pt。')
    defaults = {p: recommended_size(body, p) for p in DEFAULT_TYPOGRAPHY['subheading_sizes']}
    expanded = {p: size for pattern, size in sizes.items() for p in expand_letter_patterns([pattern])}
    # Explicit case settings override the migrated case-insensitive size.
    expanded.update({p: size for p, size in sizes.items() if expand_letter_patterns([p]) == {p}})
    return dict(body_size=body, subheading_sizes={**defaults, **expanded})


def subheading_size(settings, item):
    pattern = heading_pattern(item)
    return settings['subheading_sizes'].get(pattern, recommended_size(settings['body_size'], pattern))
