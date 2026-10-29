"""Wrap literal prose in simple mixed displays without shrinking the letters."""
import re


def prose_math_parts(content):
    """Split only top-level, plain ``\\text{...}`` arguments.

    Nested math text, delimiter pairs, arrays, tags and styled TeX retain the
    regular math renderer. No formula token is interpreted or rewritten.
    """
    if re.search(r'\\(?:begin|end|left|right|tag)\b|\\\\|[$&%#]', content):
        return None
    parts, cursor, depth, n = [], 0, 0, 0
    while n < len(content):
        if content[n] == '\\':
            match = re.match(r'\\text\s*\{([^{}\\$]*)\}', content[n:]) if depth == 0 else None
            if match:
                if n > cursor:
                    parts.append(('math', content[cursor:n]))
                parts.append(('text', match[1]))
                n += match.end()
                cursor = n
                continue
            command = re.match(r'\\(?:[A-Za-z]+|.)', content[n:])
            n += command.end() if command else 1
            continue
        if content[n] == '{':
            depth += 1
        elif content[n] == '}':
            depth -= 1
            if depth < 0:
                return None
        n += 1
    if depth:
        return None
    if cursor < len(content):
        parts.append(('math', content[cursor:]))
    prose = ''.join(value for kind, value in parts if kind == 'text')
    if (len(re.findall(r'[A-Za-z]+', prose)) < 6
            or not any(kind == 'math' and value.strip() for kind, value in parts)):
        return None
    # Nested text or a style command must not be split away from its owner.
    if any(re.search(r'\\(?:text\w*|mathrm|operatorname)\b', value)
           for kind, value in parts if kind == 'math'):
        return None
    return parts
