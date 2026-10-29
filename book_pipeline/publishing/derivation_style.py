"""Keep single-column displayed derivations out of array's text style."""
import re


def display_derivation_arrays(tex):
    """Add only style declarations to flat, relation-led derivation rows.

    Matrices, cases, nested environments and explicit style choices retain
    their original rendering. No delimiters, row breaks or math are removed.
    """
    count = 0

    def array(match):
        nonlocal count
        before, after = tex[:match.start()].strip(), tex[match.end():].strip()
        # Only a whole displayed array, optionally in one delimiter pair.
        # An array inside a script, matrix, cases or style group is not a
        # top-level derivation and must keep its inherited math style.
        if before or after:
            if (not re.fullmatch(r'\\left\s*(?:[|(\[.]|\\(?:lvert|lVert|langle|\{))', before)
                    or not re.fullmatch(r'\\right\s*(?:[|)\].]|\\(?:rvert|rVert|rangle|\}))', after)):
                return match[0]
        body = match['body']
        if re.search(r'\\(?:begin|end|textstyle|scriptstyle|scriptscriptstyle|displaystyle)\b|(?<!\\)%', body):
            return match[0]
        rows, start, depth, i = [], 0, 0, 0
        while i < len(body):
            char = body[i]
            if char == '\\':
                if i+1 < len(body) and body[i+1] == '\\' and depth == 0:
                    # Optional row spacing/star syntax belongs to the break,
                    # so leave these arrays to their existing renderer.
                    if re.match(r'\s*[\[*]', body[i+2:]):
                        return match[0]
                    rows.append(body[start:i]); start = i+2
                i += 2
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth < 0:
                    return match[0]
            elif char == '&' and depth == 0:
                return match[0]
            i += 1
        rows.append(body[start:])
        if (depth or len(rows) < 2 or any(not r.strip() for r in rows)
                or not all(re.match(r'\s*(?:[=<>≤≥≈]|\\(?:leq?|geq?|equiv|approx|sim|simeq)\b)', r)
                           for r in rows[1:])
                or not re.search(r'\\(?:frac|sum|prod|int)\b', body)):
            return match[0]
        count += 1
        return match['open'] + r'\\'.join(r'\displaystyle ' + row for row in rows) + match['close']

    rendered = re.sub(
        r'(?P<open>\\begin\{array\}\s*\{\s*[lcr]\s*\})(?P<body>.*?)(?P<close>\\end\{array\})',
        array, tex, flags=re.S)
    return rendered, count
