"""Keep source-evidenced native equation rows out of a single shrinking box."""
from pathlib import Path
import re

from book_pipeline.ingestion.math_syntax import formula_tags, math_compatibility
from book_pipeline.structure.headings import valid_box


def spaced_rows(body):
    """Candidate rows separated by top-level spacing, never inside a group."""
    stack, cuts, i = [], [], 0
    while i < len(body):
        char = body[i]
        if char == '\\':
            command = re.match(r'\\(?:[a-zA-Z]+|.)', body[i:])
            if command:
                if command[0] == r'\qquad' and not stack:
                    cuts.append((i, i+len(command[0])))
                i += len(command[0]); continue
        if char in '{[(':
            stack.append(char)
        elif char in '}])':
            if not stack or stack.pop() != {'}':'{', ']':'[', ')':'('}[char]:
                return [body]
        i += 1
    if stack or not cuts:
        return [body]
    rows, start = [], 0
    for lo, hi in cuts:
        rows.append(body[start:lo]); start = hi
    return [*rows, body[start:]]


def native_math_rows(atom, content, source_pdf):
    if (atom['kind'] != 'display_math' or atom.get('extraction_method') != 'native'
            or content != atom['content'] or atom.get('recovery_issue')
            or not source_pdf or not Path(source_pdf).is_file()
            or not valid_box(atom.get('bbox'))
            or re.search(r'\\(?:begin|end)\b|(?<!\\)[%&]', content)):
        return None
    # Tag offsets must refer to these exact bytes, before compatibility edits.
    if math_compatibility(content)[0] != content:
        return None
    tags = formula_tags(content)
    if len(tags) > 1 or (tags and content[tags[0]['end']:].strip()):
        return None
    body = content[:tags[0]['start']] if tags else content
    spacing = False
    if r'\\' in body:
        lines = body.strip().split(r'\\')
        # Do not interpret mixed row delimiters or row-spacing arguments.
        if any('\n' in line.strip() or line.lstrip().startswith('[') for line in lines):
            return None
    else:
        lines = body.strip().splitlines()
        if len(lines) == 1:
            lines = spaced_rows(body.strip())
            spacing = len(lines) > 1
    if not 2 <= len(lines) <= 12 or len(body) > 3000:
        return None
    rows = []
    for line in lines:
        stack, relations, i = [], [], 0
        while i < len(line):
            char = line[i]
            if char == '\\':
                command = re.match(r'\\(?:[a-zA-Z]+|.)', line[i:])
                i += len(command[0]) if command else 1
                continue
            if char in '{[(':
                stack.append(char)
            elif char in '}])':
                if not stack or stack.pop() != {'}': '{', ']': '[', ')': '('}[char]:
                    return None
            elif char == '=' and not stack:
                relations.append(i)
            i += 1
        if stack or len(relations) != 1:
            return None
        position = relations[0]
        if not line[position+1:].strip() or (position and line[position-1] in '<>!='):
            return None
        rows.append(line[:position] + '&' + line[position:])
    # Newlines alone are ordinary TeX whitespace. Independent native equals
    # signs must witness multiple physical baselines inside this formula box.
    import pymupdf
    with pymupdf.open(source_pdf) as pdf:
        if not 0 <= atom['page'] < len(pdf):
            return None
        page = pdf[atom['page']]
        b = atom['bbox']
        box = pymupdf.Rect(b[0]*page.rect.width, b[1]*page.rect.height,
                           b[2]*page.rect.width, b[3]*page.rect.height) + (-.75, -.75, .75, .75)
        equals = [(c['origin'][1], s['size'], c['origin'][0]) for block in page.get_text('rawdict')['blocks']
                  for line in block.get('lines', []) if line.get('dir') == (1., 0.)
                  for s in line['spans'] for c in s['chars'] if c['c'] == '='
                  and box.contains(pymupdf.Point((c['bbox'][0]+c['bbox'][2])/2,
                                               (c['bbox'][1]+c['bbox'][3])/2))]
    # Summation limits contain smaller equals signs on their own baselines.
    # Only full-size relations witness equation rows; nested full-size equals
    # still make the evidence ambiguous and retain the conservative fallback.
    if equals:
        size = max(size for _, size, _ in equals)
        equals = [(y, s, x) for y, s, x in equals if s >= .85*size]
    if len(equals) != len(rows):
        return None
    baselines = []
    for y, size, x in sorted(equals):
        if not baselines or y-baselines[-1] > .4*size:
            baselines.append(y)
    if len(baselines) < 2:
        return None
    # Spacing alone often denotes side-by-side equations. Recover rows only
    # when every relation is on its own baseline and their x positions align.
    if spacing and (len(baselines) != len(lines)
                    or max(x for _,_,x in equals)-min(x for _,_,x in equals) > .5*size):
        return None
    return r'\begin{aligned}' + r'\\'.join(rows) + r'\end{aligned}' + (
        content[tags[0]['start']:] if tags else '')
