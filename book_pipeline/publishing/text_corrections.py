"""Validate exact source-bound render edits before applying any of them."""
from collections import defaultdict
from copy import deepcopy
import re

from book_pipeline.config import fingerprint

METHOD = 'exact_nonoverlapping_plain_label_corrections_v1'


def validate_corrections(index, correction):
    if not isinstance(correction, dict) or correction.get('source_sha256') != fingerprint(index.source):
        raise ValueError('Render text corrections do not match the edited source')
    items = correction.get('items')
    if not isinstance(items, list):
        raise ValueError('Render text corrections require an items list')
    intervals = defaultdict(list)
    for row in items:
        if not isinstance(row, dict) or not isinstance(row.get('atom'), str) or row['atom'] not in index.ids:
            raise ValueError('Render text correction requires an existing atom')
        atom = index.atoms[index.ids[row['atom']]]
        lo, hi = row.get('start_offset'), row.get('end_offset')
        if (any(isinstance(v, bool) or not isinstance(v, int) for v in (lo, hi))
                or atom['kind'] not in {'text', 'math', 'display_math'}
                or not 0 <= lo < hi <= len(atom['content'])
                or atom['content'][lo:hi] != row.get('expected')
                or not isinstance(row.get('replacement'), str)):
            raise ValueError('Render text correction does not match its exact source interval')
        if atom['kind'] != 'text':
            # Checking only the two endpoints would also accept an interval
            # spanning "label} + mathematical content + \\text{label".
            labels = [m.span(1) for m in re.finditer(r'(?<!\\)\\text\{([^{}\\$%&#_^~]*)\}', atom['content'])]
            if (lo, hi) not in labels or re.search(r'[{}\\$%&#_^~]', row['replacement']):
                raise ValueError('Math text corrections must address a complete plain text label')
        intervals[row['atom']].append((lo, hi))
    for spans in intervals.values():
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError('Render text corrections overlap in the original source')
    return deepcopy(items)
