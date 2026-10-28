"""Recover explicitly labelled formula grids from source geometry."""
import re

from book_pipeline.ingestion.math_syntax import formula_tags
from book_pipeline.ingestion.source import corrected_content


def horizontal_math_rows(index, decisions, barriers=(), skipped=()):
    """Keep consecutive (a), (b), ... cells on their evidenced source row.

    Require isolated display groups, disjoint horizontal boxes, aligned tops,
    consecutive explicit item labels and an enclosing prose column. A semantic
    boundary, exclusion, page, panel or ambiguous layout stops the row. No math
    is rewritten; individual cells still use the normal measured math renderer.
    """
    edits = {}
    for edit in decisions.get('corrections', []):
        edits.setdefault(edit['atom'], []).append(edit)
    repairs = decisions.get('formula_repairs', {})
    pending, rows = [], []

    def flush():
        if 2 <= len(pending) <= 4:
            boxes = [item['bbox'] for item in pending]
            left, right = boxes[0][0], boxes[-1][2]
            page = pending[0]['page']
            columns = [a.get('block_bbox') or a.get('bbox') for a in index.atoms
                       if a['page'] == page and a['kind'] == 'text'
                       and a['block_type'] in {'text', 'paragraph'} and not a.get('source_panel')]
            columns = [b for b in columns if b and b[0] <= left and right <= b[2]]
            column = max(columns, key=lambda b:b[2]-b[0], default=None)
            if column:
                width, cursor, cells = column[2]-column[0], column[0], []
                for item in pending:
                    box = item['bbox']
                    cells.append(dict(atom=item['id'], bbox=box,
                                      gap=(box[0]-cursor)/width, width=(box[2]-box[0])/width))
                    cursor = box[2]
                first, last = (index.ids[item['id']] for item in (pending[0], pending[-1]))
                rows.append(dict(method='source_horizontal_labelled_math_row',
                                 lo=index.prefix[first], hi=index.prefix[last+1],
                                 page=page, column_bbox=column, cells=cells))
        pending.clear()

    for n, atom in enumerate(index.atoms):
        lo, hi = index.prefix[n:n+2]
        content = repairs.get(atom['id'], corrected_content(atom, edits.get(atom['id'], [])))
        label = re.match(r'^\s*\\(?:text|mathrm)\{\(([a-z]|[0-9]{1,2})\)\}', content)
        box = atom.get('bbox')
        eligible = (atom['kind'] == 'display_math' and label and len(content) <= 800
                    and not formula_tags(content) and r'\tag' not in content
                    and index.groups[atom['group']] == [n]
                    and box and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
                    and box[2]-box[0] <= .4 and box[3]-box[1] <= .12
                    and not any(lo < b < hi for b in barriers)
                    and not any(a < hi and z > lo for a,z,*_ in skipped))
        if not eligible:
            flush()
            continue
        value = label[1]
        ordinal = ord(value) if value.isalpha() else int(value)
        if pending:
            previous = pending[-1]
            a = previous['bbox']
            overlap = min(a[3],box[3])-max(a[1],box[1])
            if (lo in barriers or previous['page'] != atom['page']
                    or previous.get('source_panel') != atom.get('source_panel')
                    or a[2]+.005 > box[0]
                    or overlap < .8*min(a[3]-a[1],box[3]-box[1])
                    or abs(a[1]-box[1]) > .25*max(a[3]-a[1],box[3]-box[1])
                    or ordinal != previous['_row_ordinal']+1
                    or value.isalpha() != previous['_row_alpha']):
                flush()
        pending.append({**atom, '_row_ordinal':ordinal, '_row_alpha':value.isalpha()})
    flush()
    return rows
