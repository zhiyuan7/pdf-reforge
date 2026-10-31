"""Offline heading registry from recognized source title blocks only."""
import re

from .headings import group_blocks, parse_number, span_for, valid_box


KINDS = ('part', 'chapter', 'section', 'subsection', 'subsubsection', 'paragraph')


def explicit_label(text):
    # A reference/heading such as “附录B.3.2” carries decimal depth; the
    # permissive appendix-root parser must not promote it to a new chapter.
    match = re.match(r'^\s*(?:附录\s*|Appendix\s+)([A-Z](?:\s*[.．]\s*\d+)+)(?![.．\d])\s*(.*)$', text, re.S)
    if match:
        number = re.sub(r'\s+', '', match[1]).replace('．', '.')
        return dict(kind=KINDS[min(5, number.count('.')+1)], number=number, title=match[2].strip())
    parsed = parse_number(text)
    if parsed['kind']:
        return parsed
    match = re.match(r'^\s*(?:Chapter|章节)\s*(\d+)\s*(.*)$', text, re.S | re.I)
    if match:
        return dict(kind='chapter', number=match[1], title=match[2].strip())
    match = re.match(r'^\s*Appendix\s+([A-Z])\s*(.*)$', text, re.S)
    if match:
        return dict(kind='chapter', number=match[1], title=match[2].strip())
    match = re.match(r'^\s*((?:\d+|[A-Z])(?:\s*[.．]\s*\d+)*)(?=\s|[^A-Za-z\d.．]|$)\s*(.*)$', text, re.S)
    if match:
        number = re.sub(r'\s+', '', match[1]).replace('．', '.')
        if number.isalpha():
            return dict(kind=None, number='', title=text.strip())
        depth = number.count('.')
        return dict(kind=KINDS[min(5, depth+1)], number=number, title=match[2].strip())
    return parsed


def recognized_heading_registry(index, excluded_pages=(), *, running_headers=None):
    """Keep inferred structure explicit, without semantic enrichment or new text.

    An explicit chapter/appendix label may be a separate source block from its
    title. Merge only adjacent headings with compatible page-local geometry;
    every original atom remains inside the single heading's source interval.
    """
    blocks = group_blocks(index)
    if running_headers is None:
        from .running_headers import running_header_rows
        running_headers = running_header_rows(index)
    header_groups = {r['group'] for r in running_headers}
    from .run_in_titles import run_in_title_rows, mixed_run_in_title_spans
    from book_pipeline.publishing.native_margin_notes import native_margin_evidence
    run_in_groups = {r['group'] for r in [*run_in_title_rows(index), *mixed_run_in_title_spans(index)]}
    margin_groups = set(native_margin_evidence(index))
    has_part = any(b['type'] in {'doc_title', 'paragraph_title'}
                   and b['page'] not in excluded_pages
                   and explicit_label(b['text'])['kind'] == 'part' for b in blocks)
    result = []
    consumed = set()
    for n, block in enumerate(blocks):
        if (n in consumed or block['group'] in run_in_groups | margin_groups | header_groups or block['page'] in excluded_pages
                or block['type'] not in {'doc_title', 'paragraph_title'}
                or not block['text'].strip()):
            continue
        lo, hi = index.span(span_for(index, block))
        parsed = explicit_label(block['text'])
        kind = parsed['kind'] or ('chapter' if block['type'] == 'doc_title' else 'subsection')
        text = block['text'].strip()
        groups = [block['group']]
        if kind in {'chapter', 'part'} and parsed['number'] and not parsed['title'] and n+1 < len(blocks):
            following = blocks[n+1]
            b, c = block.get('bbox'), following.get('bbox')
            if (following['type'] in {'doc_title', 'paragraph_title'}
                    and following['page'] == block['page'] and following['text'].strip()
                    and not explicit_label(following['text'])['kind']
                    and valid_box(b) and valid_box(c)
                    and 0 <= c[1]-b[3] <= .06 and abs(c[0]-b[0]) <= .025
                    and index.span(span_for(index, following))[0] == hi):
                hi = index.span(span_for(index, following))[1]
                text += ' ' + following['text'].strip()
                groups.append(following['group'])
                consumed.add(n+1)
        level = KINDS.index(kind)+1-(0 if has_part else 1)
        key = 'source-'+block['group']
        result.append(dict(kind=kind, id=key, anchor=key, lo=lo, hi=hi,
            heading_level=max(1, level), label=text, name=text,
            start=index.address(lo), end=index.address(hi, True),
            group=block['group'], source_groups=groups, page=block['page'], number=parsed['number']))
    return result
