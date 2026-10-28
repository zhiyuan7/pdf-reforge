"""Exact, source-bound math annotations for visually reviewed table cells.

Only wrap existing literal bytes in <eq>; never accept replacement TeX or
infer variables from capitalization. Raw OCR HTML and source offsets stay intact.
"""
from html.parser import HTMLParser
import re

from book_pipeline.config import fingerprint


class CellText(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=False)
        self.lines = [0]
        self.lines.extend(m.end() for m in re.finditer('\n', html))
        self.row, self.column, self.cell, self.eq = -1, -1, False, 0
        self.parts = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row += 1
            self.column = -1
        elif tag in {'td', 'th'}:
            self.column += 1
            self.cell = True
        elif tag == 'eq':
            self.eq += 1

    def handle_endtag(self, tag):
        if tag in {'td', 'th'}:
            self.cell = False
        elif tag == 'eq':
            self.eq -= 1

    def handle_data(self, data):
        if self.cell and not self.eq:
            line, column = self.getpos()
            self.parts.append((self.row, self.column, self.lines[line-1]+column, data))


def validate_reviewed_table_math(index, review):
    return validate_table_ranges(index, review, 'table_math', 'reviewed_literal_table_math_v1')


def validate_reviewed_table_styles(index, review):
    return validate_table_ranges(index, review, 'table_styles', 'reviewed_literal_table_style_v1')


def validate_table_ranges(index, review, field, method):
    if not isinstance(review, dict) or review.get('source_sha256') != fingerprint(index.source):
        raise ValueError('Reviewed table math does not match the edited source')
    if not isinstance(review.get('evidence'), str) or not review['evidence'].strip():
        raise ValueError('Reviewed table math requires visual evidence')
    rows = review.get(field, [])
    if not isinstance(rows, list):
        raise ValueError('Reviewed table math requires a list')
    result, parsed = {}, {}
    for row in rows:
        if field == 'table_styles' and (not isinstance(row, dict)
                or type(row.get('italic')) is not bool or type(row.get('bold')) is not bool
                or not (row['italic'] or row['bold'])):
            raise ValueError('Reviewed table styles require explicit positive font choices')
        if (not isinstance(row, dict) or not isinstance(row.get('atom'), str)
                or row['atom'] not in index.ids):
            raise ValueError('Reviewed table math must address an existing table')
        atom = index.atoms[index.ids[row['atom']]]
        if atom['kind'] != 'table':
            raise ValueError('Reviewed table math must address an existing table')
        quote = row.get('quote')
        if (not isinstance(quote, str) or not quote or quote != quote.strip()
                or re.search(r'[<>&\\${}%#\n]', quote)
                or any(type(row.get(k)) is not int or row[k] < 0
                       for k in ('row', 'column', 'occurrence'))):
            raise ValueError('Invalid reviewed table math quote or cell')
        if atom['id'] not in parsed:
            from book_pipeline.publishing.latex import table_tex
            if table_tex(atom['content']) is None:
                raise ValueError('Reviewed table math requires a supported regular table')
            parsed[atom['id']] = CellText(atom['content']).parts
        matches = [(offset+m.start(), offset+m.end())
                   for r,c,offset,text in parsed[atom['id']]
                   if (r,c) == (row['row'], row['column'])
                   for m in re.finditer(re.escape(quote), text)]
        if row['occurrence'] >= len(matches):
            raise ValueError('Reviewed table math quote is absent from the plain cell text')
        a,z = matches[row['occurrence']]
        previous = result.setdefault(atom['id'], [])
        if any(r['start'] < z and a < r['end'] for r in previous):
            raise ValueError('Reviewed table math ranges overlap')
        previous.append(dict(start=a, end=z, quote=quote, row=row['row'], column=row['column'],
                             evidence=review['evidence'], method=method,
                             **({k:row[k] for k in ('italic','bold')} if field == 'table_styles' else {})))
    return {atom: sorted(rows, key=lambda r:r['start']) for atom,rows in result.items()}


def annotate_table_math(html, ranges):
    for row in reversed(ranges):
        a,z = row['start'],row['end']
        if html[a:z] != row['quote']:
            raise ValueError('Reviewed table math no longer matches the rendered table')
        if row['method'] == 'reviewed_literal_table_style_v1':
            tags = (['i'] if row['italic'] else []) + (['b'] if row['bold'] else [])
        else:
            tags = ['eq']
        html = html[:a]+''.join('<'+t+'>' for t in tags)+html[a:z]+''.join('</'+t+'>' for t in reversed(tags))+html[z:]
    return html
