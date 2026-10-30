"""Page-note decisions for rendering source atoms without an edited-text projection."""
from book_pipeline.structure.section_review import split_source
from book_pipeline.structure.footnotes import classify_footnotes


def source_page_footnotes(index, exclusions=()):
    plan = split_source(index, dict(headings=[]))
    # Excluded source blocks cannot own either a note or a reference. This
    # prevents an original contents page from creating an invisible callout.
    plan['blocks'] = [b for b in plan['blocks']
                      if not any(e['lo'] < b['hi'] and b['lo'] < e['hi'] for e in exclusions)]
    notes = classify_footnotes(index, plan)['footnotes']['items']
    result = []
    for note in notes:
        item = {**note, 'anchor': 'book-'+note['id'],
                'reference_anchor': 'ref-'+note['id'],
                'source_lo': note['lo'], 'source_hi': note['hi'], 'insert_after': note['lo']}
        if note['status'] == 'matched':
            reference = note['reference']
            item['reference'] = {**reference, 'edited_lo': reference['lo'], 'edited_hi': reference['hi']}
            item['insert_after'] = reference['insert_after']
            item['marker_lo'] = note['lo']+note['marker']['start']
            item['marker_hi'] = note['lo']+note['marker']['end']
        result.append(item)
    return result
