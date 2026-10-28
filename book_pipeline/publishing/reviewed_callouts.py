"""Source-bound anchors for manually reviewed scan callouts omitted by OCR."""
from book_pipeline.ingestion.panels import intersection_fraction
from book_pipeline.ingestion.table_prose import valid_box


def missing_reference(index, note, row, evidence):
    if (not isinstance(evidence,str) or not evidence.strip() or not note.get('marker')
            or note.get('status') != 'unmatched' or note.get('reason') != 'reference_missing'):
        raise ValueError('A missing callout needs visual evidence and an unmatched labeled note')
    atom = index.atoms[index.ids[row['atom']]]
    offset,before,after = row.get('offset'),row.get('before'),row.get('after')
    if (atom['kind'] != 'text' or atom['page']+1 != note['page']
            or atom.get('block_type') not in {'text','paragraph'}
            or type(offset) is not int or not 0 < offset <= len(atom['content'])
            or not isinstance(before,str) or not 1 <= len(before) <= 80
            or not isinstance(after,str) or len(after) > 80 or len(before+after) < 12
            or atom['content'][max(0,offset-len(before)):offset] != before
            or atom['content'][offset:offset+len(after)] != after
            or atom['content'].count(before+after) != 1
            or not valid_box(row.get('bbox'))
            or intersection_fraction(row['bbox'],atom.get('block_bbox') or atom.get('bbox')) < .98):
        raise ValueError('Reviewed missing callout must bind unique exact same-page prose and visible bounds')
    point = index.prefix[index.ids[atom['id']]]+offset
    return dict(atom=atom['id'],block=atom['group'],lo=point,hi=point,start_offset=offset,end_offset=offset,
                insert_after=point,kind='text',text=note['marker']['text'],page=note['page'],
                reviewed_missing_callout=True,evidence=evidence,marker_bbox=row['bbox'])
