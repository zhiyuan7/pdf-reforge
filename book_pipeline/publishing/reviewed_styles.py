"""Source-bound typography additions backed by a recorded visual review."""
from book_pipeline.config import fingerprint
from book_pipeline.publishing.raster_styles import style_ranges


def validate_reviewed_styles(index, review, corrections=()):
    if not isinstance(review, dict) or review.get('source_sha256') != fingerprint(index.source):
        raise ValueError('Reviewed styles do not match the edited source')
    if not isinstance(review.get('evidence'), str) or not review['evidence'].strip():
        raise ValueError('Reviewed styles require visual evidence')
    items = [dict(atom=a['id'], text=a['content']) for a in index.atoms if a['kind'] == 'text']
    answer = dict(runs=review.get('runs'), uncertain_atoms=[])
    ranges = style_ranges(answer, items)
    for atom, runs in ranges.items():
        for run in runs:
            if any(c['atom'] == atom and c['start_offset'] < run['end']
                   and run['start'] < c['end_offset'] for c in corrections):
                raise ValueError('Reviewed style overlaps a text correction')
            run['evidence'] = review['evidence']
    return ranges


def apply_reviewed_styles(text, automatic, reviewed):
    """A source-bound visual review takes precedence on exactly its bytes."""
    cuts = sorted({n for r in [*automatic, *reviewed] for n in (r['start'], r['end'])})
    result = []
    for a, z in zip(cuts, cuts[1:]):
        active = [r for r in reviewed if r['start'] <= a < z <= r['end']]
        if not active:
            active = [r for r in automatic if r['start'] <= a < z <= r['end']]
        if active:
            result.append(dict(active[0], start=a, end=z, quote=text[a:z]))
    return result
