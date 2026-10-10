"""Recover known rejected candidates in anchored, consecutive sibling runs."""
from bisect import bisect_left
from collections import defaultdict


METHOD = 'numbering_continuity_v1'


def continuity_promotions(entries):
    """Never invent candidates or bridge a gap, reset, form or heading scope.

    Entries have source order, numbering values, original confirmation and
    eligibility. Each consecutive run needs a confirmed title as its anchor.
    Compound labels are reviewed first so recovered parents bound simple lists.
    """
    entries = sorted(entries, key=lambda e: e['lo'])
    accepted = {e['b'] for e in entries if e['confirmed']}
    promotions = {}

    def review(items, parents=None):
        groups = defaultdict(list)
        active_parents, segments = {}, defaultdict(int)
        parent_positions = {sid: [p['lo'] for p in scope] for sid, scope in (parents or {}).items()}
        for entry in items:
            values = entry['values']
            parent = values[:-1]
            if parents is not None:
                scope = parents.get(entry['section'], [])
                position = bisect_left(parent_positions.get(entry['section'], []), entry['lo'])
                parent = scope[position - 1]['b'] if position else None
            token, ordinal, case = values[-1]
            family = entry['section'], entry['keyword'], token, case
            if family in active_parents and active_parents[family] != parent:
                segments[family] += 1
            active_parents[family] = parent
            key = entry['section'], entry['keyword'], parent, token, case, segments[family]
            groups[key].append(entry)

        def finish(run, parent):
            anchors = [e['b'] for e in run if e['b'] in accepted]
            if not anchors:
                return
            for entry in run:
                if entry['b'] not in accepted:
                    promotions[entry['b']] = dict(method=METHOD, anchors=anchors,
                                                  sequence=[e['b'] for e in run], parent=parent)
                    accepted.add(entry['b'])

        for key, siblings in groups.items():
            run = []
            for entry in siblings:
                if not entry['eligible']:
                    finish(run, key[2])
                    run = []
                    continue
                if run and entry['values'][-1][1] != run[-1]['values'][-1][1] + 1:
                    finish(run, key[2])
                    run = []
                run.append(entry)
            finish(run, key[2])

    review([e for e in entries if len(e['values']) > 1])
    parents = defaultdict(list)
    for entry in entries:
        if len(entry['values']) > 1 and entry['b'] in accepted:
            parents[entry['section']].append(entry)
    review([e for e in entries if len(e['values']) == 1], parents)
    return promotions
