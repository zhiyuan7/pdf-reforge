"""End complete heading subtrees; a parent's first child follows its introduction."""

RANK = {'part':0,'chapter':1,'section':2,'subsection':3,'subsubsection':4,'paragraph':5,'exercise_group':3}


class HeadingPagination:
    def __init__(self, headings, kinds=None, *, layout=None):
        self.breaks = {}
        active = []
        # Numeric YAML policy is authoritative. The kinds argument remains for
        # callers of the earlier interface; production always supplies layout.
        selected = lambda h: layout.level(h) in layout.settings['page_break_after_levels'] if layout else h['kind'] in (kinds or [])
        rank = lambda h:layout.level(h) if layout else h.get('heading_level',RANK[h['kind']])
        for heading in sorted(headings,key=lambda h:h['lo']):
            level = rank(heading)
            closed = []
            while active and rank(active[-1]) >= level:
                closed.append(active.pop())
            if any(selected(item) for item in closed):
                self.breaks[heading['lo']] = [item['anchor'] for item in closed if selected(item)]
            active.append(heading)

    def before(self, heading):
        return '\n\\par\\clearpage\n' if heading['lo'] in self.breaks else ''
