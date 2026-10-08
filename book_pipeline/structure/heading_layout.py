"""Book-relative numbered heading levels and the user's YAML presentation policy."""
from copy import deepcopy
import re

from .pagination import RANK

DEFAULT_LAYOUT = {
    'standalone_levels': [1],
    'centered_levels': [1, 2],
    'page_break_after_levels': [2, 3],
    'label_title_gap_em': 1.0,
}


def validate_heading_layout(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(DEFAULT_LAYOUT):
        raise ValueError('Unknown or invalid heading_layout setting')
    layout = {**deepcopy(DEFAULT_LAYOUT), **value}
    for key in ('standalone_levels', 'centered_levels', 'page_break_after_levels'):
        levels = layout[key]
        if not isinstance(levels, list) or any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 6 for n in levels):
            raise ValueError('heading_layout.' + key + ' must contain heading levels 1..6')
        layout[key] = sorted(set(levels))
    gap = layout['label_title_gap_em']
    if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not 0 <= gap <= 4:
        raise ValueError('heading_layout.label_title_gap_em must be 0..4')
    return layout


class HeadingLayout:
    def __init__(self, headings, settings=None):
        self.settings = validate_heading_layout(settings)
        self.explicit_levels=[{'toc_id':h.get('toc_id'),'group':h.get('group'),'kind':h['kind'],'level':h['heading_level']} for h in headings if 'heading_level' in h]
        ranks = sorted({RANK[h['kind']] for h in headings})
        numbered = {rank: n for n, rank in enumerate(ranks, 1)}
        self.levels = {kind: numbered[rank] for kind, rank in RANK.items() if rank in numbered}

    def level(self, heading):
        if 'heading_level' in heading:
            level=heading['heading_level']
            if isinstance(level,bool) or not isinstance(level,int) or not 1<=level<=6:
                raise ValueError('Explicit heading_level must be 1..6')
            return level
        return self.levels[heading['kind']]

    def standalone(self, heading):
        return self.level(heading) in self.settings['standalone_levels']

    @staticmethod
    def label_span(text):
        """Locate a real source prefix; no generated numbering or title rewrite."""
        match = re.match(
            r'^\s*(?:第\s*[0-9零〇一二三四五六七八九十百千万]+\s*[篇章]|'
            r'附录\s*(?:[A-Z]|[零〇一二三四五六七八九十百]+|\d+)|(?:§|\\S)\s*\d+[.、]?|(?:\d+(?:[.．]\d+)*|[A-Z](?:[.．]\d+)+)[.、]?)', text)
        if not match:
            return None
        end = match.end()
        body = end + len(text[end:]) - len(text[end:].lstrip())
        return (end, body) if body < len(text) else None

    def tex_setup(self):
        lines = [r'\renewcommand{\BookHeadingGap}{\hspace{' + f"{self.settings['label_title_gap_em']:g}" + 'em}}']
        for level in range(1, 7):
            align = 'center' if level in self.settings['centered_levels'] else 'left'
            single = int(level in self.settings['standalone_levels'])
            lines.append(r'\BookHeadingLayout{' + str(level) + '}{' + align + '}{' + str(single) + '}')
        return '\n'.join(lines)

    def describe(self):
        return {'settings': self.settings, 'source_kind_to_level': self.levels,
                'explicit_heading_levels':self.explicit_levels,
                'level_basis': 'explicit catalogue levels; non-body levels shifted by one' if self.explicit_levels else 'distinct source hierarchy ranks present in this book, largest first'}
