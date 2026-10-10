"""Validated presentation settings, independent of recognition and source text."""
import re

COLOR_KINDS = {
    "definition", "notation", "theorem", "lemma", "corollary", "proposition",
    "algorithm", "example", "remark", "numbered",
}
DEFAULT_COLORS = {
    "definition": "#00A652", "notation": "#00A652", "theorem": "#FF8618",
    "lemma": "#FF8618", "corollary": "#FF8618", "proposition": "#FF8618",
    "algorithm": "#00AEF7", "example": "#525A65", "remark": "#737B86",
    "numbered": "#737B86",
}


def validate_colors(value):
    if not isinstance(value, dict) or set(value) - COLOR_KINDS:
        raise ValueError("Unknown special block type")
    result = {}
    for key, color in value.items():
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Colors must use #RRGGBB")
        result[key] = color.upper()
    return result


def selected_color(settings, item):
    if item["kind"] not in COLOR_KINDS:
        return None
    return settings.get(item["kind"])


def color_definitions(settings):
    colors = sorted(set(settings.values()))
    return "\n".join(r"\definecolor{BookColor" + c[1:] + "}{HTML}{" + c[1:] + "}" for c in colors)
