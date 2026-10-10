"""Helpers used by the current PDF Reforge workflow."""
from __future__ import annotations
from pathlib import Path
import re
from .footnotes import CIRCLED, SUPERSCRIPTS

LEVELS = {"part": 1, "chapter": 2, "section": 3, "subsection": 4, "subsubsection": 5, "paragraph": 6}
HEADING_END_PUNCTUATION = frozenset('。．｡.!！?？,，;；:：、…⋯)）]］}｝》〉】〕〗〙〛”’」』»›〞〟')
HEADING_NOTE_MARK = re.compile(
    rf'(?:[{CIRCLED}{SUPERSCRIPTS}†‡※]|\\(?:textcircled|circled)\s*\{{\s*[0-9０-９]+\s*\}}|'
    rf'\^\s*(?:\{{\s*[0-9０-９{CIRCLED}{SUPERSCRIPTS}*†‡]+\s*\}}|[0-9０-９{CIRCLED}{SUPERSCRIPTS}*†‡]+))')


def heading_tail_end(text, end):
    """Include adjacent title punctuation/note marks, stopping before body text."""
    cursor = end
    while cursor < len(text):
        if text[cursor].isspace() and text[cursor] not in '\r\n':
            cursor += 1
            continue
        marker = HEADING_NOTE_MARK.match(text, cursor)
        if marker:
            cursor = marker.end()
        elif text[cursor] in HEADING_END_PUNCTUATION:
            cursor += 1
        else:
            break
        end = cursor
    return end


def source_heading_end(index, lo, hi):
    """Complete a saved title within its last source group; keep math atomic."""
    atom = index.atoms[index.ids[index.address(hi, end=True)['atom']]]
    limit = index.prefix[index.groups[atom['group']][-1] + 1]
    end = lo + heading_tail_end(index.text(lo, limit), hi - lo)
    try:
        index.position(index.address(end, end=True))
    except ValueError:
        return hi
    return end


def compact(text):
    return re.sub(r"\s+", "", text).replace("．", ".").replace('（','(').replace('）',')')


def load_metadata(pdf, path=None):
    import yaml
    path = Path(path).expanduser().resolve() if path else Path(pdf).resolve().with_suffix(".book.yaml")
    if not path.is_file():
        raise ValueError(f"Missing user metadata YAML: {path}; copy book_metadata.example.yaml and fill book.title / book.author")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid metadata YAML: {path}: {error}") from error
    return validate_metadata(value, path)


def validate_metadata(value, path="user input"):
    """Apply the same metadata contract to a form and a YAML file."""
    if not isinstance(value, dict) or set(value) - {"book", "exclude_pages", "heading_rules", "heading_layout", "overrides"}:
        raise ValueError("Metadata YAML needs book, optional exclude_pages, heading_rules, heading_layout and overrides only")
    book = value.get("book")
    if not isinstance(book, dict) or set(book) != {"title", "author"}:
        raise ValueError("book must contain title and author")
    if any(not isinstance(book[k], str) or not book[k].strip() for k in book):
        raise ValueError("book.title and book.author must be nonempty user-entered strings")
    excluded = value.get("exclude_pages", [])
    if not isinstance(excluded, list) or any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in excluded):
        raise ValueError("exclude_pages must be a list of positive physical PDF page numbers")
    rules = {"center_tolerance": .04, "same_height_tolerance": .12}
    custom = value.get("heading_rules", {})
    if not isinstance(custom, dict) or set(custom) - set(rules):
        raise ValueError("Unknown heading_rules setting")
    rules.update(custom)
    for key, maximum in (("center_tolerance", .15), ("same_height_tolerance", .4)):
        v = rules[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= maximum:
            raise ValueError(f"heading_rules.{key} must be > 0 and <= {maximum}")
    overrides = value.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("overrides must map group IDs to manual decisions")
    for group, entry in overrides.items():
        if not isinstance(group, str) or not isinstance(entry, dict) or entry.get("kind") not in {*LEVELS, "ignore"}:
            raise ValueError("Invalid manual heading override")
        if set(entry) - {"kind", "title", "number", "start_offset", "end_offset"}:
            raise ValueError("Unknown override field: " + group)
        for key in ("title", "number"):
            if key in entry and not isinstance(entry[key], str):
                raise ValueError(f"override {key} must be a string: {group}")
        for key in ("start_offset", "end_offset"):
            if key in entry and (isinstance(entry[key], bool) or not isinstance(entry[key], int) or entry[key] < 0):
                raise ValueError(f"override {key} must be a nonnegative integer: {group}")
    from .heading_layout import validate_heading_layout
    return {"path": str(path), "book": book, "exclude_pages": sorted(set(excluded)),
            "heading_rules": rules, "heading_layout": validate_heading_layout(value.get('heading_layout')),
            "overrides": overrides}


def parse_number(text):
    """Only explicit structural labels; a bare '1.' does not establish depth."""
    text = text.strip()
    match = re.match(r"附录\s*([A-Z]|[零〇一二三四五六七八九十百]+|\d+)\s*(.*)$", text, re.S)
    if match:
        return {"kind": "chapter", "number": match[1] if match[1].isalpha() and match[1].isascii() else '附录'+match[1],
                "title": match[2].strip()}
    match = re.match(r"第\s*([0-9零〇一二三四五六七八九十百]+)\s*([篇章])\s*(.*)$", text, re.S)
    if match:
        return {"kind": "part" if match[2] == "篇" else "chapter", "number": match[1], "title": match[3].strip()}
    match = re.match(r"([零〇一二三四五六七八九十百]+)章\s+(.*)$", text, re.S)
    if match:
        return {"kind": "chapter", "number": match[1], "title": match[2].strip()}
    match = re.match(r"(?:§|\\S)\s*(\d+)(?![.．]?\d)(?:[.、]\s*|\s+|(?=[\u3400-\u9fffA-Za-z]))(.*)$", text, re.S)
    if match:
        return {"kind": "section", "number": match[1], "title": match[2].strip(),
                "number_scope": "chapter_local"}
    match = re.match(r"(\d+(?:[.．]\d+){1,4})(?![.．\d])(?:\s+|(?=[\u3400-\u9fffA-Za-z]))(.*)$", text, re.S)
    if match:
        depth = len(re.split(r"[.．]", match[1]))
        return {"kind": {2: "section", 3: "subsection", 4: "subsubsection", 5: "paragraph"}[depth],
                "number": match[1].replace("．", "."), "title": match[2].strip()}
    return {"kind": None, "number": "", "title": text}


def group_blocks(index):
    result = []
    for group, indices in index.groups.items():
        atoms = [index.atoms[i] for i in indices]
        atom = atoms[0]
        box = atom.get("block_bbox") or atom.get("bbox")
        result.append({"group": group, "page": atom["page"] + 1, "type": atom["block_type"],
                       "text": "".join(a["content"] for a in atoms), "bbox": box,
                       "kinds": sorted({a["kind"] for a in atoms}), "indices": indices})
    return result


def valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4 and
            all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box) and
            0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)


def span_for(index, block, override=None):
    indices = block["indices"]
    lo, hi = index.prefix[indices[0]], index.prefix[indices[-1] + 1]
    if override:
        start, end = override.get("start_offset", 0), override.get("end_offset", hi - lo)
        if not 0 <= start < end <= hi - lo:
            raise ValueError("Manual override outside source group: " + block["group"])
        lo, hi = lo + start, lo + end
        # Also rejects a boundary inside an atomic inline formula.
        index.position(index.address(lo)); index.position(index.address(hi, end=True))
    return {"start": index.address(lo), "end": index.address(hi, end=True)}
