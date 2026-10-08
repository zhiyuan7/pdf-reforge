"""Helpers used by the current PDF Reforge workflow."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import re
from book_pipeline.ingestion.source import corrected_content

METHOD = "paragraph_boundary_text"


CLOSING_QUOTES = '\"\'”’»›」』〞'


MARKER = r"[0-9０-９①-⑳㉑-㊿❶-❿⁰¹²³⁴⁵⁶⁷⁸⁹*†‡]+"


FOOTNOTE_SUFFIX = re.compile(
    rf"(?:[①-⑳㉑-㊿❶-❿⁰¹²³⁴⁵⁶⁷⁸⁹*†‡]+"
    rf"|\[\s*{MARKER}\s*\]|［\s*{MARKER}\s*］"
    rf"|\(\s*{MARKER}\s*\)|（\s*{MARKER}\s*）"
    rf"|(?:\$\s*)?(?:\{{\s*\}}\s*)?\^\s*(?:\{{\s*{MARKER}\s*\}}|{MARKER})(?:\s*\$)?"
    rf"|\\(?:textsuperscript|textcircled)\s*\{{\s*{MARKER}\s*\}}"
    rf"|\\footnotemark(?:\s*\[\s*{MARKER}\s*\])?"
    rf"|<sup>\s*{MARKER}\s*</sup>)$"
)


DECORATIVE = {"header", "footer", "page_number", "blank_scanning_artifact"}


def ends_in_period(text):
    """Inspect punctuation without modifying any source characters."""
    tail = text.rstrip()
    while tail:
        if tail[-1] in CLOSING_QUOTES:
            tail = tail[:-1].rstrip()
            continue
        marker = FOOTNOTE_SUFFIX.search(tail)
        if marker:
            tail = tail[:marker.start()].rstrip()
            continue
        break
    return tail.endswith(("。", "."))


def page_boundaries(index, decisions):
    """Never search another pair when the first/last body blocks are blocked."""
    exclusions = [(index.span(e), set(e.get("reasons", [e["reason"]])))
                  for e in decisions.get("exclusions", [])]
    structural = decisions.get("objects", []) + decisions.get("conflicts", [])
    starts, ends, headings = [], [], []
    for item in structural:
        lo = index.position(item["start"])
        hi = index.position(item["end"]) if item.get("end") else index.prefix[-1]
        starts.append(lo)
        if not item.get("anchor_only"):
            ends.append(hi)
        if item["kind"] in {"part", "chapter", "section", "subsection", "subsubsection", "paragraph", "exercise_group"}:
            headings.append((lo, hi))
    starts.extend(index.position(c["start"]) for c in decisions.get("numbered_candidates", []))
    blocks = defaultdict(list)
    ignored = set()
    for group, indices in index.groups.items():
        lo, hi = index.prefix[indices[0]], index.prefix[indices[-1] + 1]
        atoms = [index.atoms[i] for i in indices]
        covered = [reasons for (a, b), reasons in exclusions if a <= lo and hi <= b]
        text = "".join(corrected_content(a, decisions.get("corrections", [])) for a in atoms)
        if covered or not text.strip() and all(a["kind"] == "text" for a in atoms):
            # Deleted TOC/user pages remain hard barriers even though invisible.
            if not covered or all(reasons <= DECORATIVE for reasons in covered):
                ignored.update(indices)
            continue
        if (len({a["page"] for a in atoms}) != 1 or
                any(a["kind"] not in {"text", "math"} or a["block_type"] != "text" or a["role"] != "text" for a in atoms) or
                any(lo < b and hi > a for (a, b), _ in exclusions) or
                any(lo < b and hi > a for a, b in headings)):
            continue
        blocks[atoms[0]["page"]].append(dict(group=group, page=atoms[0]["page"] + 1,
            source_ids=[a["id"] for a in atoms], text=text, lo=lo, hi=hi,
            first=indices[0], last=indices[-1]))
    pages = sorted(index.source.get("selected_pages", {a["page"] for a in index.atoms}))
    result = []
    for previous, following in zip(pages, pages[1:]):
        row = {"pages": [previous + 1, following + 1], "status": "retained"}
        if following != previous + 1:
            row["reason"] = "nonconsecutive_pages"
        elif not blocks[previous] or not blocks[following]:
            row["reason"] = "missing_body_block"
        else:
            left, right = blocks[previous][-1], blocks[following][0]
            row.update(left=left["source_ids"][-1], right=right["source_ids"][0], blocks=(left, right))
            if ends_in_period(left["text"]):
                row["reason"] = "period_ending"
            elif (any(i not in ignored for i in range(left["last"] + 1, right["first"])) or
                  any(left["lo"] < p < right["hi"] for p in starts + ends) or
                  any(left["lo"] < b and right["hi"] > a and not reasons <= DECORATIVE
                      for (a, b), reasons in exclusions)):
                row["reason"] = "structural_barrier"
            else:
                row.update(status="candidate", reason="no_period")
        result.append(row)
    return result


def supported_joins(index, decisions):
    """Apply the same guards to cached/externally supplied renderer decisions."""
    if not decisions.get("joins"):
        return [], []
    pairs = {(r["left"], r["right"]) for r in page_boundaries(index, decisions) if r["status"] == "candidate"}
    valid, rejected, seen = [], [], set()
    for join in decisions["joins"]:
        pair = (join.get("left"), join.get("right")) if isinstance(join, dict) else (None, None)
        confirmed = (isinstance(join, dict) and join.get("method") == METHOD and
                     join.get("certainty") == "confirmed" and isinstance(join.get("evidence"), str) and join["evidence"].strip())
        if not confirmed or not all(isinstance(v, str) for v in pair) or pair not in pairs:
            rejected.append({"reason": "paragraph_join_rejected; retained original breaks", "join": deepcopy(join)})
        elif pair not in seen:
            valid.append(join)
            seen.add(pair)
    return valid, rejected
