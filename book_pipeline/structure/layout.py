"""Helpers used by the current PDF Reforge workflow."""
from __future__ import annotations
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
import re
from statistics import median
from book_pipeline.structure.headings import compact, group_blocks, parse_number, span_for, valid_box

DEFAULT_RULES = {"top_margin": .12, "bottom_margin": .12, "position_tolerance": .025,
                 "repeat_page_gap": 4, "min_repeat_pages": 2}


MARGINAL_TYPES = {"header", "footer", "page_number"}


REASONS = MARGINAL_TYPES | {"original_toc"}


PROTECTED_ROLES = {"page_footnote", "footnote", "image_caption", "chart_caption", "table_caption",
                   "image_footnote", "chart_footnote", "table_footnote", "table_note", "figure_note"}


TOC_TITLE = re.compile(r"^(?:目录|目次|contents|tableofcontents)$", re.I)


PREFACE = re.compile(r"序言|前言|序章|致谢|preface|foreword|acknowledg", re.I)


COPYRIGHT = re.compile(r"ISBN|版权|版權|版权所有|圖書在版|图书在版|出版发行|copyright|all rights reserved", re.I)


CONTENT_LABEL = re.compile(r"^\s*(?:(?:定义|记号|定理|引理|推论|命题|例题|例|算法|习题|练习|图|表|公式)\s*\d|(?:证明|解答)\s*[:：]|[①-⑳†‡*＊]\s*\S)")


def content_role(role):
    return role in PROTECTED_ROLES or "footnote" in role or "caption" in role or role.endswith("_note")


def printed_number(text):
    """A complete Arabic/Roman page label, never a number inside a sentence."""
    text = text.strip()
    text = re.sub(r"^[-–—]\s*(.*?)\s*[-–—]$", r"\1", text)
    if re.fullmatch(r"\d{1,5}", text):
        return "arabic", int(text)
    text = text.upper()
    if not text or not re.fullmatch(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})", text):
        return None
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    number = sum(-values[c] if i + 1 < len(text) and values[c] < values[text[i + 1]] else values[c]
                 for i, c in enumerate(text))
    return "roman", number


def toc_lines(text, known_titles=()):
    """Keep source offsets so prose sharing a TOC block remains intact."""
    rows = []
    for match in re.finditer(r"[^\n]+", text):
        # MinerU also concatenates entries using two trailing spaces.
        for part in re.finditer(r"(\S.*?)(?:[ \t]*[.．…·]{2,}[ \t]*|[ \t]+)(\d{1,5}|[IVXLCDMivxlcdm]+)(?=[ \t]{2,}\S|[ \t]*$)", match[0]):
            line, entry = part[0], part
            if not printed_number(entry[2]):
                continue
            title = entry[1].strip()
            if len(title) > 160 or re.search(r"[。！？；]", title) or not re.search(r"[A-Za-z\u3400-\u9fff]", title):
                continue
            lo = match.start() + part.start()
            rows.append({**parse_number(title), "printed_page": printed_number(entry[2])[1],
                         "printed_page_style": printed_number(entry[2])[0], "lo": lo, "hi": lo + len(line)})
    # Scanned textbooks often concatenate lower TOC items without leaders and
    # the next major entry into one string. Recover only explicit structural
    # labels immediately preceding a leader; do not consume the earlier prose.
    label = re.compile(r'(?:第\s*[0-9零〇一二三四五六七八九十百]+\s*[篇章]|'
                       r'[零〇一二三四五六七八九十百]+章\s|§\s*\d+[.、]?|'
                       r'附录\s*(?:[A-Z]|[零〇一二三四五六七八九十百]+|\d+))')
    previous = 0
    leader_ends=[]
    for leader in re.finditer(r'[.．…·]{2,}\s*(\d{1,5}|[IVXLCDMivxlcdm]+)(?=\s|$|[^\d.．]|\d{1,2}[.．])', text):
        prefix=text[previous:leader.start()]
        labels=list(label.finditer(prefix))
        lo=None
        if labels:
            lo=previous+labels[-1].start()
        else:
            positions=[i for i,ch in enumerate(prefix) if not ch.isspace()]
            normalized=compact(prefix)
            matches=[t for t in known_titles if t and normalized.endswith(t)]
            if matches:
                lo=previous+positions[len(normalized)-len(max(matches,key=len))]
                # A known body title omits its parsed dotted number. Recover
                # that adjacent number too, otherwise the TOC leaves "1.1"
                # behind and a pure contents page appears to contain prose.
                head=text[previous:lo]
                if re.fullmatch(r'\s*\d+(?:\s*[.．]\s*\d+)+\s*',head):
                    lo=previous+len(head)-len(head.lstrip())
        if lo is not None and printed_number(leader[1]):
            title=text[lo:leader.start()].strip()
            parsed=parse_number(title)
            if len(title)<=160 and not re.search(r'[。！？；]',title):
                row={**parsed,'printed_page':printed_number(leader[1])[1],
                     'printed_page_style':printed_number(leader[1])[0],'lo':lo,'hi':leader.end()}
                rows=[r for r in rows if not (r['lo']<row['hi'] and r['hi']>row['lo'])]
                rows.append(row)
        previous=leader.end()
        if printed_number(leader[1]):leader_ends.append(leader.end())
    previous=0
    for marker in re.finditer(r'[(（]\s*(\d{1,5})\s*[)）]',text):
        start=max([previous]+[end for end in leader_ends if end<=marker.start()])
        prefix=text[start:marker.start()]
        labels=list(re.finditer(r'\d+[.．、]\s*|习题\s*|练习\s*',prefix))
        if labels:
            lo=start+labels[-1].start();title=text[lo:marker.start()].strip()
            if (len(title)<=160 and not re.search(r'[。！？；]',title) and
                    not any(lo<r['hi'] and marker.end()>r['lo'] for r in rows)):
                rows.append({**parse_number(title),'printed_page':int(marker[1]),'printed_page_style':'arabic',
                             'lo':lo,'hi':marker.end(),'local_toc_item':True})
        previous=marker.end()
    rows.sort(key=lambda r:r['lo'])
    return rows


def analyze_toc(index, excluded_pages=()):
    """Confirm entry regions before using their text to corroborate headings."""
    blocks = group_blocks(index)
    excluded_pages = set(excluded_pages)
    by_page, rows, titles = defaultdict(list), {}, set()
    body_titles = {compact(parse_number(b["text"])["title"]) for b in blocks
                   if b["type"] in {"paragraph_title", "doc_title"} and not TOC_TITLE.fullmatch(compact(b["text"]))}
    for b in blocks:
        by_page[b["page"]].append(b)
        if b["page"] in excluded_pages:
            continue
        if TOC_TITLE.fullmatch(compact(b["text"])):
            titles.add(b["page"])
        atoms = [index.atoms[i] for i in b["indices"]]
        if b["type"] not in MARGINAL_TYPES and all(a["kind"] in {"text", "math"} and not content_role(a["role"]) for a in atoms):
            rows[b["group"]] = toc_lines(b["text"],body_titles)
    count = {p: sum(len(rows.get(b["group"], [])) for b in bs) for p, bs in by_page.items()}
    candidates = {p for p, n in count.items() if n >= 2 and p not in excluded_pages}
    confirmed, evidence = set(), {}
    for p in sorted(candidates):
        typed = any(b["type"] == "index" and rows.get(b["group"]) for b in by_page[p])
        matches = sum(compact(e["title"]) in body_titles for b in by_page[p] for e in rows.get(b["group"], []))
        rules = ["toc_entry_structure"]
        if p in titles:
            rules.append("toc_title")
        if typed:
            rules.append("mineru_index")
        if matches >= 2:
            rules.append("toc_matches_body_titles")
        if (p in titles and (typed or count[p] >= 3)) or (typed and matches >= 2):
            confirmed.add(p); evidence[p] = rules
    # An anchored consecutive run can continue without a second TOC title.
    changed = True
    while changed:
        changed = False
        for p in sorted(candidates - confirmed):
            neighbors = [q for q in (p - 1, p + 1) if q in confirmed]
            boxes = [b["bbox"] for b in by_page[p] if rows.get(b["group"]) and valid_box(b["bbox"])]
            similar = any(abs(b[0] - other["bbox"][0]) <= .05 and abs(b[2] - other["bbox"][2]) <= .05
                          for b in boxes for q in neighbors for other in by_page[q]
                          if rows.get(other["group"]) and valid_box(other["bbox"]))
            typed = any(b["type"] == "index" and rows.get(b["group"]) for b in by_page[p])
            if neighbors and similar and (typed or count[p] >= 3):
                confirmed.add(p); evidence[p] = ["toc_entry_structure", "consecutive_toc_layout"]
                if typed:evidence[p].append("mineru_index")
                changed = True
    regions, entries, issues, pure_pages = [], [], [], []
    chapter = section = None
    for p in sorted(by_page):
        if p not in confirmed:
            for b in by_page[p]:
                if b["type"] == "index" and p not in excluded_pages:
                    issues.append({"stage": "layout", "page": p, "group": b["group"],
                                   "reason": "toc_evidence_insufficient", "action": "retained_content"})
            continue
        page_regions = []
        for b in by_page[p]:
            for row in rows.get(b["group"], []):
                try:
                    row_span=span_for(index,b,{"start_offset":row['lo'],"end_offset":row['hi']})
                except ValueError:
                    continue  # Never split an atomic inline formula.
                if row["kind"] == "chapter":chapter = row["number"];section=None
                if row['kind']=='section':section=row['number']
                entries.append({k: v for k, v in {**row, "chapter": chapter, "section":section,"toc_page": p, "group": b["group"]}.items()
                                if k not in {"lo", "hi"}})
                region = {"group": b["group"], "page": p, "reason": "original_toc",
                          "rules": evidence[p], "evidence": "; ".join(evidence[p]),
                          "rule_evidence": {"mineru_type": b["type"], "bbox": b["bbox"],
                                            "toc_pages": sorted(confirmed), "entry_title": row["title"],
                                            "printed_page": row["printed_page"]},
                          **row_span}
                page_regions.append(region)
            if TOC_TITLE.fullmatch(compact(b["text"])) and b["type"] not in MARGINAL_TYPES:
                page_regions.append({"group": b["group"], "page": p, "reason": "original_toc",
                                     "rules": ["confirmed_toc_title"], "evidence": "confirmed_toc_title",
                                     "rule_evidence": {"mineru_type": b["type"], "bbox": b["bbox"], "toc_pages": sorted(confirmed)},
                                     **span_for(index, b)})
        # Section labels belong to the TOC only when all other content is accounted for.
        def remainder(b):
            lo, hi = index.span(span_for(index, b))
            ranges = sorted(index.span(r) for r in page_regions if r["group"] == b["group"])
            chunks, pos = [], lo
            for start, end in ranges:
                chunks.append(index.text(pos, start)); pos = end
            chunks.append(index.text(pos, hi))
            return "".join(chunks).strip()
        remaining = [b for b in by_page[p] if remainder(b)]
        labels = [b for b in remaining if b["type"] in {"paragraph_title", "text"}
                  and parse_number(b["text"])["kind"] in {"part", "chapter"} and len(b["text"]) < 100]
        margins = [b for b in remaining if b["type"] in MARGINAL_TYPES and valid_box(b["bbox"])
                   and (b["bbox"][3] <= .12 or b["bbox"][1] >= .88)]
        if all(b in labels or b in margins for b in remaining):
            pure_pages.append(p)
            for b in labels:
                parsed = parse_number(b["text"])
                entries.append({**parsed, "chapter": chapter, "printed_page": None, "toc_page": p, "group": b["group"]})
                page_regions.append({"group": b["group"], "page": p, "reason": "original_toc",
                                     "rules": ["toc_section_label", "pure_toc_layout"],
                                     "rule_evidence": {"mineru_type": b["type"], "bbox": b["bbox"], "pure_toc_page": True},
                                     "evidence": "toc_section_label; pure_toc_layout", **span_for(index, b)})
        regions.extend(page_regions)
    return {"entries": entries, "regions": regions, "pages": sorted(confirmed), "pure_pages": pure_pages, "issues": issues}


def body_region(blocks, toc):
    by_page = defaultdict(list)
    toc_groups = {r["group"] for r in toc["regions"]}
    for b in blocks:
        if b["type"] == "text" and b["group"] not in toc_groups and valid_box(b["bbox"]) and len(b["text"]) >= 60:
            by_page[b["page"]].append(b["bbox"])
    return {"top": median(min(b[1] for b in bs) for bs in by_page.values()) if by_page else None,
            "bottom": median(max(b[3] for b in bs) for bs in by_page.values()) if by_page else None,
            "calibration_pages": len(by_page)}


def page_natures(blocks, metadata, toc):
    by_page = defaultdict(list)
    for b in blocks:by_page[b["page"]].append(b)
    result, in_preface = {}, False
    for p, bs in sorted(by_page.items()):
        if p in toc["pages"]:
            result[p] = "toc"; in_preface = False; continue
        has_chapter = any(parse_number(b["text"])["kind"] in {"part", "chapter"} and valid_box(b["bbox"])
                          and .12 < b["bbox"][1] < .85 for b in bs)
        if has_chapter:in_preface = False
        title_blocks = [b for b in bs if b["type"] in {"paragraph_title", "doc_title"}]
        if any(PREFACE.search(b["text"]) and len(b["text"]) < 80 for b in title_blocks):in_preface = True
        page_text = "\n".join(b["text"] for b in bs)
        title = compact(metadata.get("book", {}).get("title", ""))
        cover = any((compact(b["text"]) == title or p<=3 and len(compact(b['text']))>=4 and
                     compact(b['text']) in title and b['type']=='doc_title' and not has_chapter) and (b["type"] == "doc_title" or
                    p <= 3 and not has_chapter and len(bs) <= 8) for b in bs) if title else False
        result[p] = "title_or_cover" if cover else "copyright" if COPYRIGHT.search(page_text) else "preface" if in_preface else "body"
    return result


def analyze_layout(index, metadata=None, headings=(), toc=None, rules=None):
    """Three outcomes; missing evidence never authorizes a model call or removal."""
    metadata = metadata or {}
    rules = {**DEFAULT_RULES, **(rules or {})}
    toc = toc or analyze_toc(index, metadata.get("exclude_pages", []))
    blocks = group_blocks(index)
    region = body_region(blocks, toc)
    top, bottom = rules["top_margin"], 1 - rules["bottom_margin"]
    if region["top"] is not None and region["top"] > .04:top = min(top, region["top"] - .01)
    if region["bottom"] is not None and region["bottom"] < .96:bottom = max(bottom, region["bottom"] + .01)
    natures = page_natures(blocks, metadata, toc)
    heading_groups = {a["group"] for h in headings for a, _, _ in index.fragments(*index.span(h))}
    chapter_pages = sorted({index.page_for(index.position(h["start"])) + 1 for h in headings if h["kind"] in {"part", "chapter"}})
    toc_by_group = defaultdict(list)
    for r in toc["regions"]:toc_by_group[r["group"]].append(r)
    candidates, text_peers, number_peers = {}, defaultdict(list), defaultdict(list)
    for b in blocks:
        box = b["bbox"]
        side = "top" if valid_box(box) and box[3] <= top else "bottom" if valid_box(box) and box[1] >= bottom else None
        protection = []
        atoms = [index.atoms[i] for i in b["indices"]]
        if b["group"] in heading_groups:protection.append("confirmed_heading")
        if any(content_role(a["role"]) or content_role(a["block_type"]) for a in atoms):protection.append("footnote_or_caption")
        if any(a["kind"] != "text" for a in atoms):protection.append("formula_image_table_or_associated_content")
        if CONTENT_LABEL.match(b["text"]):protection.append("content_label_or_footnote_marker")
        explicit = parse_number(b["text"])
        if not side and explicit["kind"] and len(b["text"]) < 150 and not re.search(r"[。！？；]", b["text"]):
            protection.append("structural_title_in_body")
        if natures[b["page"]] in {"title_or_cover", "copyright"}:
            protection.append("cover_title_copyright_content")
        if natures[b["page"]] == "preface" and b["type"] == "footer":protection.append("preface_footer_content")
        number = printed_number(b["text"])
        c = {**b, "side": side, "number": number, "protection": protection}
        candidates[b["group"]] = c
        if side and not protection and len(b["text"]) <= 120 and b["text"].strip():
            text_peers[(side, compact(b["text"]))].append(c)
            if number:number_peers[(side, number[0])].append(c)

    def nearby(c, pool):
        box, page = c["bbox"], c["page"]
        # Compare odd/even tracks separately; an identical adjacent-page track
        # is useful additional evidence, never a requirement across the book.
        tracks = defaultdict(list)
        for other in pool:
            ob = other["bbox"]
            if (other["group"] != c["group"] and 0 < abs(other["page"] - page) <= rules["repeat_page_gap"]
                    and abs(ob[0] - box[0]) <= rules["position_tolerance"]
                    and abs(ob[1] - box[1]) <= rules["position_tolerance"]
                    and abs(ob[3] - box[3]) <= rules["position_tolerance"]):
                tracks[other["page"] % 2].append(other)
        same = tracks[page % 2]
        other = tracks[1 - page % 2]
        return same + other

    decisions = []
    for c in candidates.values():
        typ, side, text = c["type"], c["side"], c["text"]
        record = {"group": c["group"], "page": c["page"], "text": text, "type": typ, "bbox": c["bbox"],
                  "status": "retained", "reason": "ordinary_content", "rules": [], "conflicts": [],
                  "evidence": {"mineru_type": typ, "position": side, "page_nature": natures[c["page"]],
                               "protection": c["protection"], "physical_parity": "odd" if c["page"] % 2 else "even",
                               "chapter_scope": bisect_right(chapter_pages, c["page"]), "repeat_pages": [], "sequence_pages": []}}
        if c["page"] in metadata.get("exclude_pages", []):
            record.update(status="excluded_by_user", reason="user_excluded_page", rules=["explicit_user_page_override"])
        elif c["group"] in toc_by_group:
            record.update(status="excluded_regions", reason="original_toc", rules=["confirmed_toc_regions"])
        elif c["protection"]:
            record.update(reason="protected_content", rules=c["protection"])
        else:
            repeat = nearby(c, text_peers.get((side, compact(text)), [])) if side else []
            repeats = sorted({c["page"], *(b["page"] for b in repeat)}) if repeat else []
            recurring = len(repeats) >= rules["min_repeat_pages"]
            seq, contrary = [], []
            if side and c["number"]:
                for peer in nearby(c, number_peers[(side, c["number"][0])]):
                    delta = peer["number"][1] - c["number"][1]
                    (seq if delta == peer["page"] - c["page"] else contrary).append(peer["page"])
            record["evidence"].update(repeat_pages=repeats, sequence_pages=sorted(seq), inconsistent_number_pages=sorted(contrary))
            if side:record["rules"].append(side + "_margin_position")
            if recurring:record["rules"].append("local_position_repeat_with_parity_tracks")
            if c["number"]:record["rules"].append("complete_" + c["number"][0] + "_page_label")
            if seq:record["rules"].append("page_number_increments_with_physical_pages")
            if not text.strip() or text.strip() == "[NO TEXT]":
                record.update(status="uncertain" if typ in MARGINAL_TYPES else "retained", reason="missing_readable_text")
            elif c["number"] and side and (typ == "page_number" or seq):
                if not seq and len(contrary) >= 2:
                    record.update(status="uncertain", reason="page_number_sequence_conflict", conflicts=["page_label_vs_sequence"])
                else:record.update(status="excluded", reason="page_number")
            elif typ == "header" and side == "top":
                if len(text) <= 120 and not re.search(r"[。！？；]", text):
                    record.update(status="excluded", reason="header", rules=record["rules"] + ["mineru_header"])
                else:record.update(status="uncertain", reason="header_body_conflict", conflicts=["header_type_vs_prose"])
            elif recurring and side and not c["number"]:
                if typ in {"paragraph_title", "doc_title"} or re.search(r"[。！？；]", text):
                    record.update(status="uncertain", reason="repeat_content_conflict", conflicts=["repetition_vs_content_structure"])
                else:record.update(status="excluded", reason="header" if side == "top" else "footer")
            elif typ in MARGINAL_TYPES or typ == "index":
                record.update(status="uncertain", reason="layout_evidence_insufficient")
                if valid_box(c["bbox"]) and typ in MARGINAL_TYPES and (
                        not side or typ == "header" and side != "top" or typ == "footer" and side != "bottom"):
                    record.update(reason="type_position_conflict", conflicts=["mineru_type_vs_position"])
                elif typ == "page_number" and not c["number"]:
                    record.update(reason="page_number_form_conflict", conflicts=["page_number_type_vs_text"])
                elif typ == "index":record["reason"] = "toc_evidence_insufficient"
            elif side and c["number"]:
                record.update(status="uncertain", reason="page_label_without_type_or_sequence")
        decisions.append(record)
    result = {"method": "joint_layout_rules", "rules": rules, "global_page_count": index.source["source_page_count"],
              "body_region": region, "candidate_regions": {"top_end": top, "bottom_start": bottom},
              "page_natures": {str(p): nature for p, nature in natures.items()}, "toc": toc,
              "decisions": decisions, "model_reviews": []}
    return refresh_layout(index, result)


def refresh_layout(index, result):
    blocks = {b["group"]: b for b in group_blocks(index)}
    exclusions = [deepcopy(r) for r in result["toc"]["regions"] if r["start"]["atom"] in index.ids]
    issues = []
    for d in result["decisions"]:
        if d["status"] == "excluded":
            exclusions.append({**span_for(index, blocks[d["group"]]), "group": d["group"], "reason": d["reason"],
                               "evidence": "; ".join(d["rules"]), "rules": d["rules"], "rule_evidence": d["evidence"],
                               "decision_source": "model_conflict_review" if d.get("review") else "layout_rules"})
        elif d["status"] == "uncertain":
            issues.append({"stage": "layout", "page": d["page"], "group": d["group"], "reason": d["reason"],
                           "conflicts": d["conflicts"], "evidence": d["evidence"], "action": "retained_content"})
    result.update(exclusions=exclusions, issues=issues,
                  conflicts=[deepcopy(d) for d in result["decisions"] if d["status"] == "uncertain" and d["conflicts"]])
    return result


def project_layout(result, index):
    selected = set(index.source["selected_pages"])
    projected = deepcopy(result)
    for key in ("decisions", "issues", "conflicts", "model_reviews"):
        projected[key] = [d for d in result[key] if d["page"] - 1 in selected]
    projected["exclusions"] = [e for e in result["exclusions"] if e["start"]["atom"] in index.ids]
    return projected


def apply_layout_to_outline(outline, layout):
    result = deepcopy(outline)
    result["layout"] = layout
    result["exclusions"] = [e for e in result["exclusions"] if e["reason"] not in REASONS] + deepcopy(layout["exclusions"])
    result["issues"] = [e for e in result["issues"] if e.get("stage") != "layout"] + deepcopy(layout["issues"])
    return result
