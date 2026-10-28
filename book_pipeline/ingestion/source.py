"""Lossless source atoms and code-point addressed, half-open spans."""
from __future__ import annotations

import bisect
from copy import deepcopy
from pathlib import Path
import re

from book_pipeline.config import digest, read

MATERIALIZATION_METHOD = 'source_roles_roundtrip_v5'


def selected_pages(spec, page_count):
    if spec in (None, "all"):
        return list(range(page_count))
    values = set()
    for part in spec.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:-(\d+))?\s*", part)
        if not match:
            raise ValueError("pages must be all or 1-3,5")
        lo, hi = int(match[1]), int(match[2] or match[1])
        if lo < 1 or hi < lo or hi > page_count:
            raise ValueError("pages outside original PDF")
        values.update(range(lo - 1, hi))
    return sorted(values)


def flatten(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(flatten(v) for v in value)
    if isinstance(value, dict):
        return flatten(value.get("content", ""))
    return ""


def safe_asset(directory, name):
    path = (Path(directory) / name).resolve()
    if not path.is_relative_to(Path(directory).resolve()) or not path.is_file():
        raise ValueError("Missing or unsafe MinerU asset: " + name)
    return str(path)


def load_source(pdf, mineru_dir, page_spec=None):
    import pymupdf
    pdf, mineru_dir = Path(pdf).resolve(), Path(mineru_dir).resolve()
    source_hash = digest(pdf)
    manifest = read(mineru_dir / "manifest.json")
    if manifest.get("input_sha256") != source_hash:
        raise ValueError("MinerU manifest input_sha256 does not match PDF")
    middle = read(mineru_dir / "middle_json.json")
    if middle.get("schema") != "docvortex.middle" or middle.get("schema_version") != "2.0":
        raise ValueError("Unsupported MinerU schema; expected docvortex.middle 2.0")
    with pymupdf.open(pdf) as document:
        count = len(document)
        dimensions = {i: [page.rect.width, page.rect.height] for i, page in enumerate(document)}
    # A page-limited OCR run retains the original PDF and physical indices.
    # Only an explicit manifest selection may narrow the default scope; never
    # silently accept missing pages by inferring scope from recognized blocks.
    declared = manifest.get('source_page_numbers')
    if page_spec is None and declared is not None:
        if (not isinstance(declared, list) or not declared
                or any(type(n) is not int or not 1 <= n <= count for n in declared)
                or declared != sorted(set(declared))):
            raise ValueError('Invalid manifest source_page_numbers')
        chosen = [n-1 for n in declared]
    else:
        chosen = selected_pages(page_spec, count)
    pages_by_id = {p["page_idx"]: p for p in middle["pages"]}
    if len(pages_by_id) != len(middle["pages"]) or not set(chosen) <= set(pages_by_id):
        raise ValueError("Duplicate or missing selected MinerU page indices")
    atoms, pages = [], []
    extraction = {p['page']-1:p['method'] for p in manifest.get('pages',[]) if isinstance(p,dict) and 'method' in p}
    for number in chosen:
        page_atoms = []
        for block in pages_by_id[number].get("blocks", []):
            group = f"p{number + 1:06d}-b{block['index']:06d}"
            leaves = []
            def visit(node, inherited_role):
                if isinstance(node, str):
                    if node:
                        leaves.append(("text", node, {}, inherited_role))
                elif isinstance(node, list):
                    for child in node:
                        visit(child, inherited_role)
                elif isinstance(node, dict):
                    role = node.get("type", inherited_role)
                    if role in ("equation_inline", "inline_equation"):
                        leaves.append(("math", flatten(node), node, inherited_role))
                    elif role == "equation":
                        leaves.append(("display_math", flatten(node), node, role))
                    elif role in ("image_body", "chart_body"):
                        leaves.append(("image", "", node, role))
                    elif role == "table_body":
                        leaves.append(("table", flatten(node), node, role))
                    elif isinstance(node.get('content'), str):
                        if node['content']:
                            leaves.append(('text', node['content'], node, inherited_role if role == 'text' else role))
                    else:
                        visit(node.get("content", ""), inherited_role if role == "text" else role)
            if block["type"] in ("image", "chart") and isinstance(block.get("content"), str):
                leaves.append(("image", "", block, block["type"]))
            else:
                visit(block, block["type"])
            for i, (kind, content, node, role) in enumerate(leaves):
                atom = {"id": f"{group}-s{i:04d}", "group": group, "page": number,
                        "kind": kind, "role": role, "block_type": block["type"],
                        "content": content, "length": max(1, len(content)),
                        "bbox": node.get("bbox", block.get("bbox")),
                        "level_hint": block.get("level"), "block_bbox": block.get("bbox")}
                method = extraction.get(number, 'ocr' if manifest.get('mineru_version') else None)
                if method:
                    atom['extraction_method'] = method
                if kind == 'text' and node.get('source_text_runs'):
                    atom['source_text_runs'] = deepcopy(node['source_text_runs'])
                for key in ('writing_direction', 'ocr_table_recovery', 'recovery_issue', 'source_prose_end_bbox'):
                    if block.get(key): atom[key] = block[key]
                if block.get('source_panel'):
                    atom.update(source_panel=block['source_panel'], source_panel_title=block.get('source_panel_title',False))
                if node.get("image_path"):
                    atom["asset"] = safe_asset(mineru_dir, node["image_path"])
                    atom["asset_sha256"] = digest(atom["asset"])
                if kind == "image" and not atom.get("asset"):
                    raise ValueError("Image block has no asset: " + atom["id"])
                atoms.append(atom)
                page_atoms.append(atom["id"])
        pages.append({"page": number, "atoms": page_atoms, "size_pt": dimensions[number]})
    return {"source_pdf": str(pdf), "input_sha256": source_hash,
            "mineru_dir": str(mineru_dir), "mineru_sha256": digest(mineru_dir / "middle_json.json"),
            "source_page_count": count, "selected_pages": chosen, "pages": pages, "atoms": atoms,
            **({"ingestion_method": manifest["method"]} if manifest.get("method") else {}),
            **({"table_prose_recovery": manifest["table_prose_recovery"]} if manifest.get("table_prose_recovery") else {}),
            **({"translation": manifest["translation"]} if manifest.get("translation") else {}),
            **({'scan_page_frames':manifest['scan_page_frames']} if manifest.get('scan_page_frames') else {})}


class SourceIndex:
    def __init__(self, source):
        self.source = source
        self.atoms = source["atoms"]
        self.ids = {a["id"]: i for i, a in enumerate(self.atoms)}
        self.groups = {}
        for i, atom in enumerate(self.atoms):
            self.groups.setdefault(atom["group"], []).append(i)
        if len(self.ids) != len(self.atoms):
            raise ValueError("Duplicate source atom IDs")
        self.prefix = [0]
        for atom in self.atoms:
            self.prefix.append(self.prefix[-1] + atom["length"])

    def position(self, pos):
        if isinstance(pos, dict) and pos.get("atom") not in self.ids and pos.get("atom") in self.groups:
            group = self.groups[pos["atom"]]
            if pos.get("offset") == 0:
                pos = {"atom": self.atoms[group[0]]["id"], "offset": 0}
            elif pos.get("offset") == "end":
                pos = {"atom": self.atoms[group[-1]]["id"], "offset": self.atoms[group[-1]]["length"]}
        if not isinstance(pos, dict) or pos.get("atom") not in self.ids:
            raise ValueError("Unknown source position")
        i = self.ids[pos["atom"]]
        offset = pos.get("offset")
        atom = self.atoms[i]
        if offset == "end":
            offset = atom["length"]
        elif offset is None and atom["kind"] == "text":
            fields = [key for key in ("before", "after") if key in pos]
            if len(fields) != 1 or not isinstance(pos[fields[0]], str) or not pos[fields[0]]:
                raise ValueError("Text position requires offset or an exact before/after quote")
            quote = pos[fields[0]]
            if atom["content"].count(quote) != 1:
                raise ValueError("Boundary quote must match exactly once in its source atom")
            offset = atom["content"].index(quote) + (len(quote) if fields[0] == "after" else 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= self.atoms[i]["length"]:
            raise ValueError("Source offset outside atom")
        if self.atoms[i]["kind"] != "text" and offset not in (0, self.atoms[i]["length"]):
            raise ValueError("Cannot split an atomic formula/image/table")
        return self.prefix[i] + offset

    def span(self, item):
        start, end = self.position(item["start"]), self.position(item["end"])
        if end <= start:
            raise ValueError("Empty or reversed source span")
        return start, end

    def address(self, value, end=False):
        if value == self.prefix[-1]:
            i = len(self.atoms) - 1
            return {"atom": self.atoms[i]["id"], "offset": self.atoms[i]["length"]}
        i = bisect.bisect_left(self.prefix, value) - 1 if end and value else bisect.bisect_right(self.prefix, value) - 1
        return {"atom": self.atoms[i]["id"], "offset": value - self.prefix[i]}

    def fragments(self, start, end):
        for i in range(max(0, bisect.bisect_right(self.prefix, start) - 1), len(self.atoms)):
            if self.prefix[i] >= end:
                break
            a = self.atoms[i]
            lo, hi = max(start, self.prefix[i]) - self.prefix[i], min(end, self.prefix[i + 1]) - self.prefix[i]
            if lo < hi:
                yield a, lo, hi

    def text(self, start, end):
        return "".join(a["content"][lo:hi] for a, lo, hi in self.fragments(start, end))

    def page_for(self, value):
        i = min(len(self.atoms) - 1, bisect.bisect_right(self.prefix, value) - 1)
        return self.atoms[max(0, i)]["page"]

    def continuous(self, start, end):
        pages = sorted({a["page"] for a, _, _ in self.fragments(start, end)})
        return all(b == a + 1 for a, b in zip(pages, pages[1:]))

    def public_atoms(self, page_numbers):
        # Model-facing pages match physical PDF page numbers and stable IDs.
        return [{**{k: v for k, v in a.items() if k not in ("asset", "asset_sha256")}, "page": a["page"] + 1}
                for a in self.atoms if a["page"] in page_numbers]


def corrected_atoms(source, corrections):
    result = deepcopy(source)
    by_id = {a["id"]: a for a in result["atoms"]}
    for correction in sorted(corrections, key=lambda c: (c["atom"], -c["start_offset"])):
        a = by_id[correction["atom"]]
        lo, hi = correction["start_offset"], correction["end_offset"]
        if a['content'][lo:hi] != correction['replacement']:
            a.pop('source_text_runs', None)  # Ink offsets describe the unedited source bytes.
        a["content"] = a["content"][:lo] + correction["replacement"] + a["content"][hi:]
    return result


def corrected_content(atom, corrections):
    text = atom["content"]
    for c in sorted((c for c in corrections if c["atom"] == atom["id"]), key=lambda c: -c["start_offset"]):
        text = text[:c["start_offset"]] + c["replacement"] + text[c["end_offset"]:]
    return text


def materialize_source(index, directory):
    """Persist corrected atoms in the common extraction schema, before translation."""
    import os
    import shutil
    from book_pipeline.config import save
    directory = Path(directory).resolve()
    origin = Path(index.source['mineru_dir']).resolve()
    if directory == origin or directory.is_relative_to(origin):
        raise ValueError('Corrected source needs a separate directory')
    directory.mkdir(parents=True, exist_ok=True)
    stamp = index.source['mineru_sha256']
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if (manifest.get('corrected_source_sha256') != stamp
                or manifest.get('materialization_method') != MATERIALIZATION_METHOD):
            raise ValueError('Corrected source cache does not match current corrections')
        if manifest.get('middle_sha256') != digest(directory / 'middle_json.json'):
            raise ValueError('Corrected source cache changed')
        return directory
    pages = {page['page']: dict(page_idx=page['page'], blocks=[]) for page in index.source['pages']}
    for positions in index.groups.values():
        first = index.atoms[positions[0]]
        nodes = []
        for pos in positions:
            atom = index.atoms[pos]
            kind = {'math':'equation_inline', 'display_math':'equation', 'image':'image_body',
                    'table':'table_body'}.get(atom['kind'], atom['role'])
            if kind == first['block_type'] and kind not in {'equation', 'image_body', 'table_body'}:
                kind = 'text'
            node = dict(type=kind, content=atom['content'], bbox=atom.get('bbox'))
            if atom.get('source_text_runs'):
                node['source_text_runs'] = deepcopy(atom['source_text_runs'])
            if atom.get('asset'):
                asset = Path(atom['asset'])
                relative = asset.relative_to(origin)
                target = directory / 'assets' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    try: os.link(asset, target)
                    except OSError: shutil.copy2(asset, target)
                node['image_path'] = str(target.relative_to(directory))
            # Inline equations inherit their role from their parent in the
            # common schema. Preserve caption/code ownership across a round
            # trip instead of turning a caption variable into body text.
            if atom['kind'] == 'math' and atom['role'] != first['block_type']:
                node = dict(type=atom['role'], content=[node])
            nodes.append(node)
        page = pages[first['page']]
        block = dict(index=len(page['blocks']), type=first['block_type'],
                     bbox=first.get('block_bbox'), level=first.get('level_hint'), content=nodes)
        if first.get('source_panel'):
            block.update(source_panel=first['source_panel'], source_panel_title=first.get('source_panel_title',False))
        for key in ('writing_direction', 'ocr_table_recovery', 'recovery_issue', 'source_prose_end_bbox'):
            if first.get(key): block[key] = first[key]
        page['blocks'].append(block)
    page_count = index.source.get('source_page_count')
    if page_count is None:
        # Older saved SourceIndex objects may predate this metadata. Verify
        # against the original PDF rather than guessing from recognized pages.
        import pymupdf
        with pymupdf.open(index.source['source_pdf']) as document:
            page_count = len(document)
    complete = sorted(pages) == list(range(page_count))
    save(directory / 'middle_json.json', dict(schema='docvortex.middle', schema_version='2.0',
         is_full_document=complete, pages=list(pages.values())))
    manifest = read(origin / 'manifest.json')
    save(manifest_path, {**manifest, 'corrected_source_sha256':stamp,
                        'source_page_numbers':[n+1 for n in sorted(pages)],
                        'is_full_document':complete,
                        'materialization_method':MATERIALIZATION_METHOD,
                        'formula_corrections':index.source.get('formula_corrections'),
                        'middle_sha256':digest(directory / 'middle_json.json')})
    return directory
