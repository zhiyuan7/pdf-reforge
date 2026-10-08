"""Helpers used by the current PDF Reforge workflow."""
from __future__ import annotations

DEFAULT_LABELS = {
    "定义": "definition", "记号": "notation", "定理": "theorem", "引理": "lemma",
    "推论": "corollary", "命题": "proposition", "例题": "example", "例": "example",
    "算法": "algorithm", "习题": "exercise", "练习": "exercise", "问题": "exercise",
    "注释": "remark", "注记": "remark", "注": "remark", "公理": "numbered",
    "结论": "numbered", "性质": "numbered", "准则": "numbered",
    "Definition": "definition", "Theorem": "theorem", "Lemma": "lemma",
    "Corollary": "corollary", "Proposition": "proposition", "Example": "example",
    "Algorithm": "algorithm", "Exercise": "exercise", "Remark": "remark", "Axiom": "numbered",
}


BOX_KINDS = set(DEFAULT_LABELS.values())


CAPTION_ROLES = {f"{kind}_{role}" for kind in ("image", "chart", "table")
                 for role in ("caption", "footnote")}
