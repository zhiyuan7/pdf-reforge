"""Dependency-free MinerU progress events for opt-in callers."""
import json
import os

PREFIX = "@@PDF_PROGRESS@@"


def emit(phase, done=0, total=None, unit="页", detail=""):
    if os.environ.get("PDF_PIPELINE_PROGRESS") == "1":
        print(PREFIX + json.dumps(dict(phase=phase, done=done, total=total, unit=unit, detail=detail),
                                 ensure_ascii=False), flush=True)
