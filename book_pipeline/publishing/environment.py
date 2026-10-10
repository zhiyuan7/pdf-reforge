"""Environment operations, independent of workflow orchestration."""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from book_pipeline.ai.client import APIFailure
from book_pipeline.config import ROOT, executable


def doctor(config, require_key=True):
    import pymupdf
    from PIL import Image
    import openai
    tools = {name: executable(name, config[name]) for name in ("xelatex", "latexmk")}
    missing = [name for name, path in tools.items() if not path]
    if missing:
        raise ValueError("Missing TeX tools: " + ", ".join(missing) + "; run .venv/bin/python install_book_tex.py")
    if require_key and not os.environ.get(config["api_key_env"]):
        raise APIFailure("Missing environment variable " + config["api_key_env"])
    kpse = str(Path(tools["xelatex"]).parent / "kpsewhich")
    for name in ("ctex.sty", "newtxmath.sty", "txsyc.tfm", "txsyc.pfb", "tcolorbox.sty", "biblatex.sty", "needspace.sty", "adjustbox.sty", "apptools.sty", "adforn.sty", "fixtounicode.sty", "mathtools.sty", "arydshln.sty", "zref-abspage.sty", "FandolFang-Regular.otf"):
        command = subprocess.run([kpse, name], capture_output=True, text=True)
        if command.returncode or not command.stdout.strip():
            raise ValueError("Missing TeX package: " + name + "; run install_book_tex.py")
    font = Path(config["caption_font_file"])
    if not font.is_absolute():
        font = ROOT / font
    if not font.is_file():
        raise ValueError("Missing caption font: " + str(font))
    return tools
