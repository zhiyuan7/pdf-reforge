#!/usr/bin/env python3
"""Install isolated XeLaTeX/latexmk dependencies without sudo or shell profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
PACKAGES = "ctex xecjk fandol zhnumber tcolorbox biblatex needspace adjustbox newtx txfonts tex-gyre xstring etoolbox kvoptions titlesec appendix apptools adforn fixtounicode tocloft mwe makecell hologo fancyvrb enumitem footmisc multirow booktabs csquotes psnfss manfnt bbding wasysym esint setspace comment mathtools arydshln".split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/tlnet")
    parser.add_argument("--release", default="v2026.10", help="Pinned TinyTeX release")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        parser.error("This installer supports Linux x86_64/aarch64; install TeX Live and latexmk manually on other systems")
    tools = ROOT / ".tools"
    tools.mkdir(exist_ok=True)
    tex = tools / ".TinyTeX"
    if not tex.is_dir():
        arch = "x86_64" if platform.machine() == "x86_64" else "arm64"
        name = f"TinyTeX-linux-{arch}-{args.release}.tar.xz"
        url = f"https://github.com/rstudio/tinytex-releases/releases/download/{args.release}/{name}"
        archive = tools / name
        print("Downloading " + url, flush=True)
        with urlopen(url, timeout=60) as response, archive.open("wb") as out:
            while data := response.read(1024 * 1024):
                out.write(data)
        with tarfile.open(archive) as bundle:
            bundle.extractall(tools, filter="data")
        archive.unlink()
    tlmgr = next(tex.glob("bin/*/tlmgr")).absolute()
    subprocess.run([str(tlmgr), "option", "repository", args.repository], check=True)
    subprocess.run([str(tlmgr), "install", *PACKAGES], check=True)
    print("Installed local TeX: " + str(tlmgr.parent))


if __name__ == "__main__":
    main()
