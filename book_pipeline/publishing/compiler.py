"""Helpers used by the current PDF Reforge workflow."""
from __future__ import annotations
import os, re, subprocess, time
from pathlib import Path

def compile_project(project, tools, timeout=600, syntax_only=False):
    project = Path(project)
    # An interrupted TeX/biber pass can leave truncated \@writefile entries or
    # invalid bibliography XML. Never let disposable sidecars block a resume.
    # Only remove known compiler outputs, preserving all source and assets.
    for suffix in ("aux", "out", "toc", "lof", "lot", "bcf", "bcf-SAVE-ERROR",
                   "bbl", "blg", "run.xml", "fdb_latexmk", "fls", "xdv", "synctex.gz"):
        (project / ("main." + suffix)).unlink(missing_ok=True)
    env = dict(os.environ)
    env["PATH"] = str(Path(tools["xelatex"]).parent) + os.pathsep + env.get("PATH", "")
    started = time.monotonic()
    command = ([tools["xelatex"], "-no-pdf", "-file-line-error", "-interaction=nonstopmode", "-no-shell-escape", "main.tex"]
               if syntax_only else [tools["latexmk"], "-xelatex", "-file-line-error", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "main.tex"])
    process = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace", start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except BaseException:
        import signal
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
        raise
    (project / "compile-output.txt").write_text(output, encoding="utf-8")
    log = (project / "main.log").read_text(errors="replace") if (project / "main.log").is_file() else output
    warnings = [line.strip() for line in log.splitlines() if re.search(r"Overfull|Missing character|undefined|multiply defined|duplicate", line, re.I)]
    locations = [{"file": m[1], "line": int(m[2])} for m in re.finditer(r"(?:\./)?(chapters/generated-\d+\.tex):(\d+):", log)]
    details = [{"file": m[1], "line": int(m[2]), "message": m[3].strip()}
               for m in re.finditer(r"(?:\./)?(chapters/generated-\d+\.tex):(\d+):\s*([^\n]+)", log)]
    return {"success": process.returncode == 0 and (project / ("main.xdv" if syntax_only else "main.pdf")).is_file(), "exit_code": process.returncode, "error_locations": locations, "error_details": details,
            "warnings": warnings, "log_tail": log[-10000:], "seconds": round(time.monotonic() - started, 2)}
