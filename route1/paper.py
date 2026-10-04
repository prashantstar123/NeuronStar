#!/usr/bin/env python3
"""Compile the manuscript with the rebuilt figures (build/reproduced_manuscript.pdf).
Fig. 1, the workflow diagram, is not drawn from results; its SHA-256-checked file from paper/figures is used.

Uses pdflatex from a TeX installation (TeX Live with the REVTeX 4.2 class). The bibliography is
supplied pre-built (paper/main.bbl), so BibTeX is not needed. If pdflatex is not installed, this step
is skipped with instructions; the figures and tables are still checked.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = ("FINAL_workflow_flowchart_v3.pdf",)  # checked by SHA-256 in route1/compare.py


def main() -> int:
    if shutil.which("pdflatex") is None:
        print("[paper] SKIPPED: pdflatex was not found, so the PDF was not compiled.")
        print("        Figures and tables are still checked. To build the PDF, install TeX Live, e.g. on Ubuntu:")
        print("        sudo apt install texlive-latex-extra texlive-publishers texlive-science")
        return 0
    work = ROOT / "build/paper"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(ROOT / "paper", work, ignore=shutil.ignore_patterns("figures"))
    shutil.copytree(ROOT / "build/figures", work / "figures")
    for name in DIAGRAMS:
        shutil.copyfile(ROOT / "paper/figures" / name, work / "figures" / name)
    log = ROOT / "build/logs/pdflatex.log"
    for _ in range(2):
        with log.open("w") as stream:
            code = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                                  cwd=work, stdout=stream, stderr=subprocess.STDOUT).returncode
        if code:
            print(f"[paper] FAIL: pdflatex stopped with an error; see {log.relative_to(ROOT)}")
            return 1
    text = log.read_text(errors="replace")
    if "undefined" in text and ("Reference" in text or "Citation" in text):
        print("[paper] FAIL: undefined references or citations; see build/logs/pdflatex.log")
        return 1
    shutil.copyfile(work / "main.pdf", ROOT / "build/reproduced_manuscript.pdf")
    print("[paper] PASS manuscript compiled with the rebuilt figures: build/reproduced_manuscript.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
