#!/usr/bin/env python3
"""Route 1: rebuild every figure and table of the paper from the saved results and check them.

Steps: (1) check the environment and all supplied files, (2) redraw the figures, (3) compare each with
the published figure pixel by pixel, (4) recompute every table number and compare with the paper,
(5) compile the manuscript with the rebuilt figures (if pdflatex is available).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("check environment and files", ["route1.prepare"]),
    ("redraw figures", ["route1.figures"]),
    ("compare figures with the paper", ["route1.compare"]),
    ("recompute and check tables", ["route1.tables"]),
    ("compile manuscript", ["route1.paper"]),
]


def main() -> int:
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    start = time.time()
    for number, (title, module) in enumerate(STEPS, 1):
        print(f"\n== Step {number}/{len(STEPS)}: {title}", flush=True)
        code = subprocess.run([sys.executable, "-m", *module], cwd=ROOT, env=env).returncode
        if code:
            print(f"\nROUTE 1 FAILED at step {number} ({title}). Nothing else was run.")
            return code
    minutes = (time.time() - start) / 60
    print(f"\nROUTE 1 PASSED in {minutes:.1f} min: all 14 data-figure files are pixel-identical to the paper, "
          "the workflow diagram matches its SHA-256, and every number in Tables I and III-XII agrees.")
    manuscript = ROOT / "build/reproduced_manuscript.pdf"
    print("Rebuilt figures: build/figures/   Tables: build/tables/   "
          + ("Manuscript: build/reproduced_manuscript.pdf" if manuscript.is_file()
             else "Manuscript PDF: not built (LaTeX not installed; see step 5 above)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
