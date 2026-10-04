#!/usr/bin/env python3
"""Compare every rebuilt figure with the published one, pixel by pixel.

Both PDFs are rendered to images with the same renderer (PDFium, via pypdfium2) at 180 dpi and the
images must be identical. The PDF files themselves may differ in embedded metadata such as the
creation date, which does not change the picture. Fig. 1, the workflow diagram, is not drawn from results;
its file is checked by its SHA-256 instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = {"FINAL_workflow_flowchart_v3.pdf": "d2740be9725457542770ace17fa605518b070b75f65a908a9b4268e6ad4d81d5"}


def raster(path: Path, dpi: int) -> np.ndarray:
    document = pdfium.PdfDocument(str(path))
    try:
        bitmap = document[0].render(scale=dpi / 72.0)
        return np.array(bitmap.to_numpy(), copy=True)
    finally:
        document.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuilt-dir", type=Path, default=ROOT / "build/figures")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "paper/figures")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--report", type=Path, default=ROOT / "build/figure_comparison.json")
    arguments = parser.parse_args()
    results, ok = {}, True
    for reference in sorted(arguments.reference_dir.glob("*.pdf")):
        if reference.name in DIAGRAMS:
            same = hashlib.sha256(reference.read_bytes()).hexdigest() == DIAGRAMS[reference.name]
            results[reference.name] = "SHA-256 VERIFIED (diagram)" if same else "SHA-256 MISMATCH"
            ok &= same
            print(f"[compare] {'PASS' if same else 'FAIL'} {reference.name} (workflow diagram, SHA-256)")
            continue
        rebuilt = arguments.rebuilt_dir / reference.name
        if not rebuilt.is_file():
            results[reference.name] = "MISSING"
            ok = False
            print(f"[compare] FAIL {reference.name}: not rebuilt")
            continue
        a, b = raster(reference, arguments.dpi), raster(rebuilt, arguments.dpi)
        same = a.shape == b.shape and np.array_equal(a, b)
        results[reference.name] = "PIXEL-IDENTICAL" if same else "DIFFERENT"
        ok &= same
        print(f"[compare] {'PASS' if same else 'FAIL'} {reference.name}"
              f"{'' if same else ' (differs from the published figure)'}")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps({"dpi": arguments.dpi, "renderer": f"pypdfium2 {version('pypdfium2')}",
                                            "figures": results}, indent=1) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
