#!/usr/bin/env python3
"""Check the installation and every supplied file, and unpack the compressed observation files.

Run by './reproduce.sh check' and automatically at the start of './reproduce.sh route1'.
Every data and result file must match the SHA-256 recorded in data/MANIFEST.json and
results/MANIFEST.json. Compressed observation files (.gz) are unpacked once next to the archive
and verified. Nothing is downloaded; nothing is changed except the unpacked copies.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {  # exact versions used to draw the published figures
    "numpy": "2.4.6", "scipy": "1.17.1", "matplotlib": "3.11.1", "seaborn": "0.13.2", "pandas": "3.0.5",
    "pypdfium2": None,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_packages() -> list[str]:
    problems = []
    if sys.version_info[:2] != (3, 11):
        problems.append(f"Python 3.11 is required, found {sys.version.split()[0]}")
    for module, version in REQUIRED.items():
        try:
            imported = importlib.import_module(module)
        except ImportError:
            name = module
            problems.append(f"missing Python package: {name}")
            continue
        if version is not None and getattr(imported, "__version__", None) != version:
            problems.append(f"{module} {version} is required, found {getattr(imported, '__version__', '?')}")
    return problems


def unpack(entry: dict) -> None:
    target = ROOT / entry["path"]
    if target.is_file() and sha256(target) == entry["sha256"]:
        return
    archive = ROOT / entry["stored_as"]
    if sha256(archive) != entry["stored_sha256"]:
        raise SystemExit(f"FAIL: {archive.relative_to(ROOT)} is damaged (checksum mismatch); download the package again")
    partial = target.with_name(target.name + ".partial")
    with gzip.open(archive, "rb") as source, partial.open("wb") as out:
        shutil.copyfileobj(source, out, 8 * 1024 * 1024)
    if sha256(partial) != entry["sha256"]:
        partial.unlink()
        raise SystemExit(f"FAIL: unpacked {target.name} does not match its recorded checksum")
    os.replace(partial, target)
    print(f"[check] unpacked {entry['path']}")


def main() -> int:
    problems = check_packages()
    if problems:
        print("FAIL: the Python environment is not the one this package needs:")
        for problem in problems:
            print("  -", problem)
        print("Create it with:  conda env create -f environment-route1.yml   (see README, step 2)")
        return 1
    print("[check] PASS Python environment (exact package versions)")
    count = 0
    for manifest in ("data/MANIFEST.json", "results/MANIFEST.json"):
        for entry in json.loads((ROOT / manifest).read_text())["files"]:
            if "stored_as" in entry:
                unpack(entry)
            path = ROOT / entry["path"]
            if not path.is_file():
                print(f"FAIL: missing file {entry['path']}")
                return 1
            if sha256(path) != entry["sha256"]:
                print(f"FAIL: {entry['path']} does not match its recorded checksum")
                return 1
            count += 1
    print(f"[check] PASS all {count} data and result files match their recorded checksums")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
