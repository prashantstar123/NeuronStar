#!/usr/bin/env python3
"""Download the large Route 2 files from the GitHub release and verify them.

Small check inputs live in git (route2/fixtures). Large ones (frozen flows, stage archives, banks) are
attached to the GitHub release named in route2/downloads.d/*.json. Each file is downloaded once into
route2/downloads/ and must match its listed SHA-256; a damaged or partial file is never used. Files larger
than 1.9 GB are stored as numbered parts; they are joined after download and the whole file is verified.

  PYTHONPATH=$PWD python -m route2.fetch                      # every file the checks need
  PYTHONPATH=$PWD python -m route2.fetch --group tsnpe_nucleonic
  PYTHONPATH=$PWD python -m route2.fetch --group full_rerun   # optional: the paper's banks and caches for a full rerun
  PYTHONPATH=$PWD python -m route2.fetch --list               # what is available, with sizes

ROUTE2_LOCAL_MIRROR=/path/to/dir makes it copy from a local directory instead of downloading.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from route2.common import ROOT, sha256

DOWNLOADS = ROOT / "route2/downloads"
LISTS = ROOT / "route2/downloads.d"
BASE_URL = "https://github.com/prashantstar123/NeuronStar/releases/download/{release}/{name}"


def catalogue() -> dict[str, dict]:
    """Every downloadable file, keyed by name, with its group (list file) and whether the group is optional."""
    files = {}
    for listing in sorted(LISTS.glob("*.json")):
        record = json.loads(listing.read_text())
        for entry in record["files"]:
            if entry["name"] in files:
                raise SystemExit(f"FAIL: {entry['name']} is listed twice")
            files[entry["name"]] = {**entry, "group": listing.stem, "release": record["release"],
                                    "optional": bool(record.get("optional", False))}
    return files


API = "https://api.github.com/repos/prashantstar123/NeuronStar/releases/tags/{release}"
_ASSET_IDS: dict[str, dict[str, int]] = {}


def _token() -> str | None:
    """A GitHub token (GITHUB_TOKEN, GH_TOKEN or the gh login), used only if the plain download is refused."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    try:
        import subprocess

        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=20)
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _download(request: urllib.request.Request, target: Path) -> None:
    with urllib.request.urlopen(request) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out, 8 * 1024 * 1024)


def _get(name: str, release: str, target: Path) -> None:
    mirror = os.environ.get("ROUTE2_LOCAL_MIRROR")
    if mirror:
        shutil.copyfile(Path(mirror) / name, target)
        return
    try:  # public repository: plain download link
        _download(urllib.request.Request(BASE_URL.format(release=release, name=name)), target)
        return
    except urllib.error.HTTPError as error:
        token = _token()
        if error.code not in (401, 403, 404) or not token:
            raise SystemExit(f"FAIL: cannot download {name} ({error}); check the internet connection and "
                             f"run the command again") from error
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    if release not in _ASSET_IDS:  # the release API with the token
        with urllib.request.urlopen(urllib.request.Request(API.format(release=release), headers=headers)) as response:
            _ASSET_IDS[release] = {asset["name"]: asset["id"] for asset in json.load(response)["assets"]}
    asset = _ASSET_IDS[release].get(name)
    if asset is None:
        raise SystemExit(f"FAIL: {name} is not attached to release {release}")
    url = f"https://api.github.com/repos/prashantstar123/NeuronStar/releases/assets/{asset}"
    _download(urllib.request.Request(url, headers={**headers, "Accept": "application/octet-stream"}), target)


def fetch(name: str, entries: dict | None = None) -> Path:
    """Return the local path of a verified download, fetching (and joining) it if needed."""
    entry = (entries or catalogue())[name]
    target = DOWNLOADS / name
    if target.is_file() and target.stat().st_size == entry["bytes"] and sha256(target) == entry["sha256"]:
        return target
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    print(f"[fetch] {name} ({entry['bytes'] / 1e6:.1f} MB)", flush=True)
    parts = entry.get("parts")
    if parts:
        with partial.open("wb") as out:
            for part in parts:
                piece = DOWNLOADS / (part["name"] + ".partial")
                _get(part["name"], entry["release"], piece)
                if piece.stat().st_size != part["bytes"] or sha256(piece) != part["sha256"]:
                    piece.unlink()
                    raise SystemExit(f"FAIL: {part['name']} does not match its listed SHA-256; run the command again")
                with piece.open("rb") as source:
                    shutil.copyfileobj(source, out, 8 * 1024 * 1024)
                piece.unlink()
    else:
        _get(name, entry["release"], partial)
    if partial.stat().st_size != entry["bytes"] or sha256(partial) != entry["sha256"]:
        partial.unlink()
        raise SystemExit(f"FAIL: {name} does not match its listed SHA-256; run the command again")
    os.replace(partial, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", help="only the files of this list (e.g. tsnpe_nucleonic, full_rerun)")
    parser.add_argument("--list", action="store_true", help="list the available files and exit")
    arguments = parser.parse_args()
    entries = catalogue()
    if arguments.list:
        for name, entry in entries.items():
            print(f"{entry['group']:18s} {entry['bytes'] / 1e6:10.1f} MB  {name}")
        return 0
    names = [name for name, entry in entries.items()
             if (entry["group"] in arguments.group if arguments.group else not entry["optional"])]
    for name in names:
        fetch(name, entries)
    total = sum(entries[name]["bytes"] for name in names)
    print(f"[fetch] PASS: {len(names)} files ({total / 1e6:.1f} MB) present and verified in route2/downloads/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
