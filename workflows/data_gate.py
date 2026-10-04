"""Checksum gate for externally staged observational data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKAGED_DATA = Path(__file__).resolve().parent / "data"
MANIFEST = PACKAGED_DATA / "observational-manifest.json"
SUBSTITUTED_PULSAR_MANIFEST = PACKAGED_DATA / "substituted-pulsar-manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_manifest(
    data_root: str | Path,
    manifest_path: str | Path,
    filenames: set[str] | None = None,
) -> dict[str, Path]:
    data_root = Path(data_root).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    paths: dict[str, Path] = {}
    failures = []
    for record in manifest["artifacts"]:
        if filenames is not None and record["filename"] not in filenames:
            continue
        path = data_root / record["filename"]
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(
                f"size mismatch for {path}: expected {record['bytes']}, "
                f"got {path.stat().st_size}"
            )
            continue
        digest = sha256(path)
        if digest != record["sha256"]:
            failures.append(
                f"hash mismatch for {path}: expected {record['sha256']}, got {digest}"
            )
            continue
        paths[record["filename"]] = path
    if failures:
        detail = "\n- ".join(failures)
        raise RuntimeError(f"observational data gate failed:\n- {detail}")
    return paths


def validate_observational_data(data_root: str | Path) -> dict[str, Path]:
    """Return validated baseline artifact paths before a calculation starts."""

    return _validate_manifest(data_root, MANIFEST)


def validate_substituted_pulsar_data(
    data_root: str | Path, filename: str | None = None
) -> dict[str, Path]:
    """Validate one or both optional substituted-pulsar sample files."""

    filenames = None if filename is None else {filename}
    paths = _validate_manifest(
        data_root, SUBSTITUTED_PULSAR_MANIFEST, filenames=filenames
    )
    if filename is not None and filename not in paths:
        raise KeyError(f"unknown substituted-pulsar artifact {filename!r}")
    return paths
