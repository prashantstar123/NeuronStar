#!/usr/bin/env python3
"""Split a clean zero-pilot TSNPE proposal shard without changing density."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=2)
    parser.add_argument(
        "--prefix",
        help="output stem; defaults to the input proposal stem",
    )
    arguments = parser.parse_args()
    if arguments.parts < 2:
        raise ValueError("candidate shard must be split into at least two parts")
    parent_report = json.loads(arguments.proposal.with_suffix(".json").read_text())
    if parent_report.get("lineage_class") != "independent_tsnpe":
        raise RuntimeError("candidate lacks independent TSNPE lineage")
    if parent_report.get("role") != "tsnpe_round_proposal":
        raise RuntimeError("candidate has the wrong artifact role")
    if parent_report.get("pilot_rows") != 0:
        raise RuntimeError("generic candidate splitter requires zero pilot rows")
    if parent_report.get("forbidden_artifacts_used"):
        raise RuntimeError("candidate declares forbidden ancestry")
    parent_hash = sha256(arguments.proposal)
    if parent_report.get("output_sha256") != parent_hash:
        raise RuntimeError("candidate hash does not match its report")

    with np.load(arguments.proposal, allow_pickle=False) as archive:
        rows = len(archive["theta"])
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key not in {"metadata", "pilot_rows", "forbidden_artifacts_used"}
            and np.asarray(archive[key]).ndim > 0
            and len(np.asarray(archive[key])) == rows
        }
        constants = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key not in arrays
            and key not in {"metadata", "pilot_rows", "forbidden_artifacts_used"}
        }
    indices = np.array_split(np.arange(rows), arguments.parts)
    prefix = arguments.prefix or arguments.proposal.stem
    manifest = []
    for part, index in enumerate(indices):
        output = arguments.output_dir / f"{prefix}_part_{part:02d}.npz"
        local_metadata = {
            **metadata,
            "training_candidate_rows": int(len(index)),
            "subshard_parent": {
                "path": str(arguments.proposal.resolve()),
                "sha256": parent_hash,
            },
            "proposal_subshard_index": part,
            "proposal_subshard_count": arguments.parts,
            "forbidden_artifacts_used": [],
        }
        atomic_savez(
            output,
            **{key: value[index] for key, value in arrays.items()},
            **constants,
            pilot_rows=np.asarray(0, dtype=np.int64),
            metadata=np.asarray(json.dumps(local_metadata, sort_keys=True)),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
        report = {
            "status": "PASS",
            **local_metadata,
            "pilot_rows": 0,
            "output": str(output.resolve()),
            "output_sha256": sha256(output),
        }
        output.with_suffix(".json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest.append(report)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "parent": {
                    "path": str(arguments.proposal.resolve()),
                    "sha256": parent_hash,
                },
                "parts": manifest,
                "forbidden_artifacts_used": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
