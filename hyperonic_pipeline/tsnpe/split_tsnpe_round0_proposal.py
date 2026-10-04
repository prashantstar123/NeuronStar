#!/usr/bin/env python3
"""Split a clean TSNPE round-zero proposal into pilot and candidate shards."""

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


def write_piece(
    path: Path,
    arrays: dict[str, np.ndarray],
    constants: dict[str, np.ndarray],
    index: np.ndarray,
    metadata: dict[str, object],
    pilot_rows: int,
) -> dict[str, object]:
    local_metadata = {
        **metadata,
        "pilot_rows": pilot_rows,
        "training_candidate_rows": int(len(index) - pilot_rows),
    }
    atomic_savez(
        path,
        **{key: value[index] for key, value in arrays.items()},
        **constants,
        pilot_rows=np.asarray(pilot_rows, dtype=np.int64),
        metadata=np.asarray(json.dumps(local_metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    report = {
        "status": "PASS",
        **local_metadata,
        "output": str(path.resolve()),
        "output_sha256": sha256(path),
    }
    path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-parts", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.candidate_parts < 1:
        raise ValueError("candidate part count must be positive")
    report_path = arguments.proposal.with_suffix(".json")
    parent_report = json.loads(report_path.read_text())
    if parent_report.get("lineage_class") != "independent_tsnpe":
        raise RuntimeError("proposal lacks independent TSNPE lineage")
    if parent_report.get("role") != "tsnpe_round_proposal":
        raise RuntimeError("input has the wrong proposal role")
    if parent_report.get("round_index") != 0:
        raise RuntimeError("only round zero can be split by this script")
    if parent_report.get("forbidden_artifacts_used"):
        raise RuntimeError("proposal declares forbidden ancestry")
    if parent_report.get("output_sha256") != sha256(arguments.proposal):
        raise RuntimeError("proposal hash does not match its report")

    with np.load(arguments.proposal, allow_pickle=False) as archive:
        pilot_rows = int(archive["pilot_rows"])
        original_metadata = json.loads(str(archive["metadata"].item()))
        row_count = len(archive["theta"])
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key not in {"metadata", "pilot_rows", "forbidden_artifacts_used"}
            and np.asarray(archive[key]).ndim > 0
            and len(np.asarray(archive[key])) == row_count
        }
        constants = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key not in arrays
            and key not in {"metadata", "pilot_rows", "forbidden_artifacts_used"}
        }
    if pilot_rows < 1 or pilot_rows >= row_count:
        raise RuntimeError("proposal has an invalid pilot size")
    parent = {
        "path": str(arguments.proposal.resolve()),
        "sha256": sha256(arguments.proposal),
    }
    base_metadata = {
        **original_metadata,
        "parent_proposal": parent,
        "split_without_density_change": True,
    }
    pilot_index = np.arange(pilot_rows)
    pilot_path = arguments.output_dir / "round_00_bridge_v3_pilot.npz"
    pilot_report = write_piece(
        pilot_path,
        arrays,
        constants,
        pilot_index,
        {**base_metadata, "proposal_piece": "pilot"},
        pilot_rows,
    )
    candidate_indices = np.array_split(
        np.arange(pilot_rows, row_count), arguments.candidate_parts
    )
    candidate_reports = []
    for part, index in enumerate(candidate_indices):
        path = arguments.output_dir / f"round_00_bridge_v3_candidate_{part:02d}.npz"
        candidate_reports.append(
            write_piece(
                path,
                arrays,
                constants,
                index,
                {
                    **base_metadata,
                    "proposal_piece": "candidate_shard",
                    "proposal_shard_index": part,
                    "proposal_shard_count": arguments.candidate_parts,
                },
                0,
            )
        )
    manifest = {
        "status": "PASS",
        "parent": parent,
        "pilot": pilot_report,
        "candidate_shards": candidate_reports,
        "forbidden_artifacts_used": [],
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "round_00_bridge_v3_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
