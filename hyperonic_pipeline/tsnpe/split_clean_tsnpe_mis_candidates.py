#!/usr/bin/env python3
"""Split a frozen clean TSNPE MIS candidate mixture without altering it."""

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


def atomic_save(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=2)
    parser.add_argument(
        "--part-counts",
        type=int,
        nargs="+",
        help="Optional explicit row counts for unequal-throughput workers.",
    )
    arguments = parser.parse_args()
    if arguments.parts < 2:
        raise ValueError("at least two shards are required")
    report = json.loads(arguments.input.with_suffix(".json").read_text())
    if (
        report.get("status") != "PASS"
        or report.get("lineage_class") != "independent_tsnpe"
        or report.get("role") not in {
            "tsnpe_full_prior_mis_candidates",
            "tsnpe_adaptive_full_prior_mis_candidates",
            "tsnpe_frozen_adaptive_flow_full_prior_proposal",
            "tsnpe_dual_stage_new_proposal_candidates",
            "tsnpe_crossvalidated_hybrid_full_prior_proposal_candidates",
        }
        or report.get("forbidden_artifacts_used")
    ):
        raise RuntimeError("candidate mixture failed clean-lineage gates")
    parent_hash = sha256(arguments.input)
    if report.get("output_sha256") != parent_hash:
        raise RuntimeError("candidate mixture hash does not match its report")

    with np.load(arguments.input, allow_pickle=False) as archive:
        rows = len(np.asarray(archive["theta"]))
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key not in {"metadata", "forbidden_artifacts_used"}
            and np.asarray(archive[key]).ndim > 0
            and len(np.asarray(archive[key])) == rows
        }
        constants = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key not in arrays
            and key not in {"metadata", "forbidden_artifacts_used"}
        }
    if arguments.part_counts is not None:
        if (
            len(arguments.part_counts) != arguments.parts
            or any(count <= 0 for count in arguments.part_counts)
            or sum(arguments.part_counts) != rows
        ):
            raise ValueError(
                "--part-counts must contain one positive count per part and sum to all rows"
            )
        boundaries = np.cumsum([0, *arguments.part_counts])
        indices = [
            np.arange(boundaries[index], boundaries[index + 1])
            for index in range(arguments.parts)
        ]
    else:
        indices = np.array_split(np.arange(rows), arguments.parts)
    shards = []
    for index, selection in enumerate(indices):
        output = arguments.output_dir / f"tsnpe_mis_candidates_part_{index:02d}.npz"
        local_metadata = {
            **metadata,
            "parent_candidate_sha256": parent_hash,
            "shard_index": index,
            "shard_count": arguments.parts,
            "shard_rows": int(len(selection)),
            "forbidden_artifacts_used": [],
        }
        atomic_save(
            output,
            **{key: value[selection] for key, value in arrays.items()},
            **constants,
            metadata=np.asarray(json.dumps(local_metadata, sort_keys=True)),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
        local_report = {
            **local_metadata,
            "status": "PASS",
            "output": str(output.resolve()),
            "output_sha256": sha256(output),
        }
        output.with_suffix(".json").write_text(
            json.dumps(local_report, indent=2, sort_keys=True) + "\n"
        )
        shards.append(local_report)
    manifest = {
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_full_prior_mis_candidate_split",
        "parent_candidate_sha256": parent_hash,
        "shards": shards,
        "forbidden_artifacts_used": [],
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
