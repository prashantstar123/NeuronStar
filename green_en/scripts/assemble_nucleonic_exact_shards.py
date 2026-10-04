#!/usr/bin/env python3
"""Validate and concatenate ordered nucleonic exact-row shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROW_FIELDS = (
    "source_index",
    "proposal_index",
    "theta",
    "prediction",
    "log_astrophysical",
    "astrophysical_components",
    "nicer_source_components",
    "target_valid",
    "radius",
    "tidal_lambda",
    "maximum_mass",
)
SCALAR_EQUAL_FIELDS = (
    "schema",
    "support_name",
    "screen_sha256",
    "template_sha256",
    "template_parent_sha256",
    "total_proposals",
    "seed",
    "prediction_tolerance_sigma",
    "reference_present",
    "forbidden_artifacts_used",
    "evaluator",
    "prediction_storage",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {arguments.output}")
    ordered = []
    for path in arguments.shard:
        with np.load(path, allow_pickle=False) as shard:
            start = int(shard["shard_start"])
            stop = int(shard["shard_stop"])
        ordered.append((start, stop, path))
    ordered.sort()
    expected = arguments.start
    for start, stop, path in ordered:
        if start != expected or stop <= start:
            raise RuntimeError(
                f"non-contiguous shard {path}: [{start},{stop}), expected {expected}"
            )
        expected = stop
    if expected != arguments.stop:
        raise RuntimeError(f"shards end at {expected}, expected {arguments.stop}")

    values: dict[str, list[np.ndarray]] = {key: [] for key in ROW_FIELDS}
    reference: dict[str, np.ndarray] = {}
    mass_grid = prior_low = prior_high = None
    parents = []
    starts = []
    finishes = []
    internal_walls = []
    workers = 0
    maximum_prediction_delta = 0.0
    maximum_normalized_prediction_delta = 0.0
    for start, stop, path in ordered:
        with np.load(path, allow_pickle=False) as shard:
            for key in ROW_FIELDS:
                array = np.asarray(shard[key])
                if len(array) != stop - start:
                    raise RuntimeError(f"{path}: {key} has the wrong row count")
                values[key].append(array)
            for key in SCALAR_EQUAL_FIELDS:
                array = np.asarray(shard[key])
                if key not in reference:
                    reference[key] = array.copy()
                elif not np.array_equal(reference[key], array):
                    raise RuntimeError(f"{path}: scalar field {key} differs")
            local_grid = np.asarray(shard["mass_grid"], dtype=np.float64)
            local_low = np.asarray(shard["prior_low"], dtype=np.float64)
            local_high = np.asarray(shard["prior_high"], dtype=np.float64)
            if mass_grid is None:
                mass_grid, prior_low, prior_high = local_grid, local_low, local_high
            elif not (
                np.array_equal(mass_grid, local_grid)
                and np.array_equal(prior_low, local_low)
                and np.array_equal(prior_high, local_high)
            ):
                raise RuntimeError(f"{path}: grid or prior differs")
            starts.append(float(shard["started_unix"]))
            finishes.append(float(shard["finished_unix"]))
            internal_walls.append(float(shard["wall_seconds"]))
            workers += int(shard["workers"])
            maximum_prediction_delta = max(
                maximum_prediction_delta, float(shard["maximum_prediction_delta"])
            )
            maximum_normalized_prediction_delta = max(
                maximum_normalized_prediction_delta,
                float(shard["maximum_normalized_prediction_delta"]),
            )
        parents.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "start": start,
                "stop": stop,
            }
        )

    merged = {key: np.concatenate(parts, axis=0) for key, parts in values.items()}
    if not np.array_equal(
        merged["source_index"], np.arange(arguments.start, arguments.stop, dtype=np.int64)
    ):
        raise RuntimeError("aggregate source_index is incomplete or reordered")
    with np.load(arguments.screen, allow_pickle=False) as screen:
        if not np.array_equal(
            merged["proposal_index"],
            np.asarray(screen["proposal_index"], dtype=np.int64)[arguments.start:arguments.stop],
        ):
            raise RuntimeError("aggregate proposal_index differs from the screen")
        if not np.array_equal(
            merged["theta"],
            np.asarray(screen["theta"], dtype=np.float64)[arguments.start:arguments.stop],
        ):
            raise RuntimeError("aggregate theta differs from the screen")

    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **merged,
                schema=reference["schema"],
                support_name=reference["support_name"],
                mass_grid=mass_grid,
                prior_low=prior_low,
                prior_high=prior_high,
                screen_sha256=reference["screen_sha256"],
                template_sha256=reference["template_sha256"],
                template_parent_sha256=reference["template_parent_sha256"],
                total_proposals=reference["total_proposals"],
                seed=reference["seed"],
                shard_start=np.int64(arguments.start),
                shard_stop=np.int64(arguments.stop),
                prediction_tolerance_sigma=reference["prediction_tolerance_sigma"],
                maximum_prediction_delta=np.float64(maximum_prediction_delta),
                maximum_normalized_prediction_delta=np.float64(
                    maximum_normalized_prediction_delta
                ),
                workers=np.int64(workers),
                hostname=np.asarray("neutron06-parallel"),
                reference_present=reference["reference_present"],
                forbidden_artifacts_used=reference["forbidden_artifacts_used"],
                started_unix=np.float64(min(starts)),
                finished_unix=np.float64(max(finishes)),
                wall_seconds=np.float64(max(finishes) - min(starts)),
                evaluator=np.asarray(
                    "scripts/evaluate_nucleonic_support_extension.py; parallel ordered shards"
                ),
                prediction_storage=reference["prediction_storage"],
                parallel_shards=np.int64(len(ordered)),
            )
        os.replace(temporary, arguments.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    report = {
        "status": "PASS",
        "schema": "nucleonic-exact-parallel-assembly-v1",
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "rows": int(arguments.stop - arguments.start),
        "target_valid_rows": int(merged["target_valid"].sum()),
        "shards": parents,
        "shard_internal_walls_seconds": internal_walls,
        "shard_critical_path_seconds": max(finishes) - min(starts),
        "workers_total": workers,
        "screen": str(arguments.screen.resolve()),
        "screen_sha256": sha256(arguments.screen),
        "template": str(arguments.template.resolve()),
        "template_sha256": sha256(arguments.template),
        "reference_present": False,
        "forbidden_artifacts_used": [],
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
