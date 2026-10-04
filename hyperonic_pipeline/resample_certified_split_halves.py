#!/usr/bin/env python3
"""Resample two disjoint halves of a certified IS posterior for stability."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--draws-per-half", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--minimum-half-ess", type=float, default=400.0)
    parser.add_argument(
        "--method",
        default="clean dual-head 9D plus 7D+2 A-NET with exact IS",
    )
    arguments = parser.parse_args()
    if arguments.draws_per_half < 1 or arguments.minimum_half_ess <= 0.0:
        raise ValueError("draw count and ESS gate must be positive")

    with np.load(arguments.certificate, allow_pickle=False) as saved:
        theta = np.asarray(saved["theta"], dtype=np.float64)
        weight = np.asarray(saved["normalized_weight"], dtype=np.float64)
        gate_pass = bool(saved["gate_pass"])
    if not gate_pass:
        raise RuntimeError("the source certificate did not pass its internal gate")
    if theta.ndim != 2 or theta.shape[1] != 9 or weight.shape != (len(theta),):
        raise RuntimeError("invalid certified posterior shapes")
    if len(theta) < 4 or not np.isclose(weight.sum(), 1.0, atol=1.0e-12):
        raise RuntimeError("invalid certified posterior weights")

    split = len(theta) // 2
    halves = (np.arange(split), np.arange(split, len(theta)))
    source_hash = sha256(arguments.certificate)
    records = []
    for half, index in enumerate(halves):
        local_weight = weight[index]
        mass = float(local_weight.sum())
        if mass <= 0.0:
            raise RuntimeError(f"posterior half {half} has zero probability mass")
        local_weight = local_weight / mass
        ess = float(1.0 / np.square(local_weight).sum())
        rng = np.random.default_rng(arguments.seed + half)
        local_draw = rng.choice(
            len(index),
            size=arguments.draws_per_half,
            replace=True,
            p=local_weight,
        )
        unique_local, counts = np.unique(local_draw, return_counts=True)
        source_index = index[unique_local]
        output = arguments.output_prefix.with_name(
            f"{arguments.output_prefix.name}_half{half}.npz"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                theta=theta[source_index],
                counts=counts.astype(np.int64),
                source_index=source_index.astype(np.int64),
                method=np.asarray(arguments.method),
                half=np.int64(half),
                half_rows=np.int64(len(index)),
                half_probability_mass=np.float64(mass),
                half_posterior_ess=np.float64(ess),
                seed=np.int64(arguments.seed + half),
                posterior_draws=np.int64(arguments.draws_per_half),
                source_certified_file=np.asarray(
                    str(arguments.certificate.resolve())
                ),
                source_sha256=np.asarray(source_hash),
            )
        os.replace(temporary, output)
        records.append(
            {
                "half": half,
                "rows": len(index),
                "probability_mass": mass,
                "posterior_ess": ess,
                "ess_gate_pass": ess >= arguments.minimum_half_ess,
                "unique_resample_rows": int(len(source_index)),
                "output": str(output.resolve()),
                "output_sha256": sha256(output),
            }
        )
    passed = all(record["ess_gate_pass"] for record in records)
    report = {
        "status": "PASS" if passed else "FAIL",
        "certificate_sha256": source_hash,
        "minimum_half_ess": arguments.minimum_half_ess,
        "draws_per_half": arguments.draws_per_half,
        "halves": records,
    }
    report_path = arguments.output_prefix.with_name(
        arguments.output_prefix.name + "_split_report.json"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
