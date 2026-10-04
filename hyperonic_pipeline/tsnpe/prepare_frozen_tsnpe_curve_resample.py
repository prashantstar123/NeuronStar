#!/usr/bin/env python3
"""Create a deterministic curve replay sample from certified clean TSNPE MIS."""

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
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=6_000)
    parser.add_argument("--seed", type=int, default=20260921)
    arguments = parser.parse_args()
    if arguments.draws < 1:
        raise ValueError("--draws must be positive")
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")

    report = json.loads(arguments.posterior.with_suffix(".json").read_text())
    source_hash = sha256(arguments.posterior)
    if (
        report.get("status") != "PASS"
        or not report.get("scientific_posterior_certified")
        or report.get("lineage_class") != "independent_tsnpe"
        or report.get("role") not in {
            "tsnpe_frozen_adaptive_flow_full_prior_mis_posterior",
            "tsnpe_dual_stage_full_prior_mis_posterior",
            "tsnpe_crossvalidated_hybrid_full_prior_mis_posterior",
        }
        or report.get("full_prior_dimension") != 9
        or report.get("target_gate", {}).get("status") != "PASS"
        or report.get("forbidden_artifacts_used")
        or report.get("output_sha256") != source_hash
    ):
        raise RuntimeError("TSNPE posterior failed scientific and provenance gates")
    with np.load(arguments.posterior, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64)
        weight = np.asarray(archive["normalized_weight"], dtype=np.float64)
        ess = float(np.asarray(archive["ESS"]).item())
        forbidden = np.asarray(archive["forbidden_artifacts_used"])
    if (
        theta.ndim != 2
        or theta.shape[1] != 9
        or weight.shape != (len(theta),)
        or not np.isfinite(theta).all()
        or not np.isfinite(weight).all()
        or np.any(weight < 0.0)
        or not np.isclose(weight.sum(), 1.0, rtol=0.0, atol=1.0e-12)
        or forbidden.size
    ):
        raise RuntimeError("certified TSNPE posterior arrays are invalid")

    sampled = np.random.default_rng(arguments.seed).choice(
        len(theta), size=arguments.draws, replace=True, p=weight
    )
    source_index, counts = np.unique(sampled, return_counts=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_certified_mass_radius_curve_resample",
        "source_sha256": source_hash,
        "source_posterior_ess": ess,
        "forbidden_artifacts_used": [],
    }
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            theta=theta[source_index],
            counts=counts.astype(np.int64),
            source_index=source_index.astype(np.int64),
            method=np.asarray("clean full-prior 9D TSNPE with exact deterministic MIS"),
            seed=np.int64(arguments.seed),
            posterior_draws=np.int64(arguments.draws),
            source_posterior_ess=np.float64(ess),
            source_certified_file=np.asarray(str(arguments.posterior.resolve())),
            source_sha256=np.asarray(source_hash),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
    os.replace(temporary, arguments.output)
    output_report = {
        **metadata,
        "posterior_draws": arguments.draws,
        "unique_rows": int(len(source_index)),
        "seed": arguments.seed,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(output_report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output_report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
