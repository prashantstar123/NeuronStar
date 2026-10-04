#!/usr/bin/env python3
"""Create a deterministic compact resample of a certified IS posterior."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--method",
        default="clean dual-head 9D plus 7D+2 A-NET with exact IS",
    )
    arguments = parser.parse_args()
    if arguments.draws < 1:
        raise ValueError("--draws must be positive")

    with np.load(arguments.certificate, allow_pickle=False) as saved:
        theta = np.asarray(saved["theta"], dtype=np.float64)
        weight = np.asarray(saved["normalized_weight"], dtype=np.float64)
        posterior_ess = float(saved["posterior_ess"])
        gate_pass = bool(saved["gate_pass"])
    if not gate_pass:
        raise RuntimeError("the source certificate did not pass its internal gate")
    if theta.ndim != 2 or theta.shape[1] != 9 or weight.shape != (len(theta),):
        raise RuntimeError("invalid certified posterior shapes")
    if (
        not np.isfinite(theta).all()
        or not np.isfinite(weight).all()
        or np.any(weight < 0.0)
        or not np.isclose(weight.sum(), 1.0, atol=1.0e-12)
    ):
        raise RuntimeError("invalid certified posterior weights")

    rng = np.random.default_rng(arguments.seed)
    sampled = rng.choice(len(theta), size=arguments.draws, replace=True, p=weight)
    source_index, counts = np.unique(sampled, return_counts=True)
    source_hash = sha256(arguments.certificate)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            theta=theta[source_index],
            counts=counts.astype(np.int64),
            source_index=source_index.astype(np.int64),
            method=np.asarray(arguments.method),
            seed=np.int64(arguments.seed),
            posterior_draws=np.int64(arguments.draws),
            source_posterior_ess=np.float64(posterior_ess),
            source_certified_file=np.asarray(str(arguments.certificate.resolve())),
            source_sha256=np.asarray(source_hash),
        )
    os.replace(temporary, arguments.output)
    report = {
        "status": "PASS",
        "posterior_draws": arguments.draws,
        "unique_rows": int(len(source_index)),
        "seed": arguments.seed,
        "source_posterior_ess": posterior_ess,
        "source_sha256": source_hash,
        "output_sha256": sha256(arguments.output),
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
