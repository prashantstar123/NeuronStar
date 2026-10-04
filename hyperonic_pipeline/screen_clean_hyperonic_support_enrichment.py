#!/usr/bin/env python3
"""Deterministically screen uniform-prior rows for A-NET support enrichment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np

from eos import get_eos
from likelihoods.nuclear import A1_OBSERVATION, A1_SIGMA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposals", type=int, default=160_000_000)
    parser.add_argument("--block", type=int, default=2_000_000)
    parser.add_argument("--cbox", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20_260_729)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists() and not arguments.force:
        raise FileExistsError(f"support screen exists: {arguments.output}")
    if min(arguments.proposals, arguments.block) < 1 or arguments.cbox <= 0:
        raise ValueError("proposal counts and cbox must be positive")

    started = time.time()
    plugin = get_eos("ddb-hyperonic")
    prior_low = np.asarray(plugin.prior_low, dtype=np.float64)
    prior_high = np.asarray(plugin.prior_high, dtype=np.float64)
    width = prior_high - prior_low
    rho_column = plugin.parameter_names.index("rho0")
    selected_theta = []
    selected_prediction = []
    block_counts = []
    proposed = 0
    block_index = 0
    while proposed < arguments.proposals:
        count = min(arguments.block, arguments.proposals - proposed)
        rng = np.random.default_rng(
            np.random.SeedSequence([arguments.seed, 1, block_index])
        )
        theta = prior_low + rng.random((count, 9)) * width
        prediction = np.empty((count, 7), dtype=np.float64)
        prediction[:, 0] = theta[:, rho_column]
        prediction[:, 1:] = plugin.nuclear_observables_batch(theta)
        keep = np.all(
            np.abs((prediction - A1_OBSERVATION) / A1_SIGMA)
            <= arguments.cbox,
            axis=1,
        )
        selected_theta.append(theta[keep])
        selected_prediction.append(prediction[keep])
        block_counts.append(int(keep.sum()))
        proposed += count
        block_index += 1
        print(
            f"[hyperonic-support-screen] {proposed}/{arguments.proposals}; "
            f"accepted={sum(block_counts)}",
            flush=True,
        )

    theta = np.concatenate(selected_theta, axis=0)
    prediction = np.concatenate(selected_prediction, axis=0)
    if len(theta) == 0:
        raise RuntimeError("support screen accepted no rows")
    if not (
        np.all(theta >= prior_low)
        and np.all(theta <= prior_high)
        and np.isfinite(theta).all()
        and np.isfinite(prediction).all()
    ):
        raise RuntimeError("support screen produced invalid rows")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray("ddb-hyperonic-clean-support-screen-v1"),
            theta=theta,
            prediction=prediction,
            prior_low=prior_low,
            prior_high=prior_high,
            observation=np.asarray(A1_OBSERVATION, dtype=np.float64),
            sigma=np.asarray(A1_SIGMA, dtype=np.float64),
            cbox=np.float64(arguments.cbox),
            seed=np.int64(arguments.seed),
            stream_id=np.int64(1),
            total_proposals=np.int64(arguments.proposals),
            block_size=np.int64(arguments.block),
            block_accepted=np.asarray(block_counts, dtype=np.int64),
            hostname=np.asarray(socket.gethostname()),
            reference_present=np.bool_(False),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
            started_unix=np.float64(started),
            finished_unix=np.float64(time.time()),
        )
    os.replace(temporary, arguments.output)
    report = {
        "status": "PASS",
        "stage": "deterministic uniform-prior nuclear-support screen",
        "schema": "ddb-hyperonic-clean-support-screen-v1",
        "proposals": arguments.proposals,
        "accepted": int(len(theta)),
        "acceptance_fraction": float(len(theta) / arguments.proposals),
        "cbox": arguments.cbox,
        "seed": arguments.seed,
        "stream_id": 1,
        "blocks": block_index,
        "block_accepted": block_counts,
        "hostname": socket.gethostname(),
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "reference_present": False,
        "forbidden_artifacts_used": [],
        "wall_seconds": time.time() - started,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
