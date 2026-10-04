#!/usr/bin/env python3
"""Deterministically screen the predeclared Green-EN support extension.

The proposal stream is split into independently seeded blocks.  Parallel and
serial executions therefore generate the same rows in the same order.  Only
the five predeclared nuclear-observable boxes are consulted; no posterior,
sampler, held-out-source, or comparison artifact is read.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path

import numpy as np


OBSERVATION = np.asarray(
    [
        0.153,
        -16.1,
        230.0,
        32.5,
        0.505714285714279,
        1.24142857142857,
        2.4857142857143,
    ],
    dtype=np.float64,
)
SIGMA = np.asarray(
    [
        0.005,
        0.2,
        40.0,
        1.8,
        0.194285714285714,
        0.608571428571429,
        1.38285714285714,
    ],
    dtype=np.float64,
)
BOX_RADIUS = 3.0
BOX_NAMES = ("A1", "K0_200", "K0_260", "Jsym_29", "Jsym_36")


def _centres() -> np.ndarray:
    centres = np.repeat(OBSERVATION[None, :], len(BOX_NAMES), axis=0)
    centres[1, 2] = 200.0
    centres[2, 2] = 260.0
    centres[3, 3] = 29.0
    centres[4, 3] = 36.0
    return centres


CENTRES = _centres()


def _iso_utc(timestamp: float | None = None) -> str:
    when = dt.datetime.fromtimestamp(
        time.time() if timestamp is None else timestamp, tz=dt.timezone.utc
    )
    return when.isoformat(timespec="microseconds")


def _iso_kst(timestamp: float | None = None) -> str:
    when = dt.datetime.fromtimestamp(
        time.time() if timestamp is None else timestamp,
        tz=dt.timezone(dt.timedelta(hours=9)),
    )
    return when.isoformat(timespec="microseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_worker_threads() -> None:
    # Each worker owns independent proposal blocks.  Limiting library thread
    # pools prevents N workers from each claiming every host core.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "True")


def _screen_worker(payload: tuple[str, int, int, int, int]):
    eos_key, seed, block_index, proposal_start, count = payload
    _configure_worker_threads()
    # Import after the per-worker environment is fixed.
    from eos import get_eos

    plugin = get_eos(eos_key)
    prior_low = np.asarray(plugin.prior_low, dtype=np.float64)
    prior_high = np.asarray(plugin.prior_high, dtype=np.float64)
    width = prior_high - prior_low
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1, block_index]))
    theta = prior_low + rng.random((count, plugin.ndim)) * width
    prediction = np.empty((count, 7), dtype=np.float64)
    prediction[:, 0] = theta[:, plugin.parameter_names.index("rho0")]
    prediction[:, 1:] = plugin.nuclear_observables_batch(theta)

    in_boxes = np.all(
        np.abs(
            (prediction[:, None, :] - CENTRES[None, :, :])
            / SIGMA[None, None, :]
        )
        <= BOX_RADIUS,
        axis=2,
    )
    keep = np.any(in_boxes, axis=1) & ~in_boxes[:, 0]
    local_index = np.flatnonzero(keep).astype(np.int64, copy=False)
    return {
        "block_index": block_index,
        "proposal_start": proposal_start,
        "count": count,
        "accepted": int(len(local_index)),
        "theta": theta[keep],
        "prediction": prediction[keep],
        "proposal_index": proposal_start + local_index,
        "retained_box_membership": in_boxes[keep],
    }


def _payloads(proposals: int, block_size: int, eos_key: str, seed: int):
    result = []
    proposed = 0
    block_index = 0
    while proposed < proposals:
        count = min(block_size, proposals - proposed)
        result.append((eos_key, seed, block_index, proposed, count))
        proposed += count
        block_index += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eos", choices=("ddb", "ddb-hyperonic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposals", type=int, default=160_000_000)
    parser.add_argument("--block-size", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite screen: {arguments.output}")
    if min(arguments.proposals, arguments.block_size, arguments.workers) < 1:
        raise ValueError("proposals, block size, and workers must be positive")
    expected_seed = {"ddb-hyperonic": 20_260_923, "ddb": 20_260_924}[
        arguments.eos
    ]
    if arguments.proposals == 160_000_000 and arguments.seed != expected_seed:
        raise ValueError(
            f"production proposal count requires predeclared seed {expected_seed}"
        )

    started_unix = time.time()
    started_monotonic = time.monotonic()
    tasks = _payloads(
        arguments.proposals,
        arguments.block_size,
        arguments.eos,
        arguments.seed,
    )
    results: dict[int, dict] = {}
    if arguments.workers == 1:
        for task in tasks:
            item = _screen_worker(task)
            results[item["block_index"]] = item
            print(
                f"[support-extension-screen] block={item['block_index'] + 1}/"
                f"{len(tasks)} accepted={item['accepted']}",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=arguments.workers, mp_context=context
        ) as executor:
            future_to_index = {
                executor.submit(_screen_worker, task): task[2] for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_index):
                item = future.result()
                results[item["block_index"]] = item
                print(
                    f"[support-extension-screen] block={item['block_index'] + 1}/"
                    f"{len(tasks)} accepted={item['accepted']}",
                    flush=True,
                )

    if set(results) != set(range(len(tasks))):
        raise RuntimeError("one or more deterministic proposal blocks are missing")
    ordered = [results[index] for index in range(len(tasks))]
    theta = np.concatenate([item["theta"] for item in ordered], axis=0)
    prediction = np.concatenate([item["prediction"] for item in ordered], axis=0)
    proposal_index = np.concatenate(
        [item["proposal_index"] for item in ordered], axis=0
    )
    membership = np.concatenate(
        [item["retained_box_membership"] for item in ordered], axis=0
    )
    block_accepted = np.asarray(
        [item["accepted"] for item in ordered], dtype=np.int64
    )
    if len(theta) == 0:
        raise RuntimeError("support-extension screen accepted no rows")
    if not (
        np.isfinite(theta).all()
        and np.isfinite(prediction).all()
        and np.all(np.diff(proposal_index) > 0)
        and not np.any(membership[:, 0])
        and np.all(np.any(membership[:, 1:], axis=1))
    ):
        raise RuntimeError("support-extension screen failed its integrity gates")

    # Import once in the parent only after workers have exited, for bounds and
    # stable metadata without mixing parent and child JAX runtimes.
    _configure_worker_threads()
    from eos import get_eos

    plugin = get_eos(arguments.eos)
    prior_low = np.asarray(plugin.prior_low, dtype=np.float64)
    prior_high = np.asarray(plugin.prior_high, dtype=np.float64)
    if not (np.all(theta >= prior_low) and np.all(theta <= prior_high)):
        raise RuntimeError("retained parameter row lies outside the full prior")

    finished_unix = time.time()
    wall_seconds = time.monotonic() - started_monotonic
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray("green-en-nuclear-support-extension-screen-v1"),
            eos=np.asarray(arguments.eos),
            support_name=np.asarray("S_ext"),
            theta=theta,
            prediction=prediction,
            proposal_index=proposal_index,
            retained_box_membership=membership,
            parameter_names=np.asarray(plugin.parameter_names),
            prior_low=prior_low,
            prior_high=prior_high,
            observation=OBSERVATION,
            sigma=SIGMA,
            centres=CENTRES,
            centre_names=np.asarray(BOX_NAMES),
            box_radius=np.float64(BOX_RADIUS),
            seed=np.int64(arguments.seed),
            rng_substream=np.int64(1),
            total_proposals=np.int64(arguments.proposals),
            block_size=np.int64(arguments.block_size),
            block_accepted=block_accepted,
            workers=np.int64(arguments.workers),
            hostname=np.asarray(socket.gethostname()),
            reference_present=np.bool_(False),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
            started_unix=np.float64(started_unix),
            finished_unix=np.float64(finished_unix),
            wall_seconds=np.float64(wall_seconds),
        )
    os.replace(temporary, arguments.output)
    report = {
        "status": "PASS",
        "stage": "predeclared deterministic Green-EN S_ext screen",
        "schema": "green-en-nuclear-support-extension-screen-v1",
        "eos": arguments.eos,
        "support_name": "S_ext",
        "proposals": arguments.proposals,
        "accepted": int(len(theta)),
        "acceptance_fraction": float(len(theta) / arguments.proposals),
        "seed": arguments.seed,
        "rng_substream": 1,
        "block_size": arguments.block_size,
        "blocks": len(tasks),
        "block_accepted": block_accepted.tolist(),
        "workers": arguments.workers,
        "hostname": socket.gethostname(),
        "started_utc": _iso_utc(started_unix),
        "started_kst": _iso_kst(started_unix),
        "finished_utc": _iso_utc(finished_unix),
        "finished_kst": _iso_kst(finished_unix),
        "wall_seconds": wall_seconds,
        "output": str(arguments.output.resolve()),
        "output_sha256": _sha256(arguments.output),
        "reference_present": False,
        "forbidden_artifacts_used": [],
        "gates": {
            "full_prior": True,
            "all_rows_in_union": True,
            "all_rows_outside_A1_box": True,
            "proposal_indices_strictly_increasing": True,
        },
    }
    report_path = arguments.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
