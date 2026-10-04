#!/usr/bin/env python3
"""Fast EOS/TOV replay with the corrected sampler-neutral portable target."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("TOV_NW", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import jax.numpy as jnp
import joblib
import numpy as np

import ddb_hyperon_eos as hyperonic_solver
import ddb_ultranest_hyp as legacy
from fast_hyperonic_likelihood_shard import _fast_forward


BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 2500)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nuclear_observables(theta: np.ndarray) -> np.ndarray:
    output = np.empty((len(theta), 6), dtype=np.float64)
    for start in range(0, len(theta), 2500):
        stop = min(start + 2500, len(theta))
        local = theta[start:stop]
        count = len(local)
        bucket = next(size for size in BUCKETS if size >= count)
        padded = (
            local
            if count == bucket
            else np.vstack([local, np.repeat(local[-1:], bucket - count, axis=0)])
        )
        output[start:stop] = np.asarray(
            legacy.rn.nmp_batch(
                jnp.asarray(padded[:, :6]), jnp.asarray(padded[:, 6])
            ),
            dtype=np.float64,
        )[:count]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--nuclear-scenario",
        choices=("A1", "K0_200", "K0_260", "Jsym_29", "Jsym_36"),
        required=True,
    )
    parser.add_argument(
        "--source-scenario",
        choices=("A1", "J0614", "J1231", "J1614"),
        required=True,
    )
    arguments = parser.parse_args()
    if arguments.workers < 1:
        raise ValueError("--workers must be positive")
    if arguments.nuclear_scenario != "A1" and arguments.source_scenario != "A1":
        raise ValueError(
            "change either the NICER source or the nuclear observation, not both"
        )

    with np.load(arguments.input, allow_pickle=False) as stored:
        theta = np.asarray(stored["theta"], dtype=np.float64)
        index = (
            np.asarray(stored["index"], dtype=np.int64)
            if "index" in stored.files
            else np.arange(len(theta), dtype=np.int64)
        )
        proposal_nuclear = str(
            stored["nuclear_scenario"].item()
            if "nuclear_scenario" in stored.files
            else "A1"
        )
        proposal_source = str(
            stored["source_scenario"].item()
            if "source_scenario" in stored.files
            else (
                stored["scenario"].item()
                if "scenario" in stored.files
                else arguments.source_scenario
            )
        )
    if theta.ndim != 2 or theta.shape[1] != 9 or index.shape != (len(theta),):
        raise RuntimeError("input must contain theta (N,9) and index (N,)")
    if proposal_nuclear != arguments.nuclear_scenario:
        raise RuntimeError("proposal and target nuclear scenarios differ")
    if proposal_source != arguments.source_scenario:
        raise RuntimeError("proposal and target source scenarios differ")

    campaign_started = time.time()
    prewarm_started = time.perf_counter()
    first = _fast_forward(theta[0])
    prewarm_seconds = time.perf_counter() - prewarm_started

    forward_started = time.perf_counter()
    if len(theta) == 1:
        cache_started = time.perf_counter()
        cached = joblib.load(arguments.target_cache)
        cache_seconds = time.perf_counter() - cache_started
        forwards = [first]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=arguments.workers) as pool:
            pending = pool.map_async(_fast_forward, theta[1:], chunksize=1)
            cache_started = time.perf_counter()
            cached = joblib.load(arguments.target_cache)
            cache_seconds = time.perf_counter() - cache_started
            rest = pending.get()
        forwards = [first, *rest]
    forward_seconds = time.perf_counter() - forward_started

    if cached.get("schema") != "ddb-hyperonic-portable-target-cache-v1":
        raise RuntimeError("unrecognized portable target-cache schema")
    if cached.get("model") != "ddb-hyperonic":
        raise RuntimeError("portable target cache has the wrong EOS model")
    if cached.get("nuclear_scenario") != arguments.nuclear_scenario:
        raise RuntimeError("portable target cache has the wrong nuclear scenario")
    if cached.get("source_scenario") != arguments.source_scenario:
        raise RuntimeError("portable target cache has the wrong source scenario")
    target = cached["target"]

    assembly_started = time.perf_counter()
    nmp = nuclear_observables(theta)
    loglike = np.full(len(theta), -1.0e100, dtype=np.float64)
    for row, forward in enumerate(forwards):
        if forward is None:
            continue
        terms = target.evaluate(
            theta[row],
            nmp[row],
            forward.density,
            forward.energy,
            forward.pressure,
            forward,
        )
        if np.isfinite(terms.total):
            loglike[row] = terms.total
    assembly_seconds = time.perf_counter() - assembly_started
    total_seconds = time.time() - campaign_started

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            schema=np.asarray("ddb-hyperonic-fast-portable-likelihood-v1"),
            theta=theta,
            index=index,
            logl=loglike,
            nuclear_scenario=np.asarray(arguments.nuclear_scenario),
            source_scenario=np.asarray(arguments.source_scenario),
            workers=np.int64(arguments.workers),
            hostname=np.asarray(socket.gethostname()),
            input_sha256=np.asarray(sha256(arguments.input)),
            target_cache_sha256=np.asarray(sha256(arguments.target_cache)),
            solver_sha256=np.asarray(
                sha256(Path(hyperonic_solver.__file__).resolve())
            ),
            evaluator_sha256=np.asarray(sha256(Path(__file__).resolve())),
            campaign_started_unix=np.float64(campaign_started),
            campaign_finished_unix=np.float64(time.time()),
            prewarm_seconds=np.float64(prewarm_seconds),
            forward_seconds=np.float64(forward_seconds),
            cache_seconds=np.float64(cache_seconds),
            assembly_seconds=np.float64(assembly_seconds),
            total_seconds=np.float64(total_seconds),
            reference_present=np.bool_(False),
        )
    os.replace(temporary, arguments.output)
    report = {
        "status": "PASS",
        "rows": len(theta),
        "valid_rows": int(np.sum(loglike > -1.0e50)),
        "nuclear_scenario": arguments.nuclear_scenario,
        "source_scenario": arguments.source_scenario,
        "workers": arguments.workers,
        "hostname": socket.gethostname(),
        "prewarm_seconds": prewarm_seconds,
        "forward_seconds": forward_seconds,
        "cache_seconds": cache_seconds,
        "assembly_seconds": assembly_seconds,
        "total_seconds": total_seconds,
        "rows_per_second": len(theta) / total_seconds,
        "input_sha256": sha256(arguments.input),
        "target_cache_sha256": sha256(arguments.target_cache),
        "output_sha256": sha256(arguments.output),
        "reference_present": False,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
