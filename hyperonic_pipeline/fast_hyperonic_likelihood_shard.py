#!/usr/bin/env python3
"""Evaluate one production hyperonic likelihood scenario with prewarmed workers.

The input NPZ contains ``theta``, global ``index``, and optionally the locked
``reference_logl``.  The production solver is compiled on the first row, then
forked workers evaluate the remaining independent rows.  The exact NICER,
GW170817, pQCD, and nuclear likelihood assembly is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
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

import ddb_ultranest_hyp as likelihood


def _fast_forward(theta: np.ndarray) -> likelihood.HyperonicRowForward | None:
    """One core/TOV forward through the production EOS implementation."""

    os.environ["TOV_NW"] = "1"
    try:
        core = likelihood.solve_hyperonic_core(theta)
        curve = likelihood._stable_mrl_curve(core.energy, core.pressure)
    except Exception:
        return None
    return likelihood.HyperonicRowForward(
        density=core.density,
        energy=core.energy,
        pressure=core.pressure,
        mass=curve.mass,
        radius=curve.radius,
        tidal_lambda=curve.tidal_lambda,
        maximum_mass=curve.maximum_mass,
    )


def _nuclear_loglike(
    theta: np.ndarray,
    data: likelihood.A1LikelihoodData,
) -> np.ndarray:
    """Run the same bucketed certified saturation-property likelihood."""

    output = np.empty(len(theta), dtype=np.float64)
    for start in range(0, len(theta), likelihood.CHUNK_SIZE):
        stop = min(start + likelihood.CHUNK_SIZE, len(theta))
        selected = theta[start:stop]
        size = len(selected)
        bucket = next(value for value in likelihood.JAX_BUCKETS if value >= size)
        padded = (
            selected
            if size == bucket
            else np.vstack(
                [selected, np.repeat(selected[-1:], bucket - size, axis=0)]
            )
        )
        nuclear_forward = np.asarray(
            likelihood.rn.nmp_batch(
                jnp.asarray(padded[:, :6]),
                jnp.asarray(padded[:, 6]),
            ),
            dtype=np.float64,
        )[:size]
        properties = np.column_stack([selected[:, 6], nuclear_forward])
        output[start:stop] = -0.5 * np.sum(
            (
                (properties - data.nuclear_observations[None, :])
                / data.nuclear_sigmas[None, :]
            )
            ** 2,
            axis=1,
        )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--nuclear-scenario",
        choices=("A1", "K0_200", "K0_260", "Jsym_29", "Jsym_36"),
        default="A1",
    )
    parser.add_argument(
        "--source-scenario",
        choices=("A1", "J0614", "J1231", "J1614"),
        default="A1",
    )
    parser.add_argument(
        "--data-cache",
        type=Path,
        help="Prebuilt fixed-likelihood cache from build_likelihood_data_cache.py.",
    )
    parser.add_argument(
        "--reference-tolerance",
        type=float,
        default=likelihood.CERT_TOLERANCE,
    )
    arguments = parser.parse_args()
    if arguments.workers < 1:
        raise ValueError("--workers must be positive")
    if arguments.nuclear_scenario != "A1" and arguments.source_scenario != "A1":
        raise ValueError(
            "change either the NICER source or the nuclear observation, not both"
        )
    imported_source_scenario = likelihood.SWAP or "A1"
    if imported_source_scenario != arguments.source_scenario:
        raise RuntimeError(
            "DDB_SWAP and --source-scenario differ: "
            f"{imported_source_scenario!r} != {arguments.source_scenario!r}"
        )

    campaign_started = time.time()
    with np.load(arguments.input, allow_pickle=False) as stored:
        theta = np.asarray(stored["theta"], dtype=np.float64)
        index = (
            np.asarray(stored["index"], dtype=np.int64)
            if "index" in stored.files
            else np.arange(len(theta), dtype=np.int64)
        )
        reference = (
            np.asarray(stored["reference_logl"], dtype=np.float64)
            if "reference_logl" in stored.files
            else (
                np.asarray(stored["logl"], dtype=np.float64)
                if "logl" in stored.files
                else None
            )
        )
    if theta.ndim != 2 or theta.shape[1] != 9 or index.shape != (len(theta),):
        raise ValueError("input must contain theta (N,9) and index (N,)")
    if reference is not None and reference.shape != (len(theta),):
        raise ValueError("reference_logl must have shape (N,)")

    # This first physical row compiles every numba kernel.  Forked children
    # inherit those native-code pages, avoiding one compilation per worker.
    prewarm_started = time.perf_counter()
    first = _fast_forward(theta[0])
    prewarm_seconds = time.perf_counter() - prewarm_started

    cache_hit = bool(arguments.data_cache and arguments.data_cache.is_file())

    def load_data() -> likelihood.A1LikelihoodData:
        if cache_hit:
            cached = joblib.load(arguments.data_cache)
            expected_configuration = arguments.source_scenario
            if cached.get("schema") != "ddb-hyperonic-likelihood-cache-v1":
                raise ValueError("unrecognized likelihood-data cache schema")
            if cached.get("configuration") != expected_configuration:
                raise ValueError(
                    "likelihood-data cache configuration does not match this shard"
                )
            cached_nuclear_scenario = cached.get("nuclear_scenario", "A1")
            if cached_nuclear_scenario != arguments.nuclear_scenario:
                raise ValueError(
                    "likelihood-data cache nuclear scenario does not match this shard"
                )
            return cached["data"]
        return likelihood.build_a1_likelihood_data(verbose=False)

    # The parent builds or loads the fixed observational objects while the already
    # forked children solve EOS rows.  Forking happens first, so JAX/KDE setup
    # is never inherited from a multithreaded parent process.
    forward_started = time.perf_counter()
    if len(theta) == 1:
        data_started = time.perf_counter()
        data = load_data()
        data_seconds = time.perf_counter() - data_started
        forwards = [first]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=arguments.workers) as pool:
            pending = pool.map_async(_fast_forward, theta[1:], chunksize=1)
            data_started = time.perf_counter()
            data = load_data()
            data_seconds = time.perf_counter() - data_started
            rest = pending.get()
        forwards = [first, *rest]
    forward_seconds = time.perf_counter() - forward_started

    assembly_started = time.perf_counter()
    nuclear = _nuclear_loglike(theta, data)
    astro = np.asarray(
        [likelihood._astro_loglike_from_forward(row, data) for row in forwards],
        dtype=np.float64,
    )
    logl = astro + nuclear
    finite = np.isfinite(logl)
    logl[~finite] = likelihood.LOG_LIKELIHOOD_FLOOR
    assembly_seconds = time.perf_counter() - assembly_started
    total_seconds = time.time() - campaign_started

    pattern_match = True
    common_count = 0
    maximum_difference = np.nan
    median_difference = np.nan
    passed = True
    if reference is not None:
        actual_valid = logl > likelihood.CERT_VALID_FLOOR
        reference_valid = reference > likelihood.CERT_VALID_FLOOR
        pattern_match = bool(np.array_equal(actual_valid, reference_valid))
        common = actual_valid & reference_valid
        common_count = int(common.sum())
        difference = logl[common] - reference[common]
        maximum_difference = (
            float(np.max(np.abs(difference))) if common.any() else np.inf
        )
        median_difference = float(np.median(difference)) if common.any() else np.nan
        passed = bool(
            pattern_match and maximum_difference <= arguments.reference_tolerance
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.output,
        theta=theta,
        index=index,
        logl=logl,
        workers=np.int64(arguments.workers),
        hostname=np.asarray(socket.gethostname()),
        input_sha256=np.asarray(_sha256(arguments.input)),
        solver_sha256=np.asarray(
            _sha256(Path(__file__).resolve().parent / "ddb_hyperon_eos.py")
        ),
        campaign_started_unix=np.float64(campaign_started),
        campaign_finished_unix=np.float64(time.time()),
        prewarm_seconds=np.float64(prewarm_seconds),
        forward_seconds=np.float64(forward_seconds),
        data_seconds=np.float64(data_seconds),
        data_cache_hit=np.bool_(cache_hit),
        assembly_seconds=np.float64(assembly_seconds),
        total_seconds=np.float64(total_seconds),
        reference_present=np.bool_(reference is not None),
        valid_pattern_match=np.bool_(pattern_match),
        reference_common=np.int64(common_count),
        reference_max_abs_dlogl=np.float64(maximum_difference),
        reference_median_dlogl=np.float64(median_difference),
        reference_pass=np.bool_(passed),
        nuclear_scenario=np.asarray(arguments.nuclear_scenario),
        source_scenario=np.asarray(arguments.source_scenario),
    )
    print(
        f"host={socket.gethostname()} rows={len(theta)} workers={arguments.workers} "
        f"prewarm_s={prewarm_seconds:.3f} forward_s={forward_seconds:.3f} "
        f"data_s={data_seconds:.3f} assembly_s={assembly_seconds:.3f} "
        f"total_s={total_seconds:.3f} rows_per_s={len(theta) / total_seconds:.3f}",
        flush=True,
    )
    if reference is not None:
        print(
            f"reference pattern={pattern_match} common={common_count} "
            f"max_abs_dlogl={maximum_difference:.9g} "
            f"median_dlogl={median_difference:.9g} "
            f"verdict={'PASS' if passed else 'FAIL'}",
            flush=True,
        )
    print(f"saved {arguments.output.resolve()}", flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
