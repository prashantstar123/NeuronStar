#!/usr/bin/env python3
"""Replay selected hyperonic stellar curves, including the terminal radius.

This is a derived-observable replay only: the input rows are fixed posterior
samples.  It uses the validated DDB-Lambda-Xi-minus EOS and certified
TOV/Love implementation from :mod:`generate_ddbhy_bank`.  The analytic-Newton
accelerator is installed with its validated fallback, exactly as in the
previous fast-band audit.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path

os.environ.setdefault("TOV_NW", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np

import generate_ddbhy_bank as bank
from fast_newton_hyperon_audit import install


PRIOR_LOW, PRIOR_HIGH = bank.extended_prior_bounds(bank.load_certified_forward())


def _solve(item: tuple[int, np.ndarray]) -> tuple:
    index, theta = item
    os.environ["TOV_NW"] = "1"
    install()
    parameters = np.asarray(theta, dtype=np.float64)
    if (
        parameters.shape != (9,)
        or not np.all(np.isfinite(parameters))
        or np.any(parameters < PRIOR_LOW)
        or np.any(parameters > PRIOR_HIGH)
    ):
        raise ValueError(f"row {index}: invalid posterior parameter vector")

    core = bank.solve_hyperonic_core(parameters)
    curve = bank._stable_mrl_curve(core.energy, core.pressure, h=bank.TOV_STEP)
    radius_grid, lambda_grid, radius_14 = bank.interpolate_curve_to_bank_grid(
        curve, bank.MASS_GRID
    )
    if (
        not np.isfinite(radius_grid).any()
        or not np.isfinite(lambda_grid).any()
        or not np.isfinite(radius_14)
        or not np.isfinite(curve.maximum_mass)
        or not np.isfinite(curve.radius[-1])
    ):
        raise ValueError(f"row {index}: invalid derived stellar curve")
    return (
        index,
        radius_grid,
        lambda_grid,
        float(curve.maximum_mass),
        float(radius_14),
        float(curve.radius[-1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    arguments = parser.parse_args()
    with np.load(arguments.input, allow_pickle=False) as stored:
        theta = np.asarray(stored["theta"], dtype=np.float64)
        method = str(stored["method"].item())
    if theta.ndim != 2 or theta.shape[1] != 9 or len(theta) == 0:
        raise ValueError("input theta must have shape (N, 9) with N > 0")
    if arguments.workers < 1:
        raise ValueError("workers must be positive")

    started = time.perf_counter()
    first = _solve((0, theta[0]))
    prewarm_seconds = time.perf_counter() - started
    parallel_started = time.perf_counter()
    if len(theta) == 1:
        rows = [first]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=arguments.workers) as pool:
            rest = pool.map(_solve, enumerate(theta[1:], start=1), chunksize=1)
        rows = [first, *rest]
    parallel_seconds = time.perf_counter() - parallel_started
    total_seconds = time.perf_counter() - started

    row_index = np.asarray([row[0] for row in rows], dtype=np.int64)
    if not np.array_equal(row_index, np.arange(len(theta))):
        raise RuntimeError("parallel replay changed row ordering")
    radius = np.asarray([row[1] for row in rows], dtype=np.float64)
    tidal_lambda = np.asarray([row[2] for row in rows], dtype=np.float64)
    maximum_mass = np.asarray([row[3] for row in rows], dtype=np.float64)
    radius_14 = np.asarray([row[4] for row in rows], dtype=np.float64)
    radius_at_maximum_mass = np.asarray([row[5] for row in rows], dtype=np.float64)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        theta=theta,
        method=np.asarray(method),
        MG=np.asarray(bank.MASS_GRID, dtype=np.float64),
        Rg=radius,
        Lg=tidal_lambda,
        MM=maximum_mass,
        R14=radius_14,
        RMAX=radius_at_maximum_mass,
        hostname=np.asarray(socket.gethostname()),
        workers=np.int64(arguments.workers),
        prewarm_seconds=np.float64(prewarm_seconds),
        parallel_seconds=np.float64(parallel_seconds),
        total_seconds=np.float64(total_seconds),
    )
    print(
        f"method={method} host={socket.gethostname()} rows={len(theta)} "
        f"workers={arguments.workers} prewarm_s={prewarm_seconds:.3f} "
        f"parallel_s={parallel_seconds:.3f} total_s={total_seconds:.3f} "
        f"rows_per_s={len(theta) / total_seconds:.3f}",
        flush=True,
    )
    print(f"saved {arguments.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
