#!/usr/bin/env python3
"""Evaluate exact nucleonic rows for the predeclared Green-EN S_ext stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]
FAST_ROOT = ROOT.parent / "fastsolver"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FAST_ROOT))

import fast_exact_rows as fast  # noqa: E402
import fast_likelihood as fast_likelihood  # noqa: E402
from eos import get_eos  # noqa: E402
from likelihoods.nuclear import A1_SIGMA  # noqa: E402
from workflows.a1_problem import A1Problem, BUCKETS  # noqa: E402


EXPECTED_SCREEN_SCHEMA = "green-en-nuclear-support-extension-screen-v1"
OUTPUT_SCHEMA = "ddb-nucleonic-green-en-support-extension-cache-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_rows(theta, prediction, target, mass_grid, plugin, workers, chunk_size):
    rows = len(theta)
    log_astro = np.full(rows, np.nan, dtype=np.float64)
    components = np.full((rows, 4), np.nan, dtype=np.float64)
    source_components = np.full((rows, 3), np.nan, dtype=np.float64)
    radius = np.full((rows, len(mass_grid)), np.nan, dtype=np.float64)
    tidal = np.full_like(radius, np.nan)
    maximum_mass = np.full(rows, np.nan, dtype=np.float64)
    gw_kernel = fast_likelihood.BatchedGWKernel(target.gw_kernel)
    eos_seconds = tov_seconds = likelihood_seconds = curve_seconds = 0.0

    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        local_theta = theta[start:stop]
        count = len(local_theta)
        bucket = next(size for size in BUCKETS if size >= count)
        padded = (
            local_theta
            if count == bucket
            else np.vstack(
                [local_theta, np.repeat(local_theta[-1:], bucket - count, axis=0)]
            )
        )
        stamp = time.perf_counter()
        density, energy, pressure = plugin.core_eos_batch(padded)
        density = density[:count]
        energy = energy[:count]
        pressure = pressure[:count]
        eos_seconds += time.perf_counter() - stamp
        row_ok = (
            np.isfinite(energy).all(axis=1)
            & np.isfinite(pressure).all(axis=1)
            & (pressure[:, -1] > 0.0)
        )

        stamp = time.perf_counter()
        indices = np.arange(count)
        chunks = np.array_split(indices, max(1, min(workers * 8, count)))
        parts = Parallel(
            n_jobs=min(workers, count), batch_size=1, pre_dispatch="2*n_jobs"
        )(
            delayed(fast._tov_chunk)(
                energy[index], pressure[index], row_ok[index], False
            )
            for index in chunks
            if len(index)
        )
        branches = [branch for part in parts for branch in part]
        tov_seconds += time.perf_counter() - stamp

        stamp = time.perf_counter()
        result = fast_likelihood.evaluate_batch(
            target,
            local_theta,
            prediction[start:stop, 1:],
            density[0],
            energy,
            pressure,
            branches,
            gw_kernel,
        )
        likelihood_seconds += time.perf_counter() - stamp

        stamp = time.perf_counter()
        for local_index, branch in enumerate(branches):
            output_index = start + local_index
            if branch is None or not np.isfinite(result["total"][local_index]):
                continue
            component = np.asarray(
                [
                    result["maximum_mass"][local_index],
                    result["nicer"][local_index],
                    result["gw170817"][local_index],
                    result["pqcd"][local_index],
                ],
                dtype=np.float64,
            )
            source = np.asarray(
                result["nicer_sources"][local_index], dtype=np.float64
            )
            value = float(component.sum())
            if not np.isfinite(value) or not np.all(np.isfinite(source)):
                continue
            if abs(float(source.sum()) - float(component[1])) > 1.0e-10:
                continue
            covered = (
                (mass_grid >= float(branch.mass.min()))
                & (mass_grid <= float(branch.mass.max()))
                & (mass_grid <= float(branch.maximum_mass))
            )
            local_radius = np.full(len(mass_grid), np.nan, dtype=np.float64)
            local_tidal = np.full(len(mass_grid), np.nan, dtype=np.float64)
            local_radius[covered] = np.interp(
                mass_grid[covered], branch.mass, branch.radius
            )
            local_tidal[covered] = np.exp(
                np.interp(
                    mass_grid[covered],
                    branch.mass,
                    np.log(np.clip(branch.tidal_lambda, 1.0e-300, None)),
                )
            )
            log_astro[output_index] = value
            components[output_index] = component
            source_components[output_index] = source
            radius[output_index] = local_radius
            tidal[output_index] = local_tidal
            maximum_mass[output_index] = float(branch.maximum_mass)
        curve_seconds += time.perf_counter() - stamp
        print(f"[nucleonic-exact] {stop}/{rows}", flush=True)

    diagnostics = {
        "chunks": int((rows + chunk_size - 1) // chunk_size),
        "chunk_size": int(chunk_size),
        "eos_s": eos_seconds,
        "tov_s": tov_seconds,
        "likelihood_s": likelihood_seconds,
        "curves_s": curve_seconds,
    }
    return (
        log_astro,
        components,
        source_components,
        radius,
        tidal,
        maximum_mass,
        diagnostics,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=2048)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite exact shard: {arguments.output}")
    if min(arguments.workers, arguments.chunk_size) < 1:
        raise ValueError("workers and chunk size must be positive")
    if arguments.chunk_size > max(BUCKETS):
        raise ValueError("chunk size exceeds the largest certified JAX bucket")

    started_unix = time.time()
    started_monotonic = time.monotonic()
    screen_sha256 = sha256(arguments.screen)
    with np.load(arguments.screen, allow_pickle=False) as screen:
        required = {
            "schema",
            "eos",
            "support_name",
            "theta",
            "prediction",
            "proposal_index",
            "total_proposals",
            "seed",
            "reference_present",
            "forbidden_artifacts_used",
        }
        missing = required - set(screen.files)
        if missing:
            raise RuntimeError(f"support screen is missing {sorted(missing)}")
        if str(screen["schema"].item()) != EXPECTED_SCREEN_SCHEMA:
            raise RuntimeError("unexpected support-extension screen schema")
        if str(screen["eos"].item()) != "ddb":
            raise RuntimeError("screen is not nucleonic DDB")
        if str(screen["support_name"].item()) != "S_ext":
            raise RuntimeError("screen is not the predeclared S_ext stream")
        if bool(screen["reference_present"]) or screen[
            "forbidden_artifacts_used"
        ].size:
            raise RuntimeError("support screen declares forbidden ancestry")
        all_theta = np.asarray(screen["theta"], dtype=np.float64)
        all_prediction = np.asarray(screen["prediction"], dtype=np.float64)
        all_proposal_index = np.asarray(screen["proposal_index"], dtype=np.int64)
        total_proposals = int(screen["total_proposals"])
        seed = int(screen["seed"])
    if total_proposals != 160_000_000 or seed != 20_260_924:
        raise RuntimeError("screen does not carry the predeclared nucleonic count/seed")
    if not 0 <= arguments.start < arguments.stop <= len(all_theta):
        raise ValueError(
            f"invalid shard [{arguments.start},{arguments.stop}) for {len(all_theta)} rows"
        )
    theta = all_theta[arguments.start : arguments.stop]
    stored_prediction = all_prediction[arguments.start : arguments.stop]
    proposal_index = all_proposal_index[arguments.start : arguments.stop]

    with np.load(arguments.template, allow_pickle=False) as template:
        if str(template["schema"].item()) != "ddb-nucleonic-exact-template-v1":
            raise RuntimeError("unexpected nucleonic template schema")
        mass_grid = np.asarray(template["mass_grid"], dtype=np.float64)
        prior_low = np.asarray(template["prior_low"], dtype=np.float64)
        prior_high = np.asarray(template["prior_high"], dtype=np.float64)
        template_parent_sha256 = str(template["parent_bank_sha256"].item())

    plugin = get_eos("ddb")
    recomputed_prediction = np.empty_like(stored_prediction)
    recomputed_prediction[:, 0] = theta[:, plugin.parameter_names.index("rho0")]
    recomputed_prediction[:, 1:] = plugin.nuclear_observables_batch(theta)
    prediction_delta = np.abs(recomputed_prediction - stored_prediction)
    maximum_prediction_delta = float(np.max(prediction_delta))
    maximum_normalized_prediction_delta = float(
        np.max(prediction_delta / A1_SIGMA)
    )
    prediction_tolerance_sigma = 1.0e-4
    if maximum_normalized_prediction_delta > prediction_tolerance_sigma:
        raise RuntimeError(
            "screen and evaluator predictions differ by "
            f"{maximum_normalized_prediction_delta:.3e} nuclear sigma"
        )
    problem = A1Problem.from_data_root(
        arguments.data_root,
        workers=arguments.workers,
        verify_data=True,
        nuclear_scenario="A1",
        source_scenario="A1",
        model="ddb",
    )
    (
        log_astrophysical,
        components,
        source_components,
        radius,
        tidal,
        maximum_mass,
        diagnostics,
    ) = evaluate_rows(
        theta,
        recomputed_prediction,
        problem.target,
        mass_grid,
        plugin,
        arguments.workers,
        arguments.chunk_size,
    )
    target_valid = (
        np.isfinite(log_astrophysical)
        & (log_astrophysical > -1.0e29)
        & np.isfinite(components).all(axis=1)
        & np.isfinite(source_components).all(axis=1)
        & np.isfinite(maximum_mass)
        & np.isfinite(radius).any(axis=1)
        & np.isfinite(tidal).any(axis=1)
    )
    if not target_valid.any():
        raise RuntimeError("exact likelihood rejected every support-extension row")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    finished_unix = time.time()
    wall_seconds = time.monotonic() - started_monotonic
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(OUTPUT_SCHEMA),
            support_name=np.asarray("S_ext"),
            source_index=np.arange(arguments.start, arguments.stop, dtype=np.int64),
            proposal_index=proposal_index,
            theta=theta,
            # The screen prediction is the canonical nuclear coordinate used
            # to define S_ext.  Retain it after the independent evaluator has
            # passed the 1e-4-sigma agreement gate above.  This prevents JAX
            # batch-size rounding from changing stored nuclear coordinates
            # when the identical row stream is sharded for speed.
            prediction=stored_prediction,
            log_astrophysical=log_astrophysical,
            astrophysical_components=components,
            nicer_source_components=source_components,
            target_valid=target_valid,
            mass_grid=mass_grid,
            radius=radius,
            tidal_lambda=tidal,
            maximum_mass=maximum_mass,
            prior_low=prior_low,
            prior_high=prior_high,
            screen_sha256=np.asarray(screen_sha256),
            template_sha256=np.asarray(sha256(arguments.template)),
            template_parent_sha256=np.asarray(template_parent_sha256),
            total_proposals=np.int64(total_proposals),
            seed=np.int64(seed),
            shard_start=np.int64(arguments.start),
            shard_stop=np.int64(arguments.stop),
            prediction_tolerance_sigma=np.float64(prediction_tolerance_sigma),
            maximum_prediction_delta=np.float64(maximum_prediction_delta),
            maximum_normalized_prediction_delta=np.float64(
                maximum_normalized_prediction_delta
            ),
            workers=np.int64(arguments.workers),
            chunk_size=np.int64(arguments.chunk_size),
            hostname=np.asarray(socket.gethostname()),
            reference_present=np.bool_(False),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
            started_unix=np.float64(started_unix),
            finished_unix=np.float64(finished_unix),
            wall_seconds=np.float64(wall_seconds),
            evaluator=np.asarray("scripts/evaluate_nucleonic_support_extension.py"),
            prediction_storage=np.asarray(
                "canonical support-screen prediction after evaluator agreement gate"
            ),
        )
    os.replace(temporary, arguments.output)
    report = {
        "status": "PASS",
        "schema": OUTPUT_SCHEMA,
        "support_name": "S_ext",
        "evaluator": "scripts/evaluate_nucleonic_support_extension.py",
        "screen": str(arguments.screen.resolve()),
        "screen_sha256": screen_sha256,
        "template": str(arguments.template.resolve()),
        "template_sha256": sha256(arguments.template),
        "template_parent_sha256": template_parent_sha256,
        "total_screen_rows": int(len(all_theta)),
        "total_proposals": total_proposals,
        "seed": seed,
        "shard_start": arguments.start,
        "shard_stop": arguments.stop,
        "shard_rows": int(len(theta)),
        "target_valid_rows": int(target_valid.sum()),
        "maximum_prediction_delta": maximum_prediction_delta,
        "maximum_normalized_prediction_delta": maximum_normalized_prediction_delta,
        "prediction_tolerance_sigma": prediction_tolerance_sigma,
        "prediction_storage": (
            "canonical support-screen prediction after evaluator agreement gate"
        ),
        "workers": arguments.workers,
        "chunk_size": arguments.chunk_size,
        "hostname": socket.gethostname(),
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "reference_present": False,
        "forbidden_artifacts_used": [],
        "wall_seconds": wall_seconds,
        "stage_diagnostics": diagnostics,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
