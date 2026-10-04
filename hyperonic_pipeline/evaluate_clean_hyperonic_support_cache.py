#!/usr/bin/env python3
"""Evaluate one shard of the deterministic hyperonic support enrichment."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from build_clean_uniform_hyperonic_training_cache import (
    evaluate_training_row,
    sha256,
)
from eos import get_eos
from likelihoods.nuclear import A1_SIGMA
from workflows.a1_problem import A1Problem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--template-bank", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        raise ValueError("workers must be positive")

    started_unix = time.time()
    with np.load(arguments.screen, allow_pickle=False) as screen:
        required = {
            "schema",
            "theta",
            "prediction",
            "total_proposals",
            "cbox",
            "seed",
            "stream_id",
            "reference_present",
            "forbidden_artifacts_used",
        }
        missing = required - set(screen.files)
        if missing:
            raise RuntimeError(f"support screen is missing {sorted(missing)}")
        if str(screen["schema"].item()) != "ddb-hyperonic-clean-support-screen-v1":
            raise RuntimeError("unexpected support-screen schema")
        if bool(screen["reference_present"]) or screen["forbidden_artifacts_used"].size:
            raise RuntimeError("support screen declares forbidden ancestry")
        all_theta = np.asarray(screen["theta"], dtype=np.float64)
        all_prediction = np.asarray(screen["prediction"], dtype=np.float64)
        total_proposals = int(screen["total_proposals"])
        cbox = float(screen["cbox"])
        seed = int(screen["seed"])
        stream_id = int(screen["stream_id"])
    if not 0 <= arguments.start < arguments.stop <= len(all_theta):
        raise ValueError(
            f"invalid shard [{arguments.start},{arguments.stop}) for {len(all_theta)} rows"
        )
    theta = all_theta[arguments.start : arguments.stop]
    stored_prediction = all_prediction[arguments.start : arguments.stop]
    with np.load(arguments.template_bank, allow_pickle=False) as template:
        if str(template["schema_version"].item()) != "ddbhy-bank-v1":
            raise RuntimeError("unexpected template-bank schema")
        mass_grid = np.asarray(template["MG"], dtype=np.float64)
        prior_low = np.asarray(template["THETA_LOW"], dtype=np.float64)
        prior_high = np.asarray(template["THETA_HIGH"], dtype=np.float64)

    problem = A1Problem.from_data_root(
        arguments.data_root,
        workers=arguments.workers,
        verify_data=True,
        nuclear_scenario="A1",
        source_scenario="A1",
        model="ddb-hyperonic",
    )
    plugin = get_eos("ddb-hyperonic")
    prediction = np.empty_like(stored_prediction)
    prediction[:, 0] = theta[:, plugin.parameter_names.index("rho0")]
    prediction[:, 1:] = plugin.nuclear_observables_batch(theta)
    prediction_delta = np.abs(prediction - stored_prediction)
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

    rows = len(theta)
    log_astrophysical = np.full(rows, np.nan, dtype=np.float64)
    components = np.full((rows, 4), np.nan, dtype=np.float64)
    source_components = np.full((rows, 3), np.nan, dtype=np.float64)
    radius = np.full((rows, len(mass_grid)), np.nan, dtype=np.float64)
    tidal = np.full_like(radius, np.nan)
    maximum_mass = np.full(rows, np.nan, dtype=np.float64)
    evaluation_chunk = 2500
    with Parallel(
        n_jobs=min(arguments.workers, rows),
        batch_size=1,
        pre_dispatch="2*n_jobs",
    ) as parallel:
        for local_start in range(0, rows, evaluation_chunk):
            local_stop = min(local_start + evaluation_chunk, rows)
            evaluated = parallel(
                delayed(evaluate_training_row)(
                    "ddb-hyperonic",
                    theta[row],
                    prediction[row, 1:],
                    problem.target,
                    mass_grid,
                )
                for row in range(local_start, local_stop)
            )
            for offset, result in enumerate(evaluated):
                row = local_start + offset
                (
                    log_astrophysical[row],
                    components[row],
                    source_components[row],
                    radius[row],
                    tidal[row],
                    maximum_mass[row],
                ) = result
            print(
                f"[support-cache] {local_stop}/{rows} exact rows",
                flush=True,
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
        raise RuntimeError("corrected target rejected every support row")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray("ddb-hyperonic-clean-support-training-cache-v1"),
            source_index=np.arange(arguments.start, arguments.stop, dtype=np.int64),
            theta=theta,
            prediction=prediction,
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
            screen_sha256=np.asarray(sha256(arguments.screen)),
            template_bank_sha256=np.asarray(sha256(arguments.template_bank)),
            total_proposals=np.int64(total_proposals),
            cbox=np.float64(cbox),
            seed=np.int64(seed),
            stream_id=np.int64(stream_id),
            shard_start=np.int64(arguments.start),
            shard_stop=np.int64(arguments.stop),
            prediction_tolerance_sigma=np.float64(prediction_tolerance_sigma),
            maximum_prediction_delta=np.float64(maximum_prediction_delta),
            maximum_normalized_prediction_delta=np.float64(
                maximum_normalized_prediction_delta
            ),
            workers=np.int64(arguments.workers),
            hostname=np.asarray(socket.gethostname()),
            reference_present=np.bool_(False),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
            started_unix=np.float64(started_unix),
            finished_unix=np.float64(time.time()),
        )
    os.replace(temporary, arguments.output)
    report = {
        "status": "PASS",
        "schema": "ddb-hyperonic-clean-support-training-cache-v1",
        "screen": str(arguments.screen.resolve()),
        "screen_sha256": sha256(arguments.screen),
        "template_bank": str(arguments.template_bank.resolve()),
        "template_bank_sha256": sha256(arguments.template_bank),
        "total_screen_rows": int(len(all_theta)),
        "total_proposals": total_proposals,
        "shard_start": arguments.start,
        "shard_stop": arguments.stop,
        "shard_rows": rows,
        "target_valid_rows": int(target_valid.sum()),
        "maximum_prediction_delta": maximum_prediction_delta,
        "maximum_normalized_prediction_delta": maximum_normalized_prediction_delta,
        "prediction_tolerance_sigma": prediction_tolerance_sigma,
        "workers": arguments.workers,
        "hostname": socket.gethostname(),
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "reference_present": False,
        "forbidden_artifacts_used": [],
        "wall_seconds": time.time() - started_unix,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
