#!/usr/bin/env python3
"""Build one sampler-independent hyperonic training-cache shard.

The input is one validated uniform-prior physics bank.  Production retains
every forward-valid row, matching the nucleonic base-bank construction.  An
optional nuclear-support cut exists only for diagnostics.  No inference
checkpoint, posterior, evidence value, or proposal geometry is consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from eos import get_eos
from likelihoods.nicer import log_likelihood_one
from likelihoods.nuclear import A1_OBSERVATION, A1_SIGMA
from tov import solve_stable_branch
from workflows.a1_problem import A1Problem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_training_row(
    model_key: str,
    parameters: np.ndarray,
    nuclear_observables: np.ndarray,
    target,
    mass_grid: np.ndarray,
):
    """Return exact target factors and fresh stable-branch curves for one row."""

    plugin = get_eos(model_key)
    blank_curve = np.full(len(mass_grid), np.nan, dtype=np.float64)
    try:
        density, energy, pressure = plugin.core_eos(parameters)
        branch = solve_stable_branch(
            energy,
            pressure,
            require_monotonic_graft=plugin.metadata.get(
                "require_monotonic_tov_graft", False
            ),
        )
        terms = target.evaluate(
            parameters,
            nuclear_observables,
            density,
            energy,
            pressure,
            branch,
        )
        source_terms = np.asarray(
            [
                log_likelihood_one(
                    branch.mass,
                    branch.radius,
                    branch.maximum_mass,
                    interpolator,
                )
                for interpolator in target.nicer_interpolators
            ],
            dtype=np.float64,
        )
        components = np.asarray(
            [terms.maximum_mass, terms.nicer, terms.gw170817, terms.pqcd],
            dtype=np.float64,
        )
        log_astrophysical = float(np.sum(components))
        if (
            not np.isfinite(terms.total)
            or not np.isfinite(log_astrophysical)
            or not np.all(np.isfinite(source_terms))
        ):
            raise FloatingPointError("non-finite corrected target")
        if abs(float(source_terms.sum()) - float(terms.nicer)) > 1.0e-10:
            raise RuntimeError("per-source NICER terms do not sum to target")
        radius = blank_curve.copy()
        tidal = blank_curve.copy()
        covered = (
            (mass_grid >= float(branch.mass.min()))
            & (mass_grid <= float(branch.mass.max()))
            & (mass_grid <= float(branch.maximum_mass))
        )
        radius[covered] = np.interp(
            mass_grid[covered], branch.mass, branch.radius
        )
        tidal[covered] = np.exp(
            np.interp(
                mass_grid[covered],
                branch.mass,
                np.log(np.clip(branch.tidal_lambda, 1.0e-300, None)),
            )
        )
        return (
            log_astrophysical,
            components,
            source_terms,
            radius,
            tidal,
            float(branch.maximum_mass),
        )
    except (ValueError, FloatingPointError, RuntimeError):
        return (
            np.nan,
            np.full(4, np.nan, dtype=np.float64),
            np.full(3, np.nan, dtype=np.float64),
            blank_curve,
            blank_curve.copy(),
            np.nan,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--support-sigma",
        type=float,
        help="Diagnostic-only nuclear support cut; omit in production.",
    )
    parser.add_argument(
        "--maximum-selected-rows",
        type=int,
        help="Development-only deterministic cap; omit in production.",
    )
    arguments = parser.parse_args()
    if arguments.workers < 1:
        raise ValueError("--workers must be positive")
    if arguments.support_sigma is not None and arguments.support_sigma <= 0:
        raise ValueError("--support-sigma must be positive")

    started_unix = time.time()
    with np.load(arguments.bank, allow_pickle=False) as bank:
        required = {
            "theta",
            "X",
            "MG",
            "Rg",
            "Lg",
            "MM",
            "valid",
            "THETA_LOW",
            "THETA_HIGH",
            "schema_version",
        }
        missing = required - set(bank.files)
        if missing:
            raise RuntimeError(f"uniform bank is missing {sorted(missing)}")
        if str(bank["schema_version"].item()) != "ddbhy-bank-v1":
            raise RuntimeError("unexpected uniform-bank schema")
        theta_all = np.asarray(bank["theta"], dtype=np.float64)
        prediction_all = np.asarray(bank["X"], dtype=np.float64)
        valid_all = np.asarray(bank["valid"], dtype=bool)
        prior_low = np.asarray(bank["THETA_LOW"], dtype=np.float64)
        prior_high = np.asarray(bank["THETA_HIGH"], dtype=np.float64)
        mass_grid = np.asarray(bank["MG"], dtype=np.float64)
        radius_all = np.asarray(bank["Rg"], dtype=np.float64)
        tidal_all = np.asarray(bank["Lg"], dtype=np.float64)
        maximum_mass_all = np.asarray(bank["MM"], dtype=np.float64)

    rows = len(theta_all)
    if theta_all.shape != (rows, 9) or prediction_all.shape != (rows, 7):
        raise RuntimeError("uniform-bank theta/X shapes are invalid")
    if radius_all.shape != (rows, len(mass_grid)) or tidal_all.shape != radius_all.shape:
        raise RuntimeError("uniform-bank curve shapes are invalid")
    if maximum_mass_all.shape != (rows,) or valid_all.shape != (rows,):
        raise RuntimeError("uniform-bank scalar shapes are invalid")
    if prior_low.shape != (9,) or prior_high.shape != (9,):
        raise RuntimeError("uniform-bank prior shapes are invalid")
    if np.any(theta_all < prior_low) or np.any(theta_all > prior_high):
        raise RuntimeError("uniform bank contains a row outside its prior box")

    selected_mask = (
        valid_all
        & np.isfinite(theta_all).all(axis=1)
        & np.isfinite(prediction_all).all(axis=1)
        & np.isfinite(maximum_mass_all)
    )
    if arguments.support_sigma is not None:
        standardized = (prediction_all - A1_OBSERVATION) / A1_SIGMA
        selected_mask &= (
            np.max(np.abs(standardized), axis=1) <= arguments.support_sigma
        )
    selected_index = np.flatnonzero(selected_mask)
    if len(selected_index) == 0:
        raise RuntimeError("the fixed support filter selected no rows")
    if arguments.maximum_selected_rows is not None:
        if arguments.maximum_selected_rows < 1:
            raise ValueError("--maximum-selected-rows must be positive")
        selected_index = selected_index[: arguments.maximum_selected_rows]

    theta = theta_all[selected_index]
    stored_prediction = prediction_all[selected_index]
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
    log_astrophysical = np.full(len(theta), np.nan, dtype=np.float64)
    components = np.full((len(theta), 4), np.nan, dtype=np.float64)
    source_components = np.full((len(theta), 3), np.nan, dtype=np.float64)
    fresh_radius = np.full((len(theta), len(mass_grid)), np.nan, dtype=np.float64)
    fresh_tidal = np.full_like(fresh_radius, np.nan)
    fresh_maximum_mass = np.full(len(theta), np.nan, dtype=np.float64)
    evaluation_chunk = 2500
    with Parallel(
        n_jobs=min(arguments.workers, len(theta)),
        batch_size=1,
        pre_dispatch="2*n_jobs",
    ) as parallel:
        for start in range(0, len(theta), evaluation_chunk):
            stop = min(start + evaluation_chunk, len(theta))
            rows_evaluated = parallel(
                delayed(evaluate_training_row)(
                    "ddb-hyperonic",
                    theta[row],
                    prediction[row, 1:],
                    problem.target,
                    mass_grid,
                )
                for row in range(start, stop)
            )
            for local, result in enumerate(rows_evaluated):
                row = start + local
                (
                    log_astrophysical[row],
                    components[row],
                    source_components[row],
                    fresh_radius[row],
                    fresh_tidal[row],
                    fresh_maximum_mass[row],
                ) = result
            print(
                f"[uniform-cache] {stop}/{len(theta)} exact rows",
                flush=True,
            )
    prediction_delta = np.abs(prediction - stored_prediction)
    maximum_prediction_delta = float(np.max(prediction_delta))
    maximum_normalized_prediction_delta = float(
        np.max(prediction_delta / A1_SIGMA)
    )
    prediction_tolerance_sigma = 1.0e-4
    if maximum_normalized_prediction_delta > prediction_tolerance_sigma:
        raise RuntimeError(
            "stored and portable nuclear predictions differ by "
            f"{maximum_normalized_prediction_delta:.3e} nuclear sigma; "
            f"gate={prediction_tolerance_sigma:.3e}"
        )
    target_valid = (
        np.isfinite(log_astrophysical)
        & (log_astrophysical > -1.0e29)
        & np.isfinite(source_components).all(axis=1)
        & np.isfinite(fresh_maximum_mass)
        & np.isfinite(fresh_radius).any(axis=1)
        & np.isfinite(fresh_tidal).any(axis=1)
    )
    if not target_valid.any():
        raise RuntimeError("corrected target rejected every selected row")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray("ddb-hyperonic-clean-uniform-training-cache-v1"),
            source_index=selected_index,
            theta=theta,
            prediction=prediction,
            log_astrophysical=log_astrophysical,
            astrophysical_components=components,
            nicer_source_components=source_components,
            target_valid=target_valid,
            mass_grid=mass_grid,
            radius=fresh_radius,
            tidal_lambda=fresh_tidal,
            maximum_mass=fresh_maximum_mass,
            prior_low=prior_low,
            prior_high=prior_high,
            source_bank_sha256=np.asarray(sha256(arguments.bank)),
            support_sigma=np.float64(
                np.nan if arguments.support_sigma is None else arguments.support_sigma
            ),
            prediction_tolerance_sigma=np.float64(prediction_tolerance_sigma),
            maximum_prediction_delta=np.float64(maximum_prediction_delta),
            maximum_normalized_prediction_delta=np.float64(
                maximum_normalized_prediction_delta
            ),
            maximum_selected_rows=np.int64(
                -1
                if arguments.maximum_selected_rows is None
                else arguments.maximum_selected_rows
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
        "schema": "ddb-hyperonic-clean-uniform-training-cache-v1",
        "source_bank": str(arguments.bank.resolve()),
        "source_bank_sha256": sha256(arguments.bank),
        "source_rows": rows,
        "selected_rows": int(len(selected_index)),
        "target_valid_rows": int(target_valid.sum()),
        "support_sigma": arguments.support_sigma,
        "production_all_forward_valid_rows": arguments.support_sigma is None,
        "maximum_selected_rows": arguments.maximum_selected_rows,
        "maximum_prediction_delta": maximum_prediction_delta,
        "maximum_normalized_prediction_delta": (
            maximum_normalized_prediction_delta
        ),
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


if __name__ == "__main__":
    main()
