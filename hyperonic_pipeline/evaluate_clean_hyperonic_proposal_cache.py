#!/usr/bin/env python3
"""Evaluate a clean normalized proposal for hyperonic A-NET scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed

from build_clean_uniform_hyperonic_training_cache import evaluate_training_row
from clean_student_t_ensemble import checkpoint_density
from eos import get_eos
from likelihoods.nuclear import A1_SIGMA
from workflows.a1_problem import A1Problem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    proposal_sha256 = sha256(arguments.proposal)
    checkpoint_sha256 = sha256(arguments.checkpoint)
    checkpoint = joblib.load(arguments.checkpoint)
    with np.load(arguments.proposal, allow_pickle=False) as proposal:
        required = {
            "schema",
            "theta",
            "logq",
            "prior_low",
            "prior_high",
            "checkpoint_sha256",
        }
        missing = required - set(proposal.files)
        if missing:
            raise RuntimeError(f"proposal is missing {sorted(missing)}")
        if str(proposal["schema"].item()) != "ddb-hyperonic-clean-proposal-v1":
            raise RuntimeError("unexpected proposal schema")
        all_theta = np.asarray(proposal["theta"], dtype=np.float64)
        all_logq = np.asarray(proposal["logq"], dtype=np.float64)
        prior_low = np.asarray(proposal["prior_low"], dtype=np.float64)
        prior_high = np.asarray(proposal["prior_high"], dtype=np.float64)
        declared_checkpoint_sha256 = str(proposal["checkpoint_sha256"].item())
    if declared_checkpoint_sha256 != checkpoint_sha256:
        raise RuntimeError("proposal and checkpoint hashes differ")
    if all_theta.shape != (len(all_theta), 9) or all_logq.shape != (len(all_theta),):
        raise RuntimeError("proposal theta/logq shapes are invalid")
    if not 0 <= arguments.start < arguments.stop <= len(all_theta):
        raise ValueError(
            f"invalid shard [{arguments.start},{arguments.stop}) for "
            f"{len(all_theta)} proposal rows"
        )
    theta = all_theta[arguments.start : arguments.stop]
    stored_logq = all_logq[arguments.start : arguments.stop]
    if not np.all((theta >= prior_low) & (theta <= prior_high)):
        raise RuntimeError("proposal contains a row outside the canonical prior")
    prior_logq = -float(np.log(prior_high - prior_low).sum())
    replay_logq = checkpoint_density(checkpoint, theta, prior_logq)
    maximum_logq_replay_error = float(np.max(np.abs(replay_logq - stored_logq)))
    density_tolerance = 1.0e-10
    if maximum_logq_replay_error > density_tolerance:
        raise RuntimeError(
            "proposal-density replay failed: "
            f"{maximum_logq_replay_error:.3e} > {density_tolerance:.3e}"
        )

    with np.load(arguments.template_bank, allow_pickle=False) as template:
        if str(template["schema_version"].item()) != "ddbhy-bank-v1":
            raise RuntimeError("unexpected template-bank schema")
        mass_grid = np.asarray(template["MG"], dtype=np.float64)
        template_low = np.asarray(template["THETA_LOW"], dtype=np.float64)
        template_high = np.asarray(template["THETA_HIGH"], dtype=np.float64)
    if not (
        np.array_equal(prior_low, template_low)
        and np.array_equal(prior_high, template_high)
    ):
        raise RuntimeError("proposal and template prior bounds differ")

    problem = A1Problem.from_data_root(
        arguments.data_root,
        workers=arguments.workers,
        verify_data=True,
        nuclear_scenario="A1",
        source_scenario="A1",
        model="ddb-hyperonic",
    )
    plugin = get_eos("ddb-hyperonic")
    prediction = np.empty((len(theta), 7), dtype=np.float64)
    prediction[:, 0] = theta[:, plugin.parameter_names.index("rho0")]
    prediction[:, 1:] = plugin.nuclear_observables_batch(theta)

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
                f"[proposal-cache] {local_stop}/{rows} exact rows",
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
        raise RuntimeError("corrected target rejected every proposal row")
    source_sum_error = float(
        np.max(
            np.abs(
                source_components[target_valid].sum(axis=1)
                - components[target_valid, 1]
            )
        )
    )
    if source_sum_error > 1.0e-10:
        raise RuntimeError(
            f"per-source NICER sum gate failed: {source_sum_error:.3e}"
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray("ddb-hyperonic-clean-proposal-training-cache-v1"),
            source_index=np.arange(arguments.start, arguments.stop, dtype=np.int64),
            theta=theta,
            proposal_logq=stored_logq,
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
            proposal_sha256=np.asarray(proposal_sha256),
            checkpoint_sha256=np.asarray(checkpoint_sha256),
            maximum_logq_replay_error=np.float64(maximum_logq_replay_error),
            density_tolerance=np.float64(density_tolerance),
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
        "schema": "ddb-hyperonic-clean-proposal-training-cache-v1",
        "proposal": str(arguments.proposal.resolve()),
        "proposal_sha256": proposal_sha256,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "template_bank_sha256": sha256(arguments.template_bank),
        "shard_start": arguments.start,
        "shard_stop": arguments.stop,
        "shard_rows": rows,
        "target_valid_rows": int(target_valid.sum()),
        "maximum_logq_replay_error": maximum_logq_replay_error,
        "density_tolerance": density_tolerance,
        "source_sum_error": source_sum_error,
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
