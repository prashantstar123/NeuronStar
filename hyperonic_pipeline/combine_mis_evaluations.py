#!/usr/bin/env python3
"""Combine exact evaluations from distinct normalized proposals by MIS.

Each input evaluation must contain draws from the checkpoint paired with it.
The utility recomputes every generating density, verifies the stored density,
and then applies the deterministic-mixture (balance-heuristic) denominator

    q_mix(theta) = sum_k (N_k / N_total) q_k(theta).

No posterior or comparison-sampler object enters this calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np

from clean_gmm import CHECKPOINT_SCHEMA
from clean_gmm_ensemble import ENSEMBLE_SCHEMA
from clean_student_t_ensemble import (
    STUDENT_ENSEMBLE_SCHEMA,
    checkpoint_density,
)


EVALUATION_SCHEMA = "ddb-hyperonic-clean-eval-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_ess(logweight: np.ndarray) -> float:
    finite = np.isfinite(logweight)
    if not finite.any():
        return 0.0
    shifted = logweight[finite] - np.max(logweight[finite])
    weight = np.exp(shifted)
    weight /= weight.sum()
    return float(1.0 / np.sum(weight * weight))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        nargs=2,
        metavar=("EVALUATION", "CHECKPOINT"),
        action="append",
        required=True,
        help="Exact evaluation and the normalized GMM checkpoint that drew it.",
    )
    parser.add_argument("--target-beta", type=float, required=True)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if len(arguments.component) < 2:
        raise ValueError("MIS requires at least two proposal components")
    if not 0.0 <= arguments.target_beta <= 1.0:
        raise ValueError("target beta must lie in [0, 1]")

    evaluations: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    lower_reference: np.ndarray | None = None
    upper_reference: np.ndarray | None = None

    for evaluation_name, checkpoint_name in arguments.component:
        evaluation_path = Path(evaluation_name)
        checkpoint_path = Path(checkpoint_name)
        checkpoint = joblib.load(checkpoint_path)
        if checkpoint.get("schema") not in {
            CHECKPOINT_SCHEMA,
            ENSEMBLE_SCHEMA,
            STUDENT_ENSEMBLE_SCHEMA,
        }:
            raise RuntimeError(f"unrecognized GMM checkpoint: {checkpoint_path}")
        checkpoint_sha = sha256(checkpoint_path)

        with np.load(evaluation_path, allow_pickle=False) as source:
            if str(source["schema"].item()) != EVALUATION_SCHEMA:
                raise RuntimeError(f"unrecognized clean evaluation: {evaluation_path}")
            theta = np.asarray(source["theta"], dtype=np.float64)
            loglike = np.asarray(source["logl"], dtype=np.float64)
            stored_logq = np.asarray(source["logq"], dtype=np.float64)
            lower = np.asarray(source["prior_low"], dtype=np.float64)
            upper = np.asarray(source["prior_high"], dtype=np.float64)
            stored_checkpoint_sha = str(source["checkpoint_sha256"].item())
            source_beta = float(source["proposal_beta"])
        if (
            theta.ndim != 2
            or theta.shape[1] != 9
            or loglike.shape != (len(theta),)
            or stored_logq.shape != (len(theta),)
        ):
            raise RuntimeError(f"invalid evaluation arrays: {evaluation_path}")
        if stored_checkpoint_sha != checkpoint_sha:
            raise RuntimeError(
                f"checkpoint hash mismatch for evaluation {evaluation_path}"
            )
        checkpoint_lower = np.asarray(checkpoint["prior_low"], dtype=np.float64)
        checkpoint_upper = np.asarray(checkpoint["prior_high"], dtype=np.float64)
        if not np.array_equal(lower, checkpoint_lower) or not np.array_equal(
            upper, checkpoint_upper
        ):
            raise RuntimeError("evaluation and checkpoint prior bounds differ")
        if lower_reference is None:
            lower_reference = lower
            upper_reference = upper
        elif not np.array_equal(lower, lower_reference) or not np.array_equal(
            upper, upper_reference
        ):
            raise RuntimeError("component prior bounds differ")
        evaluations.append(
            {
                "path": evaluation_path,
                "theta": theta,
                "logl": loglike,
                "stored_logq": stored_logq,
                "proposal_beta": source_beta,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha,
            }
        )
        checkpoints.append(checkpoint)

    assert lower_reference is not None and upper_reference is not None
    lower = lower_reference
    upper = upper_reference
    prior_logq = -float(np.log(upper - lower).sum())
    total_rows = sum(len(item["theta"]) for item in evaluations)
    fractions = np.asarray(
        [len(item["theta"]) / total_rows for item in evaluations],
        dtype=np.float64,
    )

    theta_parts: list[np.ndarray] = []
    loglike_parts: list[np.ndarray] = []
    mixture_logq_parts: list[np.ndarray] = []
    manifest: list[dict[str, object]] = []
    for source_number, source in enumerate(evaluations):
        theta = np.asarray(source["theta"], dtype=np.float64)
        component_logq = np.column_stack(
            [
                checkpoint_density(checkpoint, theta, prior_logq)
                for checkpoint in checkpoints
            ]
        )
        own_logq = component_logq[:, source_number]
        stored_logq = np.asarray(source["stored_logq"], dtype=np.float64)
        maximum_density_error = float(np.max(np.abs(own_logq - stored_logq)))
        if maximum_density_error > arguments.density_tolerance:
            raise RuntimeError(
                "stored generating density failed checkpoint reproduction: "
                f"component={source_number}, max_abs_error={maximum_density_error:.3e}"
            )
        mixture_logq = np.logaddexp.reduce(
            component_logq + np.log(fractions)[None, :], axis=1
        )
        theta_parts.append(theta)
        loglike_parts.append(np.asarray(source["logl"], dtype=np.float64))
        mixture_logq_parts.append(mixture_logq)
        evaluation_path = Path(source["path"])
        checkpoint_path = Path(source["checkpoint_path"])
        manifest.append(
            {
                "component": source_number,
                "evaluation": str(evaluation_path.resolve()),
                "evaluation_sha256": sha256(evaluation_path),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": str(source["checkpoint_sha256"]),
                "proposal_beta": float(source["proposal_beta"]),
                "rows": int(len(theta)),
                "mixture_fraction": float(fractions[source_number]),
                "maximum_own_logq_error": maximum_density_error,
            }
        )

    theta = np.concatenate(theta_parts)
    loglike = np.concatenate(loglike_parts)
    mixture_logq = np.concatenate(mixture_logq_parts)
    valid = loglike > -1.0e50
    current_ess = normalized_ess(
        prior_logq
        + arguments.target_beta * loglike[valid]
        - mixture_logq[valid]
    )
    posterior_ess = normalized_ess(
        prior_logq + loglike[valid] - mixture_logq[valid]
    )
    manifest_json = json.dumps(manifest, sort_keys=True)
    manifest_sha = hashlib.sha256(manifest_json.encode()).hexdigest()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(EVALUATION_SCHEMA),
            theta=theta,
            logq=mixture_logq,
            logl=loglike,
            prior_low=lower,
            prior_high=upper,
            proposal_beta=np.float64(arguments.target_beta),
            checkpoint_sha256=np.asarray(f"mis-manifest:{manifest_sha}"),
            proposal_sha256=np.asarray(f"mis-manifest:{manifest_sha}"),
            likelihood_sha256=np.asarray(f"mis-manifest:{manifest_sha}"),
            valid_rows=np.int64(valid.sum()),
            posterior_ess=np.float64(posterior_ess),
            source_manifest_json=np.asarray(manifest_json),
        )
    os.replace(temporary, arguments.output)

    report = {
        "status": "PASS",
        "schema": "ddb-hyperonic-clean-mis-v1",
        "rows": int(len(theta)),
        "valid_rows": int(valid.sum()),
        "target_beta": float(arguments.target_beta),
        "current_temperature_ess": current_ess,
        "posterior_ess": posterior_ess,
        "density_tolerance": float(arguments.density_tolerance),
        "source_manifest_sha256": manifest_sha,
        "output_sha256": sha256(arguments.output),
        "components": manifest,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
