#!/usr/bin/env python3
"""Fit and sample a mode-preserving normalized bridge proposal.

The learned full-dimensional GMM is mixed with the canonical prior and one or
more earlier normalized clean checkpoints.  Retaining earlier proposals as
explicit mixture components prevents an adaptive fit from silently dropping a
sparse branch.  Fresh draws are still corrected by the exact density of the
entire normalized mixture.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp

from clean_gmm import (
    CHECKPOINT_SCHEMA as BASE_CHECKPOINT_SCHEMA,
    DIMENSIONS,
    EVALUATION_SCHEMA,
    PROPOSAL_SCHEMA,
    atomic_joblib,
    atomic_savez,
    fit_model,
    flow_log_density,
    from_unconstrained,
    load_prior,
    normalized_weights,
    sha256,
)


ENSEMBLE_SCHEMA = "ddb-hyperonic-clean-gmm-ensemble-v1"


def checkpoint_density(
    checkpoint: dict[str, object],
    theta: np.ndarray,
    prior_logq: float,
) -> np.ndarray:
    """Evaluate either a base GMM checkpoint or an ensemble recursively."""

    lower = np.asarray(checkpoint["prior_low"], dtype=np.float64)
    upper = np.asarray(checkpoint["prior_high"], dtype=np.float64)
    learned = flow_log_density(
        checkpoint["model"],
        np.asarray(checkpoint["mean"], dtype=np.float64),
        np.asarray(checkpoint["std"], dtype=np.float64),
        theta,
        lower,
        upper,
    )
    schema = checkpoint.get("schema")
    if schema == BASE_CHECKPOINT_SCHEMA:
        defensive = float(checkpoint["defensive_fraction"])
        return np.logaddexp(
            math.log1p(-defensive) + learned,
            math.log(defensive) + prior_logq,
        )
    if schema != ENSEMBLE_SCHEMA:
        raise RuntimeError(f"unrecognized ancestor checkpoint schema: {schema}")
    terms = [math.log(float(checkpoint["learned_weight"])) + learned]
    prior_weight = float(checkpoint["prior_weight"])
    if prior_weight:
        terms.append(np.full(len(theta), math.log(prior_weight) + prior_logq))
    ancestors = list(checkpoint["ancestors"])
    ancestor_weights = np.asarray(checkpoint["ancestor_weights"], dtype=np.float64)
    for weight, ancestor in zip(ancestor_weights, ancestors, strict=True):
        terms.append(
            math.log(float(weight))
            + checkpoint_density(ancestor, theta, prior_logq)
        )
    return logsumexp(np.vstack(terms), axis=0)


def candidate_density(
    learned_logq: np.ndarray,
    ancestor_logq: list[np.ndarray],
    prior_logq: float,
    learned_weight: float,
    ancestor_weights: np.ndarray,
    prior_weight: float,
) -> np.ndarray:
    terms = [math.log(learned_weight) + learned_logq]
    terms.extend(
        math.log(float(weight)) + density
        for weight, density in zip(ancestor_weights, ancestor_logq, strict=True)
    )
    if prior_weight:
        terms.append(np.full(len(learned_logq), math.log(prior_weight) + prior_logq))
    return logsumexp(np.vstack(terms), axis=0)


def validate_weights(
    learned_weight: float,
    ancestor_weights: np.ndarray,
    prior_weight: float,
) -> None:
    all_weights = np.concatenate(
        [np.asarray([learned_weight, prior_weight]), ancestor_weights]
    )
    if np.any(all_weights < 0.0) or learned_weight <= 0.0:
        raise ValueError("mixture weights must be nonnegative and learned weight positive")
    if not np.isclose(all_weights.sum(), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"mixture weights sum to {all_weights.sum():.17g}, not one")


def load_ancestors(
    paths: list[Path], lower: np.ndarray, upper: np.ndarray
) -> tuple[list[dict[str, object]], list[str]]:
    checkpoints: list[dict[str, object]] = []
    hashes: list[str] = []
    for path in paths:
        checkpoint = joblib.load(path)
        if checkpoint.get("schema") not in {BASE_CHECKPOINT_SCHEMA, ENSEMBLE_SCHEMA}:
            raise RuntimeError(f"unrecognized ancestor checkpoint: {path}")
        if not np.array_equal(checkpoint["prior_low"], lower) or not np.array_equal(
            checkpoint["prior_high"], upper
        ):
            raise RuntimeError(f"ancestor prior differs: {path}")
        checkpoints.append(checkpoint)
        hashes.append(sha256(path))
    return checkpoints, hashes


def fit(arguments: argparse.Namespace) -> None:
    started = time.time()
    lower, upper = load_prior(arguments.prior)
    ancestor_weights = np.asarray(arguments.ancestor_weight, dtype=np.float64)
    if len(arguments.ancestor_checkpoint) != len(ancestor_weights):
        raise ValueError("one --ancestor-weight is required per ancestor checkpoint")
    validate_weights(
        arguments.learned_weight, ancestor_weights, arguments.prior_weight
    )
    ancestors, ancestor_hashes = load_ancestors(
        arguments.ancestor_checkpoint, lower, upper
    )
    with np.load(arguments.input, allow_pickle=False) as source:
        if str(source["schema"].item()) != EVALUATION_SCHEMA:
            raise RuntimeError("fit input is not a clean exact evaluation")
        theta = np.asarray(source["theta"], dtype=np.float64)
        loglike = np.asarray(source["logl"], dtype=np.float64)
        source_logq = np.asarray(source["logq"], dtype=np.float64)
    if (
        theta.ndim != 2
        or theta.shape[1] != DIMENSIONS
        or loglike.shape != (len(theta),)
        or source_logq.shape != (len(theta),)
    ):
        raise RuntimeError("invalid clean-evaluation arrays")
    if not np.all((theta >= lower) & (theta <= upper)):
        raise RuntimeError("input row lies outside the canonical prior")

    prior_logq = -float(np.log(upper - lower).sum())
    logtarget = prior_logq + arguments.beta * loglike
    full_weight, input_ess = normalized_weights(logtarget - source_logq)
    if input_ess < arguments.minimum_input_ess:
        raise RuntimeError(
            f"input ESS {input_ess:.1f} is below gate {arguments.minimum_input_ess:.1f}"
        )
    ancestor_logq_full = [
        checkpoint_density(checkpoint, theta, prior_logq)
        for checkpoint in ancestors
    ]

    rng = np.random.default_rng(arguments.seed)
    folds = np.array_split(rng.permutation(len(theta)), arguments.cross_validation_folds)
    cross_validation: list[dict[str, object]] = []
    for fold_number, validation_index in enumerate(folds):
        training_index = np.concatenate(
            [part for number, part in enumerate(folds) if number != fold_number]
        )
        training_weight, training_ess = normalized_weights(
            logtarget[training_index] - source_logq[training_index]
        )
        model, mean, std = fit_model(
            theta[training_index],
            training_weight,
            lower,
            upper,
            draws=arguments.cross_validation_draws,
            components=arguments.components,
            regularization=arguments.regularization,
            jitter=arguments.jitter,
            maximum_iterations=arguments.maximum_iterations,
            seed=arguments.seed + 1000 + fold_number,
        )
        learned_logq = flow_log_density(
            model, mean, std, theta[validation_index], lower, upper
        )
        candidate_logq = candidate_density(
            learned_logq,
            [density[validation_index] for density in ancestor_logq_full],
            prior_logq,
            arguments.learned_weight,
            ancestor_weights,
            arguments.prior_weight,
        )
        local_logtarget = logtarget[validation_index]
        local_source_logq = source_logq[validation_index]
        log_normalization = (
            logsumexp(local_logtarget - local_source_logq)
            - math.log(len(validation_index))
        )
        log_second_moment = (
            logsumexp(
                2.0 * local_logtarget - candidate_logq - local_source_logq
            )
            - math.log(len(validation_index))
            - 2.0 * log_normalization
        )
        predicted_fraction = float(math.exp(-log_second_moment))
        _, validation_target_ess = normalized_weights(
            local_logtarget - local_source_logq
        )
        record = {
            "fold": fold_number,
            "training_rows": int(len(training_index)),
            "validation_rows": int(len(validation_index)),
            "training_ess": training_ess,
            "validation_target_ess": validation_target_ess,
            "predicted_ess_fraction": predicted_fraction,
            "iterations": int(model.n_iter_),
            "converged": bool(model.converged_),
        }
        cross_validation.append(record)
        print(
            f"fold={fold_number} training_ess={training_ess:.1f} "
            f"validation_target_ess={validation_target_ess:.1f} "
            f"predicted_ess_fraction={predicted_fraction:.5f}",
            flush=True,
        )
        if predicted_fraction < arguments.minimum_cv_ess_fraction:
            raise RuntimeError(
                "cross-validation ESS-fraction gate failed immediately: "
                f"fold={fold_number}, value={predicted_fraction:.5f}, "
                f"required={arguments.minimum_cv_ess_fraction:.5f}"
            )

    model, mean, std = fit_model(
        theta,
        full_weight,
        lower,
        upper,
        draws=arguments.training_draws,
        components=arguments.components,
        regularization=arguments.regularization,
        jitter=arguments.jitter,
        maximum_iterations=arguments.maximum_iterations,
        seed=arguments.seed,
    )
    checkpoint = {
        "schema": ENSEMBLE_SCHEMA,
        "model": model,
        "mean": mean,
        "std": std,
        "prior_low": lower,
        "prior_high": upper,
        "beta": float(arguments.beta),
        "learned_weight": float(arguments.learned_weight),
        "prior_weight": float(arguments.prior_weight),
        "ancestor_weights": ancestor_weights,
        "ancestors": ancestors,
        "ancestor_sha256": ancestor_hashes,
        "seed": int(arguments.seed),
        "input_sha256": sha256(arguments.input),
        "input_ess": input_ess,
        "cross_validation": cross_validation,
        "configuration": {
            "dimensions": DIMENSIONS,
            "components": arguments.components,
            "covariance_type": "full",
            "regularization": arguments.regularization,
            "jitter_standard_deviations": arguments.jitter,
            "training_draws": arguments.training_draws,
            "cross_validation_draws": arguments.cross_validation_draws,
            "cross_validation_folds": arguments.cross_validation_folds,
            "maximum_iterations": arguments.maximum_iterations,
        },
        "fit_seconds": time.time() - started,
    }
    atomic_joblib(arguments.checkpoint, checkpoint)
    fractions = np.asarray(
        [entry["predicted_ess_fraction"] for entry in cross_validation],
        dtype=np.float64,
    )
    report = {
        key: value
        for key, value in checkpoint.items()
        if key
        not in {
            "model",
            "mean",
            "std",
            "prior_low",
            "prior_high",
            "ancestors",
        }
    }
    report["ancestor_weights"] = ancestor_weights.tolist()
    report.update(
        {
            "status": "PASS",
            "checkpoint": str(arguments.checkpoint.resolve()),
            "checkpoint_sha256": sha256(arguments.checkpoint),
            "minimum_cv_ess_fraction": float(fractions.min()),
            "median_cv_ess_fraction": float(np.median(fractions)),
            "final_iterations": int(model.n_iter_),
        }
    )
    arguments.checkpoint.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def sample_learned(
    checkpoint: dict[str, object], rows: int, seed: int
) -> np.ndarray:
    if rows == 0:
        return np.empty((0, DIMENSIONS), dtype=np.float64)
    model = checkpoint["model"]
    model.random_state = seed
    standardized, _ = model.sample(rows)
    return from_unconstrained(
        standardized * np.asarray(checkpoint["std"], dtype=np.float64)
        + np.asarray(checkpoint["mean"], dtype=np.float64),
        np.asarray(checkpoint["prior_low"], dtype=np.float64),
        np.asarray(checkpoint["prior_high"], dtype=np.float64),
    )


def sample_checkpoint(
    checkpoint: dict[str, object], rows: int, rng: np.random.Generator
) -> np.ndarray:
    if rows == 0:
        return np.empty((0, DIMENSIONS), dtype=np.float64)
    lower = np.asarray(checkpoint["prior_low"], dtype=np.float64)
    upper = np.asarray(checkpoint["prior_high"], dtype=np.float64)
    schema = checkpoint.get("schema")
    if schema == BASE_CHECKPOINT_SCHEMA:
        weights = np.asarray(
            [1.0 - float(checkpoint["defensive_fraction"]), float(checkpoint["defensive_fraction"])],
            dtype=np.float64,
        )
        assignment = rng.choice(2, size=rows, p=weights)
        output = np.empty((rows, DIMENSIONS), dtype=np.float64)
        learned_index = np.flatnonzero(assignment == 0)
        prior_index = np.flatnonzero(assignment == 1)
        output[learned_index] = sample_learned(
            checkpoint, len(learned_index), int(rng.integers(0, 2**31 - 1))
        )
        output[prior_index] = rng.uniform(
            lower, upper, size=(len(prior_index), DIMENSIONS)
        )
        return output
    if schema != ENSEMBLE_SCHEMA:
        raise RuntimeError(f"unrecognized checkpoint schema: {schema}")
    weights = np.concatenate(
        [
            np.asarray(
                [checkpoint["learned_weight"], checkpoint["prior_weight"]],
                dtype=np.float64,
            ),
            np.asarray(checkpoint["ancestor_weights"], dtype=np.float64),
        ]
    )
    assignment = rng.choice(len(weights), size=rows, p=weights)
    output = np.empty((rows, DIMENSIONS), dtype=np.float64)
    learned_index = np.flatnonzero(assignment == 0)
    prior_index = np.flatnonzero(assignment == 1)
    output[learned_index] = sample_learned(
        checkpoint, len(learned_index), int(rng.integers(0, 2**31 - 1))
    )
    output[prior_index] = rng.uniform(
        lower, upper, size=(len(prior_index), DIMENSIONS)
    )
    for number, ancestor in enumerate(checkpoint["ancestors"], start=2):
        index = np.flatnonzero(assignment == number)
        output[index] = sample_checkpoint(ancestor, len(index), rng)
    return output


def sample(arguments: argparse.Namespace) -> None:
    started = time.time()
    checkpoint = joblib.load(arguments.checkpoint)
    if checkpoint.get("schema") != ENSEMBLE_SCHEMA:
        raise RuntimeError("unrecognized clean ensemble checkpoint")
    lower = np.asarray(checkpoint["prior_low"], dtype=np.float64)
    upper = np.asarray(checkpoint["prior_high"], dtype=np.float64)
    prior_logq = -float(np.log(upper - lower).sum())
    rng = np.random.default_rng(arguments.seed)
    theta = sample_checkpoint(checkpoint, arguments.draws, rng)
    logq = checkpoint_density(checkpoint, theta, prior_logq)
    if not np.isfinite(theta).all() or not np.isfinite(logq).all():
        raise RuntimeError("non-finite proposal output")
    if not np.all((theta >= lower) & (theta <= upper)):
        raise RuntimeError("proposal draw outside canonical prior")
    atomic_savez(
        arguments.output,
        schema=np.asarray(PROPOSAL_SCHEMA),
        theta=theta,
        logq=logq,
        prior_low=lower,
        prior_high=upper,
        beta=np.float64(checkpoint["beta"]),
        seed=np.int64(arguments.seed),
        checkpoint_sha256=np.asarray(sha256(arguments.checkpoint)),
        source=np.asarray("full-9D normalized mode-preserving GMM ensemble"),
        learned_weight=np.float64(checkpoint["learned_weight"]),
        prior_weight=np.float64(checkpoint["prior_weight"]),
        ancestor_weights=np.asarray(checkpoint["ancestor_weights"], dtype=np.float64),
    )
    report = {
        "status": "PASS",
        "draws": arguments.draws,
        "beta": float(checkpoint["beta"]),
        "seed": arguments.seed,
        "learned_weight": float(checkpoint["learned_weight"]),
        "prior_weight": float(checkpoint["prior_weight"]),
        "ancestor_weights": np.asarray(
            checkpoint["ancestor_weights"], dtype=np.float64
        ).tolist(),
        "wall_seconds": time.time() - started,
        "checkpoint_sha256": sha256(arguments.checkpoint),
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "logq_min": float(logq.min()),
        "logq_max": float(logq.max()),
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prior", type=Path, default=Path(__file__).parent / "prior_manifest.json")
    commands = result.add_subparsers(dest="command", required=True)
    p_fit = commands.add_parser("fit")
    p_fit.add_argument("--input", type=Path, required=True)
    p_fit.add_argument("--checkpoint", type=Path, required=True)
    p_fit.add_argument("--beta", type=float, required=True)
    p_fit.add_argument("--ancestor-checkpoint", type=Path, action="append", required=True)
    p_fit.add_argument("--ancestor-weight", type=float, action="append", required=True)
    p_fit.add_argument("--learned-weight", type=float, required=True)
    p_fit.add_argument("--prior-weight", type=float, required=True)
    p_fit.add_argument("--components", type=int, default=64)
    p_fit.add_argument("--training-draws", type=int, default=300_000)
    p_fit.add_argument("--cross-validation-draws", type=int, default=220_000)
    p_fit.add_argument("--cross-validation-folds", type=int, default=5)
    p_fit.add_argument("--regularization", type=float, default=1.0e-3)
    p_fit.add_argument("--jitter", type=float, default=0.03)
    p_fit.add_argument("--minimum-input-ess", type=float, default=28_000.0)
    p_fit.add_argument("--minimum-cv-ess-fraction", type=float, default=0.10)
    p_fit.add_argument("--maximum-iterations", type=int, default=250)
    p_fit.add_argument("--seed", type=int, default=20260962)
    p_fit.set_defaults(function=fit)
    p_sample = commands.add_parser("sample")
    p_sample.add_argument("--checkpoint", type=Path, required=True)
    p_sample.add_argument("--output", type=Path, required=True)
    p_sample.add_argument("--draws", type=int, default=200_000)
    p_sample.add_argument("--seed", type=int, default=20260963)
    p_sample.set_defaults(function=sample)
    return result


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
