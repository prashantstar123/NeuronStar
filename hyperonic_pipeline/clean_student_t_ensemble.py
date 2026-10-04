#!/usr/bin/env python3
"""Fit and sample a defensive heavy-tailed bridge proposal.

The learned component is a full-dimensional mixture of multivariate Student-t
kernels in the same smooth transform of the canonical prior box used by the
clean GMM bridge.  Component locations, weights, and scale geometry are fitted
without comparison-sampler information.  The learned density is mixed with a
canonical-prior defensive component and earlier normalized clean proposals.
Every density is analytic, so fresh rows can receive the exact full mixture
importance correction.

The Student-t mixture uses a Gaussian-mixture backbone only to estimate mode
locations and local scale matrices.  Replacing its Gaussian kernels by
heavy-tailed Student-t kernels changes the normalized proposal, not the target.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import gammaln, logsumexp

from clean_gmm import (
    CHECKPOINT_SCHEMA as GMM_BASE_SCHEMA,
    DIMENSIONS,
    EVALUATION_SCHEMA,
    PROPOSAL_SCHEMA,
    atomic_joblib,
    atomic_savez,
    fit_model,
    from_unconstrained,
    load_prior,
    log_box_jacobian,
    normalized_weights,
    sha256,
    to_unconstrained,
)
from clean_gmm_ensemble import (
    ENSEMBLE_SCHEMA as GMM_ENSEMBLE_SCHEMA,
    checkpoint_density as gmm_checkpoint_density,
    sample_checkpoint as sample_gmm_checkpoint,
    validate_weights,
)


STUDENT_ENSEMBLE_SCHEMA = "ddb-hyperonic-clean-student-t-ensemble-v1"
ANET_CERTIFICATION_SCHEMA = "ddb-hyperonic-clean-anet-certification-v1"


def standardized_student_mixture_log_density(
    model: object,
    value: np.ndarray,
    degrees_of_freedom: float,
    shape_scale: float,
    *,
    batch_size: int = 80_000,
) -> np.ndarray:
    """Log density of a Student-t mixture in standardized coordinates."""

    if degrees_of_freedom <= 2.0:
        raise ValueError("Student-t degrees of freedom must exceed two")
    if shape_scale <= 0.0:
        raise ValueError("Student-t shape scale must be positive")
    value = np.asarray(value, dtype=np.float64)
    weights = np.asarray(model.weights_, dtype=np.float64)
    means = np.asarray(model.means_, dtype=np.float64)
    covariance = np.asarray(model.covariances_, dtype=np.float64) * shape_scale
    if (
        value.ndim != 2
        or value.shape[1] != DIMENSIONS
        or means.shape != (len(weights), DIMENSIONS)
        or covariance.shape != (len(weights), DIMENSIONS, DIMENSIONS)
    ):
        raise RuntimeError("invalid Student-t mixture arrays")
    precision = np.linalg.inv(covariance)
    sign, logdet = np.linalg.slogdet(covariance)
    if np.any(sign <= 0.0) or not np.isfinite(logdet).all():
        raise RuntimeError("Student-t component shape is not positive definite")
    constant = (
        gammaln(0.5 * (degrees_of_freedom + DIMENSIONS))
        - gammaln(0.5 * degrees_of_freedom)
        - 0.5 * DIMENSIONS * math.log(degrees_of_freedom * math.pi)
        - 0.5 * logdet
        + np.log(weights)
    )
    output = np.empty(len(value), dtype=np.float64)
    for start in range(0, len(value), batch_size):
        stop = min(start + batch_size, len(value))
        local = value[start:stop]
        component_logq = np.empty((len(local), len(weights)), dtype=np.float64)
        for component in range(len(weights)):
            difference = local - means[component]
            mahalanobis = np.einsum(
                "ni,ij,nj->n",
                difference,
                precision[component],
                difference,
                optimize=True,
            )
            component_logq[:, component] = constant[component] - 0.5 * (
                degrees_of_freedom + DIMENSIONS
            ) * np.log1p(mahalanobis / degrees_of_freedom)
        output[start:stop] = logsumexp(component_logq, axis=1)
    return output


def learned_log_density(
    model: object,
    mean: np.ndarray,
    std: np.ndarray,
    theta: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    degrees_of_freedom: float,
    shape_scale: float,
) -> np.ndarray:
    unconstrained = to_unconstrained(theta, lower, upper)
    standardized = (unconstrained - mean) / std
    return (
        standardized_student_mixture_log_density(
            model, standardized, degrees_of_freedom, shape_scale
        )
        - np.log(std).sum()
        + log_box_jacobian(theta, lower, upper)
    )


def checkpoint_density(
    checkpoint: dict[str, object], theta: np.ndarray, prior_logq: float
) -> np.ndarray:
    schema = checkpoint.get("schema")
    if schema in {GMM_BASE_SCHEMA, GMM_ENSEMBLE_SCHEMA}:
        return gmm_checkpoint_density(checkpoint, theta, prior_logq)
    if schema != STUDENT_ENSEMBLE_SCHEMA:
        raise RuntimeError(f"unrecognized proposal checkpoint schema: {schema}")
    lower = np.asarray(checkpoint["prior_low"], dtype=np.float64)
    upper = np.asarray(checkpoint["prior_high"], dtype=np.float64)
    learned = learned_log_density(
        checkpoint["model"],
        np.asarray(checkpoint["mean"], dtype=np.float64),
        np.asarray(checkpoint["std"], dtype=np.float64),
        theta,
        lower,
        upper,
        float(checkpoint["degrees_of_freedom"]),
        float(checkpoint["shape_scale"]),
    )
    terms = [math.log(float(checkpoint["learned_weight"])) + learned]
    prior_weight = float(checkpoint["prior_weight"])
    if prior_weight:
        terms.append(np.full(len(theta), math.log(prior_weight) + prior_logq))
    for weight, ancestor in zip(
        np.asarray(checkpoint["ancestor_weights"], dtype=np.float64),
        list(checkpoint["ancestors"]),
        strict=True,
    ):
        terms.append(
            math.log(float(weight))
            + checkpoint_density(ancestor, theta, prior_logq)
        )
    return logsumexp(np.vstack(terms), axis=0)


def candidate_density(
    learned: np.ndarray,
    ancestor_density: list[np.ndarray],
    prior_logq: float,
    learned_weight: float,
    ancestor_weights: np.ndarray,
    prior_weight: float,
) -> np.ndarray:
    terms = [math.log(learned_weight) + learned]
    terms.extend(
        math.log(float(weight)) + density
        for weight, density in zip(
            ancestor_weights, ancestor_density, strict=True
        )
    )
    if prior_weight:
        terms.append(np.full(len(learned), math.log(prior_weight) + prior_logq))
    return logsumexp(np.vstack(terms), axis=0)


def load_ancestors(
    paths: list[Path], lower: np.ndarray, upper: np.ndarray
) -> tuple[list[dict[str, object]], list[str]]:
    accepted = {GMM_BASE_SCHEMA, GMM_ENSEMBLE_SCHEMA, STUDENT_ENSEMBLE_SCHEMA}
    checkpoints: list[dict[str, object]] = []
    hashes: list[str] = []
    for path in paths:
        checkpoint = joblib.load(path)
        if checkpoint.get("schema") not in accepted:
            raise RuntimeError(f"unrecognized ancestor checkpoint: {path}")
        if not np.array_equal(checkpoint["prior_low"], lower) or not np.array_equal(
            checkpoint["prior_high"], upper
        ):
            raise RuntimeError(f"ancestor prior differs: {path}")
        checkpoints.append(checkpoint)
        hashes.append(sha256(path))
    return checkpoints, hashes


def predicted_ess_fraction(
    logtarget: np.ndarray,
    source_logq: np.ndarray,
    candidate_logq: np.ndarray,
) -> float:
    log_normalization = logsumexp(logtarget - source_logq) - math.log(len(logtarget))
    log_second_moment = (
        logsumexp(2.0 * logtarget - candidate_logq - source_logq)
        - math.log(len(logtarget))
        - 2.0 * log_normalization
    )
    return float(math.exp(-log_second_moment))


def fit(arguments: argparse.Namespace) -> None:
    started = time.time()
    lower, upper = load_prior(arguments.prior)
    ancestor_paths = arguments.ancestor_checkpoint or []
    ancestor_weights = np.asarray(arguments.ancestor_weight or [], dtype=np.float64)
    if len(ancestor_paths) != len(ancestor_weights):
        raise ValueError("one --ancestor-weight is required per ancestor checkpoint")
    validate_weights(arguments.learned_weight, ancestor_weights, arguments.prior_weight)
    ancestors, ancestor_hashes = load_ancestors(
        ancestor_paths, lower, upper
    )
    with np.load(arguments.input, allow_pickle=False) as source:
        source_schema = str(source["schema"].item())
        if source_schema not in {EVALUATION_SCHEMA, ANET_CERTIFICATION_SCHEMA}:
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
    if arguments.degrees_of_freedom <= 2.0 or arguments.shape_scale <= 0.0:
        raise ValueError("invalid Student-t degrees of freedom or shape scale")

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
    source_weight_grid = np.asarray(arguments.source_weight_grid, dtype=np.float64)
    if np.any((source_weight_grid <= 0.0) | (source_weight_grid >= 1.0)):
        raise ValueError("source-proposal mixture weights must lie strictly between zero and one")
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
        learned = learned_log_density(
            model,
            mean,
            std,
            theta[validation_index],
            lower,
            upper,
            arguments.degrees_of_freedom,
            arguments.shape_scale,
        )
        candidate_logq = candidate_density(
            learned,
            [density[validation_index] for density in ancestor_logq_full],
            prior_logq,
            arguments.learned_weight,
            ancestor_weights,
            arguments.prior_weight,
        )
        local_logtarget = logtarget[validation_index]
        local_source_logq = source_logq[validation_index]
        fraction = predicted_ess_fraction(
            local_logtarget, local_source_logq, candidate_logq
        )
        source_mixture_fraction: dict[str, float] = {}
        for source_weight in source_weight_grid:
            mixture_logq = np.logaddexp(
                math.log(float(source_weight)) + local_source_logq,
                math.log1p(-float(source_weight)) + candidate_logq,
            )
            source_mixture_fraction[f"{source_weight:.8g}"] = predicted_ess_fraction(
                local_logtarget, local_source_logq, mixture_logq
            )
        _, validation_target_ess = normalized_weights(
            local_logtarget - local_source_logq
        )
        record = {
            "fold": fold_number,
            "training_rows": int(len(training_index)),
            "validation_rows": int(len(validation_index)),
            "training_ess": training_ess,
            "validation_target_ess": validation_target_ess,
            "predicted_ess_fraction": fraction,
            "source_mixture_predicted_ess_fraction": source_mixture_fraction,
            "iterations": int(model.n_iter_),
            "converged": bool(model.converged_),
        }
        cross_validation.append(record)
        print(
            f"fold={fold_number} training_ess={training_ess:.1f} "
            f"validation_target_ess={validation_target_ess:.1f} "
            f"predicted_ess_fraction={fraction:.5f}",
            flush=True,
        )
        if fraction < arguments.minimum_cv_ess_fraction:
            raise RuntimeError(
                "Student-t cross-validation ESS-fraction gate failed: "
                f"fold={fold_number}, value={fraction:.5f}, "
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
        "schema": STUDENT_ENSEMBLE_SCHEMA,
        "model": model,
        "mean": mean,
        "std": std,
        "prior_low": lower,
        "prior_high": upper,
        "beta": float(arguments.beta),
        "degrees_of_freedom": float(arguments.degrees_of_freedom),
        "shape_scale": float(arguments.shape_scale),
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
            "kernel": "multivariate_student_t",
            "degrees_of_freedom": arguments.degrees_of_freedom,
            "shape_scale_relative_to_gaussian_backbone": arguments.shape_scale,
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
        if key not in {"model", "mean", "std", "prior_low", "prior_high", "ancestors"}
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
            "source_mixture_cross_validation": {
                f"{source_weight:.8g}": {
                    "minimum_predicted_ess_fraction": float(
                        min(
                            fold["source_mixture_predicted_ess_fraction"][f"{source_weight:.8g}"]
                            for fold in cross_validation
                        )
                    ),
                    "median_predicted_ess_fraction": float(
                        np.median(
                            [
                                fold["source_mixture_predicted_ess_fraction"][f"{source_weight:.8g}"]
                                for fold in cross_validation
                            ]
                        )
                    ),
                }
                for source_weight in source_weight_grid
            },
        }
    )
    arguments.checkpoint.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def sample_learned(
    checkpoint: dict[str, object], rows: int, rng: np.random.Generator
) -> np.ndarray:
    if rows == 0:
        return np.empty((0, DIMENSIONS), dtype=np.float64)
    model = checkpoint["model"]
    weights = np.asarray(model.weights_, dtype=np.float64)
    means = np.asarray(model.means_, dtype=np.float64)
    shape = (
        np.asarray(model.covariances_, dtype=np.float64)
        * float(checkpoint["shape_scale"])
    )
    degrees_of_freedom = float(checkpoint["degrees_of_freedom"])
    assignment = rng.choice(len(weights), size=rows, p=weights)
    standardized = np.empty((rows, DIMENSIONS), dtype=np.float64)
    for component in range(len(weights)):
        index = np.flatnonzero(assignment == component)
        if not len(index):
            continue
        normal = rng.multivariate_normal(
            np.zeros(DIMENSIONS), shape[component], size=len(index)
        )
        radial = np.sqrt(
            rng.chisquare(degrees_of_freedom, size=len(index))
            / degrees_of_freedom
        )
        standardized[index] = means[component] + normal / radial[:, None]
    unconstrained = (
        standardized * np.asarray(checkpoint["std"], dtype=np.float64)
        + np.asarray(checkpoint["mean"], dtype=np.float64)
    )
    return from_unconstrained(
        unconstrained,
        np.asarray(checkpoint["prior_low"], dtype=np.float64),
        np.asarray(checkpoint["prior_high"], dtype=np.float64),
    )


def sample_checkpoint(
    checkpoint: dict[str, object], rows: int, rng: np.random.Generator
) -> np.ndarray:
    if rows == 0:
        return np.empty((0, DIMENSIONS), dtype=np.float64)
    schema = checkpoint.get("schema")
    if schema in {GMM_BASE_SCHEMA, GMM_ENSEMBLE_SCHEMA}:
        return sample_gmm_checkpoint(checkpoint, rows, rng)
    if schema != STUDENT_ENSEMBLE_SCHEMA:
        raise RuntimeError(f"unrecognized checkpoint schema: {schema}")
    lower = np.asarray(checkpoint["prior_low"], dtype=np.float64)
    upper = np.asarray(checkpoint["prior_high"], dtype=np.float64)
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
    output[learned_index] = sample_learned(checkpoint, len(learned_index), rng)
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
    if checkpoint.get("schema") != STUDENT_ENSEMBLE_SCHEMA:
        raise RuntimeError("unrecognized clean Student-t checkpoint")
    lower = np.asarray(checkpoint["prior_low"], dtype=np.float64)
    upper = np.asarray(checkpoint["prior_high"], dtype=np.float64)
    prior_logq = -float(np.log(upper - lower).sum())
    rng = np.random.default_rng(arguments.seed)
    theta = sample_checkpoint(checkpoint, arguments.draws, rng)
    logq = checkpoint_density(checkpoint, theta, prior_logq)
    if not np.isfinite(theta).all() or not np.isfinite(logq).all():
        raise RuntimeError("non-finite Student-t proposal output")
    if not np.all((theta >= lower) & (theta <= upper)):
        raise RuntimeError("Student-t proposal draw outside canonical prior")
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
        source=np.asarray("full-9D normalized defensive Student-t ensemble"),
        learned_weight=np.float64(checkpoint["learned_weight"]),
        prior_weight=np.float64(checkpoint["prior_weight"]),
        ancestor_weights=np.asarray(checkpoint["ancestor_weights"], dtype=np.float64),
    )
    report = {
        "status": "PASS",
        "draws": arguments.draws,
        "beta": float(checkpoint["beta"]),
        "degrees_of_freedom": float(checkpoint["degrees_of_freedom"]),
        "shape_scale": float(checkpoint["shape_scale"]),
        "seed": arguments.seed,
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
    result.add_argument(
        "--prior", type=Path, default=Path(__file__).parent / "prior_manifest.json"
    )
    commands = result.add_subparsers(dest="command", required=True)
    p_fit = commands.add_parser("fit")
    p_fit.add_argument("--input", type=Path, required=True)
    p_fit.add_argument("--checkpoint", type=Path, required=True)
    p_fit.add_argument("--beta", type=float, required=True)
    p_fit.add_argument("--ancestor-checkpoint", type=Path, action="append")
    p_fit.add_argument("--ancestor-weight", type=float, action="append")
    p_fit.add_argument("--learned-weight", type=float, required=True)
    p_fit.add_argument("--prior-weight", type=float, required=True)
    p_fit.add_argument("--components", type=int, default=48)
    p_fit.add_argument("--degrees-of-freedom", type=float, default=5.0)
    p_fit.add_argument("--shape-scale", type=float, default=0.75)
    p_fit.add_argument("--training-draws", type=int, default=500_000)
    p_fit.add_argument("--cross-validation-draws", type=int, default=380_000)
    p_fit.add_argument("--cross-validation-folds", type=int, default=5)
    p_fit.add_argument("--regularization", type=float, default=1.0e-3)
    p_fit.add_argument("--jitter", type=float, default=0.025)
    p_fit.add_argument("--minimum-input-ess", type=float, default=59_999.0)
    p_fit.add_argument("--minimum-cv-ess-fraction", type=float, default=0.10)
    p_fit.add_argument(
        "--source-weight-grid",
        type=float,
        action="append",
        default=[],
        help=(
            "Optional weights for the fixed normalized source proposal in a "
            "source-plus-Student-t defensive mixture. Cross-validation reports "
            "the predicted ESS fraction for every supplied weight."
        ),
    )
    p_fit.add_argument("--maximum-iterations", type=int, default=300)
    p_fit.add_argument("--seed", type=int, default=20260976)
    p_fit.set_defaults(function=fit)
    p_sample = commands.add_parser("sample")
    p_sample.add_argument("--checkpoint", type=Path, required=True)
    p_sample.add_argument("--output", type=Path, required=True)
    p_sample.add_argument("--draws", type=int, default=200_000)
    p_sample.add_argument("--seed", type=int, default=20260977)
    p_sample.set_defaults(function=sample)
    return result


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
