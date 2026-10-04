#!/usr/bin/env python3
"""Fit and sample a normalized, full-dimensional clean bridge proposal.

The proposal is a Gaussian mixture in a smooth transform of the canonical
nine-dimensional prior box, mixed with the unchanged prior for defensive
support.  It is trained only from a proposal's own exact-likelihood evaluation.
No comparison-sampler samples, bounds, evidence values, or stopping decisions
enter this construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture


ROOT = Path(__file__).resolve().parent
DIMENSIONS = 9
CHECKPOINT_SCHEMA = "ddb-hyperonic-clean-gmm-v1"
PROPOSAL_SCHEMA = "ddb-hyperonic-clean-proposal-v1"
EVALUATION_SCHEMA = "ddb-hyperonic-clean-eval-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_joblib(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_prior(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    lower = np.asarray(payload["lower"], dtype=np.float64)
    upper = np.asarray(payload["upper"], dtype=np.float64)
    if (
        lower.shape != (DIMENSIONS,)
        or upper.shape != (DIMENSIONS,)
        or np.any(lower >= upper)
    ):
        raise RuntimeError("invalid canonical prior manifest")
    return lower, upper


def to_unconstrained(
    theta: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    box = 2.0 * (theta - lower) / (upper - lower) - 1.0
    box = np.clip(box, -1.0 + 1.0e-9, 1.0 - 1.0e-9)
    return np.arctanh(box)


def from_unconstrained(
    value: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    theta = lower + 0.5 * (np.tanh(value) + 1.0) * (upper - lower)
    return np.clip(theta, np.nextafter(lower, upper), np.nextafter(upper, lower))


def log_box_jacobian(
    theta: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    box = np.clip(
        2.0 * (theta - lower) / (upper - lower) - 1.0,
        -1.0 + 1.0e-12,
        1.0 - 1.0e-12,
    )
    return (
        np.log(2.0 / (upper - lower))[None, :]
        - np.log1p(-(box * box))
    ).sum(axis=1)


def normalized_weights(logweight: np.ndarray) -> tuple[np.ndarray, float]:
    finite = np.isfinite(logweight)
    if not finite.any():
        raise RuntimeError("all importance weights are non-finite")
    shifted = np.full_like(logweight, -np.inf, dtype=np.float64)
    shifted[finite] = logweight[finite] - logsumexp(logweight[finite])
    weight = np.exp(shifted)
    return weight, float(1.0 / np.sum(weight * weight))


def fit_model(
    theta: np.ndarray,
    weight: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    draws: int,
    components: int,
    regularization: float,
    jitter: float,
    maximum_iterations: int,
    seed: int,
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    index = rng.choice(len(theta), size=draws, replace=True, p=weight)
    unconstrained = to_unconstrained(theta[index], lower, upper)
    mean = unconstrained.mean(axis=0)
    std = unconstrained.std(axis=0)
    if np.any(~np.isfinite(std)) or np.any(std <= 1.0e-8):
        raise RuntimeError("degenerate transformed training scale")
    standardized = (unconstrained - mean) / std
    if jitter:
        standardized += rng.normal(0.0, jitter, standardized.shape)
    model = GaussianMixture(
        n_components=components,
        covariance_type="full",
        reg_covar=regularization,
        max_iter=maximum_iterations,
        n_init=1,
        init_params="k-means++",
        random_state=seed,
    )
    model.fit(standardized)
    if not model.converged_:
        raise RuntimeError("Gaussian-mixture fit did not converge")
    return model, mean, std


def flow_log_density(
    model: GaussianMixture,
    mean: np.ndarray,
    std: np.ndarray,
    theta: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    unconstrained = to_unconstrained(theta, lower, upper)
    standardized = (unconstrained - mean) / std
    return (
        model.score_samples(standardized)
        - np.log(std).sum()
        + log_box_jacobian(theta, lower, upper)
    )


def mixture_log_density(
    learned_logq: np.ndarray,
    prior_logq: float,
    defensive_fraction: float,
) -> np.ndarray:
    if defensive_fraction == 0.0:
        return learned_logq
    return np.logaddexp(
        math.log1p(-defensive_fraction) + learned_logq,
        math.log(defensive_fraction) + prior_logq,
    )


def fit(arguments: argparse.Namespace) -> None:
    started = time.time()
    lower, upper = load_prior(arguments.prior)
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
    if not 0.0 < arguments.defensive_fraction < 1.0:
        raise ValueError("defensive fraction must lie strictly between zero and one")

    prior_logq = -float(np.log(upper - lower).sum())
    logtarget = prior_logq + arguments.beta * loglike
    full_weight, input_ess = normalized_weights(logtarget - source_logq)
    if input_ess < arguments.minimum_input_ess:
        raise RuntimeError(
            f"input ESS {input_ess:.1f} is below gate {arguments.minimum_input_ess:.1f}"
        )

    rng = np.random.default_rng(arguments.seed)
    permutation = rng.permutation(len(theta))
    folds = np.array_split(permutation, arguments.cross_validation_folds)
    cross_validation: list[dict[str, object]] = []
    for fold_number, validation_index in enumerate(folds):
        training_index = np.concatenate(
            [part for number, part in enumerate(folds) if number != fold_number]
        )
        training_weight, training_ess = normalized_weights(
            logtarget[training_index] - source_logq[training_index]
        )
        fold_seed = arguments.seed + 1000 + fold_number
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
            seed=fold_seed,
        )
        learned_logq = flow_log_density(
            model,
            mean,
            std,
            theta[validation_index],
            lower,
            upper,
        )
        candidate_logq = mixture_log_density(
            learned_logq, prior_logq, arguments.defensive_fraction
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
        cross_validation.append(
            {
                "fold": fold_number,
                "training_rows": int(len(training_index)),
                "validation_rows": int(len(validation_index)),
                "training_ess": training_ess,
                "validation_target_ess": validation_target_ess,
                "predicted_ess_fraction": predicted_fraction,
                "iterations": int(model.n_iter_),
                "converged": bool(model.converged_),
            }
        )
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

    fractions = np.asarray(
        [entry["predicted_ess_fraction"] for entry in cross_validation],
        dtype=np.float64,
    )
    if np.any(fractions < arguments.minimum_cv_ess_fraction):
        raise RuntimeError(
            "cross-validation ESS-fraction gate failed: "
            f"minimum={fractions.min():.5f}, "
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
        "schema": CHECKPOINT_SCHEMA,
        "model": model,
        "mean": mean,
        "std": std,
        "prior_low": lower,
        "prior_high": upper,
        "beta": float(arguments.beta),
        "defensive_fraction": float(arguments.defensive_fraction),
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
    report = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model", "mean", "std", "prior_low", "prior_high"}
    }
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


def sample(arguments: argparse.Namespace) -> None:
    started = time.time()
    checkpoint = joblib.load(arguments.checkpoint)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("unrecognized clean GMM checkpoint")
    model: GaussianMixture = checkpoint["model"]
    mean = np.asarray(checkpoint["mean"], dtype=np.float64)
    std = np.asarray(checkpoint["std"], dtype=np.float64)
    lower = np.asarray(checkpoint["prior_low"], dtype=np.float64)
    upper = np.asarray(checkpoint["prior_high"], dtype=np.float64)
    defensive_fraction = float(checkpoint["defensive_fraction"])
    prior_rows = int(round(arguments.draws * defensive_fraction))
    learned_rows = arguments.draws - prior_rows

    model.random_state = arguments.seed
    standardized, _ = model.sample(learned_rows)
    theta = np.empty((arguments.draws, DIMENSIONS), dtype=np.float64)
    theta[:learned_rows] = from_unconstrained(
        standardized * std + mean, lower, upper
    )
    rng = np.random.default_rng(arguments.seed + 1)
    theta[learned_rows:] = rng.uniform(
        lower, upper, size=(prior_rows, DIMENSIONS)
    )
    component = np.zeros(arguments.draws, dtype=np.uint8)
    component[learned_rows:] = 1
    permutation = rng.permutation(arguments.draws)
    theta = theta[permutation]
    component = component[permutation]

    learned_logq = flow_log_density(model, mean, std, theta, lower, upper)
    prior_logq = -float(np.log(upper - lower).sum())
    logq = mixture_log_density(learned_logq, prior_logq, defensive_fraction)
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
        source=np.asarray("full-9D normalized defensive Gaussian mixture"),
        proposal_component=component,
        defensive_fraction=np.float64(defensive_fraction),
    )
    report = {
        "status": "PASS",
        "draws": arguments.draws,
        "beta": checkpoint["beta"],
        "seed": arguments.seed,
        "defensive_fraction": defensive_fraction,
        "defensive_rows": prior_rows,
        "learned_rows": learned_rows,
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
    result.add_argument("--prior", type=Path, default=ROOT / "prior_manifest.json")
    commands = result.add_subparsers(dest="command", required=True)

    p_fit = commands.add_parser("fit")
    p_fit.add_argument("--input", type=Path, required=True)
    p_fit.add_argument("--checkpoint", type=Path, required=True)
    p_fit.add_argument("--beta", type=float, required=True)
    p_fit.add_argument("--components", type=int, default=16)
    p_fit.add_argument("--training-draws", type=int, default=160_000)
    p_fit.add_argument("--cross-validation-draws", type=int, default=120_000)
    p_fit.add_argument("--cross-validation-folds", type=int, default=5)
    p_fit.add_argument("--regularization", type=float, default=1.0e-3)
    p_fit.add_argument("--jitter", type=float, default=0.05)
    p_fit.add_argument("--defensive-fraction", type=float, default=0.15)
    p_fit.add_argument("--minimum-input-ess", type=float, default=1000.0)
    p_fit.add_argument("--minimum-cv-ess-fraction", type=float, default=0.10)
    p_fit.add_argument("--maximum-iterations", type=int, default=250)
    p_fit.add_argument("--seed", type=int, default=20260927)
    p_fit.set_defaults(function=fit)

    p_sample = commands.add_parser("sample")
    p_sample.add_argument("--checkpoint", type=Path, required=True)
    p_sample.add_argument("--output", type=Path, required=True)
    p_sample.add_argument("--draws", type=int, default=60_000)
    p_sample.add_argument("--seed", type=int, default=20260928)
    p_sample.set_defaults(function=sample)
    return result


def main() -> None:
    arguments = parser().parse_args()
    if hasattr(arguments, "beta") and not 0.0 < arguments.beta <= 1.0:
        raise ValueError("beta must lie in (0, 1]")
    arguments.function(arguments)


if __name__ == "__main__":
    main()
