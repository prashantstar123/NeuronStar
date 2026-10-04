#!/usr/bin/env python3
"""Freeze a held-out-certified GMM bridge for the clean 9D TSNPE proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.mixture import GaussianMixture


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def load_stage(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    report = json.loads(path.with_suffix(".json").read_text())
    digest = sha256(path)
    if (
        report.get("lineage_class") != "independent_tsnpe"
        or report.get("status") not in {"PASS", "FAIL_ESS_GATE"}
        or report.get("target_gate", {}).get("status") != "PASS"
        or report.get("forbidden_artifacts_used")
        or report.get("output_sha256") != digest
    ):
        raise RuntimeError(f"cross-fit stage failed clean gates: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {
            key: np.asarray(archive[key], dtype=np.float64)
            for key in (
                "theta",
                "normalized_weight",
                "log_target",
                "log_proposal",
                "prior_low",
                "prior_high",
            )
        }
        if np.asarray(archive["forbidden_artifacts_used"]).size:
            raise RuntimeError(f"cross-fit stage declares forbidden ancestry: {path}")
    return report, data


def transform(theta, low, high, mean, scale):
    unit = (theta - low) / (high - low)
    if not np.all((unit > 0.0) & (unit < 1.0)):
        raise RuntimeError("bounded transform received a boundary row")
    value = (np.log(unit) - np.log1p(-unit) - mean) / scale
    jacobian = -np.log(scale).sum() - np.sum(
        np.log(high - low) + np.log(unit) + np.log1p(-unit), axis=1
    )
    return value, jacobian


def t_log_density(value, location, shape, degrees):
    dimension = value.shape[1]
    sign, logdet = np.linalg.slogdet(shape)
    if sign <= 0:
        raise RuntimeError("Student-t shape is not positive definite")
    delta = value - location
    quadratic = np.einsum("ni,ij,nj->n", delta, np.linalg.inv(shape), delta)
    constant = (
        gammaln((degrees + dimension) / 2.0)
        - gammaln(degrees / 2.0)
        - 0.5 * (dimension * np.log(degrees * np.pi) + logdet)
    )
    return constant - 0.5 * (degrees + dimension) * np.log1p(quadratic / degrees)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-rows", type=int, default=150_000)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=50_000)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--reg-covar", type=float, default=0.01)
    parser.add_argument("--gmm-fraction", type=float, default=0.80)
    parser.add_argument("--broad-fraction", type=float, default=0.15)
    parser.add_argument("--uniform-fraction", type=float, default=0.05)
    parser.add_argument("--parent-tsnpe-fraction", type=float, default=0.10)
    parser.add_argument("--broad-degrees", type=float, default=8.0)
    parser.add_argument("--broad-scale", type=float, default=1.3)
    parser.add_argument("--minimum-worst-predicted-ess", type=float, default=3_000.0)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")
    repair_fractions = np.asarray(
        [arguments.gmm_fraction, arguments.broad_fraction, arguments.uniform_fraction],
        dtype=np.float64,
    )
    if (
        arguments.components < 1
        or arguments.resamples < 1
        or arguments.candidate_rows < 1
        or not np.isclose(repair_fractions.sum(), 1.0)
        or np.any(repair_fractions <= 0.0)
        or not 0.0 < arguments.parent_tsnpe_fraction < 1.0
    ):
        raise ValueError("invalid bridge configuration")

    train_report, train = load_stage(arguments.train)
    if train_report.get("role") != "tsnpe_crossvalidation_training_stage":
        raise RuntimeError("training artifact has the wrong role")
    low = train["prior_low"]
    high = train["prior_high"]
    weight = train["normalized_weight"]
    unit = np.clip((train["theta"] - low) / (high - low), 1.0e-8, 1.0 - 1.0e-8)
    logit = np.log(unit) - np.log1p(-unit)
    mean = np.sum(weight[:, None] * logit, axis=0)
    variance = np.sum(weight[:, None] * (logit - mean) ** 2, axis=0)
    scale = np.sqrt(np.maximum(variance, 1.0e-6))
    standardized = (logit - mean) / scale
    rng = np.random.default_rng(arguments.seed)
    index = rng.choice(
        len(standardized), size=arguments.resamples, replace=True, p=weight
    )
    fit_rows = standardized[index] + rng.normal(
        0.0, arguments.jitter, size=(arguments.resamples, 9)
    )
    model = GaussianMixture(
        n_components=arguments.components,
        covariance_type="full",
        reg_covar=arguments.reg_covar,
        max_iter=300,
        n_init=1,
        random_state=arguments.seed + arguments.components,
    ).fit(fit_rows)
    if not model.converged_:
        raise RuntimeError("cross-validated GMM bridge did not converge")

    covariance = np.cov(fit_rows - fit_rows.mean(axis=0), rowvar=False)
    covariance = 0.9 * covariance + 0.1 * np.diag(np.diag(covariance))
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    covariance = (eigenvector * np.maximum(eigenvalue, 0.02)) @ eigenvector.T
    broad_location = fit_rows.mean(axis=0)
    broad_shape = covariance * arguments.broad_scale**2 * (
        (arguments.broad_degrees - 2.0) / arguments.broad_degrees
    )
    uniform_logq = -float(np.log(high - low).sum())
    parent_fraction = float(arguments.parent_tsnpe_fraction)
    validation_diagnostics = []
    for path in arguments.validation:
        report, stage = load_stage(path)
        if (
            report.get("role") != "tsnpe_crossvalidation_validation_stage"
            or not np.array_equal(stage["prior_low"], low)
            or not np.array_equal(stage["prior_high"], high)
        ):
            raise RuntimeError(f"validation artifact has the wrong role/prior: {path}")
        value, jacobian = transform(stage["theta"], low, high, mean, scale)
        logq_gmm = model.score_samples(value) + jacobian
        logq_broad = t_log_density(
            value, broad_location, broad_shape, arguments.broad_degrees
        ) + jacobian
        logq_uniform = np.full(len(value), uniform_logq, dtype=np.float64)
        logq_repair = logsumexp(
            np.column_stack(
                [
                    np.log(repair_fractions[0]) + logq_gmm,
                    np.log(repair_fractions[1]) + logq_broad,
                    np.log(repair_fractions[2]) + logq_uniform,
                ]
            ),
            axis=1,
        )
        logq_final = logsumexp(
            np.column_stack(
                [
                    np.log1p(-parent_fraction) + logq_repair,
                    np.log(parent_fraction) + stage["log_proposal"],
                ]
            ),
            axis=1,
        )
        valid = (
            np.isfinite(stage["log_target"])
            & np.isfinite(stage["log_proposal"])
            & np.isfinite(logq_final)
        )
        log_second = float(
            logsumexp(
                2.0 * stage["log_target"][valid]
                - logq_final[valid]
                - stage["log_proposal"][valid]
            )
            - np.log(len(stage["theta"]))
        )
        fraction = float(
            np.exp(2.0 * float(report["log_evidence"]) - log_second)
        )
        validation_diagnostics.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "fold": report.get("validation_fold"),
                "finite_rows": int(valid.sum()),
                "predicted_ess_fraction": fraction,
                "predicted_candidate_ess": fraction * arguments.candidate_rows,
            }
        )
    worst = min(item["predicted_candidate_ess"] for item in validation_diagnostics)
    if worst < arguments.minimum_worst_predicted_ess:
        raise RuntimeError(
            f"held-out bridge forecast {worst:.3f} is below "
            f"the required {arguments.minimum_worst_predicted_ess:.3f}"
        )

    metadata = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_crossvalidated_gmm_defensive_bridge",
        "model": "ddb-hyperonic",
        "full_prior_dimension": 9,
        "training_stage_sha256": sha256(arguments.train),
        "validation_stage_sha256": [sha256(path) for path in arguments.validation],
        "forbidden_artifacts_used": [],
    }
    atomic_save(
        arguments.output,
        gmm_weights=np.asarray(model.weights_, dtype=np.float64),
        gmm_means=np.asarray(model.means_, dtype=np.float64),
        gmm_covariances=np.asarray(model.covariances_, dtype=np.float64),
        transform_mean=mean,
        transform_scale=scale,
        broad_location=broad_location,
        broad_shape=broad_shape,
        broad_degrees=np.asarray(arguments.broad_degrees, dtype=np.float64),
        repair_fractions=repair_fractions,
        parent_tsnpe_fraction=np.asarray(parent_fraction, dtype=np.float64),
        prior_low=low,
        prior_high=high,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    report = {
        **metadata,
        "training_stage": {
            "path": str(arguments.train.resolve()),
            "sha256": sha256(arguments.train),
        },
        "validation_stages": validation_diagnostics,
        "candidate_rows": arguments.candidate_rows,
        "gmm_components": arguments.components,
        "gmm_converged": bool(model.converged_),
        "gmm_iterations": int(model.n_iter_),
        "resamples": arguments.resamples,
        "jitter": arguments.jitter,
        "reg_covar": arguments.reg_covar,
        "repair_fractions_gmm_broad_uniform": repair_fractions.tolist(),
        "parent_tsnpe_fraction": parent_fraction,
        "worst_held_out_predicted_ess": worst,
        "minimum_worst_predicted_ess": arguments.minimum_worst_predicted_ess,
        "seed": arguments.seed,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
