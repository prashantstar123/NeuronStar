#!/usr/bin/env python3
"""Generate a bounded adaptive MIS proposal from clean exact TSNPE weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, gammaln
from sklearn.mixture import GaussianMixture


A1_OBSERVATION = np.asarray(
    [0.153, -16.1, 230.0, 32.5, 0.505714285714279,
     1.24142857142857, 2.4857142857143], dtype=np.float64
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_standardized_logit(theta, low, high, mean, scale):
    unit = (theta - low) / (high - low)
    if not np.all((unit > 0.0) & (unit < 1.0)):
        raise RuntimeError("bounded transform received a boundary or out-of-prior row")
    logit = np.log(unit) - np.log1p(-unit)
    return (logit - mean) / scale, unit


def from_standardized_logit(value, low, high, mean, scale):
    unit = expit(value * scale + mean)
    theta = low + (high - low) * unit
    if not np.all((theta > low) & (theta < high)):
        raise RuntimeError("adaptive proposal generated a boundary row")
    return theta


def log_jacobian(unit, low, high, scale):
    return -np.log(scale).sum() - np.sum(
        np.log(high - low) + np.log(unit) + np.log1p(-unit), axis=1
    )


def multivariate_t_log_density(value, location, shape, degrees):
    dimension = value.shape[1]
    sign, logdet = np.linalg.slogdet(shape)
    if sign <= 0:
        raise RuntimeError("Student-t shape matrix is not positive definite")
    difference = value - location
    mahalanobis = np.einsum(
        "ni,ij,nj->n", difference, np.linalg.inv(shape), difference
    )
    constant = (
        gammaln((degrees + dimension) / 2.0)
        - gammaln(degrees / 2.0)
        - 0.5 * (dimension * np.log(degrees * np.pi) + logdet)
    )
    return constant - 0.5 * (degrees + dimension) * np.log1p(
        mahalanobis / degrees
    )


def flow_raw_log_density(estimator, prior, observed, theta, device, batch_size):
    from sbi.inference.posteriors import DirectPosterior

    posterior = DirectPosterior(estimator, prior=prior, device=str(device))
    values = []
    with torch.no_grad():
        for start in range(0, len(theta), batch_size):
            batch = torch.as_tensor(
                theta[start : start + batch_size], dtype=torch.float32, device=device
            )
            values.append(
                posterior.log_prob(batch, x=observed, norm_posterior=False)
                .detach().cpu().numpy()
            )
    return np.concatenate(values).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-one", type=Path, required=True)
    parser.add_argument("--first-run-dir", type=Path, required=True)
    parser.add_argument("--second-run-dir", type=Path, required=True)
    parser.add_argument("--density-calibration", type=Path, required=True)
    parser.add_argument("--mild-draws", type=int, default=20_000)
    parser.add_argument("--broad-draws", type=int, default=10_000)
    parser.add_argument("--gmm-components", type=int, default=4)
    parser.add_argument("--fit-resamples", type=int, default=50_000)
    parser.add_argument("--fit-jitter", type=float, default=0.03)
    parser.add_argument("--gmm-regularization", type=float, default=0.02)
    parser.add_argument("--broad-degrees", type=float, default=4.0)
    parser.add_argument("--broad-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--density-batch", type=int, default=4_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")
    if min(arguments.mild_draws, arguments.broad_draws, arguments.fit_resamples) < 1:
        raise ValueError("proposal and fit row counts must be positive")

    from sbi.utils import BoxUniform

    stage_report = json.loads(arguments.stage_one.with_suffix(".json").read_text())
    if (
        stage_report.get("status") != "FAIL_ESS_GATE"
        or stage_report.get("lineage_class") != "independent_tsnpe"
        or stage_report.get("role") != "tsnpe_full_prior_defensive_mis_posterior"
        or stage_report.get("forbidden_artifacts_used")
        or stage_report.get("target_gate", {}).get("status") != "PASS"
        or stage_report.get("output_sha256") != sha256(arguments.stage_one)
    ):
        raise RuntimeError("stage-one diagnostic failed clean exact-target gates")
    calibration = json.loads(arguments.density_calibration.read_text())
    if (
        calibration.get("status") != "PASS"
        or calibration.get("lineage_class") != "independent_tsnpe"
        or calibration.get("forbidden_artifacts_used")
        or calibration.get("candidate_sha256")
        != stage_report.get("parent_candidate_sha256")
    ):
        raise RuntimeError("flow-density calibration failed lineage gates")

    with np.load(arguments.stage_one, allow_pickle=False) as archive:
        theta_old = np.asarray(archive["theta"], dtype=np.float64)
        weight = np.asarray(archive["normalized_weight"], dtype=np.float64)
        low = np.asarray(archive["prior_low"], dtype=np.float64)
        high = np.asarray(archive["prior_high"], dtype=np.float64)
    if theta_old.shape[1:] != (9,) or weight.shape != (len(theta_old),):
        raise RuntimeError("stage-one posterior arrays have wrong shapes")
    if not np.isclose(weight.sum(), 1.0, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("stage-one normalized weights do not sum to one")

    unit_old = np.clip((theta_old - low) / (high - low), 1.0e-8, 1.0 - 1.0e-8)
    logit_old = np.log(unit_old) - np.log1p(-unit_old)
    transform_mean = np.sum(weight[:, None] * logit_old, axis=0)
    transform_variance = np.sum(
        weight[:, None] * (logit_old - transform_mean) ** 2, axis=0
    )
    transform_scale = np.sqrt(np.maximum(transform_variance, 1.0e-6))
    standardized = (logit_old - transform_mean) / transform_scale

    rng = np.random.default_rng(arguments.seed)
    fit_index = rng.choice(
        len(theta_old), size=arguments.fit_resamples, replace=True, p=weight
    )
    fit_sample = standardized[fit_index] + rng.normal(
        0.0, arguments.fit_jitter, size=(arguments.fit_resamples, 9)
    )
    gmm = GaussianMixture(
        n_components=arguments.gmm_components,
        covariance_type="full",
        reg_covar=arguments.gmm_regularization,
        max_iter=500,
        n_init=3,
        random_state=arguments.seed,
    ).fit(fit_sample)
    theta_mild = from_standardized_logit(
        gmm.sample(arguments.mild_draws)[0],
        low, high, transform_mean, transform_scale,
    )

    centered = standardized - np.sum(weight[:, None] * standardized, axis=0)
    global_covariance = (centered * weight[:, None]).T @ centered
    global_covariance = 0.9 * global_covariance + 0.1 * np.diag(
        np.diag(global_covariance)
    )
    eigenvalue, eigenvector = np.linalg.eigh(global_covariance)
    eigenvalue = np.maximum(eigenvalue, 0.02)
    global_covariance = (eigenvector * eigenvalue) @ eigenvector.T
    broad_location = np.sum(weight[:, None] * standardized, axis=0)
    # For a multivariate t, Cov = nu/(nu-2) * shape.  Choose the shape so
    # that the requested scale multiplies the fitted posterior covariance.
    broad_shape = global_covariance * arguments.broad_scale**2 * (
        (arguments.broad_degrees - 2.0) / arguments.broad_degrees
    )
    gaussian = rng.multivariate_normal(
        np.zeros(9), broad_shape, size=arguments.broad_draws
    )
    chi_square = rng.chisquare(arguments.broad_degrees, size=arguments.broad_draws)
    standardized_broad = broad_location + gaussian / np.sqrt(
        chi_square[:, None] / arguments.broad_degrees
    )
    theta_broad = from_standardized_logit(
        standardized_broad, low, high, transform_mean, transform_scale
    )
    theta = np.vstack([theta_mild, theta_broad])
    proposal_component = np.concatenate(
        [
            np.full(arguments.mild_draws, 3, dtype=np.uint8),
            np.full(arguments.broad_draws, 4, dtype=np.uint8),
        ]
    )

    standardized_new, unit_new = to_standardized_logit(
        theta, low, high, transform_mean, transform_scale
    )
    jacobian = log_jacobian(unit_new, low, high, transform_scale)
    logq_mild = gmm.score_samples(standardized_new) + jacobian
    logq_broad = multivariate_t_log_density(
        standardized_new,
        broad_location,
        broad_shape,
        arguments.broad_degrees,
    ) + jacobian
    logq_uniform = np.full(
        len(theta), -float(np.log(high - low).sum()), dtype=np.float64
    )

    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for flow-density evaluation")
    first_report = json.loads((arguments.first_run_dir / "run_report.json").read_text())
    second_report = json.loads((arguments.second_run_dir / "run_report.json").read_text())
    estimator_paths = [
        arguments.first_run_dir / "density_estimator.pt",
        arguments.second_run_dir / "density_estimator.pt",
    ]
    for report, path, calibration_flow in zip(
        (first_report, second_report), estimator_paths, calibration["flows"]
    ):
        if (
            report.get("lineage_class") != "independent_tsnpe"
            or report.get("forbidden_artifacts_used")
            or report.get("estimator_sha256") != sha256(path)
            or calibration_flow.get("estimator_sha256") != sha256(path)
        ):
            raise RuntimeError("adaptive proposal flow failed provenance gates")
    prior = BoxUniform(
        torch.as_tensor(low, dtype=torch.float32, device=device),
        torch.as_tensor(high, dtype=torch.float32, device=device),
    )
    observed = torch.as_tensor(A1_OBSERVATION, dtype=torch.float32, device=device)
    logq_flows = []
    for path, calibration_flow in zip(estimator_paths, calibration["flows"]):
        estimator = torch.load(path, map_location=device, weights_only=False)
        estimator.eval()
        raw = flow_raw_log_density(
            estimator, prior, observed, theta, device, arguments.density_batch
        )
        logq_flows.append(raw - float(calibration_flow["precision_log_factor"]))

    component_counts = np.asarray(
        [20_000, 20_000, 5_000, arguments.mild_draws, arguments.broad_draws],
        dtype=np.int64,
    )
    metadata = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_adaptive_full_prior_mis_candidates",
        "model": "ddb-hyperonic",
        "full_prior_dimension": 9,
        "method": "weighted_logit_GMM_plus_broad_Student_t",
        "stage_one": {"path": str(arguments.stage_one.resolve()), "sha256": sha256(arguments.stage_one)},
        "density_calibration": {"path": str(arguments.density_calibration.resolve()), "sha256": sha256(arguments.density_calibration)},
        "component_counts": component_counts.tolist(),
        "seed": arguments.seed,
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            theta=theta,
            logq_flow_seed_a=logq_flows[0],
            logq_flow_seed_b=logq_flows[1],
            logq_uniform_full_prior=logq_uniform,
            logq_adaptive_mild=logq_mild,
            logq_adaptive_broad=logq_broad,
            proposal_component=proposal_component,
            component_counts=component_counts,
            prior_low=low,
            prior_high=high,
            transform_mean=transform_mean,
            transform_scale=transform_scale,
            gmm_weights=gmm.weights_,
            gmm_means=gmm.means_,
            gmm_covariances=gmm.covariances_,
            broad_location=broad_location,
            broad_shape=broad_shape,
            broad_degrees=np.asarray(arguments.broad_degrees),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
    os.replace(temporary, arguments.output)
    report = {
        **metadata,
        "candidate_rows": len(theta),
        "mild_draws": arguments.mild_draws,
        "broad_draws": arguments.broad_draws,
        "fit_resamples": arguments.fit_resamples,
        "gmm_components": arguments.gmm_components,
        "gmm_converged": bool(gmm.converged_),
        "gmm_iterations": int(gmm.n_iter_),
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
