#!/usr/bin/env python3
"""Run corrected three-component MIS from a trained TSNPE estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eos.ddb import PRIOR_HIGH, PRIOR_LOW
from inference.common import ImportanceWeights, compute_importance_weights
from likelihoods.nuclear import A1_OBSERVATION
from inference.tsnpe.mixture import TruncatedGaussian, weighted_mean_covariance
from workflows import A1Problem
from workflows.resources import DEFAULT_A1_CERTIFICATE

COMPONENT_LABELS = (
    "flow",
    "broad_truncated_gaussian",
    "mild_truncated_gaussian",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uniform_log_prior() -> float:
    return -float(np.sum(np.log(PRIOR_HIGH - PRIOR_LOW)))


def target_log_density(log_likelihood: np.ndarray) -> np.ndarray:
    log_likelihood = np.asarray(log_likelihood, dtype=np.float64)
    valid = np.isfinite(log_likelihood) & (log_likelihood > -1e50)
    result = np.full(len(log_likelihood), -np.inf, dtype=np.float64)
    result[valid] = log_likelihood[valid] + uniform_log_prior()
    return result


def log_mixture_density(
    component_log_density: list[np.ndarray], counts: list[int]
) -> np.ndarray:
    if len(component_log_density) != len(counts) or not counts:
        raise ValueError("mixture components and counts do not match")
    total = int(sum(counts))
    if total <= 0 or any(count <= 0 for count in counts):
        raise ValueError("mixture component counts must be positive")
    terms = np.column_stack(
        [
            np.log(count / total) + np.asarray(log_density, dtype=np.float64)
            for count, log_density in zip(counts, component_log_density)
        ]
    )
    return logsumexp(terms, axis=1)


def evaluate_flow_log_density(posterior, observed, theta, device, batch_size=4000):
    output = []
    with torch.no_grad():
        for start in range(0, len(theta), batch_size):
            batch = torch.as_tensor(
                theta[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            output.append(
                posterior.log_prob(
                    batch, x=observed, norm_posterior=True
                )
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(output).astype(np.float64)


def likelihood(problem: A1Problem, theta: np.ndarray) -> np.ndarray:
    value = np.asarray(problem.evaluate_batch(theta), dtype=np.float64)
    return np.where(value > -1e50, value, -np.inf)


def stage_report(importance: ImportanceWeights, rows: int) -> dict:
    importance_sampling_error = float(
        np.sqrt(
            max(
                np.sum(importance.normalized_weight**2) - 1.0 / rows,
                0.0,
            )
        )
    )
    return {
        "rows": rows,
        "finite_rows": importance.finite_count,
        "ess": importance.ess,
        "ess_fraction": importance.ess / rows,
        "log_evidence": importance.log_evidence,
        "log_evidence_standard_error": importance_sampling_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--estimator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--flow-draws", type=int, default=60_000)
    parser.add_argument("--broad-draws", type=int, default=15_000)
    parser.add_argument("--mild-draws", type=int, default=30_000)
    parser.add_argument("--broad-scale", type=float, default=2.5)
    parser.add_argument("--mild-scale", type=float, default=1.5)
    parser.add_argument("--normalization-draws", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resample-seed", type=int, default=1)
    parser.add_argument("--posterior-rows", type=int, default=20_000)
    parser.add_argument("--minimum-ess", type=float, default=100.0)
    parser.add_argument("--device")
    parser.add_argument("--certificate", type=Path, default=DEFAULT_A1_CERTIFICATE)
    parser.add_argument("--skip-target-gate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.skip_target_gate and not arguments.smoke:
        raise ValueError("--skip-target-gate is permitted only with --smoke")

    from sbi.inference.posteriors import DirectPosterior
    from sbi.utils import BoxUniform

    started = time.time()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    device = arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    rng = np.random.default_rng(arguments.seed)
    problem = A1Problem.from_data_root(arguments.data_root, workers=arguments.workers)
    target_gate = (
        {"status": "SKIPPED"}
        if arguments.skip_target_gate
        else problem.certify(arguments.certificate)
    )
    if target_gate["status"] not in {"PASS", "SKIPPED"}:
        raise RuntimeError(f"shared target gate failed: {target_gate}")

    estimator = torch.load(
        arguments.estimator, map_location=device, weights_only=False
    )
    prior = BoxUniform(
        low=torch.as_tensor(PRIOR_LOW, dtype=torch.float32, device=device),
        high=torch.as_tensor(PRIOR_HIGH, dtype=torch.float32, device=device),
    )
    posterior = DirectPosterior(
        posterior_estimator=estimator, prior=prior, device=device
    )
    observed = torch.as_tensor(
        A1_OBSERVATION, dtype=torch.float32, device=device
    )

    theta_flow = (
        posterior.sample(
            (arguments.flow_draws,),
            x=observed,
            show_progress_bars=False,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    logq_flow_at_flow = evaluate_flow_log_density(
        posterior, observed, theta_flow, device
    )
    logl_flow = likelihood(problem, theta_flow)
    flow_importance = compute_importance_weights(
        target_log_density(logl_flow), logq_flow_at_flow
    )
    np.savez_compressed(
        arguments.output_dir / "stage_1_flow.npz",
        theta=theta_flow,
        logL=logl_flow,
        logq=logq_flow_at_flow,
        logw=flow_importance.log_weight,
        w=flow_importance.normalized_weight,
    )

    broad_mean, broad_covariance = weighted_mean_covariance(
        theta_flow, flow_importance.log_weight
    )
    broad = TruncatedGaussian.from_covariance(
        broad_mean,
        broad_covariance,
        PRIOR_LOW,
        PRIOR_HIGH,
        scale=arguments.broad_scale,
        rng=np.random.default_rng(arguments.seed + 101),
        normalization_draws=arguments.normalization_draws,
    )
    theta_broad = broad.sample(
        arguments.broad_draws, rng=np.random.default_rng(arguments.seed + 102)
    )
    logl_broad = likelihood(problem, theta_broad)
    logq_flow_at_broad = evaluate_flow_log_density(
        posterior, observed, theta_broad, device
    )
    theta_intermediate = np.vstack([theta_flow, theta_broad])
    logl_intermediate = np.concatenate([logl_flow, logl_broad])
    logq_flow_intermediate = np.concatenate(
        [logq_flow_at_flow, logq_flow_at_broad]
    )
    logq_broad_intermediate = broad.log_density(theta_intermediate)
    logq_intermediate = log_mixture_density(
        [logq_flow_intermediate, logq_broad_intermediate],
        [arguments.flow_draws, arguments.broad_draws],
    )
    intermediate_importance = compute_importance_weights(
        target_log_density(logl_intermediate), logq_intermediate
    )
    np.savez_compressed(
        arguments.output_dir / "stage_2_flow_broad.npz",
        theta=theta_intermediate,
        logL=logl_intermediate,
        logq=logq_intermediate,
        logw=intermediate_importance.log_weight,
        w=intermediate_importance.normalized_weight,
        proposal_component=np.concatenate(
            [
                np.zeros(arguments.flow_draws, dtype=np.uint8),
                np.ones(arguments.broad_draws, dtype=np.uint8),
            ]
        ),
    )

    mild_mean, mild_covariance = weighted_mean_covariance(
        theta_intermediate, intermediate_importance.log_weight
    )
    mild = TruncatedGaussian.from_covariance(
        mild_mean,
        mild_covariance,
        PRIOR_LOW,
        PRIOR_HIGH,
        scale=arguments.mild_scale,
        rng=np.random.default_rng(arguments.seed + 201),
        normalization_draws=arguments.normalization_draws,
    )
    theta_mild = mild.sample(
        arguments.mild_draws, rng=np.random.default_rng(arguments.seed + 202)
    )
    logl_mild = likelihood(problem, theta_mild)
    logq_flow_at_mild = evaluate_flow_log_density(
        posterior, observed, theta_mild, device
    )
    theta = np.vstack([theta_intermediate, theta_mild])
    log_likelihood = np.concatenate([logl_intermediate, logl_mild])
    logq_flow = np.concatenate(
        [logq_flow_intermediate, logq_flow_at_mild]
    )
    # The broad density below is deliberately the density that generated the
    # 15k broad rows.  It is never replaced by a refit after stage 2.
    logq_broad = broad.log_density(theta)
    logq_mild = mild.log_density(theta)
    log_proposal = log_mixture_density(
        [logq_flow, logq_broad, logq_mild],
        [arguments.flow_draws, arguments.broad_draws, arguments.mild_draws],
    )
    importance = compute_importance_weights(
        target_log_density(log_likelihood), log_proposal
    )
    posterior_sample, posterior_index = importance.resample(
        theta, size=arguments.posterior_rows, seed=arguments.resample_seed
    )
    proposal_component = np.concatenate(
        [
            np.zeros(arguments.flow_draws, dtype=np.uint8),
            np.ones(arguments.broad_draws, dtype=np.uint8),
            np.full(arguments.mild_draws, 2, dtype=np.uint8),
        ]
    )
    final_path = arguments.output_dir / "tsnpe_mis_corrected.npz"
    np.savez_compressed(
        final_path,
        theta=theta,
        logL=log_likelihood,
        logq=log_proposal,
        logq_flow=logq_flow,
        logq_broad=logq_broad,
        logq_mild=logq_mild,
        logw=importance.log_weight,
        w=importance.normalized_weight,
        proposal_component=proposal_component,
        component_labels=np.asarray(COMPONENT_LABELS),
        posterior=posterior_sample,
        posterior_index=posterior_index,
        ESS=importance.ess,
        logZ=importance.log_evidence,
        logZ_standard_error=importance.log_evidence_standard_error,
    )
    passed = importance.ess >= arguments.minimum_ess
    report = {
        "status": (
            "SMOKE_COMPLETED"
            if arguments.smoke
            else ("PASS" if passed else "FAIL_ESS_GATE")
        ),
        "method": "TSNPE+corrected_deterministic_mixture_IS",
        "target_gate": target_gate,
        "estimator": str(arguments.estimator),
        "estimator_sha256": sha256(arguments.estimator),
        "components": {
            "flow": arguments.flow_draws,
            "broad_truncated_gaussian": arguments.broad_draws,
            "mild_truncated_gaussian": arguments.mild_draws,
        },
        "generating_density_provenance_explicit": True,
        "broad_density_refit_in_final_denominator": False,
        "truncation_normalizations": {
            "broad": broad.normalization,
            "mild": mild.normalization,
        },
        "stage_1": stage_report(flow_importance, arguments.flow_draws),
        "stage_2": stage_report(
            intermediate_importance,
            arguments.flow_draws + arguments.broad_draws,
        ),
        "final": stage_report(importance, len(theta)),
        "minimum_ess": arguments.minimum_ess,
        "output": str(final_path),
        "output_sha256": sha256(final_path),
        "wall_seconds": time.time() - started,
    }
    (arguments.output_dir / "run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if arguments.smoke or passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
