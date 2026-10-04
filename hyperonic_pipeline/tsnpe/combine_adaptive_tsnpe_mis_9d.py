#!/usr/bin/env python3
"""Combine the clean TSNPE pilot and adaptive exact MIS stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform(theta, low, high, mean, scale):
    unit = (theta - low) / (high - low)
    if not np.all((unit > 0.0) & (unit < 1.0)):
        raise RuntimeError("density transform received boundary row")
    value = (np.log(unit) - np.log1p(-unit) - mean) / scale
    jacobian = -np.log(scale).sum() - np.sum(
        np.log(high - low) + np.log(unit) + np.log1p(-unit), axis=1
    )
    return value, jacobian


def gmm_log_density(value, weights, means, covariances):
    dimension = value.shape[1]
    terms = []
    for weight, mean, covariance in zip(weights, means, covariances):
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0 or weight <= 0:
            raise RuntimeError("invalid adaptive GMM parameter")
        difference = value - mean
        quadratic = np.einsum(
            "ni,ij,nj->n", difference, np.linalg.inv(covariance), difference
        )
        terms.append(
            np.log(weight)
            - 0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic)
        )
    return logsumexp(np.column_stack(terms), axis=1)


def t_log_density(value, location, shape, degrees):
    dimension = value.shape[1]
    sign, logdet = np.linalg.slogdet(shape)
    if sign <= 0:
        raise RuntimeError("invalid adaptive Student-t shape")
    difference = value - location
    quadratic = np.einsum(
        "ni,ij,nj->n", difference, np.linalg.inv(shape), difference
    )
    constant = (
        gammaln((degrees + dimension) / 2.0)
        - gammaln(degrees / 2.0)
        - 0.5 * (dimension * np.log(degrees * np.pi) + logdet)
    )
    return constant - 0.5 * (degrees + dimension) * np.log1p(
        quadratic / degrees
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-one", type=Path, required=True)
    parser.add_argument("--adaptive-candidate", type=Path, required=True)
    parser.add_argument("--adaptive-exact", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-ess", type=float, default=1_000.0)
    parser.add_argument("--posterior-rows", type=int, default=20_000)
    parser.add_argument("--resample-seed", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")

    stage_report = json.loads(arguments.stage_one.with_suffix(".json").read_text())
    if (
        stage_report.get("status") != "FAIL_ESS_GATE"
        or stage_report.get("lineage_class") != "independent_tsnpe"
        or stage_report.get("forbidden_artifacts_used")
        or stage_report.get("target_gate", {}).get("status") != "PASS"
        or stage_report.get("output_sha256") != sha256(arguments.stage_one)
    ):
        raise RuntimeError("stage-one diagnostic failed provenance gates")
    candidate_report = json.loads(
        arguments.adaptive_candidate.with_suffix(".json").read_text()
    )
    candidate_hash = sha256(arguments.adaptive_candidate)
    if (
        candidate_report.get("status") != "PASS"
        or candidate_report.get("lineage_class") != "independent_tsnpe"
        or candidate_report.get("role")
        != "tsnpe_adaptive_full_prior_mis_candidates"
        or candidate_report.get("forbidden_artifacts_used")
        or candidate_report.get("output_sha256") != candidate_hash
        or candidate_report.get("stage_one", {}).get("sha256")
        != sha256(arguments.stage_one)
    ):
        raise RuntimeError("adaptive candidate failed provenance gates")

    with np.load(arguments.stage_one, allow_pickle=False) as old:
        theta_old = np.asarray(old["theta"], dtype=np.float64)
        logl_old = np.asarray(old["log_likelihood"], dtype=np.float64)
        terms_old = np.asarray(old["likelihood_components"], dtype=np.float64)
        valid_old = np.asarray(old["valid"], dtype=bool)
        q1_old = np.asarray(old["logq_flow_seed_a_calibrated"], dtype=np.float64)
        q2_old = np.asarray(old["logq_flow_seed_b_calibrated"], dtype=np.float64)
        qu_old = np.asarray(old["logq_uniform_full_prior"], dtype=np.float64)
        component_old = np.asarray(old["proposal_component"], dtype=np.uint8)
        low = np.asarray(old["prior_low"], dtype=np.float64)
        high = np.asarray(old["prior_high"], dtype=np.float64)

    with np.load(arguments.adaptive_candidate, allow_pickle=False) as model:
        transform_mean = np.asarray(model["transform_mean"], dtype=np.float64)
        transform_scale = np.asarray(model["transform_scale"], dtype=np.float64)
        gmm_weights = np.asarray(model["gmm_weights"], dtype=np.float64)
        gmm_means = np.asarray(model["gmm_means"], dtype=np.float64)
        gmm_covariances = np.asarray(model["gmm_covariances"], dtype=np.float64)
        broad_location = np.asarray(model["broad_location"], dtype=np.float64)
        broad_shape = np.asarray(model["broad_shape"], dtype=np.float64)
        broad_degrees = float(np.asarray(model["broad_degrees"]).item())
        component_counts = np.asarray(model["component_counts"], dtype=np.int64)
        model_low = np.asarray(model["prior_low"], dtype=np.float64)
        model_high = np.asarray(model["prior_high"], dtype=np.float64)
    if not np.array_equal(low, model_low) or not np.array_equal(high, model_high):
        raise RuntimeError("adaptive and stage-one priors differ")

    new_arrays = {key: [] for key in (
        "theta", "log_likelihood", "likelihood_components", "valid",
        "logq_flow_seed_a", "logq_flow_seed_b", "logq_uniform_full_prior",
        "logq_adaptive_mild", "logq_adaptive_broad", "proposal_component",
    )}
    exact_parents = []
    shard_indices = set()
    target_gate = None
    for path in arguments.adaptive_exact:
        report = json.loads(path.with_suffix(".json").read_text())
        input_hash = sha256(path)
        if (
            report.get("status") != "PASS"
            or report.get("lineage_class") != "independent_tsnpe"
            or report.get("role") != "tsnpe_full_prior_mis_exact_shard"
            or report.get("candidate_role")
            != "tsnpe_adaptive_full_prior_mis_candidates"
            or report.get("forbidden_artifacts_used")
            or report.get("target_gate", {}).get("status") != "PASS"
            or report.get("output_sha256") != input_hash
            or report.get("parent_candidate_sha256") != candidate_hash
        ):
            raise RuntimeError(f"adaptive exact shard failed gates: {path}")
        index = int(report["shard_index"])
        if index in shard_indices:
            raise RuntimeError("duplicate adaptive exact shard")
        shard_indices.add(index)
        with np.load(path, allow_pickle=False) as archive:
            if np.asarray(archive["forbidden_artifacts_used"]).size:
                raise RuntimeError("adaptive exact archive declares forbidden ancestry")
            for key in new_arrays:
                new_arrays[key].append(np.asarray(archive[key]))
        exact_parents.append({"path": str(path.resolve()), "sha256": input_hash})
        target_gate = report["target_gate"]
    new = {key: np.concatenate(value) for key, value in new_arrays.items()}

    standardized_old, jacobian_old = transform(
        theta_old, low, high, transform_mean, transform_scale
    )
    qm_old = gmm_log_density(
        standardized_old, gmm_weights, gmm_means, gmm_covariances
    ) + jacobian_old
    qb_old = t_log_density(
        standardized_old, broad_location, broad_shape, broad_degrees
    ) + jacobian_old

    # Recompute a subset of the stored new adaptive densities from the frozen
    # model to guard against serialization or sharding errors.
    check_rows = min(256, len(new["theta"]))
    standardized_check, jacobian_check = transform(
        new["theta"][:check_rows], low, high, transform_mean, transform_scale
    )
    qm_check = gmm_log_density(
        standardized_check, gmm_weights, gmm_means, gmm_covariances
    ) + jacobian_check
    qb_check = t_log_density(
        standardized_check, broad_location, broad_shape, broad_degrees
    ) + jacobian_check
    if (
        np.max(np.abs(qm_check - new["logq_adaptive_mild"][:check_rows])) > 1.0e-8
        or np.max(np.abs(qb_check - new["logq_adaptive_broad"][:check_rows])) > 1.0e-8
    ):
        raise RuntimeError("adaptive density replay mismatch")

    theta = np.vstack([theta_old, new["theta"]])
    log_likelihood = np.concatenate([logl_old, new["log_likelihood"]])
    likelihood_components = np.vstack([terms_old, new["likelihood_components"]])
    valid = np.concatenate([valid_old, new["valid"]])
    proposal_component = np.concatenate([component_old, new["proposal_component"]])
    densities = [
        np.concatenate([q1_old, new["logq_flow_seed_a"]]),
        np.concatenate([q2_old, new["logq_flow_seed_b"]]),
        np.concatenate([qu_old, new["logq_uniform_full_prior"]]),
        np.concatenate([qm_old, new["logq_adaptive_mild"]]),
        np.concatenate([qb_old, new["logq_adaptive_broad"]]),
    ]
    rows = len(theta)
    if int(component_counts.sum()) != rows:
        raise RuntimeError("global component counts do not equal combined rows")
    if not np.array_equal(
        np.bincount(proposal_component, minlength=5), component_counts
    ):
        raise RuntimeError("global component labels do not match counts")
    fractions = component_counts.astype(np.float64) / rows
    log_proposal = logsumexp(
        np.column_stack(
            [np.log(fraction) + density for fraction, density in zip(fractions, densities)]
        ), axis=1
    )
    log_target = np.full(rows, -np.inf, dtype=np.float64)
    target_valid = valid & np.isfinite(log_likelihood)
    log_prior = -float(np.log(high - low).sum())
    log_target[target_valid] = log_likelihood[target_valid] + log_prior
    log_weight = log_target - log_proposal
    finite = np.isfinite(log_weight)
    maximum = float(np.max(log_weight[finite]))
    scaled = np.zeros(rows, dtype=np.float64)
    scaled[finite] = np.exp(log_weight[finite] - maximum)
    normalized_weight = scaled / scaled.sum()
    ess = float(1.0 / np.sum(normalized_weight**2))
    log_evidence = float(maximum + np.log(scaled.sum()) - np.log(rows))
    mean_scaled = float(np.mean(scaled))
    log_evidence_error = float(np.sqrt(np.var(scaled, ddof=1) / rows) / mean_scaled)
    posterior_index = np.random.default_rng(arguments.resample_seed).choice(
        rows, size=arguments.posterior_rows, replace=True, p=normalized_weight
    )
    posterior = theta[posterior_index]
    passed = ess >= arguments.minimum_ess

    metadata = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL_ESS_GATE",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_adaptive_full_prior_mis_posterior",
        "model": "ddb-hyperonic",
        "method": "clean_TSNPE_flows_plus_uniform_plus_adaptive_bounded_GMM_Student_t_MIS",
        "full_prior_dimension": 9,
        "component_counts": component_counts.tolist(),
        "component_fractions": fractions.tolist(),
        "stage_one": {"path": str(arguments.stage_one.resolve()), "sha256": sha256(arguments.stage_one)},
        "adaptive_candidate": {"path": str(arguments.adaptive_candidate.resolve()), "sha256": candidate_hash},
        "adaptive_exact_parents": exact_parents,
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            theta=theta,
            log_likelihood=log_likelihood,
            likelihood_components=likelihood_components,
            valid=valid,
            logq_flow_seed_a=densities[0],
            logq_flow_seed_b=densities[1],
            logq_uniform_full_prior=densities[2],
            logq_adaptive_mild=densities[3],
            logq_adaptive_broad=densities[4],
            log_proposal=log_proposal,
            log_target=log_target,
            log_weight=log_weight,
            normalized_weight=normalized_weight,
            proposal_component=proposal_component,
            component_counts=component_counts,
            posterior=posterior,
            posterior_index=posterior_index,
            ESS=np.asarray(ess),
            logZ=np.asarray(log_evidence),
            logZ_standard_error=np.asarray(log_evidence_error),
            prior_low=low,
            prior_high=high,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
    os.replace(temporary, arguments.output)
    report = {
        **metadata,
        "scientific_posterior_certified": bool(passed),
        "candidate_rows": rows,
        "finite_weight_rows": int(finite.sum()),
        "ess": ess,
        "ess_fraction": ess / rows,
        "minimum_ess": arguments.minimum_ess,
        "maximum_normalized_weight": float(normalized_weight.max()),
        "log_evidence": log_evidence,
        "log_evidence_standard_error": log_evidence_error,
        "posterior_rows": arguments.posterior_rows,
        "resample_seed": arguments.resample_seed,
        "target_gate": target_gate,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
