#!/usr/bin/env python3
"""Combine exact shards from the frozen adaptive-flow TSNPE proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--exact", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-ess", type=float, default=1_000.0)
    parser.add_argument("--posterior-rows", type=int, default=20_000)
    parser.add_argument("--resample-seed", type=int, required=True)
    arguments = parser.parse_args()
    if len(arguments.exact) < 2:
        raise ValueError("at least two independently evaluated shards are required")
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")

    candidate_report = json.loads(arguments.candidate.with_suffix(".json").read_text())
    candidate_hash = sha256(arguments.candidate)
    if (
        candidate_report.get("status") != "PASS"
        or candidate_report.get("lineage_class") != "independent_tsnpe"
        or candidate_report.get("role")
        != "tsnpe_frozen_adaptive_flow_full_prior_proposal"
        or candidate_report.get("full_prior_dimension") != 9
        or candidate_report.get("forbidden_artifacts_used")
        or candidate_report.get("output_sha256") != candidate_hash
    ):
        raise RuntimeError("frozen proposal failed clean-lineage gates")

    arrays = {
        key: []
        for key in (
            "theta",
            "log_likelihood",
            "likelihood_components",
            "valid",
            "logq_flow_ensemble",
            "logq_broad",
            "logq_uniform_full_prior",
            "proposal_component",
        )
    }
    parents = []
    indices = set()
    expected_shards = None
    counts = None
    low = None
    high = None
    target_gate = None
    for path in arguments.exact:
        report = json.loads(path.with_suffix(".json").read_text())
        path_hash = sha256(path)
        if (
            report.get("status") != "PASS"
            or report.get("lineage_class") != "independent_tsnpe"
            or report.get("role") != "tsnpe_frozen_adaptive_flow_exact_shard"
            or report.get("full_prior_dimension") != 9
            or report.get("forbidden_artifacts_used")
            or report.get("target_gate", {}).get("status") != "PASS"
            or report.get("output_sha256") != path_hash
            or report.get("parent_candidate_sha256") != candidate_hash
        ):
            raise RuntimeError(f"exact shard failed provenance gates: {path}")
        shard_index = int(report["shard_index"])
        shard_count = int(report["shard_count"])
        if shard_index in indices:
            raise RuntimeError("duplicate exact shard index")
        indices.add(shard_index)
        if expected_shards is None:
            expected_shards = shard_count
        elif expected_shards != shard_count:
            raise RuntimeError("exact shards disagree on shard count")

        with np.load(path, allow_pickle=False) as archive:
            if np.asarray(archive["forbidden_artifacts_used"]).size:
                raise RuntimeError("exact shard declares forbidden ancestry")
            for key in arrays:
                arrays[key].append(np.asarray(archive[key]))
            local_counts = np.asarray(archive["component_counts"], dtype=np.int64)
            local_low = np.asarray(archive["prior_low"], dtype=np.float64)
            local_high = np.asarray(archive["prior_high"], dtype=np.float64)
        if counts is None:
            counts = local_counts
            low = local_low
            high = local_high
        elif (
            not np.array_equal(counts, local_counts)
            or not np.array_equal(low, local_low)
            or not np.array_equal(high, local_high)
        ):
            raise RuntimeError("exact shards disagree on proposal or prior constants")
        parents.append({"path": str(path.resolve()), "sha256": path_hash})
        target_gate = report["target_gate"]

    if expected_shards != len(arguments.exact) or indices != set(range(expected_shards)):
        raise RuntimeError("exact shard set is incomplete")
    combined = {key: np.concatenate(value) for key, value in arrays.items()}
    theta = np.asarray(combined["theta"], dtype=np.float64)
    log_likelihood = np.asarray(combined["log_likelihood"], dtype=np.float64)
    likelihood_components = np.asarray(
        combined["likelihood_components"], dtype=np.float64
    )
    valid = np.asarray(combined["valid"], dtype=bool)
    proposal_component = np.asarray(combined["proposal_component"], dtype=np.uint8)
    rows = len(theta)
    if theta.shape != (rows, 9) or counts.shape != (3,) or int(counts.sum()) != rows:
        raise RuntimeError("combined proposal shapes or declared counts are inconsistent")
    if int(candidate_report.get("candidate_rows", -1)) != rows:
        raise RuntimeError("combined exact rows do not match frozen proposal report")
    if not np.array_equal(np.bincount(proposal_component, minlength=3), counts):
        raise RuntimeError("observed proposal labels do not match declared counts")
    if not np.all((theta >= low) & (theta <= high)):
        raise RuntimeError("combined exact rows leave the full prior")

    densities = np.column_stack(
        [
            np.asarray(combined["logq_flow_ensemble"], dtype=np.float64),
            np.asarray(combined["logq_broad"], dtype=np.float64),
            np.asarray(combined["logq_uniform_full_prior"], dtype=np.float64),
        ]
    )
    if densities.shape != (rows, 3) or not np.isfinite(densities).all():
        raise RuntimeError("combined proposal densities are invalid")
    fractions = counts.astype(np.float64) / rows
    log_proposal = logsumexp(np.log(fractions)[None, :] + densities, axis=1)
    log_prior = -float(np.log(high - low).sum())
    log_target = np.full(rows, -np.inf, dtype=np.float64)
    target_valid = valid & np.isfinite(log_likelihood) & (log_likelihood > -1.0e50)
    log_target[target_valid] = log_likelihood[target_valid] + log_prior
    log_weight = log_target - log_proposal
    finite = np.isfinite(log_weight)
    if not np.any(finite):
        raise RuntimeError("no finite importance weights")
    maximum = float(np.max(log_weight[finite]))
    scaled = np.zeros(rows, dtype=np.float64)
    scaled[finite] = np.exp(log_weight[finite] - maximum)
    normalized_weight = scaled / scaled.sum()
    ess = float(1.0 / np.sum(normalized_weight**2))
    evidence_scaled = float(np.mean(scaled))
    log_evidence = float(maximum + np.log(evidence_scaled))

    # Deterministic-mixture sampling uses fixed counts from each component.
    # Its variance is the stratified sum, rather than a pooled-i.i.d. estimate.
    variance_scaled = 0.0
    for component_index, fraction in enumerate(fractions):
        local = scaled[proposal_component == component_index]
        if len(local) < 2:
            raise RuntimeError("proposal component has too few rows for certification")
        variance_scaled += fraction**2 * float(np.var(local, ddof=1)) / len(local)
    log_evidence_error = float(np.sqrt(variance_scaled) / evidence_scaled)

    posterior_index = np.random.default_rng(arguments.resample_seed).choice(
        rows,
        size=arguments.posterior_rows,
        replace=True,
        p=normalized_weight,
    )
    posterior = theta[posterior_index]
    passed = ess >= arguments.minimum_ess
    metadata = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL_ESS_GATE",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_frozen_adaptive_flow_full_prior_mis_posterior",
        "model": "ddb-hyperonic",
        "method": (
            "clean_TSNPE_adaptive_flow_ensemble_plus_defensive_Student_t_"
            "plus_uniform_full_prior_exact_deterministic_MIS"
        ),
        "full_prior_dimension": 9,
        "component_counts": counts.tolist(),
        "component_fractions": fractions.tolist(),
        "candidate": {
            "path": str(arguments.candidate.resolve()),
            "sha256": candidate_hash,
        },
        "exact_parents": parents,
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
            logq_flow_ensemble=densities[:, 0],
            logq_broad=densities[:, 1],
            logq_uniform_full_prior=densities[:, 2],
            log_proposal=log_proposal,
            log_target=log_target,
            log_weight=log_weight,
            normalized_weight=normalized_weight,
            proposal_component=proposal_component,
            component_counts=counts,
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
        "valid_rows": int(target_valid.sum()),
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
