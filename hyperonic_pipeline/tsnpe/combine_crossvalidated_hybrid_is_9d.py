#!/usr/bin/env python3
"""Combine exact shards from the held-out-certified clean 9D TSNPE proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


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
    parser.add_argument("--minimum-ess", type=float, default=1_500.0)
    parser.add_argument("--posterior-rows", type=int, default=20_000)
    parser.add_argument("--resample-seed", type=int, required=True)
    arguments = parser.parse_args()
    if len(arguments.exact) < 2:
        raise ValueError("at least two exact shards are required")
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")

    candidate_report = json.loads(arguments.candidate.with_suffix(".json").read_text())
    candidate_hash = sha256(arguments.candidate)
    if (
        candidate_report.get("status") != "PASS"
        or candidate_report.get("lineage_class") != "independent_tsnpe"
        or candidate_report.get("role")
        != "tsnpe_crossvalidated_hybrid_full_prior_proposal_candidates"
        or candidate_report.get("full_prior_dimension") != 9
        or candidate_report.get("forbidden_artifacts_used")
        or candidate_report.get("output_sha256") != candidate_hash
    ):
        raise RuntimeError("cross-validated candidate failed provenance gates")

    keys = (
        "theta",
        "log_likelihood",
        "likelihood_components",
        "valid",
        "log_proposal",
        "proposal_component",
    )
    arrays = {key: [] for key in keys}
    exact_parents = []
    indices = set()
    expected_shards = None
    counts = None
    low = high = target_gate = None
    for path in arguments.exact:
        report = json.loads(path.with_suffix(".json").read_text())
        digest = sha256(path)
        if (
            report.get("status") != "PASS"
            or report.get("lineage_class") != "independent_tsnpe"
            or report.get("role") != "tsnpe_crossvalidated_hybrid_exact_shard"
            or report.get("full_prior_dimension") != 9
            or report.get("parent_candidate_sha256") != candidate_hash
            or report.get("bridge_sha256") != candidate_report.get("bridge_sha256")
            or report.get("parent_hybrid_model_sha256")
            != candidate_report.get("parent_hybrid_model_sha256")
            or report.get("old_proposal_sha256")
            != candidate_report.get("old_proposal_sha256")
            or report.get("new_proposal_sha256")
            != candidate_report.get("new_proposal_sha256")
            or report.get("target_gate", {}).get("status") != "PASS"
            or report.get("forbidden_artifacts_used")
            or report.get("output_sha256") != digest
        ):
            raise RuntimeError(f"exact shard failed provenance gates: {path}")
        index = int(report["shard_index"])
        shard_count = int(report["shard_count"])
        if index in indices:
            raise RuntimeError("duplicate exact shard index")
        indices.add(index)
        expected_shards = shard_count if expected_shards is None else expected_shards
        if expected_shards != shard_count:
            raise RuntimeError("exact shards disagree on shard count")
        with np.load(path, allow_pickle=False) as archive:
            if np.asarray(archive["forbidden_artifacts_used"]).size:
                raise RuntimeError("exact shard declares forbidden ancestry")
            for key in keys:
                arrays[key].append((index, np.asarray(archive[key])))
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
            raise RuntimeError("exact shards disagree on proposal/prior metadata")
        exact_parents.append({"path": str(path.resolve()), "sha256": digest})
        target_gate = report["target_gate"]
    if expected_shards != len(arguments.exact) or indices != set(range(expected_shards)):
        raise RuntimeError("exact shard set is incomplete")
    combined = {
        key: np.concatenate([value for _, value in sorted(parts)])
        for key, parts in arrays.items()
    }
    rows = len(combined["theta"])
    if (
        rows != int(candidate_report.get("candidate_rows", -1))
        or int(counts.sum()) != rows
        or not np.array_equal(
            np.bincount(combined["proposal_component"].astype(np.uint8), minlength=len(counts)),
            counts,
        )
    ):
        raise RuntimeError("exact rows do not reproduce candidate component counts")

    log_prior = -float(np.log(high - low).sum())
    valid = combined["valid"].astype(bool)
    log_target = np.full(rows, -np.inf, dtype=np.float64)
    target_valid = (
        valid
        & np.isfinite(combined["log_likelihood"])
        & (combined["log_likelihood"] > -1.0e50)
    )
    log_target[target_valid] = combined["log_likelihood"][target_valid] + log_prior
    log_weight = log_target - combined["log_proposal"]
    finite = np.isfinite(log_weight)
    maximum = float(np.max(log_weight[finite]))
    scaled = np.zeros(rows, dtype=np.float64)
    scaled[finite] = np.exp(log_weight[finite] - maximum)
    normalized_weight = scaled / scaled.sum()
    ess = float(1.0 / np.sum(normalized_weight**2))
    evidence_scaled = float(np.mean(scaled))
    log_evidence = float(maximum + np.log(evidence_scaled))
    log_evidence_error = float(
        np.std(scaled, ddof=1) / np.sqrt(rows) / evidence_scaled
    )
    passed = ess >= arguments.minimum_ess
    posterior_index = np.random.default_rng(arguments.resample_seed).choice(
        rows,
        size=arguments.posterior_rows,
        replace=True,
        p=normalized_weight,
    )
    posterior = combined["theta"][posterior_index]

    metadata = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL_ESS_GATE",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_crossvalidated_hybrid_full_prior_mis_posterior",
        "model": "ddb-hyperonic",
        "method": "clean_crossvalidated_GMM_defense_plus_frozen_TSNPE_exact_IS",
        "full_prior_dimension": 9,
        "candidate": {
            "path": str(arguments.candidate.resolve()),
            "sha256": candidate_hash,
        },
        "exact_parents": exact_parents,
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            theta=combined["theta"],
            log_likelihood=combined["log_likelihood"],
            likelihood_components=combined["likelihood_components"],
            valid=valid,
            log_proposal=combined["log_proposal"],
            log_target=log_target,
            log_weight=log_weight,
            normalized_weight=normalized_weight,
            proposal_component=combined["proposal_component"],
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
    output_report = {
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
        json.dumps(output_report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output_report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
