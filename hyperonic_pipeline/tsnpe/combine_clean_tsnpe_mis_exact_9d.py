#!/usr/bin/env python3
"""Combine clean exact TSNPE MIS shards and certify the weighted posterior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


ARRAY_KEYS = (
    "theta",
    "log_likelihood",
    "likelihood_components",
    "valid",
    "logq_flow_seed_a",
    "logq_flow_seed_b",
    "logq_uniform_full_prior",
    "proposal_component",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--density-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-ess", type=float, default=1_000.0)
    parser.add_argument("--posterior-rows", type=int, default=20_000)
    parser.add_argument("--resample-seed", type=int, required=True)
    arguments = parser.parse_args()
    if len(arguments.input) < 2:
        raise ValueError("at least two exact shards are required")
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")
    if arguments.minimum_ess <= 0 or arguments.posterior_rows < 1:
        raise ValueError("ESS gate and posterior row count must be positive")

    arrays = {key: [] for key in ARRAY_KEYS}
    parents = []
    input_hashes = set()
    shard_indices = set()
    parent_candidate_hash = None
    component_counts = None
    prior_low = None
    prior_high = None
    target_gate = None
    for path in arguments.input:
        report = json.loads(path.with_suffix(".json").read_text())
        if (
            report.get("status") != "PASS"
            or report.get("lineage_class") != "independent_tsnpe"
            or report.get("role") != "tsnpe_full_prior_mis_exact_shard"
            or report.get("forbidden_artifacts_used")
            or (report.get("target_gate") or {}).get("status") != "PASS"
        ):
            raise RuntimeError(f"exact shard failed certification: {path}")
        input_hash = sha256(path)
        if report.get("output_sha256") != input_hash or input_hash in input_hashes:
            raise RuntimeError(f"exact shard hash mismatch or duplicate: {path}")
        input_hashes.add(input_hash)
        shard_index = int(report["shard_index"])
        if shard_index in shard_indices:
            raise RuntimeError("duplicate shard index")
        shard_indices.add(shard_index)
        local_parent = report.get("parent_candidate_sha256")
        if parent_candidate_hash is None:
            parent_candidate_hash = local_parent
        elif local_parent != parent_candidate_hash:
            raise RuntimeError("exact shards come from different candidate mixtures")

        with np.load(path, allow_pickle=False) as archive:
            if np.asarray(archive["forbidden_artifacts_used"]).size:
                raise RuntimeError(f"archive declares forbidden ancestry: {path}")
            local_rows = len(np.asarray(archive["theta"]))
            for key in ARRAY_KEYS:
                value = np.asarray(archive[key])
                if len(value) != local_rows:
                    raise RuntimeError(f"inconsistent array length in {path}")
                arrays[key].append(value)
            local_counts = np.asarray(archive["component_counts"], dtype=np.int64)
            local_low = np.asarray(archive["prior_low"], dtype=np.float64)
            local_high = np.asarray(archive["prior_high"], dtype=np.float64)
            if component_counts is None:
                component_counts = local_counts
                prior_low, prior_high = local_low, local_high
            elif not (
                np.array_equal(component_counts, local_counts)
                and np.array_equal(prior_low, local_low)
                and np.array_equal(prior_high, local_high)
            ):
                raise RuntimeError("shards use different mixture counts or prior bounds")
        target_gate = report["target_gate"]
        parents.append({"path": str(path.resolve()), "sha256": input_hash})

    combined = {key: np.concatenate(value) for key, value in arrays.items()}
    rows = len(combined["theta"])
    if int(component_counts.sum()) != rows:
        raise RuntimeError("combined rows do not equal deterministic-mixture counts")
    observed_counts = np.bincount(combined["proposal_component"], minlength=3)
    if not np.array_equal(observed_counts, component_counts):
        raise RuntimeError("proposal-component labels do not match declared counts")
    if prior_low.shape != (9,) or prior_high.shape != (9,):
        raise RuntimeError("wrong full-prior shape")

    calibration = json.loads(arguments.density_calibration.read_text())
    if (
        calibration.get("status") != "PASS"
        or calibration.get("lineage_class") != "independent_tsnpe"
        or calibration.get("role")
        != "tsnpe_flow_density_normalization_calibration"
        or calibration.get("forbidden_artifacts_used")
        or calibration.get("candidate_sha256") != parent_candidate_hash
        or len(calibration.get("flows", [])) != 2
    ):
        raise RuntimeError("flow-density calibration failed provenance gates")
    for flow in calibration["flows"]:
        if (
            int(flow.get("precision_draws", 0)) < 1_000_000
            or not np.isfinite(float(flow.get("additive_logq_shift", np.nan)))
            or float(flow.get("encoded_factor_constancy_max_abs", np.inf)) > 5.0e-4
        ):
            raise RuntimeError("flow-density calibration failed numerical gates")
    logq1_calibrated = combined["logq_flow_seed_a"] + float(
        calibration["flows"][0]["additive_logq_shift"]
    )
    logq2_calibrated = combined["logq_flow_seed_b"] + float(
        calibration["flows"][1]["additive_logq_shift"]
    )

    fractions = component_counts.astype(np.float64) / rows
    log_proposal = logsumexp(
        np.column_stack(
            [
                np.log(fractions[0]) + logq1_calibrated,
                np.log(fractions[1]) + logq2_calibrated,
                np.log(fractions[2]) + combined["logq_uniform_full_prior"],
            ]
        ),
        axis=1,
    )
    log_target = np.full(rows, -np.inf, dtype=np.float64)
    valid = combined["valid"] & np.isfinite(combined["log_likelihood"])
    log_prior = -float(np.log(prior_high - prior_low).sum())
    log_target[valid] = combined["log_likelihood"][valid] + log_prior
    log_weight = log_target - log_proposal
    finite = np.isfinite(log_weight)
    if not finite.any():
        raise RuntimeError("no finite MIS weight")
    maximum = float(np.max(log_weight[finite]))
    scaled = np.zeros(rows, dtype=np.float64)
    scaled[finite] = np.exp(log_weight[finite] - maximum)
    normalized_weight = scaled / scaled.sum()
    ess = float(1.0 / np.sum(normalized_weight**2))
    log_evidence = float(maximum + np.log(scaled.sum()) - np.log(rows))
    mean_scaled = float(np.mean(scaled))
    log_evidence_error = float(
        np.sqrt(np.var(scaled, ddof=1) / rows) / mean_scaled
    )
    posterior_index = np.random.default_rng(arguments.resample_seed).choice(
        rows, size=arguments.posterior_rows, replace=True, p=normalized_weight
    )
    posterior = combined["theta"][posterior_index]
    passed = ess >= arguments.minimum_ess

    metadata = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL_ESS_GATE",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_full_prior_defensive_mis_posterior",
        "model": "ddb-hyperonic",
        "method": "two_clean_TSNPE_flows_plus_full_prior_defensive_MIS",
        "full_prior_dimension": 9,
        "parent_candidate_sha256": parent_candidate_hash,
        "component_counts": component_counts.tolist(),
        "component_fractions": fractions.tolist(),
        "density_calibration": {
            "path": str(arguments.density_calibration.resolve()),
            "sha256": sha256(arguments.density_calibration),
            "additive_logq_shifts": [
                float(calibration["flows"][0]["additive_logq_shift"]),
                float(calibration["flows"][1]["additive_logq_shift"]),
            ],
            "precision_draws_per_flow": [
                int(calibration["flows"][0]["precision_draws"]),
                int(calibration["flows"][1]["precision_draws"]),
            ],
        },
        "parents": parents,
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **combined,
            logq_flow_seed_a_calibrated=logq1_calibrated,
            logq_flow_seed_b_calibrated=logq2_calibrated,
            log_proposal=log_proposal,
            log_target=log_target,
            log_weight=log_weight,
            normalized_weight=normalized_weight,
            posterior=posterior,
            posterior_index=posterior_index,
            ESS=np.asarray(ess),
            logZ=np.asarray(log_evidence),
            logZ_standard_error=np.asarray(log_evidence_error),
            component_counts=component_counts,
            prior_low=prior_low,
            prior_high=prior_high,
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
