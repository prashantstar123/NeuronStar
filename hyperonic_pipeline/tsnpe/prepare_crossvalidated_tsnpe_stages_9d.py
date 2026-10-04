#!/usr/bin/env python3
"""Make disjoint clean training/validation stages from exact TSNPE MIS rows."""

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


def atomic_save(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def certified_report(path: Path) -> tuple[dict, str]:
    report = json.loads(path.with_suffix(".json").read_text())
    digest = sha256(path)
    if (
        report.get("lineage_class") != "independent_tsnpe"
        or report.get("status") not in {"PASS", "FAIL_ESS_GATE"}
        or report.get("forbidden_artifacts_used")
        or report.get("output_sha256") != digest
    ):
        raise RuntimeError(f"artifact failed clean-lineage gates: {path}")
    return report, digest


def summarize_stage(log_target: np.ndarray, log_proposal: np.ndarray) -> dict:
    finite = np.isfinite(log_target) & np.isfinite(log_proposal)
    if not finite.any():
        raise RuntimeError("stage has no finite target weights")
    log_weight = log_target - log_proposal
    maximum = float(np.max(log_weight[finite]))
    scaled = np.zeros(len(log_weight), dtype=np.float64)
    scaled[finite] = np.exp(log_weight[finite] - maximum)
    normalized = scaled / scaled.sum()
    ess = float(1.0 / np.sum(normalized**2))
    evidence_scaled = float(np.mean(scaled))
    log_evidence = float(maximum + np.log(evidence_scaled))
    error = float(
        np.std(scaled, ddof=1) / np.sqrt(len(scaled)) / evidence_scaled
    )
    return {
        "log_weight": log_weight,
        "normalized_weight": normalized,
        "finite_weight_rows": int(finite.sum()),
        "ess": ess,
        "log_evidence": log_evidence,
        "log_evidence_standard_error": error,
        "maximum_normalized_weight": float(normalized.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-exact-stage", type=Path, required=True)
    parser.add_argument("--old-cross-density", type=Path, required=True)
    parser.add_argument("--new-candidate", type=Path, required=True)
    parser.add_argument("--new-exact", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise FileExistsError(f"refusing to replace {arguments.output_dir}")

    old_report, old_hash = certified_report(arguments.old_exact_stage)
    if old_report.get("target_gate", {}).get("status") != "PASS":
        raise RuntimeError("old exact target certificate failed")
    with np.load(arguments.old_exact_stage, allow_pickle=False) as archive:
        old_theta = np.asarray(archive["theta"], dtype=np.float64)
        old_target = np.asarray(archive["log_target"], dtype=np.float64)
        old_qold = np.asarray(archive["log_proposal"], dtype=np.float64)
        low = np.asarray(archive["prior_low"], dtype=np.float64)
        high = np.asarray(archive["prior_high"], dtype=np.float64)
        if np.asarray(archive["forbidden_artifacts_used"]).size:
            raise RuntimeError("old exact stage declares forbidden ancestry")

    cross_report, cross_hash = certified_report(arguments.old_cross_density)
    if (
        cross_report.get("role") != "tsnpe_dual_stage_old_exact_cross_density"
        or cross_report.get("old_exact_stage_sha256") != old_hash
    ):
        raise RuntimeError("old cross-density has the wrong lineage")
    with np.load(arguments.old_cross_density, allow_pickle=False) as archive:
        old_qnew = np.asarray(archive["logq_new_stage"], dtype=np.float64)
        if np.asarray(archive["forbidden_artifacts_used"]).size:
            raise RuntimeError("old cross-density declares forbidden ancestry")
    if old_qnew.shape != (len(old_theta),):
        raise RuntimeError("old cross-density row mismatch")

    candidate_report, candidate_hash = certified_report(arguments.new_candidate)
    if (
        candidate_report.get("role")
        != "tsnpe_dual_stage_new_proposal_candidates"
        or candidate_report.get("full_prior_dimension") != 9
        or candidate_report.get("old_exact_stage_sha256") != old_hash
        or cross_report.get("new_proposal_sha256")
        != candidate_report.get("new_proposal_sha256")
    ):
        raise RuntimeError("new candidate has the wrong dual-stage lineage")
    with np.load(arguments.new_candidate, allow_pickle=False) as archive:
        new_theta = np.asarray(archive["theta"], dtype=np.float64)
        new_qold = np.asarray(archive["logq_old_stage"], dtype=np.float64)
        new_qnew = np.asarray(archive["logq_new_stage"], dtype=np.float64)
        new_low = np.asarray(archive["prior_low"], dtype=np.float64)
        new_high = np.asarray(archive["prior_high"], dtype=np.float64)
        if np.asarray(archive["forbidden_artifacts_used"]).size:
            raise RuntimeError("new candidate declares forbidden ancestry")
    if not np.array_equal(low, new_low) or not np.array_equal(high, new_high):
        raise RuntimeError("old and new stages use different priors")

    exact_parts = []
    exact_parents = []
    indices = set()
    expected_parts = None
    target_gate = None
    for path in arguments.new_exact:
        report, digest = certified_report(path)
        if (
            report.get("role") != "tsnpe_dual_stage_new_exact_shard"
            or report.get("parent_candidate_sha256") != candidate_hash
            or report.get("old_exact_stage_sha256") != old_hash
            or report.get("new_proposal_sha256")
            != candidate_report.get("new_proposal_sha256")
            or report.get("target_gate", {}).get("status") != "PASS"
        ):
            raise RuntimeError(f"new exact shard has the wrong lineage: {path}")
        index = int(report["shard_index"])
        count = int(report["shard_count"])
        if index in indices:
            raise RuntimeError("duplicate exact shard index")
        indices.add(index)
        expected_parts = count if expected_parts is None else expected_parts
        if expected_parts != count:
            raise RuntimeError("exact shards disagree on shard count")
        with np.load(path, allow_pickle=False) as archive:
            exact_parts.append(
                (
                    index,
                    np.asarray(archive["theta"], dtype=np.float64),
                    np.asarray(archive["log_likelihood"], dtype=np.float64),
                )
            )
            if np.asarray(archive["forbidden_artifacts_used"]).size:
                raise RuntimeError("new exact shard declares forbidden ancestry")
        exact_parents.append({"path": str(path.resolve()), "sha256": digest})
        target_gate = report["target_gate"]
    if expected_parts != len(exact_parts) or indices != set(range(expected_parts)):
        raise RuntimeError("new exact shard set is incomplete")
    exact_parts.sort(key=lambda item: item[0])
    exact_theta = np.vstack([item[1] for item in exact_parts])
    log_likelihood = np.concatenate([item[2] for item in exact_parts])
    if not np.array_equal(exact_theta, new_theta):
        raise RuntimeError("new exact rows do not replay the frozen candidate")
    log_prior = -float(np.log(high - low).sum())
    new_target = np.where(
        np.isfinite(log_likelihood) & (log_likelihood > -1.0e50),
        log_likelihood + log_prior,
        -np.inf,
    )

    subcounts = np.asarray(
        candidate_report.get("optimized_subcomponent_counts"), dtype=np.int64
    )
    if subcounts.shape != (5,) or int(subcounts.sum()) != len(new_theta):
        raise RuntimeError("optimized five-component counts are missing or invalid")
    rng = np.random.default_rng(arguments.seed)
    train_new = []
    validation_one = []
    validation_two = []
    offset = 0
    split_counts = []
    for count in subcounts:
        local = np.arange(offset, offset + int(count))
        rng.shuffle(local)
        if int(count) % 5:
            raise RuntimeError("each component count must divide exactly into 60/20/20")
        first = 3 * int(count) // 5
        second = 4 * int(count) // 5
        train_new.append(local[:first])
        validation_one.append(local[first:second])
        validation_two.append(local[second:])
        split_counts.append([first, second - first, int(count) - second])
        offset += int(count)
    train_new = np.concatenate(train_new)
    validation_one = np.concatenate(validation_one)
    validation_two = np.concatenate(validation_two)
    if len(set(train_new) | set(validation_one) | set(validation_two)) != len(new_theta):
        raise RuntimeError("cross-validation split is not a partition")

    train_old_rows = len(old_theta)
    train_new_rows = len(train_new)
    train_total = train_old_rows + train_new_rows
    fractions = np.asarray(
        [train_old_rows / train_total, train_new_rows / train_total],
        dtype=np.float64,
    )
    old_train_q = logsumexp(
        np.column_stack(
            [np.log(fractions[0]) + old_qold, np.log(fractions[1]) + old_qnew]
        ),
        axis=1,
    )
    new_train_q = logsumexp(
        np.column_stack(
            [
                np.log(fractions[0]) + new_qold[train_new],
                np.log(fractions[1]) + new_qnew[train_new],
            ]
        ),
        axis=1,
    )

    parent_metadata = {
        "schema_version": 1,
        "lineage_class": "independent_tsnpe",
        "model": "ddb-hyperonic",
        "full_prior_dimension": 9,
        "old_exact_stage_sha256": old_hash,
        "old_cross_density_sha256": cross_hash,
        "new_candidate_sha256": candidate_hash,
        "new_exact_parents": exact_parents,
        "split_seed": arguments.seed,
        "optimized_subcomponent_split_counts_train_validation1_validation2": split_counts,
        "forbidden_artifacts_used": [],
    }

    def write_stage(name, role, theta, target, proposal, extra):
        summary = summarize_stage(target, proposal)
        output = arguments.output_dir / f"{name}.npz"
        metadata = {**parent_metadata, "role": role, **extra}
        atomic_save(
            output,
            theta=theta,
            log_target=target,
            log_proposal=proposal,
            log_weight=summary["log_weight"],
            normalized_weight=summary["normalized_weight"],
            prior_low=low,
            prior_high=high,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
        report = {
            **metadata,
            "status": "PASS" if summary["ess"] >= 1000.0 else "FAIL_ESS_GATE",
            "target_gate": target_gate,
            "candidate_rows": len(theta),
            "finite_weight_rows": summary["finite_weight_rows"],
            "ess": summary["ess"],
            "log_evidence": summary["log_evidence"],
            "log_evidence_standard_error": summary["log_evidence_standard_error"],
            "maximum_normalized_weight": summary["maximum_normalized_weight"],
            "output": str(output.resolve()),
            "output_sha256": sha256(output),
        }
        output.with_suffix(".json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        return report

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    train_report = write_stage(
        "crossfit_training_stage",
        "tsnpe_crossvalidation_training_stage",
        np.vstack([old_theta, new_theta[train_new]]),
        np.concatenate([old_target, new_target[train_new]]),
        np.concatenate([old_train_q, new_train_q]),
        {
            "old_rows": train_old_rows,
            "new_rows": train_new_rows,
            "deterministic_mixture_fractions": fractions.tolist(),
        },
    )
    validation_reports = []
    for number, selection in enumerate((validation_one, validation_two), start=1):
        validation_reports.append(
            write_stage(
                f"crossfit_validation_stage_{number}",
                "tsnpe_crossvalidation_validation_stage",
                new_theta[selection],
                new_target[selection],
                new_qnew[selection],
                {"new_rows": int(len(selection)), "validation_fold": number},
            )
        )
    manifest = {
        **parent_metadata,
        "status": "PASS",
        "role": "tsnpe_crossvalidation_partition_manifest",
        "training": train_report,
        "validations": validation_reports,
    }
    (arguments.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
