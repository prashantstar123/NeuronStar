#!/usr/bin/env python3
"""Evaluate the exact hyperonic target on one clean TSNPE MIS shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
if RUNTIME.is_dir():
    sys.path.insert(0, str(RUNTIME))

from workflows import A1Problem
from workflows.resources import DEFAULT_HYPERONIC_TARGET_CERTIFICATE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--certificate", type=Path, default=DEFAULT_HYPERONIC_TARGET_CERTIFICATE
    )
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")
    if arguments.workers < 1:
        raise ValueError("workers must be positive")

    report_path = arguments.candidate.with_suffix(".json")
    report = json.loads(report_path.read_text())
    if (
        report.get("status") != "PASS"
        or report.get("lineage_class") != "independent_tsnpe"
        or report.get("role") not in {
            "tsnpe_full_prior_mis_candidates",
            "tsnpe_adaptive_full_prior_mis_candidates",
        }
        or report.get("forbidden_artifacts_used")
    ):
        raise RuntimeError("MIS shard failed clean-lineage gates")
    if report.get("output_sha256") != sha256(arguments.candidate):
        raise RuntimeError("MIS shard hash does not match its report")

    with np.load(arguments.candidate, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64)
        logq1 = np.asarray(archive["logq_flow_seed_a"], dtype=np.float64)
        logq2 = np.asarray(archive["logq_flow_seed_b"], dtype=np.float64)
        logqu = np.asarray(archive["logq_uniform_full_prior"], dtype=np.float64)
        logq_mild = (
            np.asarray(archive["logq_adaptive_mild"], dtype=np.float64)
            if "logq_adaptive_mild" in archive.files
            else None
        )
        logq_broad = (
            np.asarray(archive["logq_adaptive_broad"], dtype=np.float64)
            if "logq_adaptive_broad" in archive.files
            else None
        )
        component = np.asarray(archive["proposal_component"], dtype=np.uint8)
        counts = np.asarray(archive["component_counts"], dtype=np.int64)
        prior_low = np.asarray(archive["prior_low"], dtype=np.float64)
        prior_high = np.asarray(archive["prior_high"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata"].item()))
        forbidden = np.asarray(archive["forbidden_artifacts_used"])
    rows = len(theta)
    if theta.shape != (rows, 9):
        raise RuntimeError("candidate theta has the wrong shape")
    if any(value.shape != (rows,) for value in (logq1, logq2, logqu, component)):
        raise RuntimeError("candidate density arrays have inconsistent lengths")
    if counts.shape not in {(3,), (5,)} or int(counts.sum()) < rows:
        raise RuntimeError("invalid deterministic-mixture component counts")
    if prior_low.shape != (9,) or prior_high.shape != (9,):
        raise RuntimeError("invalid full-prior bounds")
    if forbidden.size or metadata.get("forbidden_artifacts_used"):
        raise RuntimeError("candidate archive declares forbidden ancestry")
    if not np.isfinite(theta).all() or not np.all(
        (theta >= prior_low) & (theta <= prior_high)
    ):
        raise RuntimeError("candidate contains invalid or out-of-prior rows")
    if not np.isfinite(logq1).all() or not np.isfinite(logq2).all() or not np.isfinite(logqu).all():
        raise RuntimeError("candidate contains a non-finite proposal density")
    if report.get("role") == "tsnpe_adaptive_full_prior_mis_candidates":
        if (
            logq_mild is None
            or logq_broad is None
            or logq_mild.shape != (rows,)
            or logq_broad.shape != (rows,)
            or not np.isfinite(logq_mild).all()
            or not np.isfinite(logq_broad).all()
        ):
            raise RuntimeError("adaptive candidate lacks finite adaptive densities")

    started = time.time()
    problem = A1Problem.from_data_root(
        arguments.data_root, workers=arguments.workers, model="ddb-hyperonic"
    )
    target_gate = problem.certify(arguments.certificate)
    if target_gate.get("status") != "PASS":
        raise RuntimeError(f"shared target certificate failed: {target_gate}")
    log_likelihood, likelihood_components = problem.evaluate_batch(
        theta, return_components=True
    )
    log_likelihood = np.asarray(log_likelihood, dtype=np.float64)
    likelihood_components = np.asarray(likelihood_components, dtype=np.float64)
    valid = np.isfinite(log_likelihood) & (log_likelihood > -1.0e50)

    output_metadata = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_full_prior_mis_exact_shard",
        "model": "ddb-hyperonic",
        "full_prior_dimension": 9,
        "candidate": str(arguments.candidate.resolve()),
        "candidate_sha256": sha256(arguments.candidate),
        "candidate_role": report.get("role"),
        "parent_candidate_sha256": metadata.get("parent_candidate_sha256"),
        "shard_index": metadata.get("shard_index"),
        "shard_count": metadata.get("shard_count"),
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    density_payload = {}
    if logq_mild is not None:
        density_payload["logq_adaptive_mild"] = logq_mild
        density_payload["logq_adaptive_broad"] = logq_broad
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            theta=theta,
            log_likelihood=log_likelihood,
            likelihood_components=likelihood_components,
            valid=valid,
            logq_flow_seed_a=logq1,
            logq_flow_seed_b=logq2,
            logq_uniform_full_prior=logqu,
            **density_payload,
            proposal_component=component,
            component_counts=counts,
            prior_low=prior_low,
            prior_high=prior_high,
            metadata=np.asarray(json.dumps(output_metadata, sort_keys=True)),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
    os.replace(temporary, arguments.output)
    output_report = {
        **output_metadata,
        "candidate_rows": rows,
        "valid_rows": int(valid.sum()),
        "target_gate": target_gate,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "wall_seconds": time.time() - started,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(output_report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output_report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
