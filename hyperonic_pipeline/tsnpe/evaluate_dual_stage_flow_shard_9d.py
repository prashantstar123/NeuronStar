#!/usr/bin/env python3
"""Evaluate one new-candidate shard for clean two-generation TSNPE MIS."""

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

    candidate_report = json.loads(arguments.candidate.with_suffix(".json").read_text())
    candidate_hash = sha256(arguments.candidate)
    if (
        candidate_report.get("status") != "PASS"
        or candidate_report.get("lineage_class") != "independent_tsnpe"
        or candidate_report.get("role") != "tsnpe_dual_stage_new_proposal_candidates"
        or candidate_report.get("full_prior_dimension") != 9
        or candidate_report.get("forbidden_artifacts_used")
        or candidate_report.get("output_sha256") != candidate_hash
    ):
        raise RuntimeError("dual-stage candidate shard failed provenance gates")
    with np.load(arguments.candidate, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64)
        qold = np.asarray(archive["logq_old_stage"], dtype=np.float64)
        qnew = np.asarray(archive["logq_new_stage"], dtype=np.float64)
        component = np.asarray(archive["proposal_component"], dtype=np.uint8)
        counts = np.asarray(archive["component_counts"], dtype=np.int64)
        low = np.asarray(archive["prior_low"], dtype=np.float64)
        high = np.asarray(archive["prior_high"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata"].item()))
        forbidden = np.asarray(archive["forbidden_artifacts_used"])
    rows = len(theta)
    if theta.shape != (rows, 9) or any(
        value.shape != (rows,) for value in (qold, qnew, component)
    ):
        raise RuntimeError("dual-stage candidate arrays have inconsistent shapes")
    if counts.shape != (3,) or int(counts.sum()) < rows:
        raise RuntimeError("invalid global new-proposal component counts")
    if forbidden.size or metadata.get("forbidden_artifacts_used"):
        raise RuntimeError("dual-stage candidate declares forbidden ancestry")
    if not np.isfinite(theta).all() or not np.all((theta >= low) & (theta <= high)):
        raise RuntimeError("dual-stage candidate contains invalid theta")
    if not np.isfinite(qold).all() or not np.isfinite(qnew).all():
        raise RuntimeError("dual-stage candidate contains non-finite density")

    started = time.time()
    problem = A1Problem.from_data_root(
        arguments.data_root, workers=arguments.workers, model="ddb-hyperonic"
    )
    target_gate = problem.certify(arguments.certificate)
    if target_gate.get("status") != "PASS":
        raise RuntimeError(f"shared target certificate failed: {target_gate}")
    log_likelihood, components = problem.evaluate_batch(theta, return_components=True)
    log_likelihood = np.asarray(log_likelihood, dtype=np.float64)
    components = np.asarray(components, dtype=np.float64)
    valid = np.isfinite(log_likelihood) & (log_likelihood > -1.0e50)

    output_metadata = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_dual_stage_new_exact_shard",
        "model": "ddb-hyperonic",
        "full_prior_dimension": 9,
        "candidate_sha256": candidate_hash,
        "parent_candidate_sha256": metadata.get("parent_candidate_sha256"),
        "old_proposal_sha256": metadata.get("old_proposal_sha256"),
        "new_proposal_sha256": metadata.get("new_proposal_sha256"),
        "old_exact_stage_sha256": metadata.get("old_exact_stage_sha256"),
        "shard_index": metadata.get("shard_index"),
        "shard_count": metadata.get("shard_count"),
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            theta=theta,
            log_likelihood=log_likelihood,
            likelihood_components=components,
            valid=valid,
            logq_old_stage=qold,
            logq_new_stage=qnew,
            proposal_component=component,
            component_counts=counts,
            prior_low=low,
            prior_high=high,
            metadata=np.asarray(json.dumps(output_metadata, sort_keys=True)),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
    os.replace(temporary, arguments.output)
    report = {
        **output_metadata,
        "candidate_rows": rows,
        "valid_rows": int(valid.sum()),
        "target_gate": target_gate,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "wall_seconds": time.time() - started,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
