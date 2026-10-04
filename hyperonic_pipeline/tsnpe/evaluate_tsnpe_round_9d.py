#!/usr/bin/env python3
"""Evaluate and accept/reject one clean hyperonic TSNPE proposal round."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = ROOT / "runtime"
if RUNTIME_ROOT.is_dir():
    sys.path.insert(0, str(RUNTIME_ROOT))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ddb-amortized-inference-paper"))

from likelihoods.nuclear import A1_SIGMA
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
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--accept-seed", type=int, required=True)
    parser.add_argument(
        "--reference-margin",
        type=float,
        default=1.0,
        help=(
            "fixed log-likelihood safety margin above the independent "
            "round-zero pilot maximum"
        ),
    )
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument(
        "--certificate", type=Path, default=DEFAULT_HYPERONIC_TARGET_CERTIFICATE
    )
    arguments = parser.parse_args()
    if arguments.workers < 1:
        raise ValueError("workers must be positive")
    if arguments.reference_margin <= 0.0:
        raise ValueError("reference margin must be positive")
    if arguments.output.exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")

    proposal_report_path = arguments.proposal.with_suffix(".json")
    proposal_report = json.loads(proposal_report_path.read_text())
    if proposal_report.get("lineage_class") != "independent_tsnpe":
        raise RuntimeError("proposal lacks independent TSNPE lineage")
    if proposal_report.get("role") != "tsnpe_round_proposal":
        raise RuntimeError("input is not a TSNPE round proposal")
    if proposal_report.get("forbidden_artifacts_used"):
        raise RuntimeError("proposal declares forbidden ancestry")
    if proposal_report.get("output_sha256") != sha256(arguments.proposal):
        raise RuntimeError("proposal hash does not match its report")

    with np.load(arguments.proposal, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64)
        proposal_log_density = (
            np.asarray(archive["logq"], dtype=np.float64)
            if "logq" in archive.files
            else None
        )
        prior_low = np.asarray(archive["prior_low"], dtype=np.float64)
        prior_high = np.asarray(archive["prior_high"], dtype=np.float64)
        pilot_rows = int(archive["pilot_rows"])
        metadata = json.loads(str(archive["metadata"].item()))
        forbidden = np.asarray(archive["forbidden_artifacts_used"])
    round_index = int(metadata["round_index"])
    proposal_mode = str(metadata.get("proposal_mode", "restricted_uniform_prior"))
    if forbidden.size or metadata.get("forbidden_artifacts_used"):
        raise RuntimeError("proposal archive declares forbidden ancestry")
    if theta.ndim != 2 or theta.shape[1] != 9:
        raise RuntimeError("proposal theta must have shape (N,9)")
    if proposal_log_density is not None and proposal_log_density.shape != (len(theta),):
        raise RuntimeError("proposal log density has the wrong shape")
    if prior_low.shape != (9,) or prior_high.shape != (9,):
        raise RuntimeError("proposal prior bounds have wrong shape")
    if not np.isfinite(theta).all() or not np.all(
        (theta >= prior_low) & (theta <= prior_high)
    ):
        raise RuntimeError("proposal contains an invalid or out-of-prior row")
    if round_index > 0 and pilot_rows > 0:
        raise RuntimeError("only round zero may contain pilot rows")
    if round_index == 0 and pilot_rows == 0 and arguments.reference_report is None:
        raise RuntimeError("a round-zero continuation shard requires its pilot report")

    started = time.time()
    problem = A1Problem.from_data_root(
        arguments.data_root, workers=arguments.workers, model="ddb-hyperonic"
    )
    target_gate = problem.certify(arguments.certificate)
    if target_gate.get("status") != "PASS":
        raise RuntimeError(f"shared target gate failed: {target_gate}")
    prediction, log_astro, components = problem.evaluate_evidence_cache_batch(theta)
    log_mmax = components[:, 0]
    log_other = np.sum(components[:, 1:], axis=1)
    valid = (
        np.isfinite(prediction).all(axis=1)
        & np.isfinite(log_mmax)
        & np.isfinite(log_other)
    )

    if proposal_mode == "normalized_defensive_astrophysical_bridge":
        if proposal_log_density is None or not np.isfinite(proposal_log_density).all():
            raise RuntimeError("normalized bridge proposal lacks a finite density")
        prior_log_density = -float(np.log(prior_high - prior_low).sum())
        log_acceptance_ratio = prior_log_density + log_astro - proposal_log_density
        reference_quantity = log_acceptance_ratio
    elif proposal_mode == "restricted_uniform_prior":
        if proposal_log_density is not None:
            raise RuntimeError("restricted-prior proposal unexpectedly stores logq")
        log_acceptance_ratio = log_mmax + log_other
        reference_quantity = log_other
    else:
        raise RuntimeError(f"unsupported TSNPE proposal mode: {proposal_mode}")

    reference_parent = None
    if round_index == 0 and pilot_rows > 0:
        pilot_valid = valid[:pilot_rows]
        if not pilot_valid.any():
            raise RuntimeError("round-zero pilot has no valid exact-likelihood row")
        log_reference = float(
            np.max(reference_quantity[:pilot_rows][pilot_valid])
            + arguments.reference_margin
        )
        restricted_prior_log_reference = float(
            np.max(log_other[:pilot_rows][pilot_valid])
            + arguments.reference_margin
        )
        reference_margin = float(arguments.reference_margin)
    else:
        if arguments.reference_report is None:
            raise ValueError("this exact round requires the frozen round-zero reference")
        reference = json.loads(arguments.reference_report.read_text())
        if reference.get("lineage_class") != "independent_tsnpe":
            raise RuntimeError("acceptance reference lacks clean TSNPE lineage")
        if reference.get("round_index") != 0:
            raise RuntimeError("acceptance reference is not from round zero")
        if reference.get("forbidden_artifacts_used"):
            raise RuntimeError("acceptance reference declares forbidden ancestry")
        if proposal_mode == "restricted_uniform_prior":
            log_reference = float(
                reference.get("restricted_prior_log_reference", reference["log_reference"])
            )
        else:
            log_reference = float(reference["log_reference"])
        restricted_prior_log_reference = float(
            reference.get("restricted_prior_log_reference", log_reference)
        )
        reference_margin = float(reference["reference_margin"])
        reference_parent = {
            "path": str(arguments.reference_report.resolve()),
            "sha256": sha256(arguments.reference_report),
        }

    reference_exceedance = valid & (reference_quantity > log_reference)
    # The accept/reject simulator is proportional to the exact astrophysical
    # likelihood only while the frozen reference is an upper envelope.  Never
    # silently hide a failed envelope behind probability clipping.
    if reference_exceedance.any():
        raise RuntimeError(
            "frozen TSNPE acceptance reference was exceeded by "
            f"{int(reference_exceedance.sum())} rows; maximum excess is "
            f"{float(np.max(reference_quantity[reference_exceedance] - log_reference)):.6g}"
        )

    if proposal_mode == "normalized_defensive_astrophysical_bridge":
        log_acceptance = np.minimum(log_acceptance_ratio - log_reference, 0.0)
    else:
        log_acceptance = log_mmax + np.minimum(log_other - log_reference, 0.0)
    probability = np.where(valid, np.exp(np.minimum(log_acceptance, 0.0)), 0.0)
    rng = np.random.default_rng(arguments.accept_seed)
    accepted = valid & (rng.random(len(theta)) < probability)
    training_mask = accepted.copy()
    training_mask[:pilot_rows] = False
    x = prediction + rng.normal(0.0, 1.0, prediction.shape) * A1_SIGMA[None, :]
    x[~training_mask] = np.nan

    output_metadata = {
        "schema_version": 1,
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_exact_round",
        "model": "ddb-hyperonic",
        "round_index": round_index,
        "proposal_mode": proposal_mode,
        "full_prior_dimension": 9,
        "proposal": str(arguments.proposal.resolve()),
        "proposal_sha256": sha256(arguments.proposal),
        "reference_parent": reference_parent,
        "accept_seed": arguments.accept_seed,
        "reference_margin": reference_margin,
        "restricted_prior_log_reference": restricted_prior_log_reference,
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        theta=theta,
        prediction=prediction,
        astrophysical_components=components,
        x=x,
        valid=valid,
        accepted=accepted,
        training_mask=training_mask,
        log_reference=np.asarray(log_reference),
        restricted_prior_log_reference=np.asarray(restricted_prior_log_reference),
        prior_low=prior_low,
        prior_high=prior_high,
        metadata=np.asarray(json.dumps(output_metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    report = {
        "status": "PASS",
        **output_metadata,
        "candidate_rows": int(len(theta)),
        "pilot_rows": pilot_rows,
        "valid_rows": int(valid.sum()),
        "accepted_rows_including_pilot": int(accepted.sum()),
        "training_rows": int(training_mask.sum()),
        "acceptance_fraction": float(training_mask.sum() / max(len(theta) - pilot_rows, 1)),
        "log_reference": log_reference,
        "reference_exceedance_rows": 0,
        "maximum_reference_quantity_minus_reference": float(
            np.max(reference_quantity[valid] - log_reference)
            if valid.any()
            else -np.inf
        ),
        "target_gate": target_gate,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "wall_seconds": time.time() - started,
    }
    report_path = arguments.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
