#!/usr/bin/env python3
"""Certify independent batches drawn from one frozen A-NET proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


PROPOSAL_SCHEMAS = {
    "ddb-hyperonic-clean-parity-anet-mixture-proposal-v1",
    "ddb-hyperonic-clean-dual-parity-anet-mixture-v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(source, key):
    return source[key].item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        nargs=2,
        metavar=("PROPOSAL", "LIKELIHOOD"),
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-posterior-ess", type=float, required=True)
    parser.add_argument("--maximum-normalized-weight", type=float, default=0.01)
    arguments = parser.parse_args()
    if len(arguments.batch) < 2:
        raise ValueError("batch certification requires at least two batches")
    if arguments.output.suffix != ".npz":
        raise ValueError("--output must end in .npz so the JSON receipt is separate")

    theta_parts = []
    logq_parts = []
    loglike_parts = []
    manifest = []
    reference_metadata = None

    for proposal_name, likelihood_name in arguments.batch:
        proposal_path = Path(proposal_name)
        likelihood_path = Path(likelihood_name)
        with np.load(proposal_path, allow_pickle=False) as proposal:
            proposal_schema = str(scalar(proposal, "schema"))
            if proposal_schema not in PROPOSAL_SCHEMAS:
                raise RuntimeError(f"unexpected proposal schema: {proposal_path}")
            theta = np.asarray(proposal["theta"], dtype=np.float64)
            logq = np.asarray(proposal["logq"], dtype=np.float64)
            lower = np.asarray(proposal["prior_low"], dtype=np.float64)
            upper = np.asarray(proposal["prior_high"], dtype=np.float64)
            held_out = list(
                np.asarray(proposal["held_out_sources_absent"]).astype(str)
            )
            forbidden = np.asarray(proposal["forbidden_artifacts_used"])
            dual_head = (
                proposal_schema
                == "ddb-hyperonic-clean-dual-parity-anet-mixture-v1"
            )
            metadata = {
                "proposal_schema": proposal_schema,
                "head9_checkpoint_sha256": str(
                    scalar(
                        proposal,
                        "head9_checkpoint_sha256"
                        if dual_head
                        else "checkpoint_sha256",
                    )
                ),
                "head7_checkpoint_sha256": (
                    str(scalar(proposal, "head7_checkpoint_sha256"))
                    if dual_head
                    else ""
                ),
                "head9_standardization_sha256": str(
                    scalar(
                        proposal,
                        "head9_standardization_sha256"
                        if dual_head
                        else "standardization_sha256",
                    )
                ),
                "head7_standardization_sha256": (
                    str(scalar(proposal, "head7_standardization_sha256"))
                    if dual_head
                    else ""
                ),
                "student_checkpoint_sha256": str(
                    scalar(proposal, "student_checkpoint_sha256")
                ),
                "nuclear_scenario": str(scalar(proposal, "nuclear_scenario")),
                "source_scenario": str(scalar(proposal, "source_scenario")),
                "component_weights": (
                    np.asarray(proposal["component_weights"], dtype=np.float64)
                    if dual_head
                    else np.asarray(
                        [
                            float(scalar(proposal, "flow_weight")),
                            1.0 - float(scalar(proposal, "flow_weight")),
                        ],
                        dtype=np.float64,
                    )
                ),
                "flow_steps": int(scalar(proposal, "flow_steps")),
                "cloud_seed": int(scalar(proposal, "cloud_seed")),
                "held_out_sources_absent": held_out,
                "evaluated_source_artifact_absent_from_training": bool(
                    scalar(
                        proposal,
                        "evaluated_source_artifact_absent_from_training",
                    )
                ),
                "prior_low": lower,
                "prior_high": upper,
                "cloud": np.asarray(proposal["cloud"], dtype=np.float64),
                "nuclear_observation": np.asarray(
                    proposal["nuclear_observation"], dtype=np.float64
                ),
            }
            proposal_seed = int(scalar(proposal, "proposal_seed"))
            proposal_reference = bool(scalar(proposal, "reference_present"))

        with np.load(likelihood_path, allow_pickle=False) as likelihood:
            evaluated_theta = np.asarray(likelihood["theta"], dtype=np.float64)
            index = np.asarray(likelihood["index"], dtype=np.int64)
            loglike = np.asarray(likelihood["logl"], dtype=np.float64)
            likelihood_nuclear = str(scalar(likelihood, "nuclear_scenario"))
            likelihood_source = str(scalar(likelihood, "source_scenario"))
            likelihood_reference = bool(scalar(likelihood, "reference_present"))
            likelihood_proposal_sha = str(
                scalar(likelihood, "proposal_sha256")
            )
            certificate_sha = (
                str(scalar(likelihood, "certificate_sha256"))
                if "certificate_sha256" in likelihood.files
                else None
            )

        rows = len(theta)
        if theta.shape != (rows, 9) or logq.shape != (rows,):
            raise RuntimeError(f"invalid proposal arrays: {proposal_path}")
        if evaluated_theta.shape != theta.shape or loglike.shape != (rows,):
            raise RuntimeError(f"invalid likelihood arrays: {likelihood_path}")
        if not np.array_equal(index, np.arange(rows, dtype=np.int64)):
            raise RuntimeError(f"likelihood row order differs: {likelihood_path}")
        if not np.array_equal(theta, evaluated_theta):
            raise RuntimeError(f"proposal/likelihood theta differs: {proposal_path}")
        if likelihood_proposal_sha != sha256(proposal_path):
            raise RuntimeError(
                f"likelihood does not reference this proposal: {likelihood_path}"
            )
        if metadata["nuclear_scenario"] != likelihood_nuclear or metadata[
            "source_scenario"
        ] != likelihood_source:
            raise RuntimeError("proposal and likelihood scenarios differ")
        if proposal_reference or likelihood_reference or forbidden.size:
            raise RuntimeError("forbidden reference ancestry is present")
        if held_out != ["J0614", "J1231", "J1614"]:
            raise RuntimeError("held-out source exclusion is not locked")
        if metadata["source_scenario"] != "A1" and not metadata[
            "evaluated_source_artifact_absent_from_training"
        ]:
            raise RuntimeError("evaluated held-out source entered training")
        if not np.isfinite(theta).all() or not np.isfinite(logq).all():
            raise RuntimeError("non-finite proposal values")
        if not np.isfinite(loglike).all():
            raise RuntimeError("non-finite likelihood values")
        if not np.all((theta >= lower) & (theta <= upper)):
            raise RuntimeError("proposal row lies outside the canonical prior")

        if reference_metadata is None:
            reference_metadata = metadata
        else:
            for key in (
                "proposal_schema",
                "head9_checkpoint_sha256",
                "head7_checkpoint_sha256",
                "head9_standardization_sha256",
                "head7_standardization_sha256",
                "student_checkpoint_sha256",
                "nuclear_scenario",
                "source_scenario",
                "flow_steps",
                "cloud_seed",
                "held_out_sources_absent",
                "evaluated_source_artifact_absent_from_training",
            ):
                if metadata[key] != reference_metadata[key]:
                    raise RuntimeError(f"batch metadata differs for {key}")
            for key in (
                "prior_low",
                "prior_high",
                "component_weights",
                "cloud",
                "nuclear_observation",
            ):
                if not np.array_equal(metadata[key], reference_metadata[key]):
                    raise RuntimeError(f"batch array metadata differs for {key}")

        theta_parts.append(theta)
        logq_parts.append(logq)
        loglike_parts.append(loglike)
        manifest.append(
            {
                "proposal": str(proposal_path.resolve()),
                "proposal_sha256": sha256(proposal_path),
                "likelihood": str(likelihood_path.resolve()),
                "likelihood_sha256": sha256(likelihood_path),
                "target_certificate_sha256": certificate_sha,
                "proposal_seed": proposal_seed,
                "rows": rows,
                "valid_rows": int(np.sum(loglike > -1.0e50)),
            }
        )

    assert reference_metadata is not None
    theta = np.concatenate(theta_parts)
    logq = np.concatenate(logq_parts)
    loglike = np.concatenate(loglike_parts)
    lower = np.asarray(reference_metadata["prior_low"])
    upper = np.asarray(reference_metadata["prior_high"])
    valid = loglike > -1.0e50
    logprior = -float(np.log(upper - lower).sum())
    logweight = np.full(len(theta), -np.inf, dtype=np.float64)
    logweight[valid] = logprior + loglike[valid] - logq[valid]
    normalizer = logsumexp(logweight[valid])
    weight = np.zeros(len(theta), dtype=np.float64)
    weight[valid] = np.exp(logweight[valid] - normalizer)
    posterior_ess = float(1.0 / np.sum(weight * weight))
    maximum_weight = float(weight.max())
    log_evidence = float(normalizer - np.log(len(theta)))
    log_evidence_error = float(
        math.sqrt(max(len(theta) / posterior_ess - 1.0, 0.0) / len(theta))
    )
    ess_pass = posterior_ess >= arguments.minimum_posterior_ess
    weight_pass = maximum_weight <= arguments.maximum_normalized_weight
    passed = ess_pass and weight_pass
    manifest_json = json.dumps(manifest, sort_keys=True)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(
                "ddb-hyperonic-clean-anet-scenario-batch-certification-v2"
            ),
            theta=theta,
            logq=logq,
            logl=loglike,
            normalized_weight=weight,
            prior_low=lower,
            prior_high=upper,
            nuclear_scenario=np.asarray(reference_metadata["nuclear_scenario"]),
            source_scenario=np.asarray(reference_metadata["source_scenario"]),
            learned_dim=np.int64(9),
            component_weights=np.asarray(
                reference_metadata["component_weights"], dtype=np.float64
            ),
            head9_checkpoint_sha256=np.asarray(
                reference_metadata["head9_checkpoint_sha256"]
            ),
            head7_checkpoint_sha256=np.asarray(
                reference_metadata["head7_checkpoint_sha256"]
            ),
            student_checkpoint_sha256=np.asarray(
                reference_metadata["student_checkpoint_sha256"]
            ),
            batch_manifest_json=np.asarray(manifest_json),
            valid_rows=np.int64(valid.sum()),
            posterior_ess=np.float64(posterior_ess),
            log_evidence=np.float64(log_evidence),
            log_evidence_standard_error=np.float64(log_evidence_error),
            maximum_normalized_weight=np.float64(maximum_weight),
            minimum_posterior_ess=np.float64(arguments.minimum_posterior_ess),
            maximum_normalized_weight_gate=np.float64(
                arguments.maximum_normalized_weight
            ),
            gate_pass=np.bool_(passed),
            reference_present=np.bool_(False),
        )
    os.replace(temporary, arguments.output)

    report = {
        "status": "PASS" if passed else "FAIL",
        "nuclear_scenario": reference_metadata["nuclear_scenario"],
        "source_scenario": reference_metadata["source_scenario"],
        "rows": len(theta),
        "valid_rows": int(valid.sum()),
        "posterior_ess": posterior_ess,
        "minimum_posterior_ess": arguments.minimum_posterior_ess,
        "ess_gate_pass": ess_pass,
        "maximum_normalized_weight": maximum_weight,
        "maximum_normalized_weight_gate": arguments.maximum_normalized_weight,
        "maximum_weight_gate_pass": weight_pass,
        "log_evidence": log_evidence,
        "log_evidence_standard_error": log_evidence_error,
        "proposal_schema": reference_metadata["proposal_schema"],
        "component_weights": np.asarray(
            reference_metadata["component_weights"]
        ).tolist(),
        "head9_checkpoint_sha256": reference_metadata[
            "head9_checkpoint_sha256"
        ],
        "head7_checkpoint_sha256": (
            reference_metadata["head7_checkpoint_sha256"] or None
        ),
        "student_checkpoint_sha256": reference_metadata[
            "student_checkpoint_sha256"
        ],
        "batches": manifest,
        "output_sha256": sha256(arguments.output),
        "reference_present": False,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
