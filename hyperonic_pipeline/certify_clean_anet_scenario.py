#!/usr/bin/env python3
"""Certify a frozen hyperonic A-NET scenario by exact full-support IS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logsumexp(value: np.ndarray) -> float:
    maximum = float(np.max(value))
    return maximum + float(np.log(np.exp(value - maximum).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--likelihood", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-posterior-ess", type=float, required=True)
    parser.add_argument("--maximum-normalized-weight", type=float, default=0.01)
    arguments = parser.parse_args()

    with np.load(arguments.proposal, allow_pickle=False) as proposal:
        schema = str(proposal["schema"].item())
        accepted_schemas = {
            "ddb-hyperonic-clean-anet-mixture-proposal-v1",
            "ddb-hyperonic-clean-parity-anet-mixture-proposal-v1",
            "ddb-hyperonic-clean-dual-parity-anet-mixture-v1",
        }
        if schema not in accepted_schemas:
            raise RuntimeError(f"unexpected proposal schema: {schema}")
        theta = np.asarray(proposal["theta"], dtype=np.float64)
        logq = np.asarray(proposal["logq"], dtype=np.float64)
        lower = np.asarray(proposal["prior_low"], dtype=np.float64)
        upper = np.asarray(proposal["prior_high"], dtype=np.float64)
        learned_dim = (
            int(proposal["learned_dim"])
            if "learned_dim" in proposal.files
            else 9
        )
        dual_head = schema == "ddb-hyperonic-clean-dual-parity-anet-mixture-v1"
        head9_checkpoint_sha = (
            str(proposal["head9_checkpoint_sha256"].item())
            if dual_head
            else str(proposal["checkpoint_sha256"].item())
        )
        head7_checkpoint_sha = (
            str(proposal["head7_checkpoint_sha256"].item())
            if dual_head
            else ""
        )
        student_checkpoint_sha = str(
            proposal["student_checkpoint_sha256"].item()
        )
        # The immutable A1 proposal predates the split nuclear/source metadata
        # fields and records its target under the single ``scenario`` key.
        # Scenario proposals carry the newer explicit pair.  This compatibility
        # branch only reads metadata; it never changes samples or densities.
        nuclear_scenario = (
            str(proposal["nuclear_scenario"].item())
            if "nuclear_scenario" in proposal.files
            else "A1"
        )
        source_scenario = (
            str(proposal["source_scenario"].item())
            if "source_scenario" in proposal.files
            else str(proposal["scenario"].item())
        )
        component_weights = (
            np.asarray(proposal["component_weights"], dtype=np.float64)
            if dual_head
            else np.asarray(
                [float(proposal["flow_weight"]), 1.0 - float(proposal["flow_weight"])],
                dtype=np.float64,
            )
        )
        proposal_reference_present = (
            bool(proposal["reference_present"])
            if "reference_present" in proposal.files
            else False
        )
        proposal_forbidden = (
            np.asarray(proposal["forbidden_artifacts_used"])
            if "forbidden_artifacts_used" in proposal.files
            else np.asarray([], dtype="U1")
        )
        held_out_sources_absent = (
            list(np.asarray(proposal["held_out_sources_absent"]).astype(str))
            if "held_out_sources_absent" in proposal.files
            else []
        )
        evaluated_source_absent = (
            bool(proposal["evaluated_source_artifact_absent_from_training"])
            if "evaluated_source_artifact_absent_from_training" in proposal.files
            else source_scenario == "A1"
        )

    with np.load(arguments.likelihood, allow_pickle=False) as evaluated:
        theta_evaluated = np.asarray(evaluated["theta"], dtype=np.float64)
        index = np.asarray(evaluated["index"], dtype=np.int64)
        loglike = np.asarray(evaluated["logl"], dtype=np.float64)
        likelihood_nuclear = str(evaluated["nuclear_scenario"].item())
        likelihood_source = str(evaluated["source_scenario"].item())
        reference_present = bool(evaluated["reference_present"])

    rows = len(theta)
    if theta.shape != (rows, 9) or logq.shape != (rows,):
        raise RuntimeError("invalid A-NET proposal shapes")
    if lower.shape != (9,) or upper.shape != (9,) or np.any(upper <= lower):
        raise RuntimeError("invalid canonical-prior bounds")
    if theta_evaluated.shape != theta.shape or loglike.shape != (rows,):
        raise RuntimeError("invalid exact-likelihood shapes")
    if not np.array_equal(index, np.arange(rows, dtype=np.int64)):
        raise RuntimeError("exact-likelihood rows are not in proposal order")
    if not np.array_equal(theta, theta_evaluated):
        raise RuntimeError("exact-likelihood parameters differ from the proposal")
    if (nuclear_scenario, source_scenario) != (
        likelihood_nuclear,
        likelihood_source,
    ):
        raise RuntimeError("proposal and exact-likelihood scenarios differ")
    if reference_present:
        raise RuntimeError("forbidden reference values are present")
    if proposal_reference_present or proposal_forbidden.size:
        raise RuntimeError("proposal declares forbidden reference ancestry")
    if schema in {
        "ddb-hyperonic-clean-parity-anet-mixture-proposal-v1",
        "ddb-hyperonic-clean-dual-parity-anet-mixture-v1",
    }:
        if held_out_sources_absent != ["J0614", "J1231", "J1614"]:
            raise RuntimeError("proposal does not lock all held-out sources absent")
        if source_scenario != "A1" and not evaluated_source_absent:
            raise RuntimeError("evaluated source was not certified absent from training")
    if not np.isfinite(theta).all() or not np.isfinite(logq).all():
        raise RuntimeError("non-finite proposal values")
    if not np.isfinite(loglike).all():
        raise RuntimeError("non-finite exact likelihood values")
    if not np.all((theta >= lower) & (theta <= upper)):
        raise RuntimeError("proposal row outside the canonical prior")

    valid = loglike > -1.0e50
    if not valid.any():
        raise RuntimeError("the exact likelihood rejected every proposal row")
    logprior = -float(np.log(upper - lower).sum())
    logweight = np.full(rows, -np.inf, dtype=np.float64)
    logweight[valid] = logprior + loglike[valid] - logq[valid]
    normalizer = logsumexp(logweight[valid])
    weight = np.zeros(rows, dtype=np.float64)
    weight[valid] = np.exp(logweight[valid] - normalizer)
    posterior_ess = float(1.0 / np.sum(weight * weight))
    log_evidence = float(normalizer - np.log(rows))
    maximum_weight = float(weight.max())
    log_evidence_standard_error = float(
        math.sqrt(max(rows / posterior_ess - 1.0, 0.0) / rows)
    )
    ess_pass = posterior_ess >= arguments.minimum_posterior_ess
    weight_pass = maximum_weight <= arguments.maximum_normalized_weight
    passed = ess_pass and weight_pass

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray("ddb-hyperonic-clean-anet-scenario-certification-v1"),
            theta=theta,
            logq=logq,
            logl=loglike,
            normalized_weight=weight,
            prior_low=lower,
            prior_high=upper,
            nuclear_scenario=np.asarray(nuclear_scenario),
            source_scenario=np.asarray(source_scenario),
            learned_dim=np.int64(learned_dim),
            component_weights=component_weights,
            head9_checkpoint_sha256=np.asarray(head9_checkpoint_sha),
            head7_checkpoint_sha256=np.asarray(head7_checkpoint_sha),
            student_checkpoint_sha256=np.asarray(student_checkpoint_sha),
            proposal_sha256=np.asarray(sha256(arguments.proposal)),
            likelihood_sha256=np.asarray(sha256(arguments.likelihood)),
            valid_rows=np.int64(valid.sum()),
            posterior_ess=np.float64(posterior_ess),
            log_evidence=np.float64(log_evidence),
            log_evidence_standard_error=np.float64(log_evidence_standard_error),
            maximum_normalized_weight=np.float64(maximum_weight),
            minimum_posterior_ess=np.float64(arguments.minimum_posterior_ess),
            maximum_normalized_weight_gate=np.float64(
                arguments.maximum_normalized_weight
            ),
            gate_pass=np.bool_(passed),
        )
    os.replace(temporary, arguments.output)

    report = {
        "status": "PASS" if passed else "FAIL",
        "nuclear_scenario": nuclear_scenario,
        "source_scenario": source_scenario,
        "learned_dim": learned_dim,
        "proposal_schema": schema,
        "component_weights": component_weights.tolist(),
        "rows": rows,
        "valid_rows": int(valid.sum()),
        "posterior_ess": posterior_ess,
        "minimum_posterior_ess": arguments.minimum_posterior_ess,
        "ess_gate_pass": ess_pass,
        "maximum_normalized_weight": maximum_weight,
        "maximum_normalized_weight_gate": arguments.maximum_normalized_weight,
        "maximum_weight_gate_pass": weight_pass,
        "log_evidence": log_evidence,
        "log_evidence_standard_error": log_evidence_standard_error,
        "head9_checkpoint_sha256": head9_checkpoint_sha,
        "head7_checkpoint_sha256": head7_checkpoint_sha or None,
        "student_checkpoint_sha256": student_checkpoint_sha,
        "proposal_sha256": sha256(arguments.proposal),
        "likelihood_sha256": sha256(arguments.likelihood),
        "output_sha256": sha256(arguments.output),
        "reference_present": False,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
