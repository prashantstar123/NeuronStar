#!/usr/bin/env python3
"""Run frozen A-NET proposal generation plus exact corrected IS."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.anet import FMPEProposal, mass_radius_clouds
from inference.common import compute_importance_weights
from workflows import A1Problem
from workflows.nuclear_scenarios import NUCLEAR_SCENARIO_NAMES
from workflows.data_gate import sha256
from workflows.resources import default_target_certificate
from workflows.source_scenarios import (
    SOURCE_SCENARIO_NAMES,
    source_substitution,
)

CHECKPOINT_DIR = ROOT / "inference" / "anet" / "checkpoints"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("ddb",), default="ddb")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--standardization", type=Path)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--scenario", choices=NUCLEAR_SCENARIO_NAMES, default="A1"
    )
    parser.add_argument(
        "--source-scenario", choices=SOURCE_SCENARIO_NAMES, default="A1"
    )
    parser.add_argument("--proposals", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resample-seed", type=int, default=1)
    parser.add_argument("--minimum-ess", type=float, default=40.0)
    parser.add_argument("--device")
    parser.add_argument("--skip-target-gate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    certificate = arguments.certificate or default_target_certificate(
        arguments.source_scenario, arguments.model
    )
    if arguments.skip_target_gate and not arguments.smoke:
        raise ValueError("--skip-target-gate is permitted only with --smoke")

    started = time.time()
    problem = A1Problem.from_data_root(
        arguments.data_root,
        workers=arguments.workers,
        nuclear_scenario=arguments.scenario,
        source_scenario=arguments.source_scenario,
        source_data_root=arguments.source_data_root,
        model=arguments.model,
    )
    if not arguments.skip_target_gate:
        gate = problem.certify(certificate)
        gate["certificate_sha256"] = sha256(certificate)
        if gate["status"] != "PASS":
            raise RuntimeError(f"shared target gate failed: {gate}")
    else:
        gate = {"status": "SKIPPED"}

    replacement = source_substitution(arguments.source_scenario)
    source_artifact = ""
    source_artifact_sha256 = ""
    if replacement is not None:
        source_root = arguments.source_data_root or arguments.data_root
        source_artifact = replacement.filename
        source_artifact_sha256 = sha256(source_root / replacement.filename)
        gate["source_artifact"] = source_artifact
        gate["source_artifact_sha256"] = source_artifact_sha256

    checkpoint = arguments.checkpoint or CHECKPOINT_DIR / "fmpe9c_net.pt"
    standardization = arguments.standardization or (
        CHECKPOINT_DIR / "fmpe9c_std.npz"
    )
    proposal = FMPEProposal(
        checkpoint,
        standardization,
        device=arguments.device,
    )
    clouds = mass_radius_clouds(
        arguments.data_root,
        points=proposal.cloud_points,
        seed=0,
        source_scenario=arguments.source_scenario,
        source_data_root=arguments.source_data_root,
    )
    checkpoint_sha256 = sha256(checkpoint)
    observation = problem.target.nuclear_observation
    theta, log_proposal = proposal.sample_with_log_density(
        clouds, observation, size=arguments.proposals, seed=arguments.seed
    )
    in_bounds = np.all(
        (theta >= problem.prior_low) & (theta <= problem.prior_high), axis=1
    )
    log_likelihood = np.full(arguments.proposals, -np.inf, dtype=np.float64)
    log_likelihood[in_bounds] = problem.evaluate_batch(theta[in_bounds])
    log_likelihood[log_likelihood <= -1e50] = -np.inf
    log_prior = np.full(arguments.proposals, -np.inf, dtype=np.float64)
    log_prior[in_bounds] = -np.sum(
        np.log(problem.prior_high - problem.prior_low)
    )
    importance = compute_importance_weights(
        log_likelihood + log_prior, log_proposal
    )
    posterior, posterior_index = importance.resample(
        theta, size=arguments.proposals, seed=arguments.resample_seed
    )
    passed = importance.ess >= arguments.minimum_ess
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        theta=posterior,
        posterior_index=posterior_index,
        raw=theta,
        w=importance.normalized_weight,
        logw=importance.log_weight,
        logl=log_likelihood,
        logq=log_proposal,
        ess=importance.ess,
        logz=importance.log_evidence,
        logzerr=importance.log_evidence_standard_error,
        target_gate=np.array(json.dumps(gate, sort_keys=True)),
        model=np.array("ddb"),
        checkpoint_sha256=np.array(checkpoint_sha256),
        nuclear_scenario=np.array(arguments.scenario),
        source_scenario=np.array(arguments.source_scenario),
        source_artifact=np.array(source_artifact),
        source_artifact_sha256=np.array(source_artifact_sha256),
        evaluated_source_artifact_absent_from_training=np.asarray(True),
        nuclear_observation=np.asarray(observation, dtype=np.float64),
    )
    report = {
        "status": (
            "SMOKE_COMPLETED"
            if arguments.smoke
            else ("PASS" if passed else "FAIL_ESS_GATE")
        ),
        "method": "A-NET+IS",
        "model": "ddb",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "nuclear_scenario": arguments.scenario,
        "source_scenario": arguments.source_scenario,
        "proposals": arguments.proposals,
        "in_bounds": int(in_bounds.sum()),
        "finite_weights": importance.finite_count,
        "ess": importance.ess,
        "minimum_ess": arguments.minimum_ess,
        "log_evidence": importance.log_evidence,
        "log_evidence_standard_error": importance.log_evidence_standard_error,
        "wall_seconds": time.time() - started,
        "output": str(arguments.output),
        "target_gate": gate,
    }
    if replacement is not None:
        report["source_artifact"] = source_artifact
        report["source_artifact_sha256"] = source_artifact_sha256
    report_path = arguments.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if arguments.smoke or passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
