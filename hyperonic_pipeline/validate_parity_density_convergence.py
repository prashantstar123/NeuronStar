#!/usr/bin/env python3
"""Check Heun-step convergence of a frozen clean parity A-NET density."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from inference.anet.proposal import mass_radius_clouds
from sample_clean_hyperonic_parity_mixture import ParityFMPE
from workflows.nuclear_scenarios import NUCLEAR_SCENARIO_NAMES, nuclear_observation
from workflows.source_scenarios import SOURCE_SCENARIO_NAMES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--standardization", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--source-scenario", choices=SOURCE_SCENARIO_NAMES, default="A1")
    parser.add_argument("--nuclear-scenario", choices=NUCLEAR_SCENARIO_NAMES, default="A1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--coarse-steps", type=int, default=512)
    parser.add_argument("--fine-steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--cloud-seed", type=int, default=777001)
    parser.add_argument("--maximum-p95-logq-difference", type=float, default=0.05)
    parser.add_argument("--maximum-logq-difference", type=float, default=0.50)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.rows < 1 or not 1 <= arguments.coarse_steps < arguments.fine_steps:
        raise ValueError("require positive rows and 1 <= coarse steps < fine steps")

    training = json.loads(arguments.training_report.read_text())
    if training.get("status") != "PROPOSAL_READY_FOR_IS":
        raise RuntimeError("training checkpoint is not ready for proposal validation")
    if training.get("reference_present") is not False:
        raise RuntimeError("training receipt lacks a negative reference declaration")
    if training.get("forbidden_artifacts_used") != []:
        raise RuntimeError("training receipt declares forbidden ancestry")
    if training.get("checkpoint_sha256") != sha256(arguments.checkpoint):
        raise RuntimeError("training receipt/checkpoint hash mismatch")
    if training.get("standardization_sha256") != sha256(arguments.standardization):
        raise RuntimeError("training receipt/standardization hash mismatch")

    coarse = ParityFMPE(
        arguments.checkpoint,
        arguments.standardization,
        device=arguments.device,
        steps=arguments.coarse_steps,
    )
    fine = ParityFMPE(
        arguments.checkpoint,
        arguments.standardization,
        device=arguments.device,
        steps=arguments.fine_steps,
    )
    clouds = mass_radius_clouds(
        arguments.data_root,
        points=fine.cloud_points,
        seed=arguments.cloud_seed,
        source_scenario=arguments.source_scenario,
        source_data_root=arguments.source_data_root,
        verify_data=True,
    )
    observation = nuclear_observation(arguments.nuclear_scenario)
    coarse_generator = torch.Generator(device=coarse.device).manual_seed(arguments.seed)
    fine_generator = torch.Generator(device=fine.device).manual_seed(arguments.seed)
    theta_coarse = coarse.sample(
        clouds, observation, arguments.rows, coarse_generator
    )
    theta_fine = fine.sample(clouds, observation, arguments.rows, fine_generator)
    logq_coarse = coarse.log_density(
        theta_fine, clouds, observation, arguments.rows
    )
    logq_fine = fine.log_density(theta_fine, clouds, observation, arguments.rows)
    if not all(
        np.isfinite(value).all()
        for value in (theta_coarse, theta_fine, logq_coarse, logq_fine)
    ):
        raise RuntimeError("non-finite convergence output")

    logq_difference = np.abs(logq_fine - logq_coarse)
    normalized_theta_difference = np.abs(theta_fine - theta_coarse) / (
        fine.prior_high - fine.prior_low
    )
    p95_logq = float(np.quantile(logq_difference, 0.95))
    maximum_logq = float(logq_difference.max())
    passed = (
        p95_logq <= arguments.maximum_p95_logq_difference
        and maximum_logq <= arguments.maximum_logq_difference
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "learned_dim": fine.learned_dim,
        "rows": arguments.rows,
        "coarse_steps": arguments.coarse_steps,
        "fine_steps": arguments.fine_steps,
        "p50_absolute_logq_difference": float(np.quantile(logq_difference, 0.50)),
        "p95_absolute_logq_difference": p95_logq,
        "maximum_absolute_logq_difference": maximum_logq,
        "maximum_p95_logq_difference_gate": arguments.maximum_p95_logq_difference,
        "maximum_logq_difference_gate": arguments.maximum_logq_difference,
        "p95_normalized_theta_difference": float(
            np.quantile(normalized_theta_difference, 0.95)
        ),
        "maximum_normalized_theta_difference": float(
            normalized_theta_difference.max()
        ),
        "checkpoint_sha256": sha256(arguments.checkpoint),
        "standardization_sha256": sha256(arguments.standardization),
        "source_scenario": arguments.source_scenario,
        "nuclear_scenario": arguments.nuclear_scenario,
        "reference_present": False,
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
