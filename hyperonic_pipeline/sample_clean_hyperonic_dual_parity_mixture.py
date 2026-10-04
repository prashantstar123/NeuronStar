#!/usr/bin/env python3
"""Sample the frozen clean 9D/7D parity A-NET deterministic mixture.

The seven-dimensional head learns the nucleonic coordinates and retains the
two hyperon couplings at their canonical uniform-prior density.  The
nine-dimensional head learns all coordinates.  A clean defensive Student
component supplies heavy-tail coverage.  Every draw is scored under every
component, so the saved proposal density is the exact deterministic-mixture
density used by downstream importance sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import logsumexp

from clean_student_t_ensemble import (
    STUDENT_ENSEMBLE_SCHEMA,
    checkpoint_density as student_density,
    sample_checkpoint as sample_student,
)
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


def validate_training(report_path: Path, checkpoint: Path, standardization: Path):
    report = json.loads(report_path.read_text())
    if report.get("reference_present") is not False:
        raise RuntimeError(f"{report_path} lacks a negative reference declaration")
    if report.get("forbidden_artifacts_used") != []:
        raise RuntimeError(f"{report_path} declares forbidden ancestry")
    if report.get("held_out_sources_absent") != ["J0614", "J1231", "J1614"]:
        raise RuntimeError(f"{report_path} does not exclude all held-out sources")
    if report.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError(f"checkpoint hash mismatch for {checkpoint}")
    if report.get("standardization_sha256") != sha256(standardization):
        raise RuntimeError(f"standardization hash mismatch for {standardization}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for dimension in (9, 7):
        parser.add_argument(f"--head{dimension}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--head{dimension}-standardization", type=Path, required=True)
        parser.add_argument(f"--head{dimension}-report", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--source-scenario", choices=SOURCE_SCENARIO_NAMES, required=True)
    parser.add_argument("--nuclear-scenario", choices=NUCLEAR_SCENARIO_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=25_000)
    parser.add_argument("--head9-weight", type=float, default=0.35)
    parser.add_argument("--head7-weight", type=float, default=0.55)
    parser.add_argument("--student-weight", type=float, default=0.10)
    parser.add_argument("--cloud-seed", type=int, default=777_001)
    parser.add_argument("--proposal-seed", type=int, default=20_260_944)
    parser.add_argument("--flow-steps", type=int, default=1024)
    parser.add_argument("--draw-batch", type=int, default=5_000)
    parser.add_argument("--density-batch", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    if arguments.source_scenario != "A1" and arguments.nuclear_scenario != "A1":
        raise ValueError("change either NICER source or nuclear observation, not both")
    weights = np.asarray(
        [arguments.head9_weight, arguments.head7_weight, arguments.student_weight],
        dtype=np.float64,
    )
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=1.0e-12):
        raise ValueError("the three positive mixture weights must sum to one")
    if min(
        arguments.draws,
        arguments.flow_steps,
        arguments.draw_batch,
        arguments.density_batch,
    ) < 1:
        raise ValueError("draw and integration settings must be positive")
    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    reports = {
        dimension: validate_training(
            getattr(arguments, f"head{dimension}_report"),
            getattr(arguments, f"head{dimension}_checkpoint"),
            getattr(arguments, f"head{dimension}_standardization"),
        )
        for dimension in (9, 7)
    }
    heads = {
        dimension: ParityFMPE(
            getattr(arguments, f"head{dimension}_checkpoint"),
            getattr(arguments, f"head{dimension}_standardization"),
            device=arguments.device,
            steps=arguments.flow_steps,
        )
        for dimension in (9, 7)
    }
    if heads[9].learned_dim != 9 or heads[7].learned_dim != 7:
        raise RuntimeError("the A-NET heads do not have the declared dimensions")
    for key in ("prior_low", "prior_high", "cloud_mean", "cloud_scale", "x_mean", "x_scale"):
        if not np.array_equal(getattr(heads[9], key), getattr(heads[7], key)):
            raise RuntimeError(f"the two A-NET heads disagree on {key}")
    if reports[9].get("scenarios_sha256") != reports[7].get("scenarios_sha256"):
        raise RuntimeError("the two A-NET heads were not trained on the same scenarios")

    student = joblib.load(arguments.student_checkpoint)
    if student.get("schema") != STUDENT_ENSEMBLE_SCHEMA:
        raise RuntimeError("unrecognized defensive Student checkpoint")
    if not (
        np.array_equal(np.asarray(student["prior_low"]), heads[9].prior_low)
        and np.array_equal(np.asarray(student["prior_high"]), heads[9].prior_high)
    ):
        raise RuntimeError("A-NET and defensive Student priors differ")

    clouds = mass_radius_clouds(
        arguments.data_root,
        points=heads[9].cloud_points,
        seed=arguments.cloud_seed,
        source_scenario=arguments.source_scenario,
        source_data_root=arguments.source_data_root,
        verify_data=True,
    )
    observation = nuclear_observation(arguments.nuclear_scenario)
    started = time.time()
    assignment_rng = np.random.default_rng(arguments.proposal_seed)
    component = assignment_rng.choice(3, size=arguments.draws, p=weights).astype(np.int8)
    theta = np.empty((arguments.draws, 9), dtype=np.float64)
    for component_index, dimension in enumerate((9, 7)):
        selected = np.flatnonzero(component == component_index)
        generator = torch.Generator(device=heads[dimension].device).manual_seed(
            arguments.proposal_seed + 10 + component_index
        )
        for start in range(0, len(selected), arguments.draw_batch):
            local = selected[start : start + arguments.draw_batch]
            theta[local] = heads[dimension].sample(
                clouds, observation, len(local), generator
            )
            print(
                f"[dual-parity-draw] head{dimension} {min(start + arguments.draw_batch, len(selected))}/{len(selected)}",
                flush=True,
            )
    student_index = np.flatnonzero(component == 2)
    theta[student_index] = sample_student(
        student,
        len(student_index),
        np.random.default_rng(arguments.proposal_seed + 12),
    )
    draw_seconds = time.time() - started

    density_started = time.time()
    head9_logq = heads[9].log_density(
        theta, clouds, observation, arguments.density_batch
    )
    head7_logq = heads[7].log_density(
        theta, clouds, observation, arguments.density_batch
    )
    prior_logq = -float(np.log(heads[9].prior_high - heads[9].prior_low).sum())
    student_logq = student_density(student, theta, prior_logq)
    mixture_logq = logsumexp(
        np.vstack(
            [
                math.log(weights[0]) + head9_logq,
                math.log(weights[1]) + head7_logq,
                math.log(weights[2]) + student_logq,
            ]
        ),
        axis=0,
    )
    density_seconds = time.time() - density_started
    if not np.isfinite(theta).all() or not np.isfinite(mixture_logq).all():
        raise RuntimeError("non-finite dual-head proposal output")
    if not np.all(
        (theta >= heads[9].prior_low) & (theta <= heads[9].prior_high)
    ):
        raise RuntimeError("dual-head proposal draw lies outside the canonical prior")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    hashes = {
        f"head{dimension}_checkpoint": sha256(
            getattr(arguments, f"head{dimension}_checkpoint")
        )
        for dimension in (9, 7)
    }
    hashes.update(
        {
            f"head{dimension}_standardization": sha256(
                getattr(arguments, f"head{dimension}_standardization")
            )
            for dimension in (9, 7)
        }
    )
    hashes["student_checkpoint"] = sha256(arguments.student_checkpoint)
    hashes["dual_sampler"] = sha256(Path(__file__).resolve())
    hashes["parity_density_backend"] = sha256(
        Path(__file__).resolve().with_name(
            "sample_clean_hyperonic_parity_mixture.py"
        )
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray("ddb-hyperonic-clean-dual-parity-anet-mixture-v1"),
            theta=theta,
            logq=mixture_logq,
            head9_logq=head9_logq,
            head7_logq=head7_logq,
            student_logq=student_logq,
            component=component,
            component_weights=weights,
            cloud=clouds,
            prior_low=heads[9].prior_low,
            prior_high=heads[9].prior_high,
            nuclear_scenario=np.asarray(arguments.nuclear_scenario),
            source_scenario=np.asarray(arguments.source_scenario),
            nuclear_observation=observation,
            head9_checkpoint_sha256=np.asarray(hashes["head9_checkpoint"]),
            head7_checkpoint_sha256=np.asarray(hashes["head7_checkpoint"]),
            head9_standardization_sha256=np.asarray(
                hashes["head9_standardization"]
            ),
            head7_standardization_sha256=np.asarray(
                hashes["head7_standardization"]
            ),
            student_checkpoint_sha256=np.asarray(hashes["student_checkpoint"]),
            dual_sampler_sha256=np.asarray(hashes["dual_sampler"]),
            parity_density_backend_sha256=np.asarray(
                hashes["parity_density_backend"]
            ),
            scenario_sha256=np.asarray(reports[9].get("scenarios_sha256", "")),
            held_out_sources_absent=np.asarray(["J0614", "J1231", "J1614"]),
            evaluated_source_artifact_absent_from_training=np.bool_(
                arguments.source_scenario != "A1"
            ),
            flow_steps=np.int64(arguments.flow_steps),
            cloud_seed=np.int64(arguments.cloud_seed),
            proposal_seed=np.int64(arguments.proposal_seed),
            reference_present=np.bool_(False),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
    os.replace(temporary, arguments.output)
    report = {
        "status": "PASS_PROPOSAL_ONLY_PENDING_EXACT_IS",
        "source_scenario": arguments.source_scenario,
        "nuclear_scenario": arguments.nuclear_scenario,
        "draws": arguments.draws,
        "component_rows": {
            "head9": int(np.sum(component == 0)),
            "head7_plus_uniform2": int(np.sum(component == 1)),
            "defensive_student": int(np.sum(component == 2)),
        },
        "component_weights": weights.tolist(),
        "flow_steps": arguments.flow_steps,
        "draw_seconds": draw_seconds,
        "density_seconds": density_seconds,
        "wall_seconds": time.time() - started,
        "scenario_sha256": reports[9].get("scenarios_sha256"),
        "input_sha256": hashes,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "held_out_sources_absent": ["J0614", "J1231", "J1614"],
        "reference_present": False,
        "forbidden_artifacts_used": [],
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
