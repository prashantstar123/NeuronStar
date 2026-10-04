#!/usr/bin/env python3
"""Run or certify UltraNest on the portable corrected A1 target."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eos import available_eos
from inference.common import compute_importance_weights
from workflows import A1Problem
from workflows.nuclear_scenarios import NUCLEAR_SCENARIO_NAMES
from workflows.resources import default_target_certificate
from workflows.source_scenarios import (
    SOURCE_SCENARIO_NAMES,
    source_substitution,
)



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", choices=available_eos(), default="ddb")
    parser.add_argument(
        "--scenario", choices=NUCLEAR_SCENARIO_NAMES, default="A1"
    )
    parser.add_argument(
        "--source-scenario", choices=SOURCE_SCENARIO_NAMES, default="A1"
    )
    parser.add_argument("--min-live", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "run a small prior-probe wiring test; the resulting posterior is "
            "labelled SMOKE_COMPLETED and is not a scientific UltraNest run"
        ),
    )
    parser.add_argument(
        "--resume", choices=("resume", "overwrite", "subfolder"), default="resume"
    )
    arguments = parser.parse_args()
    certificate = arguments.certificate or default_target_certificate(
        arguments.source_scenario,
        arguments.model,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    setup_started = time.time()
    problem = A1Problem.from_data_root(
        arguments.data_root,
        workers=arguments.workers,
        verify_data=True,
        nuclear_scenario=arguments.scenario,
        source_scenario=arguments.source_scenario,
        source_data_root=arguments.source_data_root,
        model=arguments.model,
    )
    setup_seconds = time.time() - setup_started
    gate = problem.certify(certificate)
    gate["certificate_sha256"] = sha256(certificate)
    gate["setup_seconds"] = setup_seconds
    replacement = source_substitution(arguments.source_scenario)
    source_artifact = ""
    source_artifact_sha256 = ""
    if replacement is not None:
        source_root = arguments.source_data_root or arguments.data_root
        source_artifact = replacement.filename
        source_artifact_sha256 = sha256(source_root / replacement.filename)
        gate["source_artifact"] = source_artifact
        gate["source_artifact_sha256"] = source_artifact_sha256
    write_json(arguments.output_dir / "target_certification.json", gate)
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
    if gate["status"] != "PASS":
        raise RuntimeError("shared corrected target certification failed")
    if arguments.gate_only:
        return 0

    scenario_parts = [
        name.lower()
        for name in (arguments.scenario, arguments.source_scenario)
        if name != "A1"
    ]
    scenario_slug = "__".join(scenario_parts) or "a1"
    summary_path = arguments.output_dir / f"ultranest_{scenario_slug}_posterior.npz"
    if arguments.smoke:
        # This deliberately does not impersonate a nested-sampling result.  It
        # exercises the identical prior transform and corrected target, emits
        # the production-compatible posterior schema, and is unambiguously
        # barred from scientific acceptance by its status.
        rng = np.random.default_rng(arguments.seed)
        proposed = problem.prior_transform(rng.random((128, len(problem.parameter_names))))
        log_likelihood = np.asarray(problem.evaluate_batch(proposed), dtype=np.float64)
        log_prior = -float(np.sum(np.log(problem.prior_high - problem.prior_low)))
        importance = compute_importance_weights(
            log_likelihood + log_prior,
            np.full(len(proposed), log_prior, dtype=np.float64),
        )
        samples, _ = importance.resample(proposed, size=128, seed=arguments.seed + 1)
        np.savez_compressed(
            summary_path,
            samples=samples,
            logz=importance.log_evidence,
            logzerr=importance.log_evidence_standard_error,
            ncall=len(proposed),
            niter=0,
            ess=importance.ess,
            wall_seconds=0.0,
            seed=arguments.seed,
            min_live=0,
            target_certificate_sha256=np.array(sha256(certificate)),
            nuclear_scenario=np.array(arguments.scenario),
            source_scenario=np.array(arguments.source_scenario),
            model=np.array(arguments.model),
            source_artifact=np.array(source_artifact),
            source_artifact_sha256=np.array(source_artifact_sha256),
            nuclear_observation=np.asarray(
                problem.target.nuclear_observation, dtype=np.float64
            ),
            smoke_prior_probe=np.bool_(True),
        )
        report = {
            "status": "SMOKE_COMPLETED",
            "backend": "UltraNest wiring / prior-probe smoke",
            "scientific_ultranest_run": False,
            "model": arguments.model,
            "nuclear_scenario": arguments.scenario,
            "source_scenario": arguments.source_scenario,
            "logz": importance.log_evidence,
            "logzerr": importance.log_evidence_standard_error,
            "ncall": len(proposed),
            "niter": 0,
            "ess": importance.ess,
            "posterior_rows": len(samples),
            "summary": str(summary_path),
            "summary_sha256": sha256(summary_path),
            "target_certification": gate,
        }
        if replacement is not None:
            report["source_artifact"] = source_artifact
            report["source_artifact_sha256"] = source_artifact_sha256
        write_json(arguments.output_dir / "run_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0

    import ultranest

    np.random.seed(arguments.seed)
    log_directory = arguments.output_dir / "ultranest_log"
    sampler = ultranest.ReactiveNestedSampler(
        list(problem.parameter_names),
        problem.evaluate_batch,
        problem.prior_transform,
        vectorized=True,
        log_dir=str(log_directory),
        resume=arguments.resume,
        ndraw_min=2048,
        ndraw_max=65536,
    )
    started = time.time()
    result = sampler.run(
        min_num_live_points=arguments.min_live,
        show_status=False,
        viz_callback=False,
    )
    wall_seconds = time.time() - started
    samples = np.asarray(result["samples"], dtype=np.float64)
    np.savez_compressed(
        summary_path,
        samples=samples,
        logz=float(result["logz"]),
        logzerr=float(result["logzerr"]),
        ncall=int(result["ncall"]),
        niter=int(result["niter"]),
        ess=float(result.get("ess", np.nan)),
        wall_seconds=wall_seconds,
        seed=arguments.seed,
        min_live=arguments.min_live,
        target_certificate_sha256=np.array(sha256(certificate)),
        nuclear_scenario=np.array(arguments.scenario),
        source_scenario=np.array(arguments.source_scenario),
        model=np.array(arguments.model),
        source_artifact=np.array(source_artifact),
        source_artifact_sha256=np.array(source_artifact_sha256),
        nuclear_observation=np.asarray(
            problem.target.nuclear_observation, dtype=np.float64
        ),
    )
    report = {
        "status": "PASS",
        "backend": "UltraNest",
        "model": arguments.model,
        "nuclear_scenario": arguments.scenario,
        "source_scenario": arguments.source_scenario,
        "logz": float(result["logz"]),
        "logzerr": float(result["logzerr"]),
        "ncall": int(result["ncall"]),
        "niter": int(result["niter"]),
        "ess": float(result.get("ess", np.nan)),
        "posterior_rows": int(len(samples)),
        "wall_seconds": wall_seconds,
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "target_certification": gate,
    }
    if replacement is not None:
        report["source_artifact"] = source_artifact
        report["source_artifact_sha256"] = source_artifact_sha256
    write_json(arguments.output_dir / "run_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
