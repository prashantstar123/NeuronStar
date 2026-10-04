#!/usr/bin/env python3
"""Evaluate exact hyperonic rows for the predeclared S_ext screen.

This is a schema adapter around the separately certified accelerated evaluator
in ``fastsolver/fast_exact_rows.py``.  The physics path remains
compiled hyperonic EOS -> unchanged exact CPU TOV -> exact batched likelihood.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FAST_ROOT = ROOT.parent / "fastsolver"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FAST_ROOT))

import fast_exact_rows as fast  # noqa: E402
from eos import get_eos  # noqa: E402
from likelihoods.nuclear import A1_SIGMA  # noqa: E402
from workflows.a1_problem import A1Problem  # noqa: E402


EXPECTED_SCREEN_SCHEMA = "green-en-nuclear-support-extension-screen-v1"
OUTPUT_SCHEMA = "ddb-hyperonic-green-en-support-extension-cache-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--template-bank", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument(
        "--eos-backend", choices=("compiled", "certified"), default="compiled"
    )
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite exact shard: {arguments.output}")
    if arguments.workers < 1:
        raise ValueError("workers must be positive")

    started_unix = time.time()
    started_monotonic = time.monotonic()
    screen_sha256 = sha256(arguments.screen)
    with np.load(arguments.screen, allow_pickle=False) as screen:
        required = {
            "schema",
            "eos",
            "support_name",
            "theta",
            "prediction",
            "proposal_index",
            "total_proposals",
            "seed",
            "reference_present",
            "forbidden_artifacts_used",
        }
        missing = required - set(screen.files)
        if missing:
            raise RuntimeError(f"support screen is missing {sorted(missing)}")
        if str(screen["schema"].item()) != EXPECTED_SCREEN_SCHEMA:
            raise RuntimeError("unexpected support-extension screen schema")
        if str(screen["eos"].item()) != "ddb-hyperonic":
            raise RuntimeError("screen is not hyperonic DDB")
        if str(screen["support_name"].item()) != "S_ext":
            raise RuntimeError("screen is not the predeclared S_ext stream")
        if bool(screen["reference_present"]) or screen[
            "forbidden_artifacts_used"
        ].size:
            raise RuntimeError("support screen declares forbidden ancestry")
        all_theta = np.asarray(screen["theta"], dtype=np.float64)
        all_prediction = np.asarray(screen["prediction"], dtype=np.float64)
        all_proposal_index = np.asarray(screen["proposal_index"], dtype=np.int64)
        total_proposals = int(screen["total_proposals"])
        seed = int(screen["seed"])
    if total_proposals != 160_000_000 or seed != 20_260_923:
        raise RuntimeError("screen does not carry the predeclared hyperonic count/seed")
    if not 0 <= arguments.start < arguments.stop <= len(all_theta):
        raise ValueError(
            f"invalid shard [{arguments.start},{arguments.stop}) for {len(all_theta)} rows"
        )

    theta = all_theta[arguments.start : arguments.stop]
    stored_prediction = all_prediction[arguments.start : arguments.stop]
    proposal_index = all_proposal_index[arguments.start : arguments.stop]
    with np.load(arguments.template_bank, allow_pickle=False) as template:
        if str(template["schema_version"].item()) != "ddbhy-bank-v1":
            raise RuntimeError("unexpected template-bank schema")
        mass_grid = np.asarray(template["MG"], dtype=np.float64)
        prior_low = np.asarray(template["THETA_LOW"], dtype=np.float64)
        prior_high = np.asarray(template["THETA_HIGH"], dtype=np.float64)

    problem = A1Problem.from_data_root(
        arguments.data_root,
        workers=arguments.workers,
        verify_data=True,
        nuclear_scenario="A1",
        source_scenario="A1",
        model="ddb-hyperonic",
    )
    plugin = get_eos("ddb-hyperonic")
    prediction = np.empty_like(stored_prediction)
    prediction[:, 0] = theta[:, plugin.parameter_names.index("rho0")]
    prediction[:, 1:] = plugin.nuclear_observables_batch(theta)
    prediction_delta = np.abs(prediction - stored_prediction)
    maximum_prediction_delta = float(np.max(prediction_delta))
    maximum_normalized_prediction_delta = float(
        np.max(prediction_delta / A1_SIGMA)
    )
    prediction_tolerance_sigma = 1.0e-4
    if maximum_normalized_prediction_delta > prediction_tolerance_sigma:
        raise RuntimeError(
            "screen and evaluator predictions differ by "
            f"{maximum_normalized_prediction_delta:.3e} nuclear sigma"
        )

    (
        log_astrophysical,
        components,
        source_components,
        radius,
        tidal,
        maximum_mass,
        diagnostics,
    ) = fast.evaluate_rows(
        theta,
        prediction[:, 1:],
        problem.target,
        mass_grid,
        plugin,
        arguments.workers,
        eos_backend=arguments.eos_backend,
    )
    target_valid = (
        np.isfinite(log_astrophysical)
        & (log_astrophysical > -1.0e29)
        & np.isfinite(components).all(axis=1)
        & np.isfinite(source_components).all(axis=1)
        & np.isfinite(maximum_mass)
        & np.isfinite(radius).any(axis=1)
        & np.isfinite(tidal).any(axis=1)
    )
    if not target_valid.any():
        raise RuntimeError("exact likelihood rejected every support-extension row")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    template_sha = sha256(FAST_ROOT / "hyp_solver_template.py")
    finished_unix = time.time()
    wall_seconds = time.monotonic() - started_monotonic
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(OUTPUT_SCHEMA),
            support_name=np.asarray("S_ext"),
            source_index=np.arange(arguments.start, arguments.stop, dtype=np.int64),
            proposal_index=proposal_index,
            theta=theta,
            prediction=prediction,
            log_astrophysical=log_astrophysical,
            astrophysical_components=components,
            nicer_source_components=source_components,
            target_valid=target_valid,
            mass_grid=mass_grid,
            radius=radius,
            tidal_lambda=tidal,
            maximum_mass=maximum_mass,
            prior_low=prior_low,
            prior_high=prior_high,
            screen_sha256=np.asarray(screen_sha256),
            template_bank_sha256=np.asarray(sha256(arguments.template_bank)),
            total_proposals=np.int64(total_proposals),
            seed=np.int64(seed),
            shard_start=np.int64(arguments.start),
            shard_stop=np.int64(arguments.stop),
            prediction_tolerance_sigma=np.float64(prediction_tolerance_sigma),
            maximum_prediction_delta=np.float64(maximum_prediction_delta),
            maximum_normalized_prediction_delta=np.float64(
                maximum_normalized_prediction_delta
            ),
            workers=np.int64(arguments.workers),
            hostname=np.asarray(socket.gethostname()),
            reference_present=np.bool_(False),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
            started_unix=np.float64(started_unix),
            finished_unix=np.float64(finished_unix),
            wall_seconds=np.float64(wall_seconds),
            evaluator=np.asarray(
                "scripts/evaluate_hyperonic_support_extension.py"
            ),
            certified_kernel=np.asarray(
                "fastsolver/fast_exact_rows.py"
            ),
            hyp_solver_template_sha256=np.asarray(template_sha),
            eos_backend=np.asarray(arguments.eos_backend),
            fast_fallback_rows=np.int64(diagnostics["fallback_rows"]),
            fast_rows_with_emulated_exit=np.int64(
                diagnostics["rows_with_emulated_exit"]
            ),
        )
    os.replace(temporary, arguments.output)
    report = {
        "status": "PASS",
        "schema": OUTPUT_SCHEMA,
        "support_name": "S_ext",
        "evaluator": "scripts/evaluate_hyperonic_support_extension.py",
        "certified_kernel": "fastsolver/fast_exact_rows.py",
        "hyp_solver_template_sha256": template_sha,
        "eos_backend": arguments.eos_backend,
        "screen": str(arguments.screen.resolve()),
        "screen_sha256": screen_sha256,
        "template_bank": str(arguments.template_bank.resolve()),
        "template_bank_sha256": sha256(arguments.template_bank),
        "total_screen_rows": int(len(all_theta)),
        "total_proposals": total_proposals,
        "seed": seed,
        "shard_start": arguments.start,
        "shard_stop": arguments.stop,
        "shard_rows": int(len(theta)),
        "target_valid_rows": int(target_valid.sum()),
        "maximum_prediction_delta": maximum_prediction_delta,
        "maximum_normalized_prediction_delta": maximum_normalized_prediction_delta,
        "prediction_tolerance_sigma": prediction_tolerance_sigma,
        "workers": arguments.workers,
        "hostname": socket.gethostname(),
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "reference_present": False,
        "forbidden_artifacts_used": [],
        "wall_seconds": wall_seconds,
        "stage_diagnostics": diagnostics,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
