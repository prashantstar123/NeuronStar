#!/usr/bin/env python3
"""Post-freeze exact-bank check for either support-extension v2 bank.

This calculation reads no EN checkpoint and no conventional-sampler output.
It certifies finite-bank uncertainty for the five nuclear configurations and
the three held-out NICER substitutions after the six neural checkpoints have
been externally frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.evidence_network.cache import DEFAULT_NUCLEAR_SIGMA
from inference.evidence_network.conditional.batched_nicer import BatchedShiftedNICER
from inference.evidence_network.conditional.build_dataset import weighted_source_table
from inference.evidence_network.conditional.build_source_weight_cache_v2 import (
    square_root_endpoint,
)
from likelihoods.nicer import build_nicer_grid
from workflows.nuclear_scenarios import NUCLEAR_SCENARIO_NAMES, nuclear_observation
from workflows.source_scenarios import SOURCE_SUBSTITUTIONS


LOG_SENTINEL_CUTOFF = -1.0e20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_log_value(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    return np.isfinite(array) & (array > LOG_SENTINEL_CUTOFF)


def valid_stellar_curve(
    radius: np.ndarray, mass_grid: np.ndarray, maximum_mass: np.ndarray
) -> np.ndarray:
    covered = np.asarray(mass_grid) >= 1.0
    if int(covered.sum()) < 2:
        raise RuntimeError("mass grid has fewer than two nodes above 1 Msun")
    usable = (
        np.isfinite(np.asarray(radius)[:, covered])
        & (
            np.asarray(mass_grid)[covered][None, :]
            <= np.asarray(maximum_mass)[:, None]
        )
    )
    return np.isfinite(maximum_mass) & (usable.sum(axis=1) >= 2)


def evidence_summary(log_weight: np.ndarray, denominator: float) -> dict:
    retained = valid_log_value(log_weight)
    if not retained.any():
        raise RuntimeError("evidence estimator has no valid log-weight rows")
    normalization = float(logsumexp(log_weight[retained]))
    normalized = np.exp(log_weight[retained] - normalization)
    ess = float(1.0 / np.sum(normalized**2))
    return {
        "log_evidence": normalization - denominator,
        "effective_sample_size": ess,
        "naive_monte_carlo_error": float(1.0 / np.sqrt(ess)),
        "retained_rows": int(retained.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--row-chunk", type=int, default=25_000)
    parser.add_argument("--quadrature-points", type=int, default=40)
    parser.add_argument("--minimum-effective-sample-size", type=float, default=20.0)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)

    started = time.perf_counter()
    with np.load(arguments.bank, allow_pickle=False) as source:
        required = {
            "schema", "model", "X", "R", "MG", "MM", "LA", "LPC",
            "NIC", "lpc_all_lse", "metadata",
        }
        missing = required - set(source.files)
        if missing:
            raise RuntimeError(f"support-extension bank lacks {sorted(missing)}")
        if str(source["schema"].item()) != "conditional-en-independent-physics-v2":
            raise RuntimeError("unexpected support-extension bank schema")
        model = str(source["model"].item())
        metadata = json.loads(str(source["metadata"].item()))
        if metadata.get("forbidden_artifacts_used") or metadata.get(
            "held_out_sources_used"
        ):
            raise RuntimeError("support-extension bank ancestry gate failed")
        prediction = np.asarray(source["X"], dtype=np.float64)
        radius = np.asarray(source["R"], dtype=np.float32)
        mass_grid = np.asarray(source["MG"], dtype=np.float32)
        maximum_mass = np.asarray(source["MM"], dtype=np.float32)
        log_astro = np.asarray(source["LA"], dtype=np.float64)
        correction = np.asarray(source["LPC"], dtype=np.float64)
        baseline_nicer = np.asarray(source["NIC"], dtype=np.float64)
        denominator = float(source["lpc_all_lse"])

    if baseline_nicer.shape != (len(prediction), 3):
        raise RuntimeError("embedded baseline NICER table has wrong shape")
    curve_is_valid = valid_stellar_curve(radius, mass_grid, maximum_mass)
    base = correction + log_astro

    nuclear_terms: dict[str, np.ndarray] = {}
    nuclear_results = {}
    for name in NUCLEAR_SCENARIO_NAMES:
        observation = np.asarray(nuclear_observation(name), dtype=np.float64)
        nuclear = -0.5 * np.sum(
            ((prediction - observation) / DEFAULT_NUCLEAR_SIGMA) ** 2, axis=1
        )
        nuclear_terms[name] = nuclear
        nuclear_results[name] = evidence_summary(base + nuclear, denominator)

    interpolators = {}
    centres = {}
    source_receipts = {}
    grid_options = dict(nM=150, nR=150, max_rows=20_000, bw=0.08, seed=0)
    for case, specification in SOURCE_SUBSTITUTIONS.items():
        path = arguments.heldout_root / specification.filename
        if not path.is_file():
            raise FileNotFoundError(path)
        interpolators[case] = build_nicer_grid(
            path,
            mcol=specification.mass_column,
            rcol=specification.radius_column,
            wcol_or_None=specification.weight_column,
            **grid_options,
        )
        _, _, _, centre = weighted_source_table(path, specification)
        centres[case] = (float(centre[0]), float(centre[1]))
        source_receipts[case] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "slot": int(specification.slot),
            "replaces": specification.replaces,
        }

    covered = mass_grid >= 1.0
    device = torch.device(arguments.device)
    evaluator = BatchedShiftedNICER(
        interpolators,
        radius[:, covered],
        mass_grid[covered],
        maximum_mass,
        centres,
        device=str(device),
        row_chunk=arguments.row_chunk,
        quadrature_points=arguments.quadrature_points,
    )
    base_a1 = torch.as_tensor(
        base + nuclear_terms["A1"], dtype=torch.float64, device=device
    )
    baseline_gpu = torch.as_tensor(
        baseline_nicer, dtype=torch.float64, device=device
    )
    valid_curve_gpu = torch.as_tensor(curve_is_valid, device=device)
    identity = torch.as_tensor(
        [[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32, device=device
    )
    values = {
        case: np.full(len(prediction), -np.inf, dtype=np.float64)
        for case in SOURCE_SUBSTITUTIONS
    }
    endpoint_nodes = 0
    for start, stop in evaluator.chunks():
        mass, curve_radius, valid, lower, upper = evaluator.curve_quadrature(
            start, stop
        )
        fixed_radius, affected = square_root_endpoint(
            mass,
            curve_radius,
            radius[start:stop, covered],
            mass_grid[covered],
            maximum_mass[start:stop],
            device,
        )
        endpoint_nodes += int(affected.sum().item())
        quadrature = (mass, fixed_radius, valid, lower, upper)
        for case, specification in SOURCE_SUBSTITUTIONS.items():
            replacement = evaluator.evaluate_chunk(
                case, identity, quadrature
            )[0].double()
            baseline = baseline_gpu[start:stop, specification.slot]
            valid_rows = (
                valid_curve_gpu[start:stop]
                & torch.isfinite(base_a1[start:stop])
                & (base_a1[start:stop] > LOG_SENTINEL_CUTOFF)
                & torch.isfinite(baseline)
                & (baseline > LOG_SENTINEL_CUTOFF)
                & torch.isfinite(replacement)
                & (replacement > LOG_SENTINEL_CUTOFF)
            )
            log_weight = torch.where(
                valid_rows,
                base_a1[start:stop] - baseline + replacement,
                torch.full_like(replacement, -torch.inf),
            )
            values[case][start:stop] = log_weight.cpu().numpy()

    heldout_results = {
        case: {
            **evidence_summary(log_weight, denominator),
            **source_receipts[case],
        }
        for case, log_weight in values.items()
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    minimum_ess = min(
        row["effective_sample_size"]
        for row in (*nuclear_results.values(), *heldout_results.values())
    )
    report = {
        "status": (
            "PASS"
            if minimum_ess >= arguments.minimum_effective_sample_size
            else "FAIL_LOW_EFFECTIVE_SAMPLE_SIZE"
        ),
        "role": "exact importance-sampling evidence of every Table VI row on the Evidence Network bank, computed after the three networks were frozen",
        "model": model,
        "bank": str(arguments.bank.resolve()),
        "bank_sha256": sha256(arguments.bank),
        "nuclear_results": nuclear_results,
        "heldout_results": heldout_results,
        "minimum_effective_sample_size": minimum_ess,
        "minimum_effective_sample_size_gate": arguments.minimum_effective_sample_size,
        "wall_seconds": time.perf_counter() - started,
        "endpoint_rule": "square-root continuation of the last stable branch to Mmax",
        "endpoint_nodes": endpoint_nodes,
        "valid_stellar_curve_rows": int(curve_is_valid.sum()),
        "invalid_stellar_curve_rows": int((~curve_is_valid).sum()),
        "finite_negative_sentinel_rows": int(
            (np.isfinite(log_astro) & (log_astro <= LOG_SENTINEL_CUTOFF)).sum()
        ),
        "replacement_arithmetic": (
            "base minus exact embedded baseline-source term plus evaluated "
            "replacement-source term; invalid curves and finite sentinels excluded"
        ),
        "forbidden_artifacts_used": [],
        "comparison_sampler_outputs_used": [],
        "en_checkpoints_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
