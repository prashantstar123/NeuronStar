#!/usr/bin/env python3
"""Assemble the predeclared S_ext Green-EN banks and enforce coverage gates.

This program adds one independent full-prior support stream to an already
frozen physics bank.  It never reads UltraNest, A-NET, TSNPE, or an
older Evidence-Network prediction.  The two sector-specific routes share the
same five nuclear boxes and the same coverage tests, but preserve their
original counting-mixture definitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


OBS = np.asarray(
    [
        0.153,
        -16.1,
        230.0,
        32.5,
        0.505714285714279,
        1.24142857142857,
        2.4857142857143,
    ],
    dtype=np.float64,
)
SIG = np.asarray(
    [
        0.005,
        0.2,
        40.0,
        1.8,
        0.194285714285714,
        0.608571428571429,
        1.38285714285714,
    ],
    dtype=np.float64,
)
CASE_NAMES = ("A1", "K0_200", "K0_260", "Jsym_29", "Jsym_36")
CENTRES = np.repeat(OBS[None, :], len(CASE_NAMES), axis=0)
CENTRES[1, 2] = 200.0
CENTRES[2, 2] = 260.0
CENTRES[3, 3] = 29.0
CENTRES[4, 3] = 36.0
BOX_RADIUS = 3.0
EXTENSION_PROPOSALS = 160_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def memberships(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    standardized = np.abs(
        (prediction[:, None, :] - CENTRES[None, :, :]) / SIG[None, None, :]
    )
    boxes = np.all(standardized <= BOX_RADIUS, axis=2)
    b3 = boxes[:, 0]
    region = np.any(boxes, axis=1)
    return b3, region, region & ~b3


def _stable_weights(log_weight: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(log_weight)
    if not finite.any():
        return -np.inf, 0.0
    normalization = float(logsumexp(log_weight[finite]))
    normalized = np.exp(log_weight[finite] - normalization)
    return normalization, float(1.0 / np.sum(normalized * normalized))


def coverage_metrics(
    uniform_prediction: np.ndarray,
    prediction: np.ndarray,
    log_astro: np.ndarray,
    correction: np.ndarray,
    denominator: float,
) -> tuple[dict[str, dict[str, float]], dict[str, bool]]:
    _, uniform_region, _ = memberships(uniform_prediction)
    _, bank_region, _ = memberships(prediction)
    metrics: dict[str, dict[str, float]] = {}
    gates: dict[str, bool] = {}
    for name, centre in zip(CASE_NAMES, CENTRES, strict=True):
        uniform_kernel = -0.5 * np.sum(
            ((uniform_prediction - centre[None, :]) / SIG[None, :]) ** 2,
            axis=1,
        )
        finite_uniform = np.isfinite(uniform_kernel)
        total_kernel = float(logsumexp(uniform_kernel[finite_uniform]))
        outside = finite_uniform & ~uniform_region
        outside_fraction = (
            float(np.exp(logsumexp(uniform_kernel[outside]) - total_kernel))
            if outside.any()
            else 0.0
        )
        nuclear = -0.5 * np.sum(
            ((prediction - centre[None, :]) / SIG[None, :]) ** 2,
            axis=1,
        )
        full_log_weight = correction + log_astro + nuclear
        full_norm, full_ess = _stable_weights(full_log_weight)
        region_log_weight = np.where(bank_region, full_log_weight, -np.inf)
        region_norm, region_ess = _stable_weights(region_log_weight)
        outside_evidence_fraction = (
            float(np.exp(full_norm - full_norm)) - float(np.exp(region_norm - full_norm))
            if np.isfinite(full_norm) and np.isfinite(region_norm)
            else np.nan
        )
        metrics[name] = {
            "uniform_kernel_mass_outside_region": outside_fraction,
            "full_log_evidence": full_norm - denominator,
            "full_effective_sample_size": full_ess,
            "in_region_log_evidence": region_norm - denominator,
            "in_region_effective_sample_size": region_ess,
            "direct_evidence_fraction_outside_region": outside_evidence_fraction,
        }
        gates[f"{name}_uniform_kernel_outside_le_0p025"] = outside_fraction <= 0.025
        gates[f"{name}_in_region_ess_ge_50"] = region_ess >= 50.0
    return metrics, gates


def validate_extension(
    screen_path: Path,
    exact_path: Path,
    sector: str,
) -> dict[str, np.ndarray | int | str]:
    with np.load(screen_path, allow_pickle=False) as screen:
        if str(screen["schema"].item()) != "green-en-nuclear-support-extension-screen-v1":
            raise RuntimeError("unexpected extension-screen schema")
        if str(screen["eos"].item()) != sector:
            raise RuntimeError("extension screen belongs to another sector")
        if str(screen["support_name"].item()) != "S_ext":
            raise RuntimeError("extension screen is not S_ext")
        if int(screen["total_proposals"]) != EXTENSION_PROPOSALS:
            raise RuntimeError("extension proposal count differs from predeclaration")
        if bool(screen["reference_present"]) or screen["forbidden_artifacts_used"].size:
            raise RuntimeError("extension screen declares forbidden ancestry")
        screen_theta = np.asarray(screen["theta"], dtype=np.float64)
        screen_prediction = np.asarray(screen["prediction"], dtype=np.float64)
        screen_proposal_index = np.asarray(screen["proposal_index"], dtype=np.int64)
        screen_membership = np.asarray(screen["retained_box_membership"], dtype=bool)
        seed = int(screen["seed"])
        if not np.array_equal(np.asarray(screen["observation"], dtype=np.float64), OBS):
            raise RuntimeError("screen observation differs from predeclaration")
        if not np.array_equal(np.asarray(screen["sigma"], dtype=np.float64), SIG):
            raise RuntimeError("screen sigma differs from predeclaration")
        if not np.array_equal(np.asarray(screen["centres"], dtype=np.float64), CENTRES):
            raise RuntimeError("screen centres differ from predeclaration")
        if float(screen["box_radius"]) != BOX_RADIUS:
            raise RuntimeError("screen box radius differs from predeclaration")

    with np.load(exact_path, allow_pickle=False) as exact:
        required = {
            "support_name", "source_index", "proposal_index", "theta",
            "prediction", "log_astrophysical", "astrophysical_components",
            "nicer_source_components", "target_valid", "mass_grid", "radius",
            "maximum_mass", "screen_sha256", "total_proposals", "seed",
            "reference_present", "forbidden_artifacts_used",
        }
        missing = required - set(exact.files)
        if missing:
            raise RuntimeError(f"extension exact cache lacks {sorted(missing)}")
        if str(exact["support_name"].item()) != "S_ext":
            raise RuntimeError("extension exact cache is not S_ext")
        if str(exact["screen_sha256"].item()) != sha256(screen_path):
            raise RuntimeError("exact cache points to a different screen")
        if int(exact["total_proposals"]) != EXTENSION_PROPOSALS:
            raise RuntimeError("exact-cache proposal count differs from predeclaration")
        if int(exact["seed"]) != seed:
            raise RuntimeError("screen/exact seed mismatch")
        if bool(exact["reference_present"]) or exact["forbidden_artifacts_used"].size:
            raise RuntimeError("extension exact cache declares forbidden ancestry")
        source_index = np.asarray(exact["source_index"], dtype=np.int64)
        proposal_index = np.asarray(exact["proposal_index"], dtype=np.int64)
        theta = np.asarray(exact["theta"], dtype=np.float64)
        prediction = np.asarray(exact["prediction"], dtype=np.float64)
        log_astro = np.asarray(exact["log_astrophysical"], dtype=np.float64)
        components = np.asarray(exact["astrophysical_components"], dtype=np.float64)
        nicer = np.asarray(exact["nicer_source_components"], dtype=np.float64)
        valid = np.asarray(exact["target_valid"], dtype=bool)
        mass_grid = np.asarray(exact["mass_grid"], dtype=np.float64)
        radius = np.asarray(exact["radius"], dtype=np.float64)
        maximum_mass = np.asarray(exact["maximum_mass"], dtype=np.float64)

    rows = len(screen_theta)
    if not np.array_equal(source_index, np.arange(rows, dtype=np.int64)):
        raise RuntimeError("extension source_index is incomplete or reordered")
    if not np.array_equal(proposal_index, screen_proposal_index):
        raise RuntimeError("extension proposal indices differ from screen")
    if not np.array_equal(theta, screen_theta):
        raise RuntimeError("extension theta differs from screen")
    normalized_delta = np.max(np.abs(prediction - screen_prediction) / SIG[None, :])
    if normalized_delta > 1.0e-4:
        raise RuntimeError(f"screen/exact prediction mismatch: {normalized_delta:.3e} sigma")
    b3, region, sext = memberships(prediction)
    if b3.any() or not sext.all() or not region.all():
        raise RuntimeError("an extension row is not in S_ext")
    if not np.array_equal(screen_membership, np.all(
        np.abs((screen_prediction[:, None, :] - CENTRES[None, :, :]) / SIG[None, None, :])
        <= BOX_RADIUS,
        axis=2,
    )):
        raise RuntimeError("stored screen membership is inconsistent")
    finite = valid
    component_error = float(np.max(np.abs(components[finite].sum(1) - log_astro[finite])))
    nicer_error = float(np.max(np.abs(nicer[finite].sum(1) - components[finite, 1])))
    if component_error > 1.0e-8 or nicer_error > 1.0e-8:
        raise RuntimeError(
            f"extension likelihood decomposition fails: {component_error:.3e}, {nicer_error:.3e}"
        )
    return {
        "theta": theta,
        "prediction": prediction,
        "proposal_index": proposal_index,
        "log_astro": np.where(valid, log_astro, -np.inf),
        "components": components,
        "nicer": nicer,
        "valid": valid,
        "mass_grid": mass_grid,
        "radius": radius,
        "maximum_mass": maximum_mass,
        "seed": seed,
        "normalized_prediction_delta": float(normalized_delta),
        "component_error": component_error,
        "nicer_error": nicer_error,
    }


def direct_report(
    prediction: np.ndarray,
    log_astro: np.ndarray,
    correction: np.ndarray,
    denominator: float,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, centre in zip(CASE_NAMES, CENTRES, strict=True):
        nuclear = -0.5 * np.sum(
            ((prediction - centre[None, :]) / SIG[None, :]) ** 2,
            axis=1,
        )
        normalization, ess = _stable_weights(correction + log_astro + nuclear)
        result[name] = {
            "log_evidence": normalization - denominator,
            "effective_sample_size": ess,
            "naive_monte_carlo_error": 1.0 / math.sqrt(ess),
        }
    return result


def assemble_hyperonic(arguments: argparse.Namespace) -> dict[str, object]:
    extension = validate_extension(
        arguments.extension_screen, arguments.extension_exact, "ddb-hyperonic"
    )
    with np.load(arguments.base_bank, allow_pickle=False) as base:
        required = {"schema", "model", "X", "theta", "R", "MG", "MM", "LA",
                    "LPC", "AC", "NIC", "lpc_all_lse", "metadata"}
        missing = required - set(base.files)
        if missing:
            raise RuntimeError(f"base hyperonic bank lacks {sorted(missing)}")
        if str(base["schema"].item()) != "conditional-en-independent-physics-v2":
            raise RuntimeError("unexpected base hyperonic bank schema")
        if str(base["model"].item()) != "ddb-hyperonic":
            raise RuntimeError("base bank is not hyperonic DDB")
        base_metadata = json.loads(str(base["metadata"].item()))
        if base_metadata.get("forbidden_artifacts_used") or base_metadata.get("held_out_sources_used"):
            raise RuntimeError("base hyperonic bank ancestry gate failed")
        old_x = np.asarray(base["X"], dtype=np.float64)
        old_theta = np.asarray(base["theta"], dtype=np.float64)
        old_radius = np.asarray(base["R"], dtype=np.float32)
        old_grid = np.asarray(base["MG"], dtype=np.float64)
        old_mmax = np.asarray(base["MM"], dtype=np.float64)
        old_la = np.asarray(base["LA"], dtype=np.float64)
        old_lpc = np.asarray(base["LPC"], dtype=np.float64)
        old_components = np.asarray(base["AC"], dtype=np.float64)
        old_nicer = np.asarray(base["NIC"], dtype=np.float64)

    if not np.array_equal(old_grid.astype(np.float32), np.asarray(extension["mass_grid"]).astype(np.float32)):
        raise RuntimeError("hyperonic base and extension mass grids differ")
    old_b3, _, old_sext = memberships(old_x)
    old_expected = math.log(arguments.base_proposals) - np.log(
        arguments.base_proposals + arguments.b3_proposals * old_b3.astype(np.float64)
    )
    old_correction_error = float(np.max(np.abs(old_lpc - old_expected)))
    if old_correction_error > 1.0e-12:
        raise RuntimeError(f"base hyperonic correction does not reproduce: {old_correction_error:.3e}")

    x = np.concatenate([old_x, np.asarray(extension["prediction"])])
    theta = np.concatenate([old_theta, np.asarray(extension["theta"])])
    radius = np.concatenate([old_radius, np.asarray(extension["radius"], dtype=np.float32)])
    mmax = np.concatenate([old_mmax, np.asarray(extension["maximum_mass"])])
    la = np.concatenate([old_la, np.asarray(extension["log_astro"])])
    components = np.concatenate([old_components, np.asarray(extension["components"])])
    nicer = np.concatenate([old_nicer, np.asarray(extension["nicer"])])
    b3, region, sext = memberships(x)
    if not np.array_equal(b3[: len(old_x)], old_b3) or not np.array_equal(
        sext[: len(old_x)], old_sext
    ):
        raise RuntimeError("hyperonic membership changed during concatenation")
    correction = math.log(arguments.base_proposals) - np.log(
        arguments.base_proposals
        + arguments.b3_proposals * b3.astype(np.float64)
        + EXTENSION_PROPOSALS * sext.astype(np.float64)
    )
    denominator = math.log(arguments.base_proposals)
    finite = np.isfinite(la)
    component_error = float(np.max(np.abs(components[finite].sum(1) - la[finite])))
    nicer_error = float(np.max(np.abs(nicer[finite].sum(1) - components[finite, 1])))
    if component_error > 1.0e-8 or nicer_error > 1.0e-8:
        raise RuntimeError("assembled hyperonic likelihood decomposition fails")
    support_rows = sum(
        int(stream["selected_rows"]) for stream in base_metadata["support_streams"]
    )
    uniform_rows = len(old_x) - support_rows
    if uniform_rows != arguments.hyperonic_uniform_rows:
        raise RuntimeError(
            f"hyperonic uniform-row count is {uniform_rows}, expected "
            f"{arguments.hyperonic_uniform_rows}"
        )
    coverage, coverage_gates = coverage_metrics(
        old_x[:uniform_rows], x, la, correction, denominator
    )
    all_gates = all(coverage_gates.values())
    metadata = {
        "lineage_class": "independent_conditional_evidence_network",
        "model": "ddb-hyperonic",
        "role": "S_ext-expanded exact full-prior Green-EN physics bank",
        "base_bank": {"path": str(arguments.base_bank.resolve()), "sha256": sha256(arguments.base_bank)},
        "extension_screen": {"path": str(arguments.extension_screen.resolve()), "sha256": sha256(arguments.extension_screen)},
        "extension_exact": {"path": str(arguments.extension_exact.resolve()), "sha256": sha256(arguments.extension_exact)},
        "base_total_proposals": arguments.base_proposals,
        "b3_total_proposals": arguments.b3_proposals,
        "sext_total_proposals": EXTENSION_PROPOSALS,
        "extension_seed": int(extension["seed"]),
        "forbidden_artifacts_used": [],
        "held_out_sources_used": [],
    }
    atomic_savez(
        arguments.output,
        schema=np.asarray("conditional-en-independent-physics-v2"),
        model=np.asarray("ddb-hyperonic"),
        X=x.astype(np.float32),
        theta=theta.astype(np.float32),
        R=radius,
        MG=old_grid.astype(np.float32),
        MM=mmax.astype(np.float32),
        LA=la,
        LPC=correction,
        AC=components,
        NIC=nicer,
        lpc_all_lse=np.float64(denominator),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "status": "PASS" if all_gates else "FAIL_COVERAGE_GATE",
        "sector": "hyperonic",
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "rows": int(len(x)),
        "finite_rows": int(finite.sum()),
        "old_rows": int(len(old_x)),
        "extension_rows": int(len(x) - len(old_x)),
        "extension_valid_rows": int(np.asarray(extension["valid"]).sum()),
        "old_rows_in_sext": int(old_sext.sum()),
        "region_rows": int(region.sum()),
        "old_correction_max_abs_error": old_correction_error,
        "likelihood_component_max_abs_error": component_error,
        "nicer_component_max_abs_error": nicer_error,
        "screen_exact_maximum_prediction_delta_sigma": extension["normalized_prediction_delta"],
        "direct_evidence": direct_report(x, la, correction, denominator),
        "coverage": coverage,
        "coverage_gates": coverage_gates,
        "forbidden_artifacts_used": [],
        "held_out_sources_used": [],
    }


def load_fixed_j0740(directory: Path, rows: int) -> tuple[np.ndarray, float]:
    old = np.full(rows, np.nan, dtype=np.float64)
    fixed = np.full(rows, np.nan, dtype=np.float64)
    filled = np.zeros(rows, dtype=bool)
    paths = sorted(directory.glob("tov_terms_*.npz"))
    if not paths:
        raise RuntimeError(f"no fixed-J0740 replay shards in {directory}")
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            record_type = np.asarray(shard["record_type"])
            record_index = np.asarray(shard["record_index"], dtype=np.int64)
            select = (record_type == 0) & (record_index < rows)
            index = record_index[select]
            if filled[index].any():
                raise RuntimeError(f"duplicate fixed-J0740 indices in {path}")
            old[index] = np.asarray(shard["old_0740"], dtype=np.float64)[select]
            fixed[index] = np.asarray(shard["fixed_0740"], dtype=np.float64)[select]
            filled[index] = True
    if not filled.all() or not np.isfinite(old).all() or not np.isfinite(fixed).all():
        raise RuntimeError("fixed-J0740 replay is incomplete")
    return fixed, float(np.max(np.abs(fixed - old)))


def radius_at_1p4(radius: np.ndarray, mass_grid: np.ndarray) -> np.ndarray:
    upper = int(np.searchsorted(mass_grid, 1.4, side="left"))
    if upper == 0 or upper >= len(mass_grid):
        raise RuntimeError("mass grid does not bracket 1.4 solar masses")
    lower = upper - 1
    fraction = (1.4 - mass_grid[lower]) / (mass_grid[upper] - mass_grid[lower])
    left = radius[:, lower]
    right = radius[:, upper]
    result = left + fraction * (right - left)
    result[~(np.isfinite(left) & np.isfinite(right))] = np.nan
    return result


def corner_membership(
    prediction: np.ndarray,
    r14: np.ndarray,
    maximum_mass: np.ndarray,
    beta: np.ndarray,
    prediction_interval: np.ndarray,
    window: np.ndarray,
) -> np.ndarray:
    predicted_radius = np.column_stack(
        [np.ones(len(prediction)), prediction]
    ) @ beta
    return (
        (predicted_radius >= prediction_interval[0])
        & (predicted_radius <= prediction_interval[1])
        & np.isfinite(r14)
        & np.isfinite(maximum_mass)
        & (r14 >= window[0])
        & (r14 <= window[1])
        & (maximum_mass >= window[2])
    )


def assemble_nucleonic(arguments: argparse.Namespace) -> dict[str, object]:
    extension = validate_extension(
        arguments.extension_screen, arguments.extension_exact, "ddb"
    )
    with np.load(arguments.base_bank, allow_pickle=False) as base:
        if str(base["schema"].item()) != "conditional-en-independent-physics-v1":
            raise RuntimeError("unexpected base nucleonic bank schema")
        base_metadata = json.loads(str(base["metadata"].item()))
        if base_metadata.get("forbidden_artifacts_used") or base_metadata.get("held_out_sources_used"):
            raise RuntimeError("base nucleonic bank ancestry gate failed")
        old_x = np.asarray(base["X"], dtype=np.float64)
        old_radius = np.asarray(base["R"], dtype=np.float32)
        old_grid = np.asarray(base["MG"], dtype=np.float64)
        old_mmax = np.asarray(base["MM"], dtype=np.float64)
        old_la = np.asarray(base["LA"], dtype=np.float64)
        old_lpc = np.asarray(base["LPC"], dtype=np.float64)

    with np.load(arguments.raw_bank, allow_pickle=False) as raw:
        raw_theta = np.asarray(raw["theta"], dtype=np.float64)
        raw_x = np.asarray(raw["X"], dtype=np.float64)
        raw_mmax = np.asarray(raw["MM"], dtype=np.float64)
        raw_r14 = np.asarray(raw["R14"], dtype=np.float64)
        raw_lpc = np.asarray(raw["logw_prior"], dtype=np.float64)
        in_b = np.asarray(raw["in_SB"], dtype=bool)
        in_c = np.asarray(raw["in_SC"], dtype=bool)
        in_d = np.asarray(raw["in_SD"], dtype=bool)
        n0 = int(raw["N0"])
        mtot_b = int(raw["MtotB"])
        mtot_c = int(raw["MtotC"])
        mtot_d = int(raw["MtotD"])
        beta = np.asarray(raw["beta_r14"], dtype=np.float64)
        pred_c = np.asarray(raw["predC_int"], dtype=np.float64)
        pred_d = np.asarray(raw["predD_int"], dtype=np.float64)
        win_c = np.asarray(raw["winC"], dtype=np.float64)
        win_d = np.asarray(raw["winD"], dtype=np.float64)
    raw_valid = np.isfinite(raw_mmax)
    if int(raw_valid.sum()) != len(old_x):
        raise RuntimeError("base nucleonic rows do not match raw valid rows")
    if not np.array_equal(old_x.astype(np.float32), raw_x[raw_valid].astype(np.float32)):
        raise RuntimeError("base nucleonic X does not map to the raw bank")
    if not np.array_equal(old_lpc, raw_lpc[raw_valid]):
        raise RuntimeError("base nucleonic correction does not map to the raw bank")
    reconstructed_old_lpc = math.log(n0) - np.log(
        n0
        + mtot_b * in_b.astype(np.float64)
        + mtot_c * in_c.astype(np.float64)
        + mtot_d * in_d.astype(np.float64)
    )
    old_correction_error = float(np.max(np.abs(raw_lpc - reconstructed_old_lpc)))
    if old_correction_error > 1.0e-12:
        raise RuntimeError(f"raw nucleonic correction does not reproduce: {old_correction_error:.3e}")

    with np.load(arguments.exact_table, allow_pickle=False) as exact:
        old_j0030_all = np.asarray(exact["exact_j0030"], dtype=np.float64)
        old_j0740_all = np.asarray(exact["exact_j0740"], dtype=np.float64)
        old_j0437_all = np.asarray(exact["exact_j0437"], dtype=np.float64)
        old_gw_all = np.asarray(exact["exact_gw"], dtype=np.float64)
        old_mmax_term_all = np.asarray(exact["Rmm"], dtype=np.float64)
    fixed_j0740, fixed_shift_max = load_fixed_j0740(
        arguments.fixed_j0740_terms, len(old_x)
    )
    old_j0740 = old_j0740_all[raw_valid]
    # The replay's old column must reproduce the exact table.  Recover it once
    # more here without retaining it in the final product.
    replay_old = np.full(len(old_x), np.nan, dtype=np.float64)
    filled = np.zeros(len(old_x), dtype=bool)
    for path in sorted(arguments.fixed_j0740_terms.glob("tov_terms_*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            record_type = np.asarray(shard["record_type"])
            record_index = np.asarray(shard["record_index"], dtype=np.int64)
            select = (record_type == 0) & (record_index < len(old_x))
            index = record_index[select]
            replay_old[index] = np.asarray(shard["old_0740"], dtype=np.float64)[select]
            filled[index] = True
    replay_error = float(np.max(np.abs(replay_old - old_j0740)))
    if not filled.all() or replay_error > 2.75e-9:
        raise RuntimeError(f"old J0740 replay mismatch: {replay_error:.3e}")
    old_nicer = np.column_stack(
        [old_j0030_all[raw_valid], fixed_j0740, old_j0437_all[raw_valid]]
    )
    old_components = np.column_stack(
        [
            old_mmax_term_all[raw_valid],
            old_nicer.sum(axis=1),
            old_gw_all[raw_valid],
            old_la
            - old_mmax_term_all[raw_valid]
            - old_nicer.sum(axis=1)
            - old_gw_all[raw_valid],
        ]
    )
    old_component_error = float(np.max(np.abs(old_components.sum(1) - old_la)))
    if old_component_error > 1.0e-10:
        raise RuntimeError("old nucleonic component reconstruction fails")

    if not np.array_equal(old_grid.astype(np.float32), np.asarray(extension["mass_grid"]).astype(np.float32)):
        raise RuntimeError("nucleonic base and extension mass grids differ")
    extension_radius = np.asarray(extension["radius"], dtype=np.float64)
    extension_mmax = np.asarray(extension["maximum_mass"], dtype=np.float64)
    extension_x = np.asarray(extension["prediction"], dtype=np.float64)
    extension_r14 = radius_at_1p4(extension_radius, np.asarray(extension["mass_grid"]))
    extension_b3, _, extension_sext = memberships(extension_x)
    extension_c = corner_membership(
        extension_x, extension_r14, extension_mmax, beta, pred_c, win_c
    )
    extension_d = corner_membership(
        extension_x, extension_r14, extension_mmax, beta, pred_d, win_d
    )
    if extension_b3.any() or not extension_sext.all():
        raise RuntimeError("nucleonic extension is not disjoint from B3")
    _, _, raw_sext = memberships(raw_x)
    old_denominator_density = (
        n0
        + mtot_b * in_b.astype(np.float64)
        + mtot_c * in_c.astype(np.float64)
        + mtot_d * in_d.astype(np.float64)
        + EXTENSION_PROPOSALS * raw_sext.astype(np.float64)
    )
    extension_denominator_density = (
        n0
        + mtot_b * extension_b3.astype(np.float64)
        + mtot_c * extension_c.astype(np.float64)
        + mtot_d * extension_d.astype(np.float64)
        + EXTENSION_PROPOSALS * extension_sext.astype(np.float64)
    )
    old_new_lpc_all = math.log(n0) - np.log(old_denominator_density)
    extension_lpc_all = math.log(n0) - np.log(extension_denominator_density)
    denominator = float(logsumexp(np.concatenate([old_new_lpc_all, extension_lpc_all])))

    extension_valid = np.asarray(extension["valid"], dtype=bool)
    x = np.concatenate([old_x, extension_x[extension_valid]])
    theta = np.concatenate([raw_theta[raw_valid], np.asarray(extension["theta"])[extension_valid]])
    radius = np.concatenate(
        [old_radius, extension_radius[extension_valid].astype(np.float32)]
    )
    mmax = np.concatenate([old_mmax, extension_mmax[extension_valid]])
    la = np.concatenate([old_la, np.asarray(extension["log_astro"])[extension_valid]])
    correction = np.concatenate(
        [old_new_lpc_all[raw_valid], extension_lpc_all[extension_valid]]
    )
    components = np.concatenate(
        [old_components, np.asarray(extension["components"])[extension_valid]]
    )
    nicer = np.concatenate(
        [old_nicer, np.asarray(extension["nicer"])[extension_valid]]
    )
    component_error = float(np.max(np.abs(components.sum(1) - la)))
    nicer_error = float(np.max(np.abs(nicer.sum(1) - components[:, 1])))
    if component_error > 1.0e-8 or nicer_error > 1.0e-8:
        raise RuntimeError("assembled nucleonic likelihood decomposition fails")
    coverage, coverage_gates = coverage_metrics(
        raw_x[:n0], x, la, correction, denominator
    )
    all_gates = all(coverage_gates.values())
    metadata = {
        "lineage_class": "independent_conditional_evidence_network",
        "model": "ddb",
        "role": "S_ext-expanded exact full-prior Green-EN physics bank",
        "base_bank": {"path": str(arguments.base_bank.resolve()), "sha256": sha256(arguments.base_bank)},
        "raw_bank": {"path": str(arguments.raw_bank.resolve()), "sha256": sha256(arguments.raw_bank)},
        "exact_table": {"path": str(arguments.exact_table.resolve()), "sha256": sha256(arguments.exact_table)},
        "extension_screen": {"path": str(arguments.extension_screen.resolve()), "sha256": sha256(arguments.extension_screen)},
        "extension_exact": {"path": str(arguments.extension_exact.resolve()), "sha256": sha256(arguments.extension_exact)},
        "base_total_proposals": n0,
        "b3_total_proposals": mtot_b,
        "corner_c_total_proposals": mtot_c,
        "corner_d_total_proposals": mtot_d,
        "sext_total_proposals": EXTENSION_PROPOSALS,
        "extension_seed": int(extension["seed"]),
        "forbidden_artifacts_used": [],
        "held_out_sources_used": [],
    }
    atomic_savez(
        arguments.output,
        schema=np.asarray("conditional-en-independent-physics-v2"),
        model=np.asarray("ddb"),
        X=x.astype(np.float32),
        theta=theta.astype(np.float32),
        R=radius,
        MG=old_grid.astype(np.float32),
        MM=mmax.astype(np.float32),
        LA=la,
        LPC=correction,
        AC=components,
        NIC=nicer,
        lpc_all_lse=np.float64(denominator),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "status": "PASS" if all_gates else "FAIL_COVERAGE_GATE",
        "sector": "nucleonic",
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "rows": int(len(x)),
        "finite_rows": int(np.isfinite(la).sum()),
        "old_rows": int(len(old_x)),
        "extension_rows_retained": int(extension_valid.sum()),
        "extension_rows_screened": int(len(extension_valid)),
        "old_rows_in_sext_all": int(raw_sext.sum()),
        "old_valid_rows_in_sext": int(raw_sext[raw_valid].sum()),
        "extension_corner_c_rows": int(extension_c.sum()),
        "extension_corner_d_rows": int(extension_d.sum()),
        "old_correction_max_abs_error": old_correction_error,
        "old_component_max_abs_error": old_component_error,
        "assembled_component_max_abs_error": component_error,
        "assembled_nicer_max_abs_error": nicer_error,
        "old_j0740_replay_max_abs_error": replay_error,
        "fixed_j0740_maximum_change": fixed_shift_max,
        "screen_exact_maximum_prediction_delta_sigma": extension["normalized_prediction_delta"],
        "direct_evidence": direct_report(x, la, correction, denominator),
        "coverage": coverage,
        "coverage_gates": coverage_gates,
        "denominator_logsumexp_all_rows": denominator,
        "forbidden_artifacts_used": [],
        "held_out_sources_used": [],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="sector", required=True)

    hyperonic = subparsers.add_parser("hyperonic")
    hyperonic.add_argument("--base-bank", type=Path, required=True)
    hyperonic.add_argument("--extension-screen", type=Path, required=True)
    hyperonic.add_argument("--extension-exact", type=Path, required=True)
    hyperonic.add_argument("--output", type=Path, required=True)
    hyperonic.add_argument("--base-proposals", type=int, default=600_000)
    hyperonic.add_argument("--b3-proposals", type=int, default=320_000_000)
    hyperonic.add_argument("--hyperonic-uniform-rows", type=int, default=415_681)

    nucleonic = subparsers.add_parser("nucleonic")
    nucleonic.add_argument("--base-bank", type=Path, required=True)
    nucleonic.add_argument("--raw-bank", type=Path, required=True)
    nucleonic.add_argument("--exact-table", type=Path, required=True)
    nucleonic.add_argument("--fixed-j0740-terms", type=Path, required=True)
    nucleonic.add_argument("--extension-screen", type=Path, required=True)
    nucleonic.add_argument("--extension-exact", type=Path, required=True)
    nucleonic.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite assembled bank: {arguments.output}")
    started_unix = time.time()
    started_monotonic = time.monotonic()
    report = (
        assemble_hyperonic(arguments)
        if arguments.sector == "hyperonic"
        else assemble_nucleonic(arguments)
    )
    report.update(
        {
            "schema": "green-en-support-extension-assembly-report-v1",
            "assembler": str(Path(__file__).resolve()),
            "assembler_sha256": sha256(Path(__file__)),
            "hostname": socket.gethostname(),
            "started_unix": started_unix,
            "finished_unix": time.time(),
            "wall_seconds": time.monotonic() - started_monotonic,
        }
    )
    atomic_json(arguments.output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
