#!/usr/bin/env python3
"""Generate the paper's structured nucleonic A-NET training scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.anet.shifted_nicer import build_shifted_sources
from workflows import A1Problem
from workflows.data_gate import sha256, validate_observational_data
from workflows.source_scenarios import BASE_NICER_SOURCES


SOURCES = ("j0030", "j0740", "j0437")
IDENTITY = (0.0, 0.0, 1.0, 1.0)


def interpolate_grid(values, grid, query):
    index = np.clip(np.searchsorted(grid, query) - 1, 0, len(grid) - 2)
    fraction = (query - grid[index]) / (grid[index + 1] - grid[index])
    return values[:, index] * (1.0 - fraction) + values[:, index + 1] * fraction


def weighted_table(path, source):
    table = np.loadtxt(path)
    weight = (
        np.ones(len(table), dtype=np.float64)
        if source.weight_column is None
        else np.asarray(table[:, source.weight_column], dtype=np.float64)
    )
    weight = weight / weight.sum()
    mass = np.asarray(table[:, source.mass_column], dtype=np.float64)
    radius = np.asarray(table[:, source.radius_column], dtype=np.float64)
    centre = (
        float(np.sum(weight * mass)),
        float(np.sum(weight * radius)),
    )
    return mass, radius, weight, centre


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=2000)
    parser.add_argument("--cloud-points", type=int, default=256)
    parser.add_argument("--validation-scenarios", type=int, default=100)
    parser.add_argument("--identity-scenarios", type=int, default=15)
    parser.add_argument("--maximum-posterior-rows", type=int, default=6000)
    parser.add_argument("--minimum-bank-ess", type=float, default=250.0)
    parser.add_argument("--identity-fraction", type=float, default=0.25)
    parser.add_argument("--one-slot-fraction", type=float, default=0.50)
    parser.add_argument("--tail-boost", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists() and not arguments.force:
        raise FileExistsError(
            "scenario archive exists; resume from its validated campaign receipt "
            "or pass --force explicitly"
        )
    if arguments.scenarios <= arguments.validation_scenarios:
        raise ValueError("training scenarios must remain after validation split")
    if arguments.identity_scenarios > arguments.scenarios:
        raise ValueError("identity-scenario count exceeds all scenarios")
    if not (
        0 <= arguments.identity_fraction <= 1
        and 0 <= arguments.one_slot_fraction <= 1
        and arguments.identity_fraction + arguments.one_slot_fraction <= 1
    ):
        raise ValueError("invalid structured-scenario fractions")
    if not 0 <= arguments.tail_boost <= 1:
        raise ValueError("tail boost must lie in [0,1]")
    if arguments.cloud_points != 256:
        raise ValueError("paper nucleonic A-NET requires 256 cloud points")
    if arguments.smoke:
        arguments.scenarios = 12
        arguments.validation_scenarios = 3
        arguments.identity_scenarios = 2
        arguments.maximum_posterior_rows = 200
        arguments.minimum_bank_ess = 1.0

    started = time.time()
    with np.load(arguments.bank, allow_pickle=False) as bank:
        required = {
            "theta", "X", "Rg", "MG", "MM", "GWG", "GLAM", "PQG", "RHO"
        }
        missing = required - set(bank.files)
        if missing:
            raise ValueError(f"A-NET bank is missing {sorted(missing)}")
        theta = np.asarray(bank["theta"], dtype=np.float64)
        prediction = np.asarray(bank["X"], dtype=np.float64)
        radius_grid = np.asarray(bank["Rg"], dtype=np.float64)
        mass_grid = np.asarray(bank["MG"], dtype=np.float64)
        maximum_mass = np.asarray(bank["MM"], dtype=np.float64)
        gw_grid = np.asarray(bank["GWG"], dtype=np.float64)
        gamma_grid = np.asarray(bank["GLAM"], dtype=np.float64)
        pqcd_grid = np.asarray(bank["PQG"], dtype=np.float64)
        rho_grid = np.asarray(bank["RHO"], dtype=np.float64)
        log_prior_correction = (
            np.asarray(bank["logw_prior"], dtype=np.float64)
            if "logw_prior" in bank.files
            else np.zeros(len(theta), dtype=np.float64)
        )
        base_rows = int(bank["N0"]) if "N0" in bank.files else len(theta)
    with np.load(arguments.exact, allow_pickle=False) as exact:
        exact_nicer = {
            name: np.asarray(exact[f"exact_{name}"], dtype=np.float64)
            for name in SOURCES
        }
        exact_gw = np.asarray(exact["exact_gw"], dtype=np.float64)
    if any(len(values) != len(theta) for values in (*exact_nicer.values(), exact_gw)):
        raise ValueError("A-NET bank and exact table have different row counts")

    paths = validate_observational_data(arguments.data_root)
    problem = A1Problem.from_data_root(arguments.data_root)
    interpolators = dict(zip(SOURCES, problem.target.nicer_interpolators))
    rng = np.random.default_rng(arguments.seed)
    index_rng = np.random.default_rng(777 + arguments.seed)
    tables = {}
    centres = {}
    cloud_pool = {}
    slot_standard_deviation = {}
    for name, source in zip(SOURCES, BASE_NICER_SOURCES):
        mass, radius, weight, centre = weighted_table(
            paths[source.filename], source
        )
        centres[name] = centre
        slot_standard_deviation[name] = (
            float(np.sqrt(np.sum(weight * (mass - centre[0]) ** 2))),
            float(np.sqrt(np.sum(weight * (radius - centre[1]) ** 2))),
        )
        selected = rng.choice(len(mass), min(len(mass), 40_000), p=weight)
        cloud_pool[name] = np.column_stack([mass[selected], radius[selected]])
        tables[name] = (mass, radius, weight)

    covered = mass_grid >= 1.0
    shifted_nicer = build_shifted_sources(
        interpolators,
        radius_grid[:, covered],
        mass_grid[covered],
        maximum_mass,
        centres,
        device=arguments.device,
    )
    log_mmax = np.full(len(maximum_mass), -np.inf, dtype=np.float64)
    finite_mmax = np.isfinite(maximum_mass)
    log_mmax[finite_mmax] = -np.logaddexp(
        0.0, -(maximum_mass[finite_mmax] - 2.0) / 0.05
    )
    baseline_gw = interpolate_grid(gw_grid, gamma_grid, 0.0)
    valid = (
        np.isfinite(maximum_mass)
        & np.isfinite(radius_grid).any(axis=1)
        & np.isfinite(exact_gw)
        & np.isfinite(exact_nicer["j0030"])
        & np.isfinite(exact_nicer["j0740"])
        & np.isfinite(exact_nicer["j0437"])
    )

    mass_shift_range = (-0.5, 0.25)
    radius_shift_range = (-1.6, 1.6)
    scale_range = (0.28, 2.2)

    def draw_scale():
        lower, upper = np.log(scale_range)
        third = (upper - lower) / 3.0
        if rng.random() < arguments.tail_boost:
            value = rng.uniform(0.0, 2.0 * third)
            log_scale = lower + value if value < third else upper - (value - third)
        else:
            log_scale = rng.uniform(lower, upper)
        return float(np.exp(log_scale))

    def draw_transform():
        return (
            float(rng.uniform(*mass_shift_range)),
            float(rng.uniform(*radius_shift_range)),
            draw_scale(),
            draw_scale(),
        )

    def draw_structured():
        value = rng.random()
        if value < arguments.identity_fraction:
            return {name: IDENTITY for name in SOURCES}, 2
        if value < arguments.identity_fraction + arguments.one_slot_fraction:
            transformed = SOURCES[int(rng.integers(len(SOURCES)))]
            return {
                name: draw_transform() if name == transformed else IDENTITY
                for name in SOURCES
            }, 1
        return {name: draw_transform() for name in SOURCES}, 0

    scenario_cloud = []
    scenario_theta = []
    scenario_prediction = []
    scenario_dial = []
    scenario_transform = []
    scenario_ess = []
    scenario_mode = []
    rejected = 0
    attempts = 0
    maximum_attempts = max(10_000, 100 * arguments.scenarios)
    while len(scenario_cloud) < arguments.scenarios:
        attempts += 1
        if attempts > maximum_attempts:
            raise RuntimeError(
                "A-NET scenario generator exceeded its rejection limit; "
                "the bank does not cover the declared amortization domain"
            )
        identity_case = len(scenario_cloud) < arguments.identity_scenarios
        if identity_case:
            transforms = {name: IDENTITY for name in SOURCES}
            gamma, rho, mode = 0.0, 1.2, 2
        else:
            transforms, mode = draw_structured()
            gamma = float(rng.uniform(gamma_grid[0] + 0.02, gamma_grid[-1] - 0.02))
            rho = float(rng.uniform(rho_grid[0] + 0.02, rho_grid[-1] - 0.02))
        log_weight = (
            log_prior_correction
            + log_mmax
            + exact_gw
            + interpolate_grid(gw_grid, gamma_grid, gamma)
            - baseline_gw
            + interpolate_grid(pqcd_grid, rho_grid, rho)
        )
        for name in SOURCES:
            transform = transforms[name]
            log_weight = log_weight + (
                exact_nicer[name]
                if transform == IDENTITY
                else shifted_nicer(name, *transform)
            )
        selected_log_weight = np.where(valid, log_weight, -np.inf)
        normalization = logsumexp(selected_log_weight)
        if not np.isfinite(normalization):
            rejected += 1
            continue
        weight = np.exp(selected_log_weight - normalization)
        effective_sample_size = float(1.0 / np.sum(weight**2))
        if effective_sample_size < arguments.minimum_bank_ess and not identity_case:
            rejected += 1
            continue
        posterior_rows = int(
            np.clip(
                0.8 * effective_sample_size,
                min(200, arguments.maximum_posterior_rows),
                arguments.maximum_posterior_rows,
            )
        )
        chosen = index_rng.choice(len(theta), posterior_rows, p=weight)
        scenario_theta.append(theta[chosen])
        scenario_prediction.append(prediction[chosen])
        clouds = []
        for name in SOURCES:
            mass_shift, radius_shift, mass_scale, radius_scale = transforms[name]
            centre = np.asarray(centres[name])
            points = cloud_pool[name][
                rng.choice(len(cloud_pool[name]), arguments.cloud_points)
            ]
            clouds.append(
                np.column_stack(
                    [
                        centre[0]
                        + mass_scale * (points[:, 0] - centre[0])
                        + mass_shift,
                        centre[1]
                        + radius_scale * (points[:, 1] - centre[1])
                        + radius_shift,
                    ]
                )
            )
        scenario_cloud.append(np.stack(clouds))
        scenario_dial.append([gamma, rho])
        scenario_transform.append([transforms[name] for name in SOURCES])
        scenario_ess.append(effective_sample_size)
        scenario_mode.append(mode)
        if len(scenario_cloud) % 250 == 0:
            print(
                f"[anet-scenarios] {len(scenario_cloud)}/{arguments.scenarios} "
                f"median ESS={np.median(scenario_ess):.0f} rejected={rejected}",
                flush=True,
            )

    lengths = np.asarray([len(value) for value in scenario_theta], dtype=np.int64)
    metadata = {
        "schema_version": 1,
        "bank": str(arguments.bank),
        "bank_sha256": sha256(arguments.bank),
        "exact": str(arguments.exact),
        "exact_sha256": sha256(arguments.exact),
        "base_uniform_rows": base_rows,
        "seed": arguments.seed,
        "identity_scenarios": arguments.identity_scenarios,
        "scenario_structure": {
            "identity_fraction": arguments.identity_fraction,
            "one_slot_fraction": arguments.one_slot_fraction,
            "all_slot_fraction": 1.0
            - arguments.identity_fraction
            - arguments.one_slot_fraction,
        },
        "domain": {
            "mass_shift": list(mass_shift_range),
            "radius_shift": list(radius_shift_range),
            "scale": list(scale_range),
            "gamma": gamma_grid[[0, -1]].tolist(),
            "rho_eval": rho_grid[[0, -1]].tolist(),
        },
        "centres": {name: list(centres[name]) for name in SOURCES},
        "slot_standard_deviation": {
            name: list(slot_standard_deviation[name]) for name in SOURCES
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.output,
        gtheta=np.concatenate(scenario_prediction, axis=0),
        clouds=np.asarray(scenario_cloud),
        dial=np.asarray(scenario_dial),
        transforms=np.asarray(scenario_transform),
        ess=np.asarray(scenario_ess),
        mode=np.asarray(scenario_mode),
        ntr=np.int64(arguments.scenarios - arguments.validation_scenarios),
        nreal=np.int64(arguments.identity_scenarios),
        theta=np.concatenate(scenario_theta, axis=0),
        theta_len=lengths,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = {
        "status": "SMOKE_COMPLETED" if arguments.smoke else "PASS",
        "stage": "A-NET scenario generation",
        "scenarios": arguments.scenarios,
        "training_scenarios": arguments.scenarios - arguments.validation_scenarios,
        "posterior_rows": int(lengths.sum()),
        "minimum_ess": float(np.min(scenario_ess)),
        "median_ess": float(np.median(scenario_ess)),
        "rejected_candidates": rejected,
        "output": str(arguments.output),
        "output_sha256": sha256(arguments.output),
        "metadata": metadata,
        "wall_seconds": time.time() - started,
    }
    report_path = arguments.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
