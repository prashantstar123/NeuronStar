#!/usr/bin/env python3
"""Build nine-dimensional hyperonic A-NET scenarios by nucleonic parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from inference.anet.shifted_nicer import build_shifted_sources
from workflows import A1Problem
from workflows.data_gate import validate_observational_data
from workflows.source_scenarios import BASE_NICER_SOURCES


SOURCES = ("j0030", "j0740", "j0437")
IDENTITY = (0.0, 0.0, 1.0, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_table(path: Path, source):
    table = np.loadtxt(path)
    weight = (
        np.ones(len(table), dtype=np.float64)
        if source.weight_column is None
        else np.asarray(table[:, source.weight_column], dtype=np.float64)
    )
    if not np.isfinite(weight).all() or not (weight.sum() > 0):
        raise RuntimeError(f"invalid cloud weights in {path}")
    weight = weight / weight.sum()
    mass = np.asarray(table[:, source.mass_column], dtype=np.float64)
    radius = np.asarray(table[:, source.radius_column], dtype=np.float64)
    centre = (
        float(np.sum(weight * mass)),
        float(np.sum(weight * radius)),
    )
    return mass, radius, weight, centre


def load_caches(paths: list[Path], correction_path: Path):
    accumulated = {
        key: []
        for key in (
            "theta",
            "prediction",
            "log_astrophysical",
            "astrophysical_components",
            "nicer_source_components",
            "target_valid",
            "radius",
            "maximum_mass",
        )
    }
    mass_grid = None
    prior_low = None
    prior_high = None
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "schema",
                "theta",
                "prediction",
                "log_astrophysical",
                "astrophysical_components",
                "nicer_source_components",
                "target_valid",
                "mass_grid",
                "radius",
                "maximum_mass",
                "prior_low",
                "prior_high",
                "reference_present",
                "forbidden_artifacts_used",
            }
            missing = required - set(archive.files)
            if missing:
                raise RuntimeError(f"{path} is missing {sorted(missing)}")
            schema = str(archive["schema"].item())
            if schema not in {
                "ddb-hyperonic-clean-uniform-training-cache-v1",
                "ddb-hyperonic-clean-support-training-cache-v1",
                "ddb-hyperonic-clean-proposal-training-cache-v1",
            }:
                raise RuntimeError(f"unexpected cache schema in {path}")
            if bool(archive["reference_present"]):
                raise RuntimeError(f"reference contamination declared by {path}")
            if archive["forbidden_artifacts_used"].size:
                raise RuntimeError(f"forbidden ancestry declared by {path}")
            local_mass_grid = np.asarray(archive["mass_grid"], dtype=np.float64)
            local_low = np.asarray(archive["prior_low"], dtype=np.float64)
            local_high = np.asarray(archive["prior_high"], dtype=np.float64)
            if mass_grid is None:
                mass_grid = local_mass_grid
                prior_low = local_low
                prior_high = local_high
            elif not (
                np.array_equal(mass_grid, local_mass_grid)
                and np.array_equal(prior_low, local_low)
                and np.array_equal(prior_high, local_high)
            ):
                raise RuntimeError("cache mass grid or prior bounds differ")
            rows = len(archive["theta"])
            for key in accumulated:
                value = np.asarray(archive[key])
                accumulated[key].append(value)
            records.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "rows": rows,
                    "schema": schema,
                }
            )
    merged = {
        key: np.concatenate(values, axis=0)
        for key, values in accumulated.items()
    }
    with np.load(correction_path, allow_pickle=False) as correction:
        required = {
            "schema",
            "cache_sha256",
            "cache_rows",
            "log_prior_correction",
            "reference_present",
            "forbidden_artifacts_used",
        }
        missing = required - set(correction.files)
        if missing:
            raise RuntimeError(
                f"counting correction is missing {sorted(missing)}"
            )
        if str(correction["schema"].item()) not in {
            "ddb-hyperonic-clean-counting-correction-v1",
            "ddb-hyperonic-clean-augmented-counting-correction-v1",
        }:
            raise RuntimeError("unexpected counting-correction schema")
        if bool(correction["reference_present"]) or correction["forbidden_artifacts_used"].size:
            raise RuntimeError("counting correction declares forbidden ancestry")
        expected_hashes = np.asarray(
            [record["sha256"] for record in records]
        )
        expected_rows = np.asarray(
            [record["rows"] for record in records], dtype=np.int64
        )
        if not np.array_equal(correction["cache_sha256"], expected_hashes):
            raise RuntimeError("counting correction cache hashes do not match")
        if not np.array_equal(correction["cache_rows"], expected_rows):
            raise RuntimeError("counting correction cache row counts do not match")
        log_prior_correction = np.asarray(
            correction["log_prior_correction"], dtype=np.float64
        )
    if log_prior_correction.shape != (len(merged["theta"]),):
        raise RuntimeError("counting correction has the wrong row count")
    merged["log_prior_correction"] = log_prior_correction
    correction_record = {
        "path": str(correction_path.resolve()),
        "sha256": sha256(correction_path),
    }
    return (
        merged,
        mass_grid,
        prior_low,
        prior_high,
        records,
        correction_record,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--counting-correction", type=Path, required=True)
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
        raise FileExistsError(f"scenario archive already exists: {arguments.output}")
    if arguments.scenarios <= arguments.validation_scenarios:
        raise ValueError("training scenarios must remain after validation split")
    if arguments.identity_scenarios > arguments.scenarios:
        raise ValueError("identity-scenario count exceeds all scenarios")
    if arguments.cloud_points != 256:
        raise ValueError("nucleonic-parity A-NET requires 256 cloud points")
    if not (
        0 <= arguments.identity_fraction <= 1
        and 0 <= arguments.one_slot_fraction <= 1
        and arguments.identity_fraction + arguments.one_slot_fraction <= 1
    ):
        raise ValueError("invalid structured-scenario fractions")
    if not 0 <= arguments.tail_boost <= 1:
        raise ValueError("tail boost must lie in [0,1]")
    if arguments.smoke:
        arguments.scenarios = 12
        arguments.validation_scenarios = 3
        arguments.identity_scenarios = 2
        arguments.maximum_posterior_rows = 200
        arguments.minimum_bank_ess = 1.0

    started = time.time()
    (
        bank,
        mass_grid,
        prior_low,
        prior_high,
        cache_records,
        correction_record,
    ) = load_caches(arguments.cache, arguments.counting_correction)
    theta = np.asarray(bank["theta"], dtype=np.float64)
    prediction = np.asarray(bank["prediction"], dtype=np.float64)
    radius_grid = np.asarray(bank["radius"], dtype=np.float64)
    maximum_mass = np.asarray(bank["maximum_mass"], dtype=np.float64)
    log_astrophysical = np.asarray(bank["log_astrophysical"], dtype=np.float64)
    components = np.asarray(bank["astrophysical_components"], dtype=np.float64)
    exact_nicer_matrix = np.asarray(
        bank["nicer_source_components"], dtype=np.float64
    )
    log_prior_correction = np.asarray(
        bank["log_prior_correction"], dtype=np.float64
    )
    valid = np.asarray(bank["target_valid"], dtype=bool)
    rows = len(theta)
    expected_shapes = {
        "theta": (rows, 9),
        "prediction": (rows, 7),
        "radius": (rows, len(mass_grid)),
        "maximum_mass": (rows,),
        "log_astrophysical": (rows,),
        "components": (rows, 4),
        "source_components": (rows, 3),
        "log_prior_correction": (rows,),
        "valid": (rows,),
    }
    actual_shapes = {
        "theta": theta.shape,
        "prediction": prediction.shape,
        "radius": radius_grid.shape,
        "maximum_mass": maximum_mass.shape,
        "log_astrophysical": log_astrophysical.shape,
        "components": components.shape,
        "source_components": exact_nicer_matrix.shape,
        "log_prior_correction": log_prior_correction.shape,
        "valid": valid.shape,
    }
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            f"cache shapes differ: actual={actual_shapes}, expected={expected_shapes}"
        )
    valid &= (
        np.isfinite(theta).all(axis=1)
        & np.isfinite(prediction).all(axis=1)
        & np.isfinite(maximum_mass)
        & np.isfinite(log_astrophysical)
        & np.isfinite(components).all(axis=1)
        & np.isfinite(exact_nicer_matrix).all(axis=1)
        & np.isfinite(log_prior_correction)
        & np.isfinite(radius_grid).any(axis=1)
    )
    if not valid.any():
        raise RuntimeError("no target-valid rows remain")
    source_sum_error = float(
        np.max(
            np.abs(
                exact_nicer_matrix[valid].sum(axis=1)
                - components[valid, 1]
            )
        )
    )
    if source_sum_error > 1.0e-10:
        raise RuntimeError(
            f"per-source NICER sum gate failed: {source_sum_error:.3e}"
        )
    fixed_without_nicer = log_astrophysical - exact_nicer_matrix.sum(axis=1)

    paths = validate_observational_data(arguments.data_root)
    problem = A1Problem.from_data_root(
        arguments.data_root,
        workers=1,
        verify_data=True,
        model="ddb-hyperonic",
    )
    interpolators = dict(zip(SOURCES, problem.target.nicer_interpolators))
    rng = np.random.default_rng(arguments.seed)
    index_rng = np.random.default_rng(777 + arguments.seed)
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

    covered = mass_grid >= 1.0
    shifted_nicer = build_shifted_sources(
        interpolators,
        radius_grid[:, covered],
        mass_grid[covered],
        maximum_mass,
        centres,
        device=arguments.device,
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
                "scenario generator exceeded its rejection limit; "
                "the bank does not cover the declared amortization domain"
            )
        identity_case = len(scenario_cloud) < arguments.identity_scenarios
        if identity_case:
            transforms = {name: IDENTITY for name in SOURCES}
            mode = 2
        else:
            transforms, mode = draw_structured()
        log_weight = log_prior_correction + fixed_without_nicer
        for source_index, name in enumerate(SOURCES):
            transform = transforms[name]
            log_weight = log_weight + (
                exact_nicer_matrix[:, source_index]
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
        chosen = index_rng.choice(rows, posterior_rows, p=weight)
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
        scenario_transform.append([transforms[name] for name in SOURCES])
        scenario_ess.append(effective_sample_size)
        scenario_mode.append(mode)
        if len(scenario_cloud) % 250 == 0:
            print(
                f"[hyperonic-parity-scenarios] {len(scenario_cloud)}/"
                f"{arguments.scenarios} median ESS={np.median(scenario_ess):.0f} "
                f"rejected={rejected}",
                flush=True,
            )

    lengths = np.asarray([len(value) for value in scenario_theta], dtype=np.int64)
    metadata = {
        "schema_version": 1,
        "method": "hyperonic nucleonic-parity A-NET scenario bank",
        "cache_records": cache_records,
        "counting_correction": correction_record,
        "cache_rows": rows,
        "target_valid_rows": int(valid.sum()),
        "source_sum_max_abs_error": source_sum_error,
        "prior_low": prior_low.tolist(),
        "prior_high": prior_high.tolist(),
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
        },
        "centres": {name: list(centres[name]) for name in SOURCES},
        "slot_standard_deviation": {
            name: list(slot_standard_deviation[name]) for name in SOURCES
        },
        "amortized_inputs": ["nuclear_observation", "three_NICER_clouds"],
        "fixed_inputs": ["GW170817", "pQCD", "maximum_mass"],
        "held_out_sources_absent": ["J0614", "J1231", "J1614"],
        "reference_present": False,
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            gtheta=np.concatenate(scenario_prediction, axis=0),
            clouds=np.asarray(scenario_cloud),
            transforms=np.asarray(scenario_transform),
            ess=np.asarray(scenario_ess),
            mode=np.asarray(scenario_mode),
            ntr=np.int64(arguments.scenarios - arguments.validation_scenarios),
            nreal=np.int64(arguments.identity_scenarios),
            theta=np.concatenate(scenario_theta, axis=0),
            theta_len=lengths,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            reference_present=np.bool_(False),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
    temporary.replace(arguments.output)
    report = {
        "status": "SMOKE_COMPLETED" if arguments.smoke else "PASS",
        "stage": "hyperonic nucleonic-parity A-NET scenario generation",
        "scenarios": arguments.scenarios,
        "training_scenarios": arguments.scenarios - arguments.validation_scenarios,
        "posterior_rows": int(lengths.sum()),
        "minimum_ess": float(np.min(scenario_ess)),
        "median_ess": float(np.median(scenario_ess)),
        "rejected_candidates": rejected,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "metadata": metadata,
        "wall_seconds": time.time() - started,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
