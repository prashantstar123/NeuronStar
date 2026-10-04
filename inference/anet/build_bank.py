#!/usr/bin/env python3
"""Build the common nucleonic physics bank used by independent methods.

The production recipe mirrors the paper campaign:

* 200,000 uniform-prior rows with seed 1234;
* 460,000 additional uniform-prior rows with seed 2026;
* exact stable-branch R(M), Lambda(M), pQCD and shifted-GW grids;
* exact identity NICER/GW likelihood values from the shared corrected target;
* observation-box and radius-corner enrichment with an exact counting
  correction back to the uniform prior.

The bank contains simulator and exact-likelihood products, not a neural
checkpoint or posterior.  A-NET and TSNPE may independently consume the
appropriate rows.  Every expensive phase is sharded and resumable.  A final
output is accepted only after its arrays, row counts and scientific ESS gate
are checked.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eos import get_eos
from likelihoods.gw170817 import load_kde
from likelihoods.gw170817 import log_likelihood as gw_log_likelihood
from likelihoods.nicer import log_likelihood_one as nicer_log_likelihood_one
from likelihoods.pqcd import log_likelihood as pqcd_log_likelihood
from tov import solve_stable_branch
from workflows import A1Problem
from workflows.data_gate import sha256, validate_observational_data


MASS_GRID = np.linspace(0.5, 2.6, 200)
GAMMA_GRID = np.linspace(-0.6, 0.6, 9)
PQCD_DENSITY_GRID = np.asarray([1.0, 1.1, 1.2, 1.3, 1.4])
GW_MASS_RATIO = np.linspace(0.7, 1.0, 20)
BANK_KEYS = ("theta", "X", "Rg", "Lg", "MM", "PQG", "GWG", "R14")
EXACT_KEYS = (
    "exact_j0030",
    "exact_j0740",
    "exact_j0437",
    "exact_gw",
    "Rmm",
    "Mmax_fresh",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    return sha256(path)


def nuclear_predictions(theta: np.ndarray, batch_size: int) -> np.ndarray:
    plugin = get_eos("ddb")
    rho_column = plugin.parameter_names.index("rho0")
    output = np.empty((len(theta), 7), dtype=np.float64)
    output[:, 0] = theta[:, rho_column]
    for start in range(0, len(theta), batch_size):
        stop = min(start + batch_size, len(theta))
        local = theta[start:stop]
        output[start:stop, 1:] = plugin.nuclear_observables_batch(local)
    return output


def stellar_row(energy, pressure, density, mass_grid, rho_grid):
    try:
        branch = solve_stable_branch(energy, pressure)
    except (ValueError, FloatingPointError, RuntimeError):
        return None
    radius = np.full(len(mass_grid), np.nan, dtype=np.float64)
    tidal = np.full(len(mass_grid), np.nan, dtype=np.float64)
    covered = (
        (mass_grid >= branch.mass.min())
        & (mass_grid <= branch.mass.max())
    )
    if covered.any():
        radius[covered] = np.interp(
            mass_grid[covered], branch.mass, branch.radius
        )
        tidal[covered] = np.exp(
            np.interp(
                mass_grid[covered],
                branch.mass,
                np.log(np.clip(branch.tidal_lambda, 1e-300, None)),
            )
        )
    pqcd = np.asarray(
        [
            pqcd_log_likelihood(density, energy, pressure, rho_eval=value)
            for value in rho_grid
        ],
        dtype=np.float64,
    )
    radius_14 = (
        float(np.interp(1.4, mass_grid, radius))
        if branch.maximum_mass >= 1.4
        else np.nan
    )
    return radius, tidal, branch.maximum_mass, pqcd, radius_14


def shifted_gw_grid(
    tidal_grid: np.ndarray,
    maximum_mass: np.ndarray,
    kernel,
    chirp_mass: float,
) -> np.ndarray:
    """Vectorized paper GW grid for one forward shard."""

    count = len(tidal_grid)
    mass_ratio = GW_MASS_RATIO
    m1 = chirp_mass * (1.0 + mass_ratio) ** 0.2 / mass_ratio**0.6
    m2 = mass_ratio * m1
    finite = np.isfinite(tidal_grid)
    first = finite.argmax(axis=1)
    minimum_mass = np.where(finite.any(axis=1), MASS_GRID[first], np.inf)
    log_tidal = np.log(np.where(finite, tidal_grid, 1.0))

    def column(query_mass):
        index = np.clip(
            np.searchsorted(MASS_GRID, query_mass) - 1,
            0,
            len(MASS_GRID) - 2,
        )
        fraction = (query_mass - MASS_GRID[index]) / (
            MASS_GRID[index + 1] - MASS_GRID[index]
        )
        value = (
            log_tidal[:, index] * (1.0 - fraction)
            + log_tidal[:, index + 1] * fraction
        )
        valid = finite[:, index] & finite[:, index + 1]
        return value, valid

    columns1 = [column(value) for value in m1]
    columns2 = [column(value) for value in m2]
    lambda1 = np.exp(np.stack([value for value, _ in columns1], axis=1))
    lambda2 = np.exp(np.stack([value for value, _ in columns2], axis=1))
    bracket = np.stack([valid for _, valid in columns1], axis=1) & np.stack(
        [valid for _, valid in columns2], axis=1
    )
    admissible = (
        (m1[None, :] <= maximum_mass[:, None])
        & (m2[None, :] <= maximum_mass[:, None])
        & (m1[None, :] >= minimum_mass[:, None])
        & (m2[None, :] >= minimum_mass[:, None])
        & bracket
    )
    spacing = np.diff(mass_ratio)
    output = np.full((count, len(GAMMA_GRID)), -1e30, dtype=np.float64)
    for column_index, gamma in enumerate(GAMMA_GRID):
        scale = np.exp(-gamma)
        points = np.vstack(
            [
                np.full(count * len(mass_ratio), chirp_mass),
                np.tile(mass_ratio, count),
                (lambda1 * scale).ravel(),
                (lambda2 * scale).ravel(),
            ]
        )
        density = kernel(points).reshape(count, len(mass_ratio))
        adjacent = admissible[:, :-1] & admissible[:, 1:]
        trapezoid = 0.5 * (density[:, :-1] + density[:, 1:]) * spacing[None, :]
        output[:, column_index] = np.log(
            np.sum(np.where(adjacent, trapezoid, 0.0), axis=1) + 1e-300
        )
    return output


def forward_content(
    theta: np.ndarray,
    kernel,
    chirp_mass: float,
    workers: int,
    forward_chunk: int,
    nuclear_batch: int,
) -> dict[str, np.ndarray]:
    plugin = get_eos("ddb")
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in BANK_KEYS}
    for start in range(0, len(theta), forward_chunk):
        stop = min(start + forward_chunk, len(theta))
        local = np.asarray(theta[start:stop], dtype=np.float64)
        prediction = nuclear_predictions(local, nuclear_batch)
        density, energy, pressure = plugin.core_eos_batch(local)
        rows = Parallel(n_jobs=min(workers, len(local)), batch_size=8)(
            delayed(stellar_row)(
                energy[index],
                pressure[index],
                density[index],
                MASS_GRID,
                PQCD_DENSITY_GRID,
            )
            for index in range(len(local))
        )
        radius = np.full((len(local), len(MASS_GRID)), np.nan)
        tidal = np.full((len(local), len(MASS_GRID)), np.nan)
        maximum_mass = np.full(len(local), np.nan)
        pqcd = np.zeros((len(local), len(PQCD_DENSITY_GRID)))
        radius_14 = np.full(len(local), np.nan)
        for index, result in enumerate(rows):
            if result is None:
                continue
            radius[index], tidal[index], maximum_mass[index], pqcd[index], radius_14[index] = result
        gw = shifted_gw_grid(tidal, maximum_mass, kernel, chirp_mass)
        for key, value in (
            ("theta", local),
            ("X", prediction),
            ("Rg", radius),
            ("Lg", tidal),
            ("MM", maximum_mass),
            ("PQG", pqcd),
            ("GWG", gw),
            ("R14", radius_14),
        ):
            pieces[key].append(value)
    return {key: np.concatenate(value, axis=0) for key, value in pieces.items()}


def exact_row(density, energy, pressure, nicer, gw_kernel, chirp_mass):
    try:
        branch = solve_stable_branch(energy, pressure)
    except (ValueError, FloatingPointError, RuntimeError):
        return np.full(6, np.nan, dtype=np.float64)
    values = [
        nicer_log_likelihood_one(
            branch.mass, branch.radius, branch.maximum_mass, interpolator
        )
        for interpolator in nicer
    ]
    values.extend(
        [
            gw_log_likelihood(
                branch.mass,
                branch.tidal_lambda,
                branch.maximum_mass,
                gw_kernel,
                chirp_mass,
            ),
            float(branch.radius[-1]),
            branch.maximum_mass,
        ]
    )
    return np.asarray(values, dtype=np.float64)


def exact_content(
    theta: np.ndarray,
    problem: A1Problem,
    workers: int,
    forward_chunk: int,
) -> dict[str, np.ndarray]:
    plugin = get_eos("ddb")
    values = []
    for start in range(0, len(theta), forward_chunk):
        stop = min(start + forward_chunk, len(theta))
        local = np.asarray(theta[start:stop], dtype=np.float64)
        density, energy, pressure = plugin.core_eos_batch(local)
        result = Parallel(n_jobs=min(workers, len(local)), batch_size=8)(
            delayed(exact_row)(
                density[index],
                energy[index],
                pressure[index],
                problem.target.nicer_interpolators,
                problem.target.gw_kernel,
                problem.target.gw_chirp_mass,
            )
            for index in range(len(local))
        )
        values.append(np.stack(result, axis=0))
    matrix = np.concatenate(values, axis=0)
    return {key: matrix[:, index] for index, key in enumerate(EXACT_KEYS)}


def validate_bank_arrays(arrays: dict[str, np.ndarray], expected_rows: int) -> None:
    if set(BANK_KEYS) - set(arrays):
        raise RuntimeError("bank shard is incomplete")
    shapes = {
        "theta": (expected_rows, 7),
        "X": (expected_rows, 7),
        "Rg": (expected_rows, len(MASS_GRID)),
        "Lg": (expected_rows, len(MASS_GRID)),
        "MM": (expected_rows,),
        "PQG": (expected_rows, len(PQCD_DENSITY_GRID)),
        "GWG": (expected_rows, len(GAMMA_GRID)),
        "R14": (expected_rows,),
    }
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise RuntimeError(
                f"bank array {key} has shape {arrays[key].shape}, expected {shape}"
            )


def load_npz_dict(path: Path, keys) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in keys}


def concatenate_archives(paths: list[Path], keys) -> dict[str, np.ndarray]:
    accumulated = {key: [] for key in keys}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            for key in keys:
                accumulated[key].append(np.asarray(archive[key]))
    return {
        key: np.concatenate(values, axis=0)
        for key, values in accumulated.items()
    }


def base_stage(arguments, kernel, chirp_mass) -> Path:
    output = arguments.output_dir / "base_bank.npz"
    report_path = arguments.output_dir / "base_bank.json"
    if output.exists() and report_path.exists() and not arguments.force:
        report = json.loads(report_path.read_text())
        if report.get("output_sha256") == file_hash(output):
            return output
        raise RuntimeError("base-bank receipt exists but its hash is invalid")
    if output.exists() and not arguments.force:
        raise FileExistsError("base bank exists without a valid receipt")

    parts_dir = arguments.output_dir / "base_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    streams = (
        ("primary", arguments.base_primary, 1234),
        ("growth", arguments.base_growth, 2026),
    )
    paths = []
    started = time.time()
    for stream_name, rows, seed in streams:
        rng = np.random.default_rng(seed)
        for part, start in enumerate(range(0, rows, arguments.base_part_size)):
            count = min(arguments.base_part_size, rows - start)
            theta = get_eos("ddb").prior_transform(rng.random((count, 7)))
            path = parts_dir / f"{stream_name}_{part:04d}.npz"
            paths.append(path)
            if path.exists() and not arguments.force:
                with np.load(path, allow_pickle=False) as archive:
                    metadata_ok = (
                        str(archive["stream"].item()) == stream_name
                        and int(archive["seed"]) == seed
                        and int(archive["start"]) == start
                        and int(archive["rows"]) == count
                        and np.array_equal(archive["theta"], theta)
                    )
                    arrays = {key: np.asarray(archive[key]) for key in BANK_KEYS}
                if metadata_ok:
                    validate_bank_arrays(arrays, count)
                    continue
                raise RuntimeError(f"stale or corrupt base-bank shard: {path}")
            arrays = forward_content(
                theta,
                kernel,
                chirp_mass,
                arguments.workers,
                arguments.forward_chunk,
                arguments.nuclear_batch,
            )
            validate_bank_arrays(arrays, count)
            np.savez(
                path,
                **arrays,
                stream=np.asarray(stream_name),
                seed=np.int64(seed),
                start=np.int64(start),
                rows=np.int64(count),
            )
            print(
                f"[anet-bank] {stream_name} {start + count}/{rows}", flush=True
            )
    merged = concatenate_archives(paths, BANK_KEYS)
    expected = arguments.base_primary + arguments.base_growth
    validate_bank_arrays(merged, expected)
    np.savez(
        output,
        **merged,
        MG=MASS_GRID,
        GLAM=GAMMA_GRID,
        RHO=PQCD_DENSITY_GRID,
        Mc_obs=np.float64(chirp_mass),
        base_primary=np.int64(arguments.base_primary),
        base_growth=np.int64(arguments.base_growth),
        recipe=np.asarray("paper_two_uniform_prior_streams"),
    )
    report = {
        "status": "SMOKE_COMPLETED" if arguments.smoke else "PASS",
        "stage": "A-NET base bank",
        "rows": expected,
        "valid_stellar_rows": int(np.isfinite(merged["MM"]).sum()),
        "streams": [
            {"name": name, "rows": rows, "seed": seed}
            for name, rows, seed in streams
        ],
        "output": str(output),
        "output_sha256": file_hash(output),
        "wall_seconds": time.time() - started,
    }
    write_json(report_path, report)
    return output


def exact_stage(arguments, base: Path, problem: A1Problem) -> Path:
    output = arguments.output_dir / "base_exact.npz"
    report_path = arguments.output_dir / "base_exact.json"
    if output.exists() and report_path.exists() and not arguments.force:
        report = json.loads(report_path.read_text())
        if report.get("output_sha256") == file_hash(output):
            return output
        raise RuntimeError("base exact receipt exists but its hash is invalid")
    if output.exists() and not arguments.force:
        raise FileExistsError("base exact table exists without a valid receipt")
    with np.load(base, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64)
    parts_dir = arguments.output_dir / "exact_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    started = time.time()
    for part, start in enumerate(range(0, len(theta), arguments.exact_part_size)):
        stop = min(start + arguments.exact_part_size, len(theta))
        local = theta[start:stop]
        path = parts_dir / f"exact_{part:04d}.npz"
        paths.append(path)
        if path.exists() and not arguments.force:
            with np.load(path, allow_pickle=False) as archive:
                if (
                    int(archive["start"]) == start
                    and int(archive["rows"]) == len(local)
                    and np.array_equal(archive["theta_first"], local[0])
                    and all(archive[key].shape == (len(local),) for key in EXACT_KEYS)
                ):
                    continue
            raise RuntimeError(f"stale or corrupt exact shard: {path}")
        values = exact_content(
            local, problem, arguments.workers, arguments.forward_chunk
        )
        np.savez(
            path,
            **values,
            start=np.int64(start),
            rows=np.int64(len(local)),
            theta_first=local[0],
        )
        print(f"[anet-exact] {stop}/{len(theta)}", flush=True)
    merged = concatenate_archives(paths, EXACT_KEYS)
    if any(value.shape != (len(theta),) for value in merged.values()):
        raise RuntimeError("merged exact table has inconsistent rows")
    np.savez(output, **merged)
    report = {
        "status": "SMOKE_COMPLETED" if arguments.smoke else "PASS",
        "stage": "A-NET exact identity likelihood table",
        "rows": int(len(theta)),
        "finite_rows": int(np.isfinite(merged["exact_gw"]).sum()),
        "base_bank_sha256": file_hash(base),
        "output": str(output),
        "output_sha256": file_hash(output),
        "wall_seconds": time.time() - started,
    }
    write_json(report_path, report)
    return output


def ridge_and_support(base_arrays, observation, sigma, cbox):
    x = base_arrays["X"]
    r14 = base_arrays["R14"]
    maximum_mass = base_arrays["MM"]
    finite = np.isfinite(r14) & np.isfinite(maximum_mass)
    design = np.column_stack([np.ones(int(finite.sum())), x[finite]])
    beta = np.linalg.solve(
        design.T @ design + 1e-6 * np.eye(8), design.T @ r14[finite]
    )

    def predict(values):
        return np.column_stack([np.ones(len(values)), values]) @ beta

    predicted = predict(x)
    window_c = (9.4, 11.2, 1.9)
    window_d = (14.8, 18.0, 1.9)

    def radius_window(radius, mass, window):
        return (
            np.isfinite(radius)
            & np.isfinite(mass)
            & (radius >= window[0])
            & (radius <= window[1])
            & (mass >= window[2])
        )

    mask_c = radius_window(r14, maximum_mass, window_c)
    mask_d = radius_window(r14, maximum_mass, window_d)
    if not mask_c.any() or not mask_d.any():
        raise RuntimeError("base bank has no rows in an enrichment radius window")
    pred_c = tuple(np.percentile(predicted[mask_c], [1.0, 99.0]))
    pred_d = tuple(np.percentile(predicted[mask_d], [1.0, 99.0]))

    def in_b(values):
        return np.all(np.abs(values - observation) <= cbox * sigma, axis=1)

    def in_corner(values, radius, mass, interval, window):
        value = predict(values)
        return (
            (value >= interval[0])
            & (value <= interval[1])
            & radius_window(radius, mass, window)
        )

    return beta, window_c, window_d, pred_c, pred_d, in_b, in_corner


def enrichment_stage(
    arguments,
    base_path: Path,
    exact_path: Path,
    problem: A1Problem,
    kernel,
    chirp_mass: float,
) -> tuple[Path, Path]:
    bank_output = arguments.output_dir / "final_bank.npz"
    exact_output = arguments.output_dir / "final_exact.npz"
    report_path = arguments.output_dir / "final_bank.json"
    if bank_output.exists() and exact_output.exists() and report_path.exists() and not arguments.force:
        report = json.loads(report_path.read_text())
        if (
            report.get("bank_sha256") == file_hash(bank_output)
            and report.get("exact_sha256") == file_hash(exact_output)
        ):
            return bank_output, exact_output
        raise RuntimeError("final bank receipt exists but an output hash is invalid")
    if (bank_output.exists() or exact_output.exists()) and not arguments.force:
        raise FileExistsError("partial final bank exists without a valid receipt")

    base = load_npz_dict(base_path, BANK_KEYS)
    base_exact = load_npz_dict(exact_path, EXACT_KEYS)
    observation = np.asarray(problem.target.nuclear_observation, dtype=np.float64)
    sigma = np.asarray(problem.target.nuclear_sigma, dtype=np.float64)
    (
        beta,
        window_c,
        window_d,
        pred_c,
        pred_d,
        in_b,
        in_corner,
    ) = ridge_and_support(base, observation, sigma, arguments.cbox)
    prior_low = get_eos("ddb").prior_low
    prior_high = get_eos("ddb").prior_high
    base_rows = len(base["theta"])
    parts_dir = arguments.output_dir / "enrichment_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # Stream B: cheap nuclear-observation screening.
    b_screen_paths = []
    b_blocks = math.ceil(arguments.mtot_b / arguments.block_b)
    for block in range(b_blocks):
        count = min(arguments.block_b, arguments.mtot_b - block * arguments.block_b)
        path = parts_dir / f"B_screen_{block:04d}.npz"
        b_screen_paths.append(path)
        rng = np.random.default_rng(np.random.SeedSequence([arguments.enrich_seed, 1, block]))
        theta = prior_low + rng.random((count, 7)) * (prior_high - prior_low)
        if path.exists() and not arguments.force:
            with np.load(path, allow_pickle=False) as archive:
                if int(archive["nprop"]) == count:
                    continue
            raise RuntimeError(f"stale stream-B screen shard: {path}")
        prediction = nuclear_predictions(theta, arguments.nuclear_batch)
        accepted = in_b(prediction)
        np.savez(
            path,
            theta=theta[accepted],
            X=prediction[accepted],
            nprop=np.int64(count),
        )
        print(
            f"[anet-enrich:B-screen] {block + 1}/{b_blocks}: "
            f"{int(accepted.sum())}/{count}",
            flush=True,
        )
    screened = concatenate_archives(b_screen_paths, ("theta", "X"))
    mtot_b = sum(
        int(np.load(path, allow_pickle=False)["nprop"]) for path in b_screen_paths
    )

    b_content_paths = []
    for part, start in enumerate(range(0, len(screened["theta"]), arguments.enrich_content_chunk)):
        stop = min(start + arguments.enrich_content_chunk, len(screened["theta"]))
        path = parts_dir / f"B_content_{part:04d}.npz"
        b_content_paths.append(path)
        theta = screened["theta"][start:stop]
        if path.exists() and not arguments.force:
            with np.load(path, allow_pickle=False) as archive:
                if len(archive["theta"]) == len(theta) and (
                    len(theta) == 0 or np.array_equal(archive["theta"][0], theta[0])
                ):
                    continue
            raise RuntimeError(f"stale stream-B content shard: {path}")
        content = forward_content(
            theta,
            kernel,
            chirp_mass,
            arguments.workers,
            arguments.forward_chunk,
            arguments.nuclear_batch,
        )
        # Preserve the already screened prediction byte-for-byte.
        content["X"] = screened["X"][start:stop]
        np.savez(path, **content, nprop=np.int64(0))
        print(
            f"[anet-enrich:B-content] {stop}/{len(screened['theta'])}", flush=True
        )

    def corner_stream(stream_id, name, multiplier, interval, window):
        blocks = math.ceil(max(multiplier - 1.0, 0.0) * base_rows / arguments.block_cd)
        paths = []
        for block in range(blocks):
            path = parts_dir / f"{name}_{block:04d}.npz"
            paths.append(path)
            if path.exists() and not arguments.force:
                with np.load(path, allow_pickle=False) as archive:
                    if int(archive["nprop"]) == arguments.block_cd:
                        continue
                raise RuntimeError(f"stale stream-{name} shard: {path}")
            rng = np.random.default_rng(
                np.random.SeedSequence([arguments.enrich_seed, stream_id, block])
            )
            theta = prior_low + rng.random((arguments.block_cd, 7)) * (
                prior_high - prior_low
            )
            prediction = nuclear_predictions(theta, arguments.nuclear_batch)
            predicted = np.column_stack([np.ones(len(prediction)), prediction]) @ beta
            prescreen = (predicted >= interval[0]) & (predicted <= interval[1])
            content = forward_content(
                theta[prescreen],
                kernel,
                chirp_mass,
                arguments.workers,
                arguments.forward_chunk,
                arguments.nuclear_batch,
            )
            content["X"] = prediction[prescreen]
            keep = in_corner(
                content["X"], content["R14"], content["MM"], interval, window
            )
            content = {key: value[keep] for key, value in content.items()}
            np.savez(path, **content, nprop=np.int64(arguments.block_cd))
            print(
                f"[anet-enrich:{name}] {block + 1}/{blocks}: "
                f"{int(prescreen.sum())} prescreen -> {int(keep.sum())}",
                flush=True,
            )
        if not paths:
            return {key: np.empty((0,) + base[key].shape[1:], dtype=base[key].dtype) for key in BANK_KEYS}, 0
        content = concatenate_archives(paths, BANK_KEYS)
        total = sum(
            int(np.load(path, allow_pickle=False)["nprop"]) for path in paths
        )
        return content, total

    if b_content_paths:
        content_b = concatenate_archives(b_content_paths, BANK_KEYS)
    else:
        content_b = {
            key: np.empty((0,) + base[key].shape[1:], dtype=base[key].dtype)
            for key in BANK_KEYS
        }
    content_c, mtot_c = corner_stream(
        2, "C", arguments.mult_c, pred_c, window_c
    )
    content_d, mtot_d = corner_stream(
        3, "D", arguments.mult_d, pred_d, window_d
    )
    new = {
        key: np.concatenate(
            [content_b[key], content_c[key], content_d[key]], axis=0
        )
        for key in BANK_KEYS
    }

    # Exact identity values for every appended row, also sharded/resumable.
    exact_new_paths = []
    for part, start in enumerate(range(0, len(new["theta"]), arguments.exact_part_size)):
        stop = min(start + arguments.exact_part_size, len(new["theta"]))
        path = parts_dir / f"exact_new_{part:04d}.npz"
        exact_new_paths.append(path)
        theta = new["theta"][start:stop]
        if path.exists() and not arguments.force:
            with np.load(path, allow_pickle=False) as archive:
                if len(archive[EXACT_KEYS[0]]) == len(theta) and (
                    len(theta) == 0 or np.array_equal(archive["theta_first"], theta[0])
                ):
                    continue
            raise RuntimeError(f"stale enriched exact shard: {path}")
        values = exact_content(
            theta, problem, arguments.workers, arguments.forward_chunk
        )
        np.savez(path, **values, theta_first=theta[0])
        print(f"[anet-enrich:exact] {stop}/{len(new['theta'])}", flush=True)
    exact_new = (
        concatenate_archives(exact_new_paths, EXACT_KEYS)
        if exact_new_paths
        else {key: np.empty(0, dtype=np.float64) for key in EXACT_KEYS}
    )

    final = {
        key: np.concatenate([base[key], new[key]], axis=0) for key in BANK_KEYS
    }
    final_exact = {
        key: np.concatenate([base_exact[key], exact_new[key]], axis=0)
        for key in EXACT_KEYS
    }
    support_b = in_b(final["X"])
    support_c = in_corner(
        final["X"], final["R14"], final["MM"], pred_c, window_c
    )
    support_d = in_corner(
        final["X"], final["R14"], final["MM"], pred_d, window_d
    )
    denominator = (
        base_rows
        + np.float64(mtot_b) * support_b
        + np.float64(mtot_c) * support_c
        + np.float64(mtot_d) * support_d
    )
    log_prior_correction = np.log(base_rows) - np.log(denominator)
    uniform_log_prior = -float(
        np.sum(np.log(get_eos("ddb").prior_high - get_eos("ddb").prior_low))
    )
    # The proposal normalization constant cancels in EvidenceCache's explicit
    # denominator.  This encoding therefore lets the exact counting-enriched
    # bank be consumed directly as an importance proposal.
    log_proposal = uniform_log_prior - log_prior_correction
    np.savez(
        bank_output,
        **final,
        MG=MASS_GRID,
        GLAM=GAMMA_GRID,
        RHO=PQCD_DENSITY_GRID,
        Mc_obs=np.float64(chirp_mass),
        logw_prior=log_prior_correction,
        logq=log_proposal,
        in_SB=support_b,
        in_SC=support_c,
        in_SD=support_d,
        enrich_seed=np.int64(arguments.enrich_seed),
        N0=np.int64(base_rows),
        MtotB=np.int64(mtot_b),
        MtotC=np.int64(mtot_c),
        MtotD=np.int64(mtot_d),
        cbox=np.float64(arguments.cbox),
        beta_r14=beta,
        predC_int=np.asarray(pred_c),
        predD_int=np.asarray(pred_d),
        winC=np.asarray(window_c),
        winD=np.asarray(window_d),
        recipe=np.asarray("paper_exact_counting_enrichment"),
    )
    np.savez(exact_output, **final_exact)

    valid = (
        np.isfinite(final["MM"])
        & np.isfinite(final_exact["exact_gw"])
        & np.isfinite(final_exact["exact_j0030"])
        & np.isfinite(final_exact["exact_j0740"])
        & np.isfinite(final_exact["exact_j0437"])
    )
    standardized = (final["X"] - observation) / sigma
    log_kernel = -0.5 * np.sum(standardized**2, axis=1)
    finite_log_kernel = log_kernel[valid]
    shifted = np.full(len(final["theta"]), 0.0)
    shifted[valid] = np.exp(finite_log_kernel - np.max(finite_log_kernel))
    weight = shifted * np.exp(log_prior_correction)
    conditional_ess = float(weight.sum() ** 2 / np.sum(weight**2))
    passed = conditional_ess >= 2000.0
    if not arguments.smoke and not passed:
        raise RuntimeError(
            f"A-NET conditional bank ESS gate failed: {conditional_ess:.1f} < 2000"
        )
    report = {
        "status": "SMOKE_COMPLETED" if arguments.smoke else "PASS",
        "stage": "A-NET final enriched bank",
        "rows": int(len(final["theta"])),
        "base_rows": base_rows,
        "new_rows": int(len(new["theta"])),
        "stream_rows": {
            "B": int(len(content_b["theta"])),
            "C": int(len(content_c["theta"])),
            "D": int(len(content_d["theta"])),
        },
        "screened_proposals": {
            "B": int(mtot_b),
            "C": int(mtot_c),
            "D": int(mtot_d),
        },
        "conditional_ess_at_A1": conditional_ess,
        "conditional_ess_gate": 2000.0,
        "conditional_ess_pass": passed,
        "base_bank_sha256": file_hash(base_path),
        "base_exact_sha256": file_hash(exact_path),
        "bank": str(bank_output),
        "bank_sha256": file_hash(bank_output),
        "exact": str(exact_output),
        "exact_sha256": file_hash(exact_output),
        "wall_seconds": time.time() - started,
    }
    write_json(report_path, report)
    return bank_output, exact_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("all", "base", "exact", "enrich"), default="all"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--base-primary", type=int, default=200_000)
    parser.add_argument("--base-growth", type=int, default=460_000)
    parser.add_argument("--base-part-size", type=int, default=20_000)
    parser.add_argument("--forward-chunk", type=int, default=4_000)
    parser.add_argument("--nuclear-batch", type=int, default=20_000)
    parser.add_argument("--exact-part-size", type=int, default=20_000)
    parser.add_argument("--enrich-seed", type=int, default=20_260_729)
    parser.add_argument("--mtot-b", type=int, default=160_000_000)
    parser.add_argument("--cbox", type=float, default=3.0)
    parser.add_argument("--mult-c", type=float, default=5.0)
    parser.add_argument("--mult-d", type=float, default=3.0)
    parser.add_argument("--block-b", type=int, default=2_000_000)
    parser.add_argument("--block-cd", type=int, default=60_000)
    parser.add_argument("--enrich-content-chunk", type=int, default=20_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    counts = (
        arguments.workers,
        arguments.base_primary,
        arguments.base_growth,
        arguments.base_part_size,
        arguments.forward_chunk,
        arguments.nuclear_batch,
        arguments.exact_part_size,
        arguments.mtot_b,
        arguments.block_b,
        arguments.block_cd,
        arguments.enrich_content_chunk,
    )
    if any(value < 1 for value in counts):
        raise ValueError("all bank counts and worker settings must be positive")
    if arguments.cbox <= 0 or arguments.mult_c < 1 or arguments.mult_d < 1:
        raise ValueError("invalid A-NET enrichment settings")
    if arguments.smoke:
        arguments.base_primary = 64
        arguments.base_growth = 64
        arguments.base_part_size = 64
        arguments.forward_chunk = 64
        arguments.nuclear_batch = 64
        arguments.exact_part_size = 64
        arguments.mtot_b = 128
        arguments.block_b = 64
        arguments.cbox = 100.0
        arguments.mult_c = 1.0
        arguments.mult_d = 1.0
        arguments.enrich_content_chunk = 64

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    paths = validate_observational_data(arguments.data_root)
    kernel, chirp_mass = load_kde(
        paths["GW170817_GWTC-1.hdf5"], subsample=3500, seed=0
    )
    problem = A1Problem.from_data_root(
        arguments.data_root, workers=arguments.workers
    )
    base_path = arguments.output_dir / "base_bank.npz"
    exact_path = arguments.output_dir / "base_exact.npz"
    if arguments.phase in ("all", "base"):
        base_path = base_stage(arguments, kernel, chirp_mass)
    if arguments.phase == "base":
        return 0
    if not base_path.exists():
        raise FileNotFoundError("base bank is required before this phase")
    if arguments.phase in ("all", "exact"):
        exact_path = exact_stage(arguments, base_path, problem)
    if arguments.phase == "exact":
        return 0
    if not exact_path.exists():
        raise FileNotFoundError("base exact table is required before enrichment")
    if arguments.phase in ("all", "enrich"):
        enrichment_stage(
            arguments,
            base_path,
            exact_path,
            problem,
            kernel,
            chirp_mass,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
