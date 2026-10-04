#!/usr/bin/env python3
"""Fresh A-NET-seeded direct-theta refinement for the A1 tidal posterior.

The ordinary A-NET+IS rows are adaptation data only. A full-posterior GMM and
a low-Lambda-tail GMM are fitted in prior-scaled physical parameter space.
Fresh draws from those analytic densities plus a uniform defensive component
form the final estimator; the adaptation rows are excluded. Every fresh row is
evaluated with the shared corrected target, so no neural-flow Jacobian appears
in this estimator's proposal density.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eos import get_eos
from inference.common import compute_importance_weights
from tov import solve_stable_branch
from workflows import A1Problem
from workflows.data_gate import sha256


MASS_GRID = np.linspace(0.5, 2.5, 201)


def systematic(weight: np.ndarray, size: int) -> np.ndarray:
    weight = np.asarray(weight, dtype=np.float64)
    weight = weight / weight.sum()
    cumulative = np.cumsum(weight)
    cumulative[-1] = 1.0
    position = (np.arange(size, dtype=np.float64) + 0.5) / size
    return np.searchsorted(cumulative, position, side="left")


def weighted_quantile(value, weight, probabilities=(0.05, 0.5, 0.95)):
    value = np.asarray(value, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    if not valid.any():
        return np.full(len(probabilities), np.nan)
    value, weight = value[valid], weight[valid]
    order = np.argsort(value, kind="mergesort")
    value, weight = value[order], weight[order]
    position = (np.cumsum(weight) - 0.5 * weight) / weight.sum()
    return np.interp(probabilities, position, value)


def curve_band(curves: np.ndarray, weight: np.ndarray):
    quantile = np.full((3, curves.shape[1]), np.nan)
    count = np.zeros(curves.shape[1], dtype=np.int64)
    support = np.zeros(curves.shape[1], dtype=np.float64)
    conditional_ess = np.zeros(curves.shape[1], dtype=np.float64)
    normalized = np.where(
        np.isfinite(weight) & (weight > 0),
        np.asarray(weight, dtype=np.float64),
        0.0,
    )
    total = float(normalized.sum())
    if not np.isfinite(total) or total <= 0:
        return {
            "quantile": quantile,
            "count": count,
            "support": support,
            "conditional_ess": conditional_ess,
        }
    normalized /= total
    for column in range(curves.shape[1]):
        valid = np.isfinite(curves[:, column]) & (normalized > 0)
        count[column] = int(valid.sum())
        if not valid.any():
            continue
        local = normalized[valid]
        support[column] = float(local.sum())
        local = local / local.sum()
        conditional_ess[column] = float(1.0 / np.sum(local**2))
        quantile[:, column] = weighted_quantile(curves[valid, column], local)
    return {
        "quantile": quantile,
        "count": count,
        "support": support,
        "conditional_ess": conditional_ess,
    }


def sample_gmm(rng, size, weights, means, covariances):
    component = rng.choice(len(weights), size=size, p=weights)
    output = np.empty((size, means.shape[1]), dtype=np.float64)
    for index in range(len(weights)):
        selected = component == index
        if selected.any():
            cholesky = np.linalg.cholesky(covariances[index])
            output[selected] = means[index] + rng.standard_normal(
                (int(selected.sum()), means.shape[1])
            ) @ cholesky.T
    return output


def gmm_log_density(value, weights, means, covariances):
    dimension = value.shape[1]
    component = np.empty((len(weights), len(value)), dtype=np.float64)
    constant = dimension * np.log(2.0 * np.pi)
    for index in range(len(weights)):
        cholesky = np.linalg.cholesky(covariances[index])
        whitened = solve_triangular(
            cholesky,
            (value - means[index]).T,
            lower=True,
            check_finite=False,
        )
        log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
        component[index] = np.log(weights[index]) - 0.5 * (
            constant + log_determinant + np.sum(whitened**2, axis=0)
        )
    return logsumexp(component, axis=0)


def evaluate_row(theta, prediction, density, energy, pressure, target):
    try:
        branch = solve_stable_branch(energy, pressure)
    except (ValueError, FloatingPointError, RuntimeError):
        return None
    terms = target.evaluate(theta, prediction, density, energy, pressure, branch)
    if not np.isfinite(terms.total):
        return None
    radius = np.full(len(MASS_GRID), np.nan)
    tidal = np.full(len(MASS_GRID), np.nan)
    covered = (
        (MASS_GRID >= branch.mass.min()) & (MASS_GRID <= branch.mass.max())
    )
    radius[covered] = np.interp(MASS_GRID[covered], branch.mass, branch.radius)
    tidal[covered] = np.exp(
        np.interp(
            MASS_GRID[covered],
            branch.mass,
            np.log(np.clip(branch.tidal_lambda, 1e-300, None)),
        )
    )
    covers_14 = branch.mass.min() <= 1.4 <= branch.mass.max()
    radius_14 = (
        float(np.interp(1.4, branch.mass, branch.radius))
        if covers_14
        else np.nan
    )
    lambda_14 = (
        float(
            np.exp(
                np.interp(
                    1.4,
                    branch.mass,
                    np.log(np.clip(branch.tidal_lambda, 1e-300, None)),
                )
            )
        )
        if covers_14
        else np.nan
    )
    return (
        terms.total,
        radius,
        tidal,
        branch.maximum_mass,
        float(branch.radius[-1]),
        radius_14,
        lambda_14,
    )


def evaluate_theta(theta, problem, workers, chunk):
    plugin = get_eos("ddb")
    output = {
        key: []
        for key in ("logl", "radius", "tidal", "mmax", "rmax", "r14", "lambda14")
    }
    for start in range(0, len(theta), chunk):
        stop = min(start + chunk, len(theta))
        local = np.asarray(theta[start:stop], dtype=np.float64)
        prediction = plugin.nuclear_observables_batch(local)
        density, energy, pressure = plugin.core_eos_batch(local)
        rows = Parallel(n_jobs=min(workers, len(local)), batch_size=8)(
            delayed(evaluate_row)(
                local[index],
                prediction[index],
                density[index],
                energy[index],
                pressure[index],
                problem.target,
            )
            for index in range(len(local))
        )
        logl = np.full(len(local), -np.inf)
        radius = np.full((len(local), len(MASS_GRID)), np.nan)
        tidal = np.full((len(local), len(MASS_GRID)), np.nan)
        scalar = np.full((len(local), 4), np.nan)
        for index, result in enumerate(rows):
            if result is None:
                continue
            logl[index], radius[index], tidal[index] = result[:3]
            scalar[index] = result[3:]
        for key, value in (
            ("logl", logl),
            ("radius", radius),
            ("tidal", tidal),
            ("mmax", scalar[:, 0]),
            ("rmax", scalar[:, 1]),
            ("r14", scalar[:, 2]),
            ("lambda14", scalar[:, 3]),
        ):
            output[key].append(value)
    return {key: np.concatenate(value, axis=0) for key, value in output.items()}


def evaluate_sharded(theta, problem, output_dir, workers, part_size, chunk, force):
    part_dir = output_dir / "curve_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for part, start in enumerate(range(0, len(theta), part_size)):
        stop = min(start + part_size, len(theta))
        path = part_dir / f"part_{part:04d}.npz"
        paths.append(path)
        local = theta[start:stop]
        if path.exists() and not force:
            with np.load(path, allow_pickle=False) as archive:
                if (
                    int(archive["start"]) == start
                    and len(archive["logl"]) == len(local)
                    and np.array_equal(archive["theta_first"], local[0])
                ):
                    continue
            raise RuntimeError(f"stale tidal-refinement shard: {path}")
        values = evaluate_theta(local, problem, workers, chunk)
        np.savez(path, **values, start=np.int64(start), theta_first=local[0])
        print(f"[anet-tidal] evaluated {stop}/{len(theta)}", flush=True)
    keys = ("logl", "radius", "tidal", "mmax", "rmax", "r14", "lambda14")
    accumulated = {key: [] for key in keys}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            for key in keys:
                accumulated[key].append(np.asarray(archive[key]))
    return {key: np.concatenate(value, axis=0) for key, value in accumulated.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anet-is", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rows", type=int, default=320_000)
    parser.add_argument("--part-size", type=int, default=8_000)
    parser.add_argument("--forward-chunk", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=20_260_821)
    parser.add_argument("--minimum-ess", type=float, default=15_000.0)
    parser.add_argument("--minimum-tail-ess", type=float, default=2_000.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.workers, arguments.rows, arguments.part_size, arguments.forward_chunk) < 1:
        raise ValueError("all tidal-refinement counts must be positive")
    if arguments.smoke:
        arguments.rows = min(arguments.rows, 128)
        arguments.part_size = min(arguments.part_size, 64)
        arguments.forward_chunk = min(arguments.forward_chunk, 64)

    from sklearn.mixture import GaussianMixture

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = arguments.output_dir / "anet_tidal_refinement.npz"
    report_path = arguments.output_dir / "run_report.json"
    if final_path.exists() and report_path.exists() and not arguments.force:
        report = json.loads(report_path.read_text())
        if report.get("output_sha256") == sha256(final_path):
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
            return 0
        raise RuntimeError("tidal refinement receipt does not match its output")
    started = time.time()
    problem = A1Problem.from_data_root(arguments.data_root, workers=arguments.workers)
    plugin = get_eos("ddb")
    low, high = plugin.prior_low, plugin.prior_high
    width = high - low
    centre = 0.5 * (low + high)
    with np.load(arguments.anet_is, allow_pickle=False) as archive:
        if not {"raw", "logw"} <= set(archive.files):
            raise ValueError("A-NET+IS adaptation file needs raw theta and logw")
        adaptation_theta = np.asarray(archive["raw"], dtype=np.float64)
        adaptation_log_weight = np.asarray(archive["logw"], dtype=np.float64)
    finite_adaptation = np.isfinite(adaptation_log_weight) & np.all(
        (adaptation_theta >= low) & (adaptation_theta <= high), axis=1
    )
    if finite_adaptation.sum() < (5 if arguments.smoke else 20):
        raise RuntimeError("too few finite A-NET rows for tidal adaptation")
    adaptation_theta = adaptation_theta[finite_adaptation]
    adaptation_weight = np.exp(
        adaptation_log_weight[finite_adaptation]
        - logsumexp(adaptation_log_weight[finite_adaptation])
    )

    adaptation_curve_path = arguments.output_dir / "adaptation_curves.npz"
    if adaptation_curve_path.exists() and not arguments.force:
        with np.load(adaptation_curve_path, allow_pickle=False) as archive:
            if (
                str(archive["source_sha256"].item()) == sha256(arguments.anet_is)
                and len(archive["lambda1"]) == len(adaptation_theta)
            ):
                lambda1 = np.asarray(archive["lambda1"], dtype=np.float64)
            else:
                raise RuntimeError("stale tidal adaptation curve cache")
    else:
        adaptation_curves = evaluate_theta(
            adaptation_theta, problem, arguments.workers, arguments.forward_chunk
        )
        lambda1 = adaptation_curves["tidal"][:, int(np.argmin(np.abs(MASS_GRID - 1.0)))]
        np.savez(
            adaptation_curve_path,
            lambda1=lambda1,
            source_sha256=np.asarray(sha256(arguments.anet_is)),
        )

    finite_lambda = np.isfinite(lambda1)
    if not finite_lambda.any():
        raise RuntimeError("A-NET adaptation has no finite Lambda(1.0)")
    tail_threshold = float(
        weighted_quantile(
            lambda1[finite_lambda], adaptation_weight[finite_lambda], (0.10,)
        )[0]
    )
    tail = finite_lambda & (lambda1 <= tail_threshold)
    z = (adaptation_theta - centre) / width
    rng = np.random.default_rng(arguments.seed)
    full_fit_rows = 256 if arguments.smoke else 100_000
    tail_fit_rows = 128 if arguments.smoke else 60_000
    full_index = systematic(adaptation_weight, full_fit_rows)
    tail_rows = np.flatnonzero(tail)
    tail_index = tail_rows[systematic(adaptation_weight[tail], tail_fit_rows)]
    full_components = min(2 if arguments.smoke else 6, len(np.unique(full_index)))
    tail_components = min(2 if arguments.smoke else 4, len(np.unique(tail_index)))
    if full_components < 1 or tail_components < 1:
        raise RuntimeError("tidal adaptation cannot populate its GMM components")
    full_model = GaussianMixture(
        n_components=full_components,
        covariance_type="full",
        reg_covar=1e-6,
        tol=1e-5,
        max_iter=500,
        n_init=1 if arguments.smoke else 2,
        random_state=arguments.seed,
    ).fit(z[full_index][rng.permutation(full_fit_rows)])
    tail_model = GaussianMixture(
        n_components=tail_components,
        covariance_type="full",
        reg_covar=1e-6,
        tol=1e-5,
        max_iter=500,
        n_init=1 if arguments.smoke else 2,
        random_state=arguments.seed + 1,
    ).fit(z[tail_index][rng.permutation(tail_fit_rows)])
    if not (full_model.converged_ and tail_model.converged_):
        raise RuntimeError("tidal-refinement GMM adaptation did not converge")
    full_covariance = 1.20 * full_model.covariances_ + 1e-6 * np.eye(7)[None]
    tail_covariance = 1.35 * tail_model.covariances_ + 1e-6 * np.eye(7)[None]

    allocation = np.asarray([0.65, 0.30, 0.05])
    counts = np.floor(arguments.rows * allocation).astype(int)
    counts[-1] = arguments.rows - counts[:-1].sum()
    proposal_rng = np.random.default_rng(arguments.seed + 100)
    proposal_z = np.vstack(
        [
            sample_gmm(
                proposal_rng,
                counts[0],
                full_model.weights_,
                full_model.means_,
                full_covariance,
            ),
            sample_gmm(
                proposal_rng,
                counts[1],
                tail_model.weights_,
                tail_model.means_,
                tail_covariance,
            ),
            proposal_rng.uniform(-0.5, 0.5, size=(counts[2], 7)),
        ]
    )
    proposal_theta = centre + proposal_z * width
    logq_full = gmm_log_density(
        proposal_z, full_model.weights_, full_model.means_, full_covariance
    )
    logq_tail = gmm_log_density(
        proposal_z, tail_model.weights_, tail_model.means_, tail_covariance
    )
    inside = np.all((proposal_theta >= low) & (proposal_theta <= high), axis=1)
    logq_uniform_z = np.where(inside, 0.0, -np.inf)
    logq_z = logsumexp(
        np.vstack(
            [
                np.log(allocation[0]) + logq_full,
                np.log(allocation[1]) + logq_tail,
                np.log(allocation[2]) + logq_uniform_z,
            ]
        ),
        axis=0,
    )
    logq_physical = logq_z - np.sum(np.log(width))
    proposal_path = arguments.output_dir / "direct_theta_proposal.npz"
    np.savez(
        proposal_path,
        theta=proposal_theta,
        logq=logq_physical,
        allocation=allocation,
        adaptation_sha256=np.asarray(sha256(arguments.anet_is)),
    )

    evaluated = evaluate_sharded(
        proposal_theta,
        problem,
        arguments.output_dir,
        arguments.workers,
        arguments.part_size,
        arguments.forward_chunk,
        arguments.force,
    )
    log_prior = np.full(arguments.rows, -np.inf)
    log_prior[inside] = -np.sum(np.log(width))
    importance = compute_importance_weights(
        evaluated["logl"] + log_prior, logq_physical
    )
    tidal_band = curve_band(evaluated["tidal"], importance.normalized_weight)
    radius_band = curve_band(evaluated["radius"], importance.normalized_weight)
    parity_bands = []
    for parity in (0, 1):
        mask = np.arange(arguments.rows) % 2 == parity
        parity_bands.append(
            curve_band(evaluated["tidal"][mask], importance.normalized_weight[mask])
        )
    common = (
        np.all(np.isfinite(parity_bands[0]["quantile"]), axis=0)
        & np.all(np.isfinite(parity_bands[1]["quantile"]), axis=0)
        & (MASS_GRID <= 1.1)
    )
    parity_difference = (
        float(
            np.max(
                np.abs(
                    parity_bands[0]["quantile"][[0, 2]][:, common]
                    - parity_bands[1]["quantile"][[0, 2]][:, common]
                )
                / np.maximum(
                    parity_bands[0]["quantile"][[0, 2]][:, common], 1e-300
                )
            )
        )
        if common.any()
        else float("inf")
    )

    tail_ess = {}
    minimum_tail_ess = float("inf")
    for mass in (0.5, 0.75, 1.0, 1.1):
        column = int(np.argmin(np.abs(MASS_GRID - mass)))
        threshold = weighted_quantile(
            evaluated["tidal"][:, column], importance.normalized_weight, (0.05,)
        )[0]
        selected = (
            np.isfinite(evaluated["tidal"][:, column])
            & (evaluated["tidal"][:, column] <= threshold)
        )
        local = importance.normalized_weight[selected]
        value = float(local.sum() ** 2 / np.sum(local**2))
        tail_ess[f"{mass:.2f}"] = value
        minimum_tail_ess = min(minimum_tail_ess, value)
    internal_pass = (
        importance.ess >= arguments.minimum_ess
        and minimum_tail_ess >= arguments.minimum_tail_ess
        and parity_difference <= 0.03
    )
    if not arguments.smoke and not internal_pass:
        raise RuntimeError("A-NET tidal refinement failed an internal ESS/stability gate")
    np.savez(
        final_path,
        theta=proposal_theta,
        logq=logq_physical,
        logl=evaluated["logl"],
        logw=importance.log_weight,
        w=importance.normalized_weight,
        mass_grid=MASS_GRID,
        radius=evaluated["radius"],
        tidal_lambda=evaluated["tidal"],
        maximum_mass=evaluated["mmax"],
        radius_at_maximum_mass=evaluated["rmax"],
        radius_1p4=evaluated["r14"],
        lambda_1p4=evaluated["lambda14"],
        band_radius=radius_band["quantile"],
        support_radius=radius_band["support"],
        conditional_ess_radius=radius_band["conditional_ess"],
        band_tidal=tidal_band["quantile"],
        support_tidal=tidal_band["support"],
        conditional_ess_tidal=tidal_band["conditional_ess"],
        ESS=np.float64(importance.ess),
        logZ=np.float64(importance.log_evidence),
        logZ_standard_error=np.float64(importance.log_evidence_standard_error),
    )
    report = {
        "status": (
            "SMOKE_COMPLETED"
            if arguments.smoke
            else "REFINEMENT_READY_FOR_CROSS_METHOD_GATES"
        ),
        "method": "A-NET-seeded direct-theta importance-sampling refinement",
        "adaptation_excluded_from_final_estimator": True,
        "flow_density_used_in_final_estimator": False,
        "anet_adaptation": str(arguments.anet_is),
        "anet_adaptation_sha256": sha256(arguments.anet_is),
        "proposal": str(proposal_path),
        "proposal_sha256": sha256(proposal_path),
        "rows": arguments.rows,
        "allocation": allocation.tolist(),
        "tail_threshold_lambda_at_1p0": tail_threshold,
        "importance_ess": importance.ess,
        "minimum_ess": arguments.minimum_ess,
        "low_tail_ess": tail_ess,
        "minimum_low_tail_ess": minimum_tail_ess,
        "required_minimum_low_tail_ess": arguments.minimum_tail_ess,
        "parity_half_max_edge_fraction": parity_difference,
        "internal_gates_pass": internal_pass,
        "postcomputation_validation_not_used_in_estimator": True,
        "output": str(final_path),
        "output_sha256": sha256(final_path),
        "wall_seconds": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
