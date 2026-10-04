#!/usr/bin/env python3
"""Compare a bounded nucleonic S_ext shard with the scalar reference path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from eos import get_eos
from likelihoods.nicer import log_likelihood_one
from tov import solve_stable_branch
from workflows.a1_problem import A1Problem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _maximum_abs(left, right, mask) -> float:
    return float(np.max(np.abs(left[mask] - right[mask]))) if mask.any() else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite certificate: {arguments.output}")

    with np.load(arguments.candidate, allow_pickle=False) as candidate:
        theta = np.asarray(candidate["theta"], dtype=np.float64)
        prediction = np.asarray(candidate["prediction"], dtype=np.float64)
        mass_grid = np.asarray(candidate["mass_grid"], dtype=np.float64)
        got_valid = np.asarray(candidate["target_valid"], dtype=bool)
        got_la = np.asarray(candidate["log_astrophysical"], dtype=np.float64)
        got_components = np.asarray(
            candidate["astrophysical_components"], dtype=np.float64
        )
        got_sources = np.asarray(
            candidate["nicer_source_components"], dtype=np.float64
        )
        got_radius = np.asarray(candidate["radius"], dtype=np.float64)
        got_tidal = np.asarray(candidate["tidal_lambda"], dtype=np.float64)
        got_mmax = np.asarray(candidate["maximum_mass"], dtype=np.float64)
    if len(theta) > 256:
        raise RuntimeError("bounded adapter certificate accepts at most 256 rows")

    problem = A1Problem.from_data_root(
        arguments.data_root,
        workers=1,
        verify_data=True,
        nuclear_scenario="A1",
        source_scenario="A1",
        model="ddb",
    )
    plugin = get_eos("ddb")
    ref_valid = np.zeros(len(theta), dtype=bool)
    ref_la = np.full(len(theta), np.nan, dtype=np.float64)
    ref_components = np.full((len(theta), 4), np.nan, dtype=np.float64)
    ref_sources = np.full((len(theta), 3), np.nan, dtype=np.float64)
    ref_radius = np.full_like(got_radius, np.nan)
    ref_tidal = np.full_like(got_tidal, np.nan)
    ref_mmax = np.full(len(theta), np.nan, dtype=np.float64)

    for row in range(len(theta)):
        try:
            density, energy, pressure = plugin.core_eos(theta[row])
            branch = solve_stable_branch(energy, pressure)
        except (ValueError, FloatingPointError, RuntimeError):
            continue
        terms = problem.target.evaluate(
            theta[row], prediction[row, 1:], density, energy, pressure, branch
        )
        source = np.asarray(
            [
                log_likelihood_one(
                    branch.mass,
                    branch.radius,
                    branch.maximum_mass,
                    interpolator,
                )
                for interpolator in problem.target.nicer_interpolators
            ],
            dtype=np.float64,
        )
        components = np.asarray(
            [terms.maximum_mass, terms.nicer, terms.gw170817, terms.pqcd],
            dtype=np.float64,
        )
        value = float(components.sum())
        if (
            not np.isfinite(terms.total)
            or value <= -1.0e29
            or not np.all(np.isfinite(source))
        ):
            continue
        covered = (
            (mass_grid >= float(branch.mass.min()))
            & (mass_grid <= float(branch.mass.max()))
            & (mass_grid <= float(branch.maximum_mass))
        )
        ref_valid[row] = True
        ref_la[row] = value
        ref_components[row] = components
        ref_sources[row] = source
        ref_radius[row, covered] = np.interp(
            mass_grid[covered], branch.mass, branch.radius
        )
        ref_tidal[row, covered] = np.exp(
            np.interp(
                mass_grid[covered],
                branch.mass,
                np.log(np.clip(branch.tidal_lambda, 1.0e-300, None)),
            )
        )
        ref_mmax[row] = float(branch.maximum_mass)

    if not np.array_equal(got_valid, ref_valid):
        mismatch = np.flatnonzero(got_valid != ref_valid)
        raise RuntimeError(f"candidate/reference validity mismatch at {mismatch.tolist()}")
    mask = ref_valid
    finite_curves = mask[:, None] & np.isfinite(ref_radius) & np.isfinite(got_radius)
    finite_tidal = mask[:, None] & np.isfinite(ref_tidal) & np.isfinite(got_tidal)
    # Curve products are consumed only for target-valid rows.  The accelerated
    # evaluator may retain a finite TOV curve for a row subsequently rejected
    # by the astrophysical likelihood, whereas the scalar reference leaves the
    # entire rejected row as NaN.  Compare the curve masks on the common valid
    # set; requiring rejected-row storage to match would test bookkeeping, not
    # the certified physics output.
    nan_pattern_radius = np.array_equal(
        np.isnan(got_radius[mask]), np.isnan(ref_radius[mask])
    )
    nan_pattern_tidal = np.array_equal(
        np.isnan(got_tidal[mask]), np.isnan(ref_tidal[mask])
    )
    metrics = {
        "valid_rows": int(mask.sum()),
        "log_astrophysical_max_abs": _maximum_abs(got_la, ref_la, mask),
        "components_max_abs": _maximum_abs(
            got_components, ref_components, mask
        ),
        "nicer_sources_max_abs": _maximum_abs(
            got_sources, ref_sources, mask
        ),
        "maximum_mass_max_abs": _maximum_abs(got_mmax, ref_mmax, mask),
        "radius_max_abs": _maximum_abs(got_radius, ref_radius, finite_curves),
        "tidal_max_relative": float(
            np.max(
                np.abs(got_tidal[finite_tidal] - ref_tidal[finite_tidal])
                / np.maximum(np.abs(ref_tidal[finite_tidal]), 1.0e-300)
            )
        )
        if finite_tidal.any()
        else 0.0,
        "radius_nan_pattern_equal": nan_pattern_radius,
        "tidal_nan_pattern_equal": nan_pattern_tidal,
        "rejected_candidate_radius_finite_entries_ignored": int(
            np.isfinite(got_radius[~mask]).sum()
        ),
        "rejected_candidate_tidal_finite_entries_ignored": int(
            np.isfinite(got_tidal[~mask]).sum()
        ),
    }
    gates = {
        "validity_equal": True,
        "log_astrophysical_le_2e-7": metrics["log_astrophysical_max_abs"] <= 2e-7,
        "components_le_2e-7": metrics["components_max_abs"] <= 2e-7,
        "nicer_sources_le_2e-7": metrics["nicer_sources_max_abs"] <= 2e-7,
        "maximum_mass_le_1e-10": metrics["maximum_mass_max_abs"] <= 1e-10,
        "radius_le_1e-7_km": metrics["radius_max_abs"] <= 1e-7,
        "tidal_relative_le_2e-6": metrics["tidal_max_relative"] <= 2e-6,
        "curve_nan_patterns_equal": nan_pattern_radius and nan_pattern_tidal,
    }
    if not all(gates.values()):
        raise RuntimeError(f"nucleonic adapter certification failed: {gates}; {metrics}")
    report = {
        "status": "PASS",
        "schema": "nucleonic-support-extension-adapter-certificate-v1",
        "candidate": str(arguments.candidate.resolve()),
        "candidate_sha256": sha256(arguments.candidate),
        "rows": int(len(theta)),
        "reference": "scalar plugin.core_eos + exact CPU TOV + JointLikelihood.evaluate",
        "metrics": metrics,
        "gates": gates,
        "timing_class": "certification only; excluded from production wall time",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
