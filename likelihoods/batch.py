"""Batch-exact likelihood evaluation for accelerated cache construction.

The scalar likelihood functions remain authoritative.  These routines group
identical operations across EOS rows to remove Python and SciPy call overhead;
the frozen-target certificate is required before their output is accepted.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from likelihoods.pqcd.core import constraints
from likelihoods.pqcd.likelihood import SCALE_GRID


@njit(parallel=True, cache=True)
def _pqcd_batch(density, energy, pressure, rho_eval, scale_grid):
    result = np.empty(density.shape[0], dtype=np.float64)
    for row in prange(density.shape[0]):
        if rho_eval < density[row, 0] or rho_eval > density[row, -1]:
            result[row] = 0.0
            continue
        lo = 0
        hi = density.shape[1] - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if density[row, mid] <= rho_eval:
                lo = mid
            else:
                hi = mid
        width = density[row, hi] - density[row, lo]
        fraction = 0.0 if width <= 0.0 else (rho_eval - density[row, lo]) / width
        energy0 = (
            energy[row, lo] + fraction * (energy[row, hi] - energy[row, lo])
        ) / 1000.0
        pressure0 = (
            pressure[row, lo]
            + fraction * (pressure[row, hi] - pressure[row, lo])
        ) / 1000.0
        if not math.isfinite(energy0) or not math.isfinite(pressure0) or pressure0 <= 0.0:
            result[row] = 0.0
            continue
        count = 0
        for column in range(len(scale_grid)):
            if constraints(
                scale_grid[column], energy0, pressure0, rho_eval
            ):
                count += 1
        result[row] = math.log(count / len(scale_grid) + 1e-30)
    return result


def pqcd_log_likelihood_batch(density, energy, pressure, rho_eval=1.2):
    density = np.ascontiguousarray(density, dtype=np.float64)
    energy = np.ascontiguousarray(energy, dtype=np.float64)
    pressure = np.ascontiguousarray(pressure, dtype=np.float64)
    if density.shape != energy.shape or density.shape != pressure.shape:
        raise ValueError("batched pQCD arrays must match")
    if density.ndim != 2:
        raise ValueError("batched pQCD arrays must be two-dimensional")
    return _pqcd_batch(
        density,
        energy,
        pressure,
        float(rho_eval),
        np.ascontiguousarray(SCALE_GRID, dtype=np.float64),
    )


def nicer_log_likelihood_batch(branches, interpolators):
    """Evaluate one target's NICER factors with batched grid interpolation."""

    count = len(branches)
    output = np.full(count, -1e30, dtype=np.float64)
    if not interpolators:
        output.fill(0.0)
        return output
    valid_rows = []
    mass_grids = []
    radius_grids = []
    intervals = []
    for row, branch in enumerate(branches):
        if branch is None:
            continue
        lower = max(float(branch.mass.min()), 1.0)
        upper = min(float(branch.maximum_mass), float(branch.mass.max()))
        if upper <= lower:
            continue
        mass_grid = np.linspace(lower, upper, 40)
        valid_rows.append(row)
        mass_grids.append(mass_grid)
        radius_grids.append(np.interp(mass_grid, branch.mass, branch.radius))
        intervals.append(upper - lower)
    if not valid_rows:
        return output
    mass_grids = np.asarray(mass_grids, dtype=np.float64)
    radius_grids = np.asarray(radius_grids, dtype=np.float64)
    intervals = np.asarray(intervals, dtype=np.float64)
    points = np.column_stack([mass_grids.ravel(), radius_grids.ravel()])
    total = np.zeros(len(valid_rows), dtype=np.float64)
    for interpolator in interpolators:
        density = np.exp(interpolator(points)).reshape(len(valid_rows), 40)
        integral = np.trapezoid(density, mass_grids, axis=1) / intervals
        total += np.log(integral + 1e-300)
    output[np.asarray(valid_rows, dtype=np.int64)] = total
    return output


def gw_log_likelihood_batch(
    branches,
    kernel,
    observed_chirp_mass,
    nq=20,
    qmin=0.7,
):
    """Evaluate the unchanged GW KDE once for all row-specific query points."""

    count = len(branches)
    output = np.full(count, -1e30, dtype=np.float64)
    mass_ratio = np.linspace(qmin, 1.0, nq)
    m1 = (
        observed_chirp_mass
        * (1.0 + mass_ratio) ** 0.2
        / mass_ratio**0.6
    )
    m2 = mass_ratio * m1
    point_blocks = []
    row_records = []
    offset = 0
    for row, branch in enumerate(branches):
        if branch is None:
            continue
        minimum_mass = float(branch.mass.min())
        valid = (
            (m1 <= branch.maximum_mass)
            & (m2 <= branch.maximum_mass)
            & (m1 >= minimum_mass)
            & (m2 >= minimum_mass)
        )
        if not valid.any():
            continue
        log_lambda = np.log(branch.tidal_lambda)
        lambda1 = np.exp(np.interp(m1[valid], branch.mass, log_lambda))
        lambda2 = np.exp(np.interp(m2[valid], branch.mass, log_lambda))
        points = np.vstack(
            [
                np.full(valid.sum(), observed_chirp_mass),
                mass_ratio[valid],
                lambda1,
                lambda2,
            ]
        )
        point_blocks.append(points)
        stop = offset + points.shape[1]
        row_records.append((row, offset, stop, mass_ratio[valid]))
        offset = stop
    if not point_blocks:
        return output
    density = np.asarray(kernel(np.hstack(point_blocks)), dtype=np.float64)
    for row, start, stop, local_ratio in row_records:
        integral = np.trapezoid(density[start:stop], local_ratio)
        output[row] = np.log(integral + 1e-300)
    return output


def astrophysical_log_likelihoods_batch(
    density,
    energy,
    pressure,
    branches,
    targets,
):
    """Return one astrophysical log-likelihood vector per source target."""

    if not targets:
        raise ValueError("at least one target is required")
    primary = targets[0]
    maximum_mass = np.asarray(
        [np.nan if branch is None else branch.maximum_mass for branch in branches],
        dtype=np.float64,
    )
    maximum_term = -np.logaddexp(
        0.0,
        -(maximum_mass - primary.maximum_mass_threshold)
        / primary.maximum_mass_width,
    )
    if primary.pqcd_enabled:
        pqcd = pqcd_log_likelihood_batch(
            density, energy, pressure, rho_eval=primary.pqcd_density
        )
    else:
        pqcd = np.zeros(len(branches), dtype=np.float64)
    if primary.gw_kernel is None:
        gw = np.zeros(len(branches), dtype=np.float64)
    else:
        gw = gw_log_likelihood_batch(
            branches,
            primary.gw_kernel,
            primary.gw_chirp_mass,
        )
    common = maximum_term + pqcd + gw
    invalid = np.asarray([branch is None for branch in branches])
    outputs = []
    for target in targets:
        nicer = nicer_log_likelihood_batch(branches, target.nicer_interpolators)
        value = common + nicer
        value[invalid | ~np.isfinite(value)] = -np.inf
        outputs.append(value)
    return outputs
