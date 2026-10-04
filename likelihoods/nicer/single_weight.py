"""NICER mass-radius likelihood grid with the single-weight convention of the paper.

The double-weight implementation is kept in legacy_stack/ddb_astro_mod/nicer_like.py.
This module changes one operation only: after a weighted,
probability-proportional-to-weight subsample is drawn, the KDE is unweighted.
The sample weight therefore enters once rather than during both subsampling and
KDE fitting.  Grid bounds, random seed, sampling without replacement,
bandwidth, floor, interpolation, and the curve likelihood are unchanged.
"""

from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("CUDA_ROOT", "/tmp")

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde


@lru_cache(maxsize=32)
def build_nicer_grid(
    path,
    mcol,
    rcol,
    wcol_or_None,
    nM=400,
    nR=400,
    bw=0.08,
    Mlo=None,
    Mhi=None,
    Rlo=None,
    Rhi=None,
    max_rows=60000,
    seed=0,
):
    """Build the single-weight NICER log-density interpolator.

    Weighted files with more than ``max_rows`` entries are sampled with
    probability proportional to their supplied weights.  Those selected rows
    then enter the KDE with equal weight.  If no subsampling is required, the
    supplied weights are passed to the KDE once.
    """

    raw = np.loadtxt(path, comments="#")
    raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
    mass = raw[:, mcol].astype(np.float64)
    radius = raw[:, rcol].astype(np.float64)
    if wcol_or_None is None:
        weight = np.ones_like(mass)
    else:
        weight = raw[:, wcol_or_None].astype(np.float64)

    good = (
        np.isfinite(mass)
        & np.isfinite(radius)
        & np.isfinite(weight)
        & (weight >= 0)
    )
    mass, radius, weight = mass[good], radius[good], weight[good]
    if len(mass) == 0:
        raise ValueError(f"No usable NICER rows in {path}")
    if wcol_or_None is not None and not (weight.sum() > 0):
        raise ValueError(f"NICER weights do not have a positive sum in {path}")

    if Mlo is None:
        Mlo = np.percentile(mass, 1.0)
    if Mhi is None:
        Mhi = np.percentile(mass, 99.0)
    if Rlo is None:
        Rlo = np.percentile(radius, 1.0)
    if Rhi is None:
        Rhi = np.percentile(radius, 99.0)

    if len(mass) > max_rows:
        rng = np.random.default_rng(seed)
        probability = weight / weight.sum()
        index = rng.choice(
            len(mass), size=max_rows, replace=False, p=probability
        )
        selected_mass = mass[index]
        selected_radius = radius[index]
        # The probability-proportional subsample already carries the weights.
        selected_weight = None
    else:
        selected_mass = mass
        selected_radius = radius
        selected_weight = weight if wcol_or_None is not None else None

    kde = gaussian_kde(
        np.vstack([selected_mass, selected_radius]),
        weights=selected_weight,
        bw_method=bw,
    )

    mass_centers = np.linspace(Mlo, Mhi, nM)
    radius_centers = np.linspace(Rlo, Rhi, nR)
    mass_mesh, radius_mesh = np.meshgrid(
        mass_centers, radius_centers, indexing="ij"
    )
    points = np.vstack([mass_mesh.ravel(), radius_mesh.ravel()])
    density = kde(points).reshape(nM, nR)

    floor = density.max() * 1e-6
    log_density = np.log(np.maximum(density, floor))
    log_floor = float(np.log(floor))
    return RegularGridInterpolator(
        (mass_centers, radius_centers),
        log_density,
        method="linear",
        bounds_error=False,
        fill_value=log_floor,
    )


def log_likelihood_one(mass, radius, maximum_mass, interpolator):
    """Marginalize one NICER density along the stable mass-radius branch."""

    mass = np.asarray(mass, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    if mass.ndim != 1 or radius.shape != mass.shape or len(mass) < 2:
        raise ValueError("mass and radius must be matching one-dimensional curves")
    lower = max(float(mass.min()), 1.0)
    upper = min(float(maximum_mass), float(mass.max()))
    if upper <= lower:
        return -1e30
    mass_grid = np.linspace(lower, upper, 40)
    radius_grid = np.interp(mass_grid, mass, radius)
    density = np.exp(interpolator(np.column_stack([mass_grid, radius_grid])))
    integral = np.trapezoid(density, mass_grid) / (upper - lower)
    return float(np.log(integral + 1e-300))


def log_likelihood(mass, radius, maximum_mass, interpolators):
    """Sum the independently gridded likelihoods for all selected sources."""

    return float(
        sum(
            log_likelihood_one(mass, radius, maximum_mass, interpolator)
            for interpolator in interpolators
        )
    )


# Aliases with the names used in legacy_stack/ddb_astro_mod/nicer_like.py.
logL_nicer_one = log_likelihood_one
logL_nicer = log_likelihood
