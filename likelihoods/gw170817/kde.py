"""Validated marginalized GW170817 tidal likelihood."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import h5py
import numpy as np
from scipy.stats import gaussian_kde

Z_GW170817 = 0.0099
DATASET = "IMRPhenomPv2NRT_lowSpin_posterior"


@lru_cache(maxsize=8)
def load_kde(path: str | Path, subsample=4000, z=Z_GW170817, seed=0):
    """Build the certified four-dimensional source-frame posterior KDE."""

    with h5py.File(path, "r") as handle:
        data = handle[DATASET][:]
        m1_detector = np.asarray(
            data["m1_detector_frame_Msun"], dtype=np.float64
        )
        m2_detector = np.asarray(
            data["m2_detector_frame_Msun"], dtype=np.float64
        )
        lambda1 = np.asarray(data["lambda1"], dtype=np.float64)
        lambda2 = np.asarray(data["lambda2"], dtype=np.float64)

    m1 = m1_detector / (1.0 + z)
    m2 = m2_detector / (1.0 + z)
    swap = m2 > m1
    m1_sorted = np.where(swap, m2, m1)
    m2_sorted = np.where(swap, m1, m2)
    lambda1_sorted = np.where(swap, lambda2, lambda1)
    lambda2_sorted = np.where(swap, lambda1, lambda2)
    m1, m2 = m1_sorted, m2_sorted
    lambda1, lambda2 = lambda1_sorted, lambda2_sorted

    chirp_mass = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    mass_ratio = m2 / m1
    if subsample is not None and subsample < len(chirp_mass):
        index = np.random.default_rng(seed).choice(
            len(chirp_mass), size=subsample, replace=False
        )
        kde_data = np.vstack(
            [
                chirp_mass[index],
                mass_ratio[index],
                lambda1[index],
                lambda2[index],
            ]
        )
    else:
        kde_data = np.vstack([chirp_mass, mass_ratio, lambda1, lambda2])
    kernel = gaussian_kde(kde_data)

    chirp_kde = gaussian_kde(chirp_mass)
    grid = np.linspace(chirp_mass.min(), chirp_mass.max(), 2001)
    observed_chirp_mass = float(grid[np.argmax(chirp_kde(grid))])
    return kernel, observed_chirp_mass


def log_likelihood(
    mass,
    tidal_lambda,
    maximum_mass,
    kernel,
    observed_chirp_mass,
    nq=20,
    qmin=0.7,
):
    """Marginalize the GW KDE over mass ratio along one EOS curve."""

    mass = np.asarray(mass, dtype=np.float64)
    tidal_lambda = np.asarray(tidal_lambda, dtype=np.float64)
    mass_ratio = np.linspace(qmin, 1.0, nq)
    m1 = (
        observed_chirp_mass
        * (1.0 + mass_ratio) ** 0.2
        / mass_ratio**0.6
    )
    m2 = mass_ratio * m1
    minimum_mass = mass.min()
    valid = (
        (m1 <= maximum_mass)
        & (m2 <= maximum_mass)
        & (m1 >= minimum_mass)
        & (m2 >= minimum_mass)
    )
    if not valid.any():
        return -1e30
    log_lambda = np.log(tidal_lambda)
    lambda1 = np.exp(np.interp(m1[valid], mass, log_lambda))
    lambda2 = np.exp(np.interp(m2[valid], mass, log_lambda))
    points = np.vstack(
        [
            np.full(valid.sum(), observed_chirp_mass),
            mass_ratio[valid],
            lambda1,
            lambda2,
        ]
    )
    density = kernel(points)
    integral = np.trapezoid(density, mass_ratio[valid])
    return float(np.log(integral + 1e-300))


load_gw_kde = load_kde
logL_gw = log_likelihood
