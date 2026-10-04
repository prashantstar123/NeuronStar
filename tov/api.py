"""Model-independent public interface to the validated stellar solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import solver, tidal
from .crust import TOV_H, graft_bps_crust


@dataclass(frozen=True)
class StableBranch:
    mass: np.ndarray
    radius: np.ndarray
    tidal_lambda: np.ndarray
    maximum_mass: float


def solve_stable_branch(
    energy,
    pressure,
    step=TOV_H,
    *,
    require_monotonic_graft=False,
):
    """Solve one core EOS and return its stable M-R-Lambda branch.

    Inputs are energy density and pressure in MeV/fm^3.  The selection,
    sorting, and stable-branch conventions exactly mirror the certified paper
    forward.
    """

    energy = np.asarray(energy, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    valid = (
        np.isfinite(energy)
        & np.isfinite(pressure)
        & (energy > 0)
        & (pressure > 0)
    )
    energy = energy[valid]
    pressure = pressure[valid]
    if len(energy) < 5:
        raise ValueError("EOS has fewer than five finite positive rows")
    order = np.argsort(energy)
    energy = energy[order]
    pressure = pressure[order]
    full_energy, full_pressure = graft_bps_crust(energy, pressure)
    if require_monotonic_graft:
        if len(full_energy) < 5 or np.any(np.diff(full_energy) <= 0.0):
            raise ValueError("crust graft did not produce monotone energy density")
        if np.any(np.diff(full_pressure) <= 0.0):
            raise ValueError("crust graft did not produce monotone pressure")
    energy_geom = np.ascontiguousarray(full_energy * solver.MEV_FM3_TO_GEOM)
    pressure_geom = np.ascontiguousarray(full_pressure * solver.MEV_FM3_TO_GEOM)
    central = solver.P_C_GEOM[solver.P_C_GEOM < 0.999 * pressure_geom[-1]]
    if len(central) < 4:
        raise ValueError("EOS pressure range supports fewer than four TOV stars")
    mass, radius, tidal_lambda = tidal.mrl_curve(
        np.ascontiguousarray(central), pressure_geom, energy_geom, step
    )
    valid = (
        np.isfinite(mass)
        & np.isfinite(radius)
        & np.isfinite(tidal_lambda)
        & (mass > 0)
        & (radius > 0)
        & (tidal_lambda > 0)
    )
    mass = mass[valid]
    radius = radius[valid]
    tidal_lambda = tidal_lambda[valid]
    if len(mass) < 4:
        raise ValueError("fewer than four valid TOV solutions")
    maximum_index = int(np.argmax(mass))
    maximum_mass = float(mass[maximum_index])
    mass = mass[: maximum_index + 1]
    radius = radius[: maximum_index + 1]
    tidal_lambda = tidal_lambda[: maximum_index + 1]
    order = np.argsort(mass)
    mass = mass[order]
    radius = radius[order]
    tidal_lambda = tidal_lambda[order]
    unique_index = np.unique(mass, return_index=True)[1]
    mass = mass[unique_index]
    radius = radius[unique_index]
    tidal_lambda = tidal_lambda[unique_index]
    if len(mass) < 4:
        raise ValueError("stable branch has fewer than four unique masses")
    return StableBranch(mass, radius, tidal_lambda, maximum_mass)
