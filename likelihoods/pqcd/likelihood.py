"""Deterministic pQCD likelihood factor."""

from __future__ import annotations

import numpy as np

from .core import constraints

SCALE_GRID = np.linspace(1.0, 4.0, 400)


def log_likelihood(density, energy, pressure, rho_eval=1.2):
    """Return the log fraction of renormalization scales passing the constraint."""

    density = np.asarray(density, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    if not (density.shape == energy.shape == pressure.shape) or density.ndim != 1:
        raise ValueError("density, energy, and pressure must be matching 1D arrays")
    if rho_eval < density.min() or rho_eval > density.max():
        return 0.0
    energy0 = float(np.interp(rho_eval, density, energy)) / 1000.0
    pressure0 = float(np.interp(rho_eval, density, pressure)) / 1000.0
    if not (np.isfinite(energy0) and np.isfinite(pressure0)) or pressure0 <= 0:
        return 0.0
    count = 0
    for scale in SCALE_GRID:
        if constraints(scale, energy0, pressure0, rho_eval):
            count += 1
    return float(np.log(count / len(SCALE_GRID) + 1e-30))


logL_pqcd = log_likelihood
