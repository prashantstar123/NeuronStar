"""Nine-parameter DDB Lambda-Xi-minus equation-of-state adapter.

The nonlinear hyperonic equilibrium solver is preserved byte-for-byte in
``physics.py`` from the certified implementation.  This module supplies
only the repository's backend-independent EOS interface: prior metadata,
core-table construction, and the inherited nucleonic saturation observables.
"""

from __future__ import annotations

import numpy as np

from eos.ddb import (
    PARAMETER_NAMES as DDB_PARAMETER_NAMES,
    PRIOR_HIGH as DDB_PRIOR_HIGH,
    PRIOR_LOW as DDB_PRIOR_LOW,
    nuclear_matter_observables as ddb_nuclear_matter_observables,
    nuclear_matter_observables_batch as ddb_nuclear_matter_observables_batch,
)

from .physics import (
    DDBParameters,
    HBARC_MEV_FM,
    HyperonCouplings,
    MatterState,
    RMFMasses,
    SolverOptions,
    build_eos_table,
)


PARAMETER_NAMES = DDB_PARAMETER_NAMES + (
    "x_sigma_lambda",
    "x_sigma_xi_minus",
)
PRIOR_LOW = np.concatenate(
    [DDB_PRIOR_LOW, np.array([0.609, 0.309], dtype=np.float64)]
)
PRIOR_HIGH = np.concatenate(
    [DDB_PRIOR_HIGH, np.array([0.622, 0.322], dtype=np.float64)]
)
CORE_DENSITY_GRID = np.linspace(0.04, 1.5, 200, dtype=np.float64)


def _as_theta(theta) -> np.ndarray:
    value = np.asarray(theta, dtype=np.float64)
    if value.shape != (9,) or not np.all(np.isfinite(value)):
        raise ValueError(
            f"expected nine finite DDB-hyperonic parameters, got {value.shape}"
        )
    return value


def core_eos_with_states(
    theta,
    *,
    density_grid=CORE_DENSITY_GRID,
    options: SolverOptions | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[MatterState, ...]]:
    """Return the core table and equilibrium states for one parameter row."""

    value = _as_theta(theta)
    density = np.asarray(density_grid, dtype=np.float64)
    if density.ndim != 1 or len(density) == 0 or not np.all(np.isfinite(density)):
        raise ValueError("density_grid must be a non-empty finite vector")
    ddb = DDBParameters.from_theta(value[:7])
    hyperons = HyperonCouplings.literal_malik_providencia(
        x_sigma_lambda=float(value[7]),
        x_sigma_xi_minus=float(value[8]),
    )
    states = tuple(
        build_eos_table(
            density,
            ddb,
            RMFMasses.certified_ddb(),
            hyperons,
            options=options,
        )
    )
    if len(states) != len(density) or not all(
        isinstance(state, MatterState) for state in states
    ):
        raise ValueError("hyperonic solver returned an invalid state table")
    energy = np.asarray(
        [state.energy_density * HBARC_MEV_FM for state in states],
        dtype=np.float64,
    )
    pressure = np.asarray(
        [state.pressure * HBARC_MEV_FM for state in states],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(pressure)):
        raise ValueError("hyperonic core table contains non-finite thermodynamics")
    return density.copy(), energy, pressure, states


def core_eos(theta):
    """Return density, energy density, and pressure in the shared units."""

    density, energy, pressure, _states = core_eos_with_states(theta)
    return density, energy, pressure


def core_eos_batch(theta):
    """Evaluate the exact solver row-by-row for an ``(N, 9)`` array."""

    value = np.asarray(theta, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 9:
        raise ValueError(
            f"expected DDB-hyperonic parameter array with shape (N, 9), got {value.shape}"
        )
    rows = [core_eos(row) for row in value]
    return tuple(np.stack(items, axis=0) for items in zip(*rows))


def nuclear_matter_observables(theta):
    """Return the six saturation observables inherited from nucleonic DDB."""

    return ddb_nuclear_matter_observables(_as_theta(theta)[:7])


def nuclear_matter_observables_batch(theta):
    """Evaluate inherited saturation observables for an ``(N, 9)`` array."""

    value = np.asarray(theta, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 9:
        raise ValueError(
            f"expected DDB-hyperonic parameter array with shape (N, 9), got {value.shape}"
        )
    return ddb_nuclear_matter_observables_batch(value[:, :7])
