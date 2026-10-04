#!/usr/bin/env python3
"""Generate a certified-schema DDB-Lambda-Xi-minus physics bank.

The output keeps the table contract used by the certified nucleonic bank:

``theta``
    Nine input columns: the certified seven-parameter DDB vector followed by
    ``x_sigma_Lambda`` and ``x_sigma_Xi``.
``X``
    Seven certified nuclear observables in the order
    ``rho0, E0, K0, Jsym, P08, P12, P16``.
``MG``, ``Rg``, ``Lg``, ``MM``
    The certified fixed mass grid, radius and tidal-deformability curves, and
    maximum mass.  Unsupported mass-grid entries remain NaN, as upstream.

The hyperonic core is solved by :mod:`ddb_hyperon_eos`.  Crust grafting, TOV
integration, Love-number equations, mass grid, and nuclear observables follow
the read-only certified DDB pipeline.  Extra keys provide per-row validity,
failure accounting, timings, provenance, and optional direct-solve spot checks;
the certified consumer keys retain their names and shapes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from joblib import Parallel, delayed, parallel_config
from numba import njit
from numpy.typing import ArrayLike, NDArray

from ddb_hyperon_eos import (
    DDBParameters,
    HBARC_MEV_FM,
    HyperonCouplings,
    MatterState,
    RMFMasses,
    SolverOptions,
    build_eos_table,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

PROJECT_DIR = Path(__file__).resolve().parent
CERTIFIED_DDB_DIR = Path(
    os.environ.get(
        "CERTIFIED_DDB_DIR",
        "/home/nucleartheory/Desktop/validated_code_DDB",
    )
).resolve()
CERTIFIED_ASTRO_DIR = Path(
    os.environ.get("CERTIFIED_ASTRO_DIR", "/home/nucleartheory/ddb_astro_mod")
).resolve()

if not CERTIFIED_DDB_DIR.is_dir():
    raise FileNotFoundError(f"Certified DDB directory is missing: {CERTIFIED_DDB_DIR}")
sys.path.insert(0, str(CERTIFIED_DDB_DIR))

# These are read-only imports.  Unlike importing tidal_tov, they do not import
# JAX in every joblib worker.  The Love/TOV kernel below is copied verbatim from
# the certified ddb_astro_mod/tidal_tov.py implementation, while its RHS and
# constants remain live references to the certified module.
import ddb_crust as _certified_crust  # noqa: E402
import tov_numba as _certified_tov  # noqa: E402


THETA_NAMES = np.asarray(
    [
        "a_sigma",
        "a_omega",
        "a_rho",
        "Gamma_sigma",
        "Gamma_omega",
        "Gamma_rho",
        "rho0",
        "x_sigma_Lambda",
        "x_sigma_Xi",
    ]
)
X_NAMES = np.asarray(["rho0", "E0", "K0", "Jsym", "P08", "P12", "P16"])
MASS_GRID = np.linspace(0.5, 2.6, 200, dtype=np.float64)
CORE_DENSITY_GRID = np.linspace(0.04, 1.5, 200, dtype=np.float64)
HYPERON_LOW = np.asarray([0.609, 0.309], dtype=np.float64)
HYPERON_HIGH = np.asarray([0.622, 0.322], dtype=np.float64)
TOV_STEP = float(_certified_crust.TOV_H)

FAIL_VALID = 0
FAIL_INPUT = 1
FAIL_EOS = 2
FAIL_CORE_TABLE = 3
FAIL_TOV = 4
FAIL_MASS_GRID = 5
FAIL_NMP = 6
FAIL_INTERNAL = 7
FAILURE_LABELS = {
    FAIL_VALID: "valid",
    FAIL_INPUT: "input",
    FAIL_EOS: "eos",
    FAIL_CORE_TABLE: "core_table",
    FAIL_TOV: "tov",
    FAIL_MASS_GRID: "mass_grid",
    FAIL_NMP: "nmp",
    FAIL_INTERNAL: "internal",
}
FAILURE_REASON_DTYPE = np.dtype("S160")
CERTIFIED_CONSUMER_KEYS = frozenset(
    {"theta", "X", "MG", "Rg", "Lg", "MM", "R14", "OBS", "SIG"}
)


class RowResult(NamedTuple):
    """One parallel worker result, including failure and timing evidence."""

    index: int
    radius_grid: FloatArray
    lambda_grid: FloatArray
    maximum_mass: float
    radius_14: float
    failure_code: int
    failure_reason: str
    row_seconds: float
    eos_seconds: float
    tov_seconds: float


class CurveResult(NamedTuple):
    """Stable branch returned by the certified crust/Love/TOV route."""

    mass: FloatArray
    radius: FloatArray
    tidal_lambda: FloatArray
    maximum_mass: float


class HyperonicCoreResult(NamedTuple):
    """Shared hyperonic core forward before the certified crust/TOV step."""

    density: FloatArray
    energy: FloatArray
    pressure: FloatArray
    states: tuple[MatterState, ...]


def _blank_row(
    index: int,
    mass_grid: FloatArray,
    failure_code: int,
    failure_reason: str,
    *,
    started: float,
    eos_seconds: float = 0.0,
    tov_seconds: float = 0.0,
) -> RowResult:
    """Construct a shape-stable failed row without object arrays."""

    nan_grid = np.full(mass_grid.shape, np.nan, dtype=np.float64)
    return RowResult(
        index=index,
        radius_grid=nan_grid,
        lambda_grid=nan_grid.copy(),
        maximum_mass=np.nan,
        radius_14=np.nan,
        failure_code=failure_code,
        failure_reason=str(failure_reason).replace("\n", " ")[:240],
        row_seconds=time.perf_counter() - started,
        eos_seconds=eos_seconds,
        tov_seconds=tov_seconds,
    )


# ---------------------------------------------------------------------------
# Certified Love/TOV implementation
# ---------------------------------------------------------------------------


@njit(fastmath=True, error_model="numpy")
def _tov_single_lambda(
    p_c_geom: float,
    p_tab: FloatArray,
    eps_tab: FloatArray,
    r_init: float,
    h: float,
    n_steps: int,
) -> tuple[float, float, float]:
    """Certified Love-k2/``1/20`` TOV kernel from tidal_tov.py."""

    pi = np.pi
    p_surface = max(1.0e-20, p_tab[0] * 1.001)
    eps_c = _certified_tov.eps_of_p_lin(p_c_geom, p_tab, eps_tab)
    p = p_c_geom - (2.0 * pi / 3.0) * (eps_c + p_c_geom) * (
        eps_c + 3.0 * p_c_geom
    ) * r_init**2
    mass = (4.0 * pi / 3.0) * eps_c * r_init**3
    radius = r_init
    y_value = 2.0
    radius_surface = r_init
    mass_surface = mass
    y_surface = 2.0
    found = False
    for _ in range(n_steps):
        if found:
            break
        k1p, k1m, k1y = _certified_tov._tov_rhs(
            radius, p, mass, y_value, p_tab, eps_tab
        )
        k2p, k2m, k2y = _certified_tov._tov_rhs(
            radius + h / 2.0,
            p + h / 2.0 * k1p,
            mass + h / 2.0 * k1m,
            y_value + h / 2.0 * k1y,
            p_tab,
            eps_tab,
        )
        k3p, k3m, k3y = _certified_tov._tov_rhs(
            radius + h / 2.0,
            p + h / 2.0 * k2p,
            mass + h / 2.0 * k2m,
            y_value + h / 2.0 * k2y,
            p_tab,
            eps_tab,
        )
        k4p, k4m, k4y = _certified_tov._tov_rhs(
            radius + h,
            p + h * k3p,
            mass + h * k3m,
            y_value + h * k3y,
            p_tab,
            eps_tab,
        )
        p_new = p + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        mass_new = mass + h / 6.0 * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
        y_new = y_value + h / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        if p_new < p_surface:
            fraction = (p - p_surface) / max(p - p_new, 1.0e-30)
            fraction = min(max(fraction, 0.0), 1.0)
            radius_surface = radius + h * fraction
            mass_surface = mass + fraction * (mass_new - mass)
            y_surface = y_value + fraction * (y_new - y_value)
            found = True
        else:
            radius += h
            p = p_new
            mass = mass_new
            y_value = y_new
    if not found:
        return np.nan, np.nan, np.nan
    compactness = 2.0 * mass_surface / radius_surface
    if compactness > 0.99:
        compactness = 0.99
    f1 = 2.0 - y_surface + (y_surface - 1.0) * compactness
    numerator = (
        (1.0 / 20.0)
        * compactness**5
        * (1.0 - compactness) ** 2
        * f1
    )
    denominator = (
        compactness
        * (
            6.0
            - 3.0 * y_surface
            + 1.5 * compactness * (5.0 * y_surface - 8.0)
        )
        + 0.25
        * compactness**3
        * (
            26.0
            - 22.0 * y_surface
            + compactness * (3.0 * y_surface - 2.0)
            + compactness**2 * (1.0 + y_surface)
        )
        + 3.0
        * (1.0 - compactness) ** 2
        * f1
        * np.log1p(-compactness)
    )
    tidal_lambda = (
        (2.0 / 3.0)
        * (numerator / denominator)
        * (radius_surface / max(mass_surface, 1.0e-10)) ** 5
    )
    return (
        mass_surface,
        radius_surface * _certified_tov.KM_PER_MSOL,
        tidal_lambda,
    )


@njit(fastmath=True, error_model="numpy")
def _mrl_curve(
    central_pressures: FloatArray,
    pressure_table: FloatArray,
    energy_table: FloatArray,
    h: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate the certified mass-radius-Lambda central-pressure grid."""

    count = len(central_pressures)
    mass = np.empty(count)
    radius = np.empty(count)
    tidal_lambda = np.empty(count)
    for index in range(count):
        mass[index], radius[index], tidal_lambda[index] = _tov_single_lambda(
            central_pressures[index],
            pressure_table,
            energy_table,
            1.0e-4,
            h,
            _certified_tov.N_TOV_STEPS,
        )
    return mass, radius, tidal_lambda


def _stable_mrl_curve(
    energy_mev: ArrayLike,
    pressure_mev: ArrayLike,
    *,
    h: float = TOV_STEP,
) -> CurveResult:
    """Apply the certified BPS bridge and return the stable M-R-Lambda branch."""

    energy = np.asarray(energy_mev, dtype=np.float64)
    pressure = np.asarray(pressure_mev, dtype=np.float64)
    if energy.shape != pressure.shape or energy.ndim != 1:
        raise ValueError("Core energy and pressure must be same-shape 1-D arrays.")
    keep = (
        np.isfinite(energy)
        & np.isfinite(pressure)
        & (energy > 0.0)
        & (pressure > 0.0)
    )
    energy = energy[keep]
    pressure = pressure[keep]
    if len(energy) < 5:
        raise ValueError("Fewer than five finite positive core EOS points remain.")
    order = np.argsort(energy)
    energy = energy[order]
    pressure = pressure[order]
    energy_full, pressure_full = _certified_crust.graft_bps_crust(
        energy,
        pressure,
        n_bridge=30,
    )
    if len(energy_full) < 5 or np.any(np.diff(energy_full) <= 0.0):
        raise ValueError("Certified crust graft did not produce a monotone energy table.")
    if np.any(np.diff(pressure_full) <= 0.0):
        raise ValueError("Certified crust graft did not produce a monotone pressure table.")
    energy_geom = np.ascontiguousarray(
        energy_full * _certified_tov.MEV_FM3_TO_GEOM
    )
    pressure_geom = np.ascontiguousarray(
        pressure_full * _certified_tov.MEV_FM3_TO_GEOM
    )
    central_pressures = _certified_tov.P_C_GEOM[
        _certified_tov.P_C_GEOM < 0.999 * pressure_geom[-1]
    ]
    if len(central_pressures) < 4:
        raise ValueError("EOS table supports fewer than four certified TOV pressures.")
    mass, radius, tidal_lambda = _mrl_curve(
        np.ascontiguousarray(central_pressures),
        pressure_geom,
        energy_geom,
        h,
    )
    valid = (
        np.isfinite(mass)
        & np.isfinite(radius)
        & np.isfinite(tidal_lambda)
        & (mass > 0.0)
        & (radius > 0.0)
        & (tidal_lambda > 0.0)
    )
    mass = mass[valid]
    radius = radius[valid]
    tidal_lambda = tidal_lambda[valid]
    if len(mass) < 4:
        raise ValueError("Fewer than four finite TOV configurations remain.")
    maximum_index = int(np.argmax(mass))
    maximum_mass = float(mass[maximum_index])
    mass = mass[: maximum_index + 1]
    radius = radius[: maximum_index + 1]
    tidal_lambda = tidal_lambda[: maximum_index + 1]
    order = np.argsort(mass)
    mass = mass[order]
    radius = radius[order]
    tidal_lambda = tidal_lambda[order]
    unique = np.unique(mass, return_index=True)[1]
    mass = mass[unique]
    radius = radius[unique]
    tidal_lambda = tidal_lambda[unique]
    if len(mass) < 4:
        raise ValueError("Stable TOV branch has fewer than four unique masses.")
    return CurveResult(mass, radius, tidal_lambda, maximum_mass)


def interpolate_curve_to_bank_grid(
    curve: CurveResult,
    mass_grid: ArrayLike = MASS_GRID,
) -> tuple[FloatArray, FloatArray, float]:
    """Interpolate one stable branch with the certified bank conventions."""

    grid = np.asarray(mass_grid, dtype=np.float64)
    radius_grid = np.full(grid.shape, np.nan, dtype=np.float64)
    lambda_grid = np.full(grid.shape, np.nan, dtype=np.float64)
    supported = (grid >= curve.mass.min()) & (grid <= curve.mass.max())
    if supported.any():
        radius_grid[supported] = np.interp(
            grid[supported], curve.mass, curve.radius
        )
        lambda_grid[supported] = np.exp(
            np.interp(
                grid[supported],
                curve.mass,
                np.log(np.clip(curve.tidal_lambda, 1.0e-30, None)),
            )
        )
    radius_14 = (
        float(np.interp(1.4, curve.mass, curve.radius))
        if curve.mass.min() <= 1.4 <= curve.mass.max()
        else np.nan
    )
    return radius_grid, lambda_grid, radius_14


# ---------------------------------------------------------------------------
# Hyperonic row forward
# ---------------------------------------------------------------------------


def solve_hyperonic_core(
    theta: ArrayLike,
    *,
    density_grid: ArrayLike = CORE_DENSITY_GRID,
    options: SolverOptions | None = None,
) -> HyperonicCoreResult:
    """Solve the shared nine-parameter hyperonic core forward.

    This function deliberately does not enforce the sampling prior.  The bank
    wrapper performs its own prior-box validation, while certification tools
    may evaluate finite diagnostic points outside that box.  EOS equations,
    density continuation, units, and state audits all remain owned by
    :func:`ddb_hyperon_eos.build_eos_table`.
    """

    parameters = np.asarray(theta, dtype=np.float64)
    densities = np.asarray(density_grid, dtype=np.float64)
    if parameters.shape != (9,) or not np.all(np.isfinite(parameters)):
        raise ValueError("theta must contain nine finite parameters.")
    ddb = DDBParameters.from_theta(parameters[:7])
    hyperons = HyperonCouplings.literal_malik_providencia(
        x_sigma_lambda=float(parameters[7]),
        x_sigma_xi_minus=float(parameters[8]),
    )
    states = build_eos_table(
        densities,
        ddb,
        RMFMasses.certified_ddb(),
        hyperons,
        options=options,
    )
    if not states or not all(isinstance(state, MatterState) for state in states):
        raise ValueError("Hyperonic solver returned an empty or invalid state table.")
    energy = np.asarray(
        [state.energy_density * HBARC_MEV_FM for state in states],
        dtype=np.float64,
    )
    pressure = np.asarray(
        [state.pressure * HBARC_MEV_FM for state in states],
        dtype=np.float64,
    )
    if (
        energy.shape != densities.shape
        or pressure.shape != densities.shape
        or not np.all(np.isfinite(energy))
        or not np.all(np.isfinite(pressure))
    ):
        raise ValueError("Hyperonic core table has invalid shape or thermodynamics.")
    return HyperonicCoreResult(densities, energy, pressure, tuple(states))


def solve_bank_row(
    index: int,
    theta: ArrayLike,
    *,
    prior_low: ArrayLike,
    prior_high: ArrayLike,
    density_grid: ArrayLike = CORE_DENSITY_GRID,
    mass_grid: ArrayLike = MASS_GRID,
    options: SolverOptions | None = None,
) -> RowResult:
    """Solve one nine-parameter bank row and never raise across a batch."""

    started = time.perf_counter()
    parameters = np.asarray(theta, dtype=np.float64)
    low = np.asarray(prior_low, dtype=np.float64)
    high = np.asarray(prior_high, dtype=np.float64)
    masses = np.asarray(mass_grid, dtype=np.float64)
    if (
        parameters.shape != (9,)
        or low.shape != (9,)
        or high.shape != (9,)
        or not np.all(np.isfinite(parameters))
        or np.any(parameters < low)
        or np.any(parameters > high)
    ):
        return _blank_row(
            index,
            masses,
            FAIL_INPUT,
            "theta is nonfinite, wrong-shaped, or outside the declared prior box",
            started=started,
        )

    eos_started = time.perf_counter()
    try:
        core = solve_hyperonic_core(
            parameters,
            density_grid=density_grid,
            options=options,
        )
    except Exception as error:  # Per-row accounting must survive every solve.
        return _blank_row(
            index,
            masses,
            FAIL_EOS,
            f"{type(error).__name__}: {error}",
            started=started,
            eos_seconds=time.perf_counter() - eos_started,
        )
    eos_seconds = time.perf_counter() - eos_started
    energy = core.energy
    pressure = core.pressure

    tov_started = time.perf_counter()
    try:
        curve = _stable_mrl_curve(energy, pressure, h=TOV_STEP)
    except Exception as error:
        return _blank_row(
            index,
            masses,
            FAIL_TOV,
            f"{type(error).__name__}: {error}",
            started=started,
            eos_seconds=eos_seconds,
            tov_seconds=time.perf_counter() - tov_started,
        )
    tov_seconds = time.perf_counter() - tov_started
    radius_grid, lambda_grid, radius_14 = interpolate_curve_to_bank_grid(
        curve,
        masses,
    )
    if not np.isfinite(radius_grid).any() or not np.isfinite(lambda_grid).any():
        return RowResult(
            index=index,
            radius_grid=radius_grid,
            lambda_grid=lambda_grid,
            maximum_mass=curve.maximum_mass,
            radius_14=np.nan,
            failure_code=FAIL_MASS_GRID,
            failure_reason=(
                "stable curve does not overlap the certified fixed mass grid"
            ),
            row_seconds=time.perf_counter() - started,
            eos_seconds=eos_seconds,
            tov_seconds=tov_seconds,
        )
    return RowResult(
        index=index,
        radius_grid=radius_grid,
        lambda_grid=lambda_grid,
        maximum_mass=curve.maximum_mass,
        radius_14=radius_14,
        failure_code=FAIL_VALID,
        failure_reason="",
        row_seconds=time.perf_counter() - started,
        eos_seconds=eos_seconds,
        tov_seconds=tov_seconds,
    )


# ---------------------------------------------------------------------------
# Certified nuclear forward, bank assembly, and CLI
# ---------------------------------------------------------------------------


def load_certified_forward() -> Any:
    """Load the read-only certified JAX nuclear forward in the parent process."""

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ.setdefault("CUDA_ROOT", "/tmp")
    if str(CERTIFIED_DDB_DIR) not in sys.path:
        sys.path.insert(0, str(CERTIFIED_DDB_DIR))
    import ddb_certified_forward as certified  # noqa: PLC0415

    return certified


def extended_prior_bounds(certified: Any) -> tuple[FloatArray, FloatArray]:
    """Append the literal hyperon priors to the certified seven-D box."""

    low = np.concatenate(
        [np.asarray(certified.THETA_LOW, dtype=np.float64), HYPERON_LOW]
    )
    high = np.concatenate(
        [np.asarray(certified.THETA_HIGH, dtype=np.float64), HYPERON_HIGH]
    )
    if low.shape != (9,) or high.shape != (9,) or np.any(low >= high):
        raise RuntimeError("Certified plus hyperon prior bounds are malformed.")
    return low, high


def sample_prior(
    count: int,
    low: ArrayLike,
    high: ArrayLike,
    *,
    seed: int,
) -> FloatArray:
    """Draw deterministic uniform samples from the full nine-D prior box."""

    if count < 1:
        raise ValueError("count must be positive.")
    lower = np.asarray(low, dtype=np.float64)
    upper = np.asarray(high, dtype=np.float64)
    if lower.shape != (9,) or upper.shape != (9,) or np.any(lower >= upper):
        raise ValueError("low/high must be increasing nine-element arrays.")
    generator = np.random.default_rng(seed)
    return lower + generator.random((count, 9)) * (upper - lower)


def load_theta_file(path: Path) -> FloatArray:
    """Load a ``theta`` table from NPZ, NPY, or whitespace-delimited text."""

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "theta" not in data.files:
                raise KeyError(f"{path} has no 'theta' array.")
            theta = np.asarray(data["theta"], dtype=np.float64)
    elif suffix == ".npy":
        theta = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    else:
        theta = np.asarray(np.loadtxt(path), dtype=np.float64)
    theta = np.atleast_2d(theta)
    if theta.ndim != 2 or theta.shape[1] != 9:
        raise ValueError(f"Expected theta shape (N, 9), got {theta.shape}.")
    return theta


def compute_certified_nmp(
    theta: FloatArray,
    certified: Any,
    *,
    chunk_size: int,
) -> tuple[FloatArray, float]:
    """Evaluate the exact certified nuclear observable forward in chunks."""

    import jax.numpy as jnp  # noqa: PLC0415

    output = np.full((len(theta), 7), np.nan, dtype=np.float64)
    started = time.perf_counter()
    for start in range(0, len(theta), chunk_size):
        stop = min(start + chunk_size, len(theta))
        subset = theta[start:stop]
        finite = np.all(np.isfinite(subset), axis=1)
        if not finite.any():
            continue
        positions = np.flatnonzero(finite)
        selected = subset[positions]
        values = np.asarray(
            certified.nmp_batch(
                jnp.asarray(selected[:, :6]),
                jnp.asarray(selected[:, 6]),
            ),
            dtype=np.float64,
        )
        output[start + positions, 0] = selected[:, 6]
        output[start + positions, 1:] = values
    return output, time.perf_counter() - started


def _selected_spot_indices(valid: BoolArray, count: int) -> NDArray[np.int64]:
    """Choose deterministic low/middle/high valid rows for direct replay."""

    indices = np.flatnonzero(valid)
    if count <= 0 or len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    count = min(count, len(indices))
    positions = np.linspace(0, len(indices) - 1, count, dtype=int)
    return indices[positions].astype(np.int64)


def run_spot_checks(
    theta: FloatArray,
    nuclear: FloatArray,
    radius_grid: FloatArray,
    lambda_grid: FloatArray,
    maximum_mass: FloatArray,
    valid: BoolArray,
    *,
    count: int,
    prior_low: FloatArray,
    prior_high: FloatArray,
    certified: Any,
) -> dict[str, NDArray[Any]]:
    """Replay selected rows serially and compare against stored bank values."""

    import jax.numpy as jnp  # noqa: PLC0415

    indices = _selected_spot_indices(valid, count)
    delta_mass = np.full(len(indices), np.nan)
    delta_radius = np.full(len(indices), np.nan)
    relative_lambda = np.full(len(indices), np.nan)
    delta_nuclear = np.full(len(indices), np.nan)
    passed = np.zeros(len(indices), dtype=bool)
    for output_index, row_index in enumerate(indices):
        direct = solve_bank_row(
            int(row_index),
            theta[row_index],
            prior_low=prior_low,
            prior_high=prior_high,
        )
        direct_nmp = np.concatenate(
            [
                [theta[row_index, 6]],
                np.asarray(
                    certified.derived_one(
                        jnp.asarray(theta[row_index, :6]),
                        float(theta[row_index, 6]),
                    ),
                    dtype=np.float64,
                ),
            ]
        )
        radius_support_equal = np.array_equal(
            np.isfinite(direct.radius_grid),
            np.isfinite(radius_grid[row_index]),
        )
        lambda_support_equal = np.array_equal(
            np.isfinite(direct.lambda_grid),
            np.isfinite(lambda_grid[row_index]),
        )
        radius_mask = np.isfinite(direct.radius_grid) & np.isfinite(
            radius_grid[row_index]
        )
        lambda_mask = np.isfinite(direct.lambda_grid) & np.isfinite(
            lambda_grid[row_index]
        )
        delta_mass[output_index] = abs(
            direct.maximum_mass - maximum_mass[row_index]
        )
        delta_radius[output_index] = (
            float(
                np.max(
                    np.abs(
                        direct.radius_grid[radius_mask]
                        - radius_grid[row_index, radius_mask]
                    )
                )
            )
            if radius_mask.any()
            else np.inf
        )
        relative_lambda[output_index] = (
            float(
                np.max(
                    np.abs(
                        direct.lambda_grid[lambda_mask]
                        / lambda_grid[row_index, lambda_mask]
                        - 1.0
                    )
                )
            )
            if lambda_mask.any()
            else np.inf
        )
        delta_nuclear[output_index] = float(
            np.max(np.abs(direct_nmp - nuclear[row_index]))
        )
        passed[output_index] = bool(
            direct.failure_code == FAIL_VALID
            and radius_support_equal
            and lambda_support_equal
            and delta_mass[output_index] <= 1.0e-10
            and delta_radius[output_index] <= 1.0e-10
            and relative_lambda[output_index] <= 1.0e-10
            # K0 is a second finite difference.  Certified scalar and vmap
            # execution can differ by a few micro-MeV through operation order.
            and delta_nuclear[output_index] <= 1.0e-5
        )
        print(
            f"[spot] row={row_index} pass={passed[output_index]} "
            f"dMmax={delta_mass[output_index]:.3e} "
            f"max_dR={delta_radius[output_index]:.3e} "
            f"max_rel_dLambda={relative_lambda[output_index]:.3e} "
            f"max_dX={delta_nuclear[output_index]:.3e}",
            flush=True,
        )
    return {
        "spotcheck_indices": indices,
        "spotcheck_delta_Mmax": delta_mass,
        "spotcheck_max_delta_Rg": delta_radius,
        "spotcheck_max_relative_delta_Lg": relative_lambda,
        "spotcheck_max_delta_X": delta_nuclear,
        "spotcheck_pass": passed,
    }


def _atomic_savez(path: Path, payload: dict[str, Any]) -> None:
    """Write an NPZ through a same-directory temporary and atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_bank(
    theta: ArrayLike,
    output_path: Path,
    *,
    workers: int,
    chunk_size: int,
    seed: int,
    spot_check_count: int = 0,
) -> dict[str, Any]:
    """Generate, validate, save, and return one DDB-hyperon bank payload."""

    samples = np.asarray(theta, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 9 or len(samples) == 0:
        raise ValueError(f"Expected a nonempty theta table with shape (N, 9), got {samples.shape}.")
    if workers == 0 or workers < -1:
        raise ValueError("workers must be -1 or a positive integer.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    certified = load_certified_forward()
    prior_low, prior_high = extended_prior_bounds(certified)
    started = time.perf_counter()
    nuclear, nuclear_seconds = compute_certified_nmp(
        samples,
        certified,
        chunk_size=chunk_size,
    )

    count = len(samples)
    radius_grid = np.full((count, len(MASS_GRID)), np.nan, dtype=np.float64)
    lambda_grid = np.full_like(radius_grid, np.nan)
    maximum_mass = np.full(count, np.nan, dtype=np.float64)
    radius_14 = np.full(count, np.nan, dtype=np.float64)
    failure_code = np.full(count, FAIL_INTERNAL, dtype=np.uint8)
    # Fixed-width bytes avoid the 4x memory penalty of NumPy Unicode at bank
    # scale (160 MB rather than 960 MB per million rows for this diagnostic).
    failure_reason = np.full(
        count,
        b"worker result missing",
        dtype=FAILURE_REASON_DTYPE,
    )
    row_seconds = np.zeros(count, dtype=np.float64)
    eos_seconds = np.zeros(count, dtype=np.float64)
    tov_seconds = np.zeros(count, dtype=np.float64)

    physics_started = time.perf_counter()
    completed = 0
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(
            n_jobs=workers,
            batch_size=1,
            pre_dispatch="2*n_jobs",
        ) as parallel:
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                results = parallel(
                    delayed(solve_bank_row)(
                        index,
                        samples[index],
                        prior_low=prior_low,
                        prior_high=prior_high,
                    )
                    for index in range(start, stop)
                )
                for result in results:
                    index = result.index
                    radius_grid[index] = result.radius_grid
                    lambda_grid[index] = result.lambda_grid
                    maximum_mass[index] = result.maximum_mass
                    radius_14[index] = result.radius_14
                    failure_code[index] = result.failure_code
                    failure_reason[index] = result.failure_reason.encode(
                        "utf-8", errors="replace"
                    )[:160]
                    row_seconds[index] = result.row_seconds
                    eos_seconds[index] = result.eos_seconds
                    tov_seconds[index] = result.tov_seconds
                completed = stop
                elapsed = time.perf_counter() - physics_started
                current_valid = int(np.count_nonzero(failure_code[:stop] == FAIL_VALID))
                print(
                    f"[bank] {completed}/{count} rows; solver-valid={current_valid}; "
                    f"{completed / max(elapsed, 1.0e-12):.3f} rows/s",
                    flush=True,
                )
    physics_seconds = time.perf_counter() - physics_started

    nmp_finite = np.all(np.isfinite(nuclear), axis=1)
    nmp_failures = (failure_code == FAIL_VALID) & ~nmp_finite
    failure_code[nmp_failures] = FAIL_NMP
    failure_reason[nmp_failures] = (
        b"certified nuclear forward returned nonfinite X"
    )
    valid = (
        (failure_code == FAIL_VALID)
        & nmp_finite
        & np.isfinite(maximum_mass)
        & np.isfinite(radius_grid).any(axis=1)
        & np.isfinite(lambda_grid).any(axis=1)
    )
    inconsistent = (failure_code == FAIL_VALID) & ~valid
    failure_code[inconsistent] = FAIL_INTERNAL
    failure_reason[inconsistent] = (
        b"row passed worker but failed final schema audit"
    )

    counts = {
        label: int(np.count_nonzero(failure_code == code))
        for code, label in FAILURE_LABELS.items()
    }
    generation_seconds = time.perf_counter() - started
    spot = run_spot_checks(
        samples,
        nuclear,
        radius_grid,
        lambda_grid,
        maximum_mass,
        valid,
        count=spot_check_count,
        prior_low=prior_low,
        prior_high=prior_high,
        certified=certified,
    )
    total_seconds = time.perf_counter() - started
    resolved_workers = os.cpu_count() if workers == -1 else workers
    payload: dict[str, Any] = {
        # Certified consumer schema.
        "theta": samples,
        "X": nuclear,
        "MG": MASS_GRID,
        "Rg": radius_grid,
        "Lg": lambda_grid,
        "MM": maximum_mass,
        "R14": radius_14,
        "OBS": np.asarray(certified.OBS_MU, dtype=np.float64),
        "SIG": np.asarray(certified.OBS_SIG, dtype=np.float64),
        # Backward-compatible diagnostics and provenance.
        "valid": valid,
        "failure_code": failure_code,
        "failure_reason": failure_reason,
        "row_seconds": row_seconds,
        "eos_seconds": eos_seconds,
        "tov_seconds": tov_seconds,
        "theta_names": THETA_NAMES,
        "X_names": X_NAMES,
        "THETA_LOW": prior_low,
        "THETA_HIGH": prior_high,
        "density_grid": CORE_DENSITY_GRID,
        "tov_h": np.asarray(TOV_STEP),
        "workers": np.asarray(resolved_workers),
        "seed": np.asarray(seed),
        "n_rows": np.asarray(count),
        "n_valid": np.asarray(int(valid.sum())),
        "valid_fraction": np.asarray(float(valid.mean())),
        "failure_counts_json": np.asarray(json.dumps(counts, sort_keys=True)),
        "nmp_seconds": np.asarray(nuclear_seconds),
        "physics_seconds": np.asarray(physics_seconds),
        "generation_seconds": np.asarray(generation_seconds),
        "total_seconds_with_spotchecks": np.asarray(total_seconds),
        "rows_per_second": np.asarray(count / max(generation_seconds, 1.0e-12)),
        "physics_rows_per_second": np.asarray(
            count / max(physics_seconds, 1.0e-12)
        ),
        "schema_version": np.asarray("ddbhy-bank-v1"),
        "phase2e_validation": np.asarray(
            "provisional-pass; 20-corner project regression bank; posterior replay open-upgrade"
        ),
        **spot,
    }
    missing = CERTIFIED_CONSUMER_KEYS.difference(payload)
    if missing:
        raise RuntimeError(f"Internal schema error; missing keys: {sorted(missing)}")
    _atomic_savez(output_path, payload)
    print(
        f"[bank] saved {output_path} | valid={int(valid.sum())}/{count} "
        f"({100.0 * valid.mean():.2f}%) | "
        f"throughput={count / max(generation_seconds, 1.0e-12):.3f} rows/s | "
        f"failures={json.dumps(counts, sort_keys=True)}",
        flush=True,
    )
    return payload


def _workers_from_environment() -> int:
    """Read the certified-style worker override from ``TOV_NW``."""

    raw = os.environ.get("TOV_NW")
    if raw is None:
        return min(20, os.cpu_count() or 1)
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError(f"TOV_NW must be an integer, got {raw!r}.") from error
    if workers == 0 or workers < -1:
        raise ValueError("TOV_NW must be -1 or a positive integer.")
    return workers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--n", type=int, help="Draw N uniform-prior rows.")
    source.add_argument(
        "--theta-file",
        type=Path,
        help="NPZ ('theta' key), NPY, or text table with shape (N, 9).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "ddbhy_bank.npz",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="joblib workers; default reads TOV_NW, then min(20, CPU count).",
    )
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument(
        "--spot-check",
        type=int,
        default=0,
        metavar="K",
        help="Replay K deterministic valid rows serially before saving.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    arguments = _parser().parse_args(argv)
    certified = load_certified_forward()
    prior_low, prior_high = extended_prior_bounds(certified)
    if arguments.theta_file is not None:
        theta = load_theta_file(arguments.theta_file.resolve())
    else:
        theta = sample_prior(
            arguments.n,
            prior_low,
            prior_high,
            seed=arguments.seed,
        )
    workers = (
        _workers_from_environment()
        if arguments.workers is None
        else arguments.workers
    )
    generate_bank(
        theta,
        arguments.output.resolve(),
        workers=workers,
        chunk_size=arguments.chunk_size,
        seed=arguments.seed,
        spot_check_count=arguments.spot_check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
