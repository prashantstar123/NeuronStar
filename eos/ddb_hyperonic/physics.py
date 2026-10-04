"""Cold beta-equilibrated DDB-Lambda-Xi-minus core equation of state.

The implementation follows Malik and Providencia, Phys. Rev. D 106, 063024
(2022). The model source and public interface are described in ``README.md``.
Internally it uses the certified DDB natural-unit convention: momenta and
masses in fm^-1, densities in fm^-3, and energy density and pressure in fm^-4.
Multiplication by the certified ``HBARC_MEV_FM`` converts the latter two to
MeV/fm^3.

At exactly zero hyperon coupling scale the new solver is bypassed completely;
the injected certified nucleonic backend receives the original arguments and
its native return object is returned unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult, least_squares, root_scalar

try:  # Bank-scale acceleration for the nonlinear-equilibrium hot path.
    import numba as _numba
except ImportError:  # pragma: no cover - availability is environment-specific.
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


FloatArray: TypeAlias = NDArray[np.float64]

# These constants are copied from the read-only certified DDB forward
# (validated_code_DDB/ddb_certified_forward.py).  The three hyperonic masses are the
# values approved at the Phase-2c gate.
HBARC_MEV_FM = 197.33
CERTIFIED_NUCLEON_MASS_FM = 4.7583690772
CERTIFIED_ELECTRON_MASS_FM = 2.5896e-3
CERTIFIED_MUON_MASS_FM = 0.53544
APPROVED_LAMBDA_MASS_MEV = 1115.68
APPROVED_XI_MINUS_MASS_MEV = 1321.71
APPROVED_PHI_MASS_MEV = 1019.461


class Species(str, Enum):
    """Fermion labels used in compositions and active sets."""

    NEUTRON = "n"
    PROTON = "p"
    LAMBDA = "lambda"
    XI_MINUS = "xi_minus"
    ELECTRON = "electron"
    MUON = "muon"


class HyperonSpecies(str, Enum):
    """Hyperon labels accepted by threshold routines."""

    LAMBDA = Species.LAMBDA.value
    XI_MINUS = Species.XI_MINUS.value


_BARYONS = (
    Species.NEUTRON,
    Species.PROTON,
    Species.LAMBDA,
    Species.XI_MINUS,
)
_ISOSPIN = {
    Species.NEUTRON: -0.5,
    Species.PROTON: 0.5,
    Species.LAMBDA: 0.0,
    Species.XI_MINUS: -0.5,
}


@dataclass(frozen=True, slots=True)
class DDBParameters:
    """Seven-parameter nucleonic DDB vector.

    The field order mirrors the user's established convention

    ``(a_sigma, a_omega, a_rho, Gamma_sigma0, Gamma_omega0,
    Gamma_rho0, n0)``.

    The existing nucleonic backend remains authoritative for how these values
    are consumed.  This container exists so the hyperonic extension does not
    silently reorder them.
    """

    a_sigma: float
    a_omega: float
    a_rho: float
    gamma_sigma0: float
    gamma_omega0: float
    gamma_rho0: float
    saturation_density: float

    def __post_init__(self) -> None:
        values = np.asarray(self.as_theta(), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("All DDB parameters must be finite.")
        if min(self.a_sigma, self.a_omega, self.a_rho) < 0.0:
            raise ValueError("DDB density-dependence exponents must be nonnegative.")
        if min(self.gamma_sigma0, self.gamma_omega0, self.gamma_rho0) <= 0.0:
            raise ValueError("Saturation-density meson couplings must be positive.")
        if self.saturation_density <= 0.0:
            raise ValueError("saturation_density must be positive.")

    @classmethod
    def from_theta(cls, theta: ArrayLike) -> "DDBParameters":
        """Construct parameters from the established seven-element ordering."""

        values = np.asarray(theta, dtype=np.float64)
        if values.shape != (7,):
            raise ValueError(f"Expected theta with shape (7,), got {values.shape}.")
        return cls(*map(float, values))

    def as_theta(self) -> tuple[float, ...]:
        """Return the parameters in the established seven-element ordering."""

        return (
            self.a_sigma,
            self.a_omega,
            self.a_rho,
            self.gamma_sigma0,
            self.gamma_omega0,
            self.gamma_rho0,
            self.saturation_density,
        )


@dataclass(frozen=True, slots=True)
class RMFMasses:
    """Explicit particle and meson masses in one consistent unit system.

    ``certified_ddb`` is the only supplied numerical constructor.  It copies
    the nucleonic/leptonic/mesonic constants from the user's certified DDB
    forward and uses the explicitly gate-approved hyperon and phi masses.
    """

    neutron: float
    proton: float
    lambda_: float
    xi_minus: float
    electron: float
    muon: float
    sigma: float
    omega: float
    rho: float
    phi: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.neutron,
                self.proton,
                self.lambda_,
                self.xi_minus,
                self.electron,
                self.muon,
                self.sigma,
                self.omega,
                self.rho,
                self.phi,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("All particle and meson masses must be finite and positive.")

    @classmethod
    def certified_ddb(cls) -> "RMFMasses":
        """Construct masses in fm^-1 using the certified DDB constants."""

        return cls(
            neutron=CERTIFIED_NUCLEON_MASS_FM,
            proton=CERTIFIED_NUCLEON_MASS_FM,
            lambda_=APPROVED_LAMBDA_MASS_MEV / HBARC_MEV_FM,
            xi_minus=APPROVED_XI_MINUS_MASS_MEV / HBARC_MEV_FM,
            electron=CERTIFIED_ELECTRON_MASS_FM,
            muon=CERTIFIED_MUON_MASS_FM,
            sigma=550.0 / HBARC_MEV_FM,
            omega=783.0 / HBARC_MEV_FM,
            rho=763.0 / HBARC_MEV_FM,
            phi=APPROVED_PHI_MASS_MEV / HBARC_MEV_FM,
        )


@dataclass(frozen=True, slots=True)
class HyperonCouplings:
    """Hyperon-to-nucleon coupling ratios.

    ``literal_malik_providencia`` constructs the selected SU(6) vector sector.
    The scalar ratios are required inputs because the paper samples them over
    intervals rather than fixing central values.

    ``coupling_scale`` is an implementation/test switch.  At exactly zero it
    removes Lambda, Xi-minus, and phi and delegates to nucleonic DDB.  For any
    nonzero value the species remain physical active-set candidates.
    """

    x_sigma_lambda: float
    x_sigma_xi_minus: float
    x_omega_lambda: float = 2.0 / 3.0
    x_omega_xi_minus: float = 1.0 / 3.0
    x_rho_lambda: float = 1.0
    x_rho_xi_minus: float = 1.0
    x_phi_lambda: float = -np.sqrt(2.0) / 3.0
    x_phi_xi_minus: float = -2.0 * np.sqrt(2.0) / 3.0
    coupling_scale: float = 1.0
    include_lambda: bool = True
    include_xi_minus: bool = True

    def __post_init__(self) -> None:
        ratios = np.asarray(
            (
                self.x_sigma_lambda,
                self.x_sigma_xi_minus,
                self.x_omega_lambda,
                self.x_omega_xi_minus,
                self.x_rho_lambda,
                self.x_rho_xi_minus,
                self.x_phi_lambda,
                self.x_phi_xi_minus,
                self.coupling_scale,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(ratios)):
            raise ValueError("All hyperon coupling ratios and the scale must be finite.")
        if self.coupling_scale < 0.0:
            raise ValueError("coupling_scale must be nonnegative.")

    @classmethod
    def literal_malik_providencia(
        cls,
        *,
        x_sigma_lambda: float,
        x_sigma_xi_minus: float,
    ) -> "HyperonCouplings":
        """Return the literal scalar-ratio/SU(6)-vector coupling setup."""

        return cls(
            x_sigma_lambda=x_sigma_lambda,
            x_sigma_xi_minus=x_sigma_xi_minus,
        )

    @classmethod
    def zero(cls) -> "HyperonCouplings":
        """Return the exact nucleonic-limit switch with all ratios zero.

        The solver interprets this object as removal of hyperon degrees of
        freedom, not as an interacting system of free zero-coupled hyperons.
        """

        return cls(
            x_sigma_lambda=0.0,
            x_sigma_xi_minus=0.0,
            x_omega_lambda=0.0,
            x_omega_xi_minus=0.0,
            x_rho_lambda=0.0,
            x_rho_xi_minus=0.0,
            x_phi_lambda=0.0,
            x_phi_xi_minus=0.0,
            coupling_scale=0.0,
            include_lambda=False,
            include_xi_minus=False,
        )

    @property
    def is_nucleonic_limit(self) -> bool:
        """Whether the API must take the exact nucleonic-delegation branch."""

        return self.coupling_scale == 0.0 or not (
            self.include_lambda or self.include_xi_minus
        )

    def effective_ratio(self, bare_ratio: float) -> float:
        """Apply the diagnostic coupling scale to a bare ratio."""

        return self.coupling_scale * bare_ratio


@dataclass(frozen=True, slots=True)
class SolverOptions:
    """Numerical policy for the nonlinear equilibrium solver."""

    relative_tolerance: float = 1.0e-10
    absolute_tolerance: float = 1.0e-12
    maximum_function_evaluations: int = 2_000
    maximum_active_set_updates: int = 8
    threshold_density_tolerance: float = 1.0e-8
    field_residual_scale: float = 1.0
    density_residual_scale: float = 1.0
    chemical_potential_residual_scale: float = 1.0
    use_numba: bool = True

    def __post_init__(self) -> None:
        if self.relative_tolerance <= 0.0 or self.absolute_tolerance <= 0.0:
            raise ValueError("Solver tolerances must be positive.")
        if self.maximum_function_evaluations < 1:
            raise ValueError("maximum_function_evaluations must be positive.")
        if self.maximum_active_set_updates < 1:
            raise ValueError("maximum_active_set_updates must be positive.")
        if self.threshold_density_tolerance <= 0.0:
            raise ValueError("threshold_density_tolerance must be positive.")
        scales = (
            self.field_residual_scale,
            self.density_residual_scale,
            self.chemical_potential_residual_scale,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError("Residual scales must be finite and positive.")
        if self.use_numba and not NUMBA_AVAILABLE:
            raise RuntimeError("use_numba=True was requested, but numba is unavailable.")


@dataclass(frozen=True, slots=True)
class ActiveSet:
    """Species whose equilibrium equalities are active at one density."""

    electron: bool = True
    muon: bool = False
    lambda_: bool = False
    xi_minus: bool = False

    @property
    def fermions(self) -> tuple[Species, ...]:
        """Return the ordered fermion layout used for solver unknowns."""

        species: list[Species] = [Species.NEUTRON, Species.PROTON]
        if self.electron:
            species.append(Species.ELECTRON)
        if self.muon:
            species.append(Species.MUON)
        if self.lambda_:
            species.append(Species.LAMBDA)
        if self.xi_minus:
            species.append(Species.XI_MINUS)
        return tuple(species)


@dataclass(frozen=True, slots=True)
class MeanFields:
    """Uniform-matter mean fields in the upstream unit convention."""

    sigma: float
    omega: float
    rho: float
    phi: float


@dataclass(frozen=True, slots=True)
class Composition:
    """Number densities for every supported fermion species."""

    neutron: float
    proton: float
    lambda_: float
    xi_minus: float
    electron: float
    muon: float

    @property
    def baryon_density(self) -> float:
        """Return total baryon density."""

        return self.neutron + self.proton + self.lambda_ + self.xi_minus

    @property
    def net_charge_density(self) -> float:
        """Return positive minus negative charge density."""

        return self.proton - self.xi_minus - self.electron - self.muon


@dataclass(frozen=True, slots=True)
class ChemicalPotentials:
    """Chemical potentials for every supported fermion species."""

    neutron: float
    proton: float
    lambda_: float
    xi_minus: float
    electron: float
    muon: float


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    """Convergence evidence attached to a solved hyperonic state."""

    success: bool
    status: int
    message: str
    function_evaluations: int
    active_set_updates: int
    scaled_residual_norm: float
    physical_residual_norm: float
    continued_from_density: float | None = None
    physical_residuals: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MatterState:
    """One converged cold beta-equilibrium core-matter state."""

    baryon_density: float
    fields: MeanFields
    composition: Composition
    chemical_potentials: ChemicalPotentials
    effective_masses: Mapping[Species, float]
    rearrangement_self_energy: float
    energy_density: float
    pressure: float
    active_set: ActiveSet
    diagnostics: SolverDiagnostics


@dataclass(frozen=True, slots=True)
class CouplingSnapshot:
    """Density-dependent couplings and total-density derivatives.

    Both mappings are keyed first by baryon species and then by meson name
    (``sigma``, ``omega``, ``rho``, ``phi``).
    """

    values: Mapping[Species, Mapping[str, float]]
    derivatives: Mapping[Species, Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class HyperonThreshold:
    """A refined zero-density onset for one hyperon species."""

    species: HyperonSpecies
    baryon_density: float
    gap_below: float
    gap_above: float
    bracket: tuple[float, float]
    density_tolerance: float
    converged: bool


@runtime_checkable
class NucleonicDDBBackend(Protocol):
    """Adapter contract for exact reuse of the user's nucleonic DDB code.

    An adapter may translate ``DDBParameters`` to the established function's
    native argument form, but it must not recompute or post-process physics
    outputs.  The zero-hyperon branch returns each backend result unchanged.
    """

    def solve_point(
        self,
        baryon_density: float,
        parameters: DDBParameters,
        *,
        options: SolverOptions,
    ) -> Any:
        """Solve one nucleonic beta-equilibrium state."""

    def solve_grid(
        self,
        baryon_densities: ArrayLike,
        parameters: DDBParameters,
        *,
        options: SolverOptions,
    ) -> Any:
        """Solve a density grid with the backend's native continuation path."""


class EquilibriumConvergenceError(RuntimeError):
    """Raised when a nonlinear solve fails the declared physics gates."""


class _ActiveSetBoundaryTransition(RuntimeError):
    """Request a retry after an active threshold species reaches zero density."""

    def __init__(self, active_set: ActiveSet) -> None:
        super().__init__(f"Retry with active set {active_set}.")
        self.active_set = active_set


class NucleonicBackendRequiredError(RuntimeError):
    """Raised when the exact nucleonic branch lacks its required backend."""


def isoscalar_shape(x: ArrayLike, exponent: float) -> FloatArray:
    """Evaluate ``exp(1 - x**exponent)`` for sigma or omega.

    Phase 2c must handle the exactly constant ``exponent == 0`` case without
    evaluating a singular derivative at zero density.
    """

    values = np.asarray(x, dtype=np.float64)
    if exponent < 0.0 or not np.isfinite(exponent):
        raise ValueError("exponent must be finite and nonnegative.")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("The reduced density x must be finite and nonnegative.")
    if exponent == 0.0:
        return np.ones_like(values, dtype=np.float64)
    return np.asarray(np.exp(1.0 - np.power(values, exponent)), dtype=np.float64)


def isoscalar_shape_derivative(
    baryon_density: ArrayLike,
    *,
    exponent: float,
    saturation_density: float,
) -> FloatArray:
    """Return ``dh/dn_B`` for an isoscalar DDB shape."""

    density = np.asarray(baryon_density, dtype=np.float64)
    if saturation_density <= 0.0 or not np.isfinite(saturation_density):
        raise ValueError("saturation_density must be finite and positive.")
    if exponent < 0.0 or not np.isfinite(exponent):
        raise ValueError("exponent must be finite and nonnegative.")
    if np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("baryon_density must be finite and nonnegative.")
    if exponent == 0.0:
        return np.zeros_like(density, dtype=np.float64)
    if exponent < 1.0 and np.any(density == 0.0):
        raise ValueError(
            "The isoscalar derivative diverges at zero density for 0<a<1; "
            "the DDB model is defined on the positive core-density domain."
        )
    x = density / saturation_density
    derivative = -(exponent / saturation_density) * np.power(
        x, exponent - 1.0
    ) * isoscalar_shape(x, exponent)
    return np.asarray(derivative, dtype=np.float64)


def isovector_shape(x: ArrayLike, exponent: float) -> FloatArray:
    """Evaluate ``exp[-exponent * (x - 1)]`` for rho."""

    values = np.asarray(x, dtype=np.float64)
    if exponent < 0.0 or not np.isfinite(exponent):
        raise ValueError("exponent must be finite and nonnegative.")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("The reduced density x must be finite and nonnegative.")
    return np.asarray(np.exp(-exponent * (values - 1.0)), dtype=np.float64)


def isovector_shape_derivative(
    baryon_density: ArrayLike,
    *,
    exponent: float,
    saturation_density: float,
) -> FloatArray:
    """Return ``dh_rho/dn_B``."""

    density = np.asarray(baryon_density, dtype=np.float64)
    if saturation_density <= 0.0 or not np.isfinite(saturation_density):
        raise ValueError("saturation_density must be finite and positive.")
    if exponent < 0.0 or not np.isfinite(exponent):
        raise ValueError("exponent must be finite and nonnegative.")
    if np.any(~np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("baryon_density must be finite and nonnegative.")
    return np.asarray(
        -(exponent / saturation_density)
        * isovector_shape(density / saturation_density, exponent),
        dtype=np.float64,
    )


def evaluate_couplings(
    baryon_density: float,
    parameters: DDBParameters,
    hyperons: HyperonCouplings,
) -> CouplingSnapshot:
    """Build all baryon-meson couplings and total-density derivatives.

    The implementation applies the scalar ratios to ``Gamma_sigma``,
    the omega and phi ratios to ``Gamma_omega``, and the rho ratios to
    ``Gamma_rho``.  In particular, phi inherits the omega density dependence.
    """

    if not np.isfinite(baryon_density) or baryon_density <= 0.0:
        raise ValueError("baryon_density must be finite and positive.")

    n0 = parameters.saturation_density
    x = baryon_density / n0
    sigma = parameters.gamma_sigma0 * float(isoscalar_shape(x, parameters.a_sigma))
    omega = parameters.gamma_omega0 * float(isoscalar_shape(x, parameters.a_omega))
    rho = parameters.gamma_rho0 * float(isovector_shape(x, parameters.a_rho))
    d_sigma = parameters.gamma_sigma0 * float(
        isoscalar_shape_derivative(
            baryon_density,
            exponent=parameters.a_sigma,
            saturation_density=n0,
        )
    )
    d_omega = parameters.gamma_omega0 * float(
        isoscalar_shape_derivative(
            baryon_density,
            exponent=parameters.a_omega,
            saturation_density=n0,
        )
    )
    d_rho = parameters.gamma_rho0 * float(
        isovector_shape_derivative(
            baryon_density,
            exponent=parameters.a_rho,
            saturation_density=n0,
        )
    )

    zero = {"sigma": 0.0, "omega": 0.0, "rho": 0.0, "phi": 0.0}
    values: dict[Species, dict[str, float]] = {
        Species.NEUTRON: {"sigma": sigma, "omega": omega, "rho": rho, "phi": 0.0},
        Species.PROTON: {"sigma": sigma, "omega": omega, "rho": rho, "phi": 0.0},
        Species.LAMBDA: dict(zero),
        Species.XI_MINUS: dict(zero),
    }
    derivatives: dict[Species, dict[str, float]] = {
        Species.NEUTRON: {
            "sigma": d_sigma,
            "omega": d_omega,
            "rho": d_rho,
            "phi": 0.0,
        },
        Species.PROTON: {
            "sigma": d_sigma,
            "omega": d_omega,
            "rho": d_rho,
            "phi": 0.0,
        },
        Species.LAMBDA: dict(zero),
        Species.XI_MINUS: dict(zero),
    }

    ratio_rows = (
        (
            Species.LAMBDA,
            hyperons.include_lambda,
            hyperons.x_sigma_lambda,
            hyperons.x_omega_lambda,
            hyperons.x_rho_lambda,
            hyperons.x_phi_lambda,
        ),
        (
            Species.XI_MINUS,
            hyperons.include_xi_minus,
            hyperons.x_sigma_xi_minus,
            hyperons.x_omega_xi_minus,
            hyperons.x_rho_xi_minus,
            hyperons.x_phi_xi_minus,
        ),
    )
    for species, included, xs, xw, xr, xp in ratio_rows:
        if not included:
            continue
        ratios = {
            "sigma": hyperons.effective_ratio(xs),
            "omega": hyperons.effective_ratio(xw),
            "rho": hyperons.effective_ratio(xr),
            "phi": hyperons.effective_ratio(xp),
        }
        values[species] = {
            "sigma": ratios["sigma"] * sigma,
            "omega": ratios["omega"] * omega,
            "rho": ratios["rho"] * rho,
            "phi": ratios["phi"] * omega,
        }
        derivatives[species] = {
            "sigma": ratios["sigma"] * d_sigma,
            "omega": ratios["omega"] * d_omega,
            "rho": ratios["rho"] * d_rho,
            # Gate-approved h_phi == h_omega.
            "phi": ratios["phi"] * d_omega,
        }
    return CouplingSnapshot(values=values, derivatives=derivatives)


def number_density_from_fermi_momentum(fermi_momentum: ArrayLike) -> FloatArray:
    """Return the spin-1/2 number density ``k_F**3 / (3*pi**2)``."""

    momentum = np.asarray(fermi_momentum, dtype=np.float64)
    if np.any(~np.isfinite(momentum)) or np.any(momentum < 0.0):
        raise ValueError("Fermi momenta must be finite and nonnegative.")
    return np.asarray(momentum**3 / (3.0 * np.pi**2), dtype=np.float64)


def _fermi_inputs(
    fermi_momentum: ArrayLike, mass: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate and broadcast a Fermi momentum and positive mass."""

    momentum, particle_mass = np.broadcast_arrays(
        np.asarray(fermi_momentum, dtype=np.float64),
        np.asarray(mass, dtype=np.float64),
    )
    if np.any(~np.isfinite(momentum)) or np.any(momentum < 0.0):
        raise ValueError("Fermi momenta must be finite and nonnegative.")
    if np.any(~np.isfinite(particle_mass)) or np.any(particle_mass <= 0.0):
        raise ValueError("Fermion masses must be finite and positive.")
    return momentum, particle_mass, momentum / particle_mass


def scalar_density(fermi_momentum: ArrayLike, effective_mass: ArrayLike) -> FloatArray:
    """Return the zero-temperature scalar density for spin degeneracy two."""

    _, mass, z = _fermi_inputs(fermi_momentum, effective_mass)
    root = np.sqrt(1.0 + z * z)
    bracket = z * root - np.arcsinh(z)
    small = z < 1.0e-3
    if np.any(small):
        z2 = z * z
        series = z**3 * (
            2.0 / 3.0
            + z2
            * (
                -1.0 / 5.0
                + z2 * (3.0 / 28.0 + z2 * (-5.0 / 72.0 + z2 * 35.0 / 704.0))
            )
        )
        bracket = np.where(small, series, bracket)
    return np.asarray(mass**3 * bracket / (2.0 * np.pi**2), dtype=np.float64)


# Numeric species order used only inside the compiled equilibrium kernels:
# n, p, Lambda, Xi-minus, electron, muon.  Keeping it independent of Enum and
# mapping objects removes those Python operations from every optimizer call.
_NUMERIC_SPECIES_CODE = {
    Species.NEUTRON: 0,
    Species.PROTON: 1,
    Species.LAMBDA: 2,
    Species.XI_MINUS: 3,
    Species.ELECTRON: 4,
    Species.MUON: 5,
}
_NUMERIC_BARYONS = (
    Species.NEUTRON,
    Species.PROTON,
    Species.LAMBDA,
    Species.XI_MINUS,
)
_NUMERIC_MESONS = ("sigma", "omega", "rho", "phi")


if NUMBA_AVAILABLE:

    @_numba.njit(cache=True, error_model="numpy")
    def _scalar_density_kernel(momentum: float, mass: float) -> float:
        """Scalar-density scalar kernel, algebraically identical to the audit."""

        z = momentum / mass
        root = np.sqrt(1.0 + z * z)
        if z < 1.0e-3:
            z2 = z * z
            bracket = z**3 * (
                2.0 / 3.0
                + z2
                * (
                    -1.0 / 5.0
                    + z2
                    * (
                        3.0 / 28.0
                        + z2 * (-5.0 / 72.0 + z2 * 35.0 / 704.0)
                    )
                )
            )
        else:
            bracket = z * root - np.arcsinh(z)
        return mass**3 * bracket / (2.0 * np.pi**2)


    @_numba.njit(cache=True, error_model="numpy")
    def _scalar_density_derivatives_kernel(
        momentum: float,
        mass: float,
    ) -> tuple[float, float]:
        """Return analytic scalar-density derivatives with respect to k and m."""

        z = momentum / mass
        root = np.sqrt(1.0 + z * z)
        energy = mass * root
        derivative_momentum = mass * momentum * momentum / (
            np.pi**2 * energy
        )
        if z < 1.0e-3:
            z2 = z * z
            # This is d[m^3 B(k/m)]/dm at fixed k for the same B-series used
            # in _scalar_density_kernel.  Its leading z^3 term cancels.
            derivative_mass = mass * mass * z**5 * (
                2.0 / 5.0
                + z2
                * (-3.0 / 7.0 + z2 * (5.0 / 12.0 - z2 * 35.0 / 88.0))
            ) / (2.0 * np.pi**2)
        else:
            bracket = z * root - np.arcsinh(z)
            derivative_mass = mass * mass * (
                3.0 * bracket - 2.0 * z**3 / root
            ) / (2.0 * np.pi**2)
        return derivative_momentum, derivative_mass


    @_numba.njit(cache=True, error_model="numpy")
    def _equilibrium_residual_kernel(
        unknowns: FloatArray,
        species_codes: NDArray[np.int64],
        baryon_density: float,
        coupling_values: FloatArray,
        coupling_derivatives: FloatArray,
        particle_masses: FloatArray,
        meson_mass_squares: FloatArray,
        isospin: FloatArray,
        residual_scales: FloatArray,
    ) -> FloatArray:
        """Fused scaled equilibrium residual for one fixed active set."""

        momenta = np.zeros(6, dtype=np.float64)
        active_columns = np.full(6, -1, dtype=np.int64)
        for offset in range(species_codes.size):
            code = species_codes[offset]
            momenta[code] = unknowns[4 + offset]
            active_columns[code] = 4 + offset

        densities = np.zeros(6, dtype=np.float64)
        for code in range(6):
            momentum = momenta[code]
            densities[code] = momentum**3 / (3.0 * np.pi**2)

        effective_masses = np.empty(4, dtype=np.float64)
        scalar_densities = np.empty(4, dtype=np.float64)
        for code in range(4):
            effective_mass = (
                particle_masses[code] - coupling_values[code, 0] * unknowns[0]
            )
            effective_masses[code] = effective_mass
            scalar_densities[code] = _scalar_density_kernel(
                momenta[code], effective_mass
            )

        rearrangement = 0.0
        for code in range(4):
            density = densities[code]
            derivative = coupling_derivatives[code]
            rearrangement += (
                -derivative[0] * unknowns[0] * scalar_densities[code]
                + derivative[1] * unknowns[1] * density
                + derivative[2] * isospin[code] * unknowns[2] * density
                + derivative[3] * unknowns[3] * density
            )

        baryon_mu = np.empty(4, dtype=np.float64)
        for code in range(4):
            coupling = coupling_values[code]
            baryon_mu[code] = (
                np.hypot(momenta[code], effective_masses[code])
                + coupling[1] * unknowns[1]
                + coupling[2] * isospin[code] * unknowns[2]
                + coupling[3] * unknowns[3]
                + rearrangement
            )

        electron_active = active_columns[4] >= 0
        if electron_active:
            electron_mu = np.hypot(momenta[4], particle_masses[4])
        else:
            electron_mu = baryon_mu[0] - baryon_mu[1]
        muon_mu = np.hypot(momenta[5], particle_masses[5])

        residuals = np.empty(unknowns.size, dtype=np.float64)
        sigma_source = 0.0
        omega_source = 0.0
        rho_source = 0.0
        phi_source = 0.0
        baryon_sum = 0.0
        for code in range(4):
            coupling = coupling_values[code]
            sigma_source += coupling[0] * scalar_densities[code]
            omega_source += coupling[1] * densities[code]
            rho_source += coupling[2] * isospin[code] * densities[code]
            phi_source += coupling[3] * densities[code]
            baryon_sum += densities[code]
        field_scale = residual_scales[0]
        density_scale = residual_scales[1]
        chemical_scale = residual_scales[2]
        residuals[0] = (
            meson_mass_squares[0] * unknowns[0] - sigma_source
        ) / field_scale
        residuals[1] = (
            meson_mass_squares[1] * unknowns[1] - omega_source
        ) / field_scale
        residuals[2] = (
            meson_mass_squares[2] * unknowns[2] - rho_source
        ) / field_scale
        residuals[3] = (
            meson_mass_squares[3] * unknowns[3] - phi_source
        ) / field_scale
        residuals[4] = (baryon_sum - baryon_density) / density_scale
        residuals[5] = (
            densities[1] - densities[3] - densities[4] - densities[5]
        ) / density_scale

        row = 6
        if electron_active:
            residuals[row] = (
                baryon_mu[0] - baryon_mu[1] - electron_mu
            ) / chemical_scale
            row += 1
        if active_columns[2] >= 0:
            residuals[row] = (baryon_mu[2] - baryon_mu[0]) / chemical_scale
            row += 1
        if active_columns[3] >= 0:
            residuals[row] = (
                baryon_mu[3] - baryon_mu[0] - electron_mu
            ) / chemical_scale
            row += 1
        if active_columns[5] >= 0:
            residuals[row] = (muon_mu - electron_mu) / chemical_scale
        return residuals


    @_numba.njit(cache=True, error_model="numpy")
    def _equilibrium_jacobian_kernel(
        unknowns: FloatArray,
        species_codes: NDArray[np.int64],
        baryon_density: float,
        coupling_values: FloatArray,
        coupling_derivatives: FloatArray,
        particle_masses: FloatArray,
        meson_mass_squares: FloatArray,
        isospin: FloatArray,
        residual_scales: FloatArray,
    ) -> FloatArray:
        """Analytic Jacobian of the fused scaled equilibrium residual."""

        # baryon_density and coupling_derivatives are fixed inputs of this
        # derivative.  The latter enters the common rearrangement self-energy,
        # which cancels identically in every chemical-equilibrium difference.
        _ = baryon_density
        _ = coupling_derivatives
        column_count = unknowns.size
        momenta = np.zeros(6, dtype=np.float64)
        active_columns = np.full(6, -1, dtype=np.int64)
        for offset in range(species_codes.size):
            code = species_codes[offset]
            momenta[code] = unknowns[4 + offset]
            active_columns[code] = 4 + offset

        densities_derivative = np.zeros(6, dtype=np.float64)
        for code in range(6):
            densities_derivative[code] = momenta[code] ** 2 / np.pi**2

        effective_masses = np.empty(4, dtype=np.float64)
        fermi_energies = np.empty(4, dtype=np.float64)
        scalar_derivative_k = np.empty(4, dtype=np.float64)
        scalar_derivative_m = np.empty(4, dtype=np.float64)
        for code in range(4):
            effective_mass = (
                particle_masses[code] - coupling_values[code, 0] * unknowns[0]
            )
            effective_masses[code] = effective_mass
            fermi_energies[code] = np.hypot(momenta[code], effective_mass)
            derivative_k, derivative_m = _scalar_density_derivatives_kernel(
                momenta[code], effective_mass
            )
            scalar_derivative_k[code] = derivative_k
            scalar_derivative_m[code] = derivative_m

        jacobian = np.zeros((column_count, column_count), dtype=np.float64)
        field_scale = residual_scales[0]
        density_scale = residual_scales[1]
        chemical_scale = residual_scales[2]

        jacobian[0, 0] = meson_mass_squares[0]
        jacobian[1, 1] = meson_mass_squares[1]
        jacobian[2, 2] = meson_mass_squares[2]
        jacobian[3, 3] = meson_mass_squares[3]
        for code in range(4):
            coupling = coupling_values[code]
            jacobian[0, 0] += (
                coupling[0] * coupling[0] * scalar_derivative_m[code]
            )
            column = active_columns[code]
            if column >= 0:
                jacobian[0, column] -= coupling[0] * scalar_derivative_k[code]
                jacobian[1, column] -= coupling[1] * densities_derivative[code]
                jacobian[2, column] -= (
                    coupling[2] * isospin[code] * densities_derivative[code]
                )
                jacobian[3, column] -= coupling[3] * densities_derivative[code]
                jacobian[4, column] += densities_derivative[code]
        jacobian[5, active_columns[1]] += densities_derivative[1]
        if active_columns[3] >= 0:
            jacobian[5, active_columns[3]] -= densities_derivative[3]
        if active_columns[4] >= 0:
            jacobian[5, active_columns[4]] -= densities_derivative[4]
        if active_columns[5] >= 0:
            jacobian[5, active_columns[5]] -= densities_derivative[5]

        for column in range(column_count):
            jacobian[0, column] /= field_scale
            jacobian[1, column] /= field_scale
            jacobian[2, column] /= field_scale
            jacobian[3, column] /= field_scale
            jacobian[4, column] /= density_scale
            jacobian[5, column] /= density_scale

        # Derivatives of the baryonic chemical potentials without the common
        # rearrangement term.  The common term has coefficient zero in every
        # beta-equilibrium row, including the electron-free Xi-minus branch.
        baryon_mu_derivative = np.zeros((4, column_count), dtype=np.float64)
        for code in range(4):
            coupling = coupling_values[code]
            energy = fermi_energies[code]
            baryon_mu_derivative[code, 0] = (
                -coupling[0] * effective_masses[code] / energy
            )
            baryon_mu_derivative[code, 1] = coupling[1]
            baryon_mu_derivative[code, 2] = coupling[2] * isospin[code]
            baryon_mu_derivative[code, 3] = coupling[3]
            column = active_columns[code]
            if column >= 0:
                baryon_mu_derivative[code, column] = momenta[code] / energy

        electron_derivative = np.zeros(column_count, dtype=np.float64)
        electron_active = active_columns[4] >= 0
        if electron_active:
            column = active_columns[4]
            electron_energy = np.hypot(momenta[4], particle_masses[4])
            electron_derivative[column] = momenta[4] / electron_energy
        else:
            for column in range(column_count):
                electron_derivative[column] = (
                    baryon_mu_derivative[0, column]
                    - baryon_mu_derivative[1, column]
                )

        row = 6
        if electron_active:
            for column in range(column_count):
                jacobian[row, column] = (
                    baryon_mu_derivative[0, column]
                    - baryon_mu_derivative[1, column]
                    - electron_derivative[column]
                ) / chemical_scale
            row += 1
        if active_columns[2] >= 0:
            for column in range(column_count):
                jacobian[row, column] = (
                    baryon_mu_derivative[2, column]
                    - baryon_mu_derivative[0, column]
                ) / chemical_scale
            row += 1
        if active_columns[3] >= 0:
            for column in range(column_count):
                jacobian[row, column] = (
                    baryon_mu_derivative[3, column]
                    - baryon_mu_derivative[0, column]
                    - electron_derivative[column]
                ) / chemical_scale
            row += 1
        if active_columns[5] >= 0:
            muon_energy = np.hypot(momenta[5], particle_masses[5])
            muon_column = active_columns[5]
            for column in range(column_count):
                value = -electron_derivative[column]
                if column == muon_column:
                    value += momenta[5] / muon_energy
                jacobian[row, column] = value / chemical_scale
        return jacobian

else:  # pragma: no cover - the bank environment requires numba.

    def _equilibrium_residual_kernel(*args: Any, **kwargs: Any) -> FloatArray:
        raise RuntimeError("The compiled equilibrium kernel requires numba.")

    def _equilibrium_jacobian_kernel(*args: Any, **kwargs: Any) -> FloatArray:
        raise RuntimeError("The compiled equilibrium kernel requires numba.")


def _equilibrium_kernel_inputs(
    *,
    baryon_density: float,
    active_set: ActiveSet,
    parameters: DDBParameters,
    masses: RMFMasses,
    hyperons: HyperonCouplings,
    options: SolverOptions,
) -> tuple[
    NDArray[np.int64],
    float,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """Pack immutable numeric inputs once per fixed-active-set solve."""

    snapshot = evaluate_couplings(baryon_density, parameters, hyperons)
    species_codes = np.asarray(
        [_NUMERIC_SPECIES_CODE[species] for species in active_set.fermions],
        dtype=np.int64,
    )
    coupling_values = np.asarray(
        [
            [snapshot.values[species][meson] for meson in _NUMERIC_MESONS]
            for species in _NUMERIC_BARYONS
        ],
        dtype=np.float64,
    )
    coupling_derivatives = np.asarray(
        [
            [snapshot.derivatives[species][meson] for meson in _NUMERIC_MESONS]
            for species in _NUMERIC_BARYONS
        ],
        dtype=np.float64,
    )
    particle_masses = np.asarray(
        [
            masses.neutron,
            masses.proton,
            masses.lambda_,
            masses.xi_minus,
            masses.electron,
            masses.muon,
        ],
        dtype=np.float64,
    )
    meson_mass_squares = np.asarray(
        [masses.sigma**2, masses.omega**2, masses.rho**2, masses.phi**2],
        dtype=np.float64,
    )
    isospin = np.asarray([-0.5, 0.5, 0.0, -0.5], dtype=np.float64)
    residual_scales = np.asarray(
        [
            max(baryon_density, 1.0e-6) * options.field_residual_scale,
            max(baryon_density, 1.0e-6) * options.density_residual_scale,
            max(masses.neutron, 1.0)
            * options.chemical_potential_residual_scale,
        ],
        dtype=np.float64,
    )
    return (
        species_codes,
        float(baryon_density),
        coupling_values,
        coupling_derivatives,
        particle_masses,
        meson_mass_squares,
        isospin,
        residual_scales,
    )


def kinetic_energy_density(fermi_momentum: ArrayLike, mass: ArrayLike) -> FloatArray:
    """Return free-fermion kinetic-plus-rest energy density."""

    _, particle_mass, z = _fermi_inputs(fermi_momentum, mass)
    root = np.sqrt(1.0 + z * z)
    bracket = z * root * (2.0 * z * z + 1.0) - np.arcsinh(z)
    small = z < 1.0e-3
    if np.any(small):
        z2 = z * z
        series = z**3 * (
            8.0 / 3.0
            + z2
            * (
                4.0 / 5.0
                + z2 * (-1.0 / 7.0 + z2 * (1.0 / 18.0 - z2 * 5.0 / 176.0))
            )
        )
        bracket = np.where(small, series, bracket)
    return np.asarray(
        particle_mass**4 * bracket / (8.0 * np.pi**2), dtype=np.float64
    )


def kinetic_pressure(fermi_momentum: ArrayLike, mass: ArrayLike) -> FloatArray:
    """Return free-fermion pressure for spin degeneracy two."""

    _, particle_mass, z = _fermi_inputs(fermi_momentum, mass)
    root = np.sqrt(1.0 + z * z)
    bracket = z * root * (2.0 * z * z - 3.0) + 3.0 * np.arcsinh(z)
    small = z < 1.0e-3
    if np.any(small):
        z2 = z * z
        series = z**5 * (
            8.0 / 5.0
            + z2 * (-4.0 / 7.0 + z2 * (1.0 / 3.0 - z2 * 5.0 / 22.0))
        )
        bracket = np.where(small, series, bracket)
    return np.asarray(
        particle_mass**4 * bracket / (24.0 * np.pi**2), dtype=np.float64
    )


def _composition_density(composition: Composition, species: Species) -> float:
    """Read one species density from a composition."""

    return {
        Species.NEUTRON: composition.neutron,
        Species.PROTON: composition.proton,
        Species.LAMBDA: composition.lambda_,
        Species.XI_MINUS: composition.xi_minus,
        Species.ELECTRON: composition.electron,
        Species.MUON: composition.muon,
    }[species]


def rearrangement_self_energy(
    *,
    fields: MeanFields,
    composition: Composition,
    scalar_densities: Mapping[Species, float],
    couplings: CouplingSnapshot,
) -> float:
    """Evaluate the common hyperonic density-dependent rearrangement term."""

    result = 0.0
    for species in _BARYONS:
        density = _composition_density(composition, species)
        scalar = float(scalar_densities.get(species, 0.0))
        derivative = couplings.derivatives[species]
        result += -derivative["sigma"] * fields.sigma * scalar
        result += derivative["omega"] * fields.omega * density
        result += derivative["rho"] * _ISOSPIN[species] * fields.rho * density
        result += derivative["phi"] * fields.phi * density
    return float(result)


def _species_mass(masses: RMFMasses, species: Species) -> float:
    """Return one particle mass in the configured natural units."""

    return {
        Species.NEUTRON: masses.neutron,
        Species.PROTON: masses.proton,
        Species.LAMBDA: masses.lambda_,
        Species.XI_MINUS: masses.xi_minus,
        Species.ELECTRON: masses.electron,
        Species.MUON: masses.muon,
    }[species]


def _unpack_unknowns(
    unknowns: ArrayLike,
    active_set: ActiveSet,
) -> tuple[MeanFields, Composition, dict[Species, float]]:
    """Decode fields and active Fermi momenta from a solver vector."""

    vector = np.asarray(unknowns, dtype=np.float64)
    expected = 4 + len(active_set.fermions)
    if vector.shape != (expected,):
        raise ValueError(f"Expected {expected} unknowns, got shape {vector.shape}.")
    fields = MeanFields(*map(float, vector[:4]))
    momenta = {species: 0.0 for species in Species}
    for species, momentum in zip(active_set.fermions, vector[4:], strict=True):
        momenta[species] = float(momentum)
    densities = {
        species: float(number_density_from_fermi_momentum(momentum))
        for species, momentum in momenta.items()
    }
    composition = Composition(
        neutron=densities[Species.NEUTRON],
        proton=densities[Species.PROTON],
        lambda_=densities[Species.LAMBDA],
        xi_minus=densities[Species.XI_MINUS],
        electron=densities[Species.ELECTRON],
        muon=densities[Species.MUON],
    )
    return fields, composition, momenta


def _state_quantities(
    unknowns: ArrayLike,
    *,
    baryon_density: float,
    active_set: ActiveSet,
    parameters: DDBParameters,
    masses: RMFMasses,
    hyperons: HyperonCouplings,
) -> tuple[
    MeanFields,
    Composition,
    dict[Species, float],
    CouplingSnapshot,
    dict[Species, float],
    dict[Species, float],
    float,
    ChemicalPotentials,
]:
    """Evaluate all local quantities needed by residuals and state audits."""

    fields, composition, momenta = _unpack_unknowns(unknowns, active_set)
    coupling = evaluate_couplings(baryon_density, parameters, hyperons)
    effective_masses = {
        species: _species_mass(masses, species)
        - coupling.values[species]["sigma"] * fields.sigma
        for species in _BARYONS
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in effective_masses.values()):
        raise ValueError("A candidate solver state has a nonpositive effective baryon mass.")
    scalar_densities = {
        species: float(scalar_density(momenta[species], effective_masses[species]))
        for species in _BARYONS
    }
    rearrangement = rearrangement_self_energy(
        fields=fields,
        composition=composition,
        scalar_densities=scalar_densities,
        couplings=coupling,
    )
    baryon_mu: dict[Species, float] = {}
    for species in _BARYONS:
        row = coupling.values[species]
        fermi_energy = np.hypot(momenta[species], effective_masses[species])
        baryon_mu[species] = float(
            fermi_energy
            + row["omega"] * fields.omega
            + row["rho"] * _ISOSPIN[species] * fields.rho
            + row["phi"] * fields.phi
            + rearrangement
        )
    if active_set.electron:
        electron_mu = float(
            np.hypot(momenta[Species.ELECTRON], masses.electron)
        )
    else:
        # With no electrons, ``mu_e`` denotes the charge chemical potential.
        # It remains fixed by the baryonic weak-equilibrium sector and need not
        # equal the zero-momentum electron energy.  The inactive inequality is
        # m_e - mu_e >= 0 and is checked by the active-set update.
        electron_mu = float(
            baryon_mu[Species.NEUTRON] - baryon_mu[Species.PROTON]
        )
    muon_mu = float(np.hypot(momenta[Species.MUON], masses.muon))
    chemical = ChemicalPotentials(
        neutron=baryon_mu[Species.NEUTRON],
        proton=baryon_mu[Species.PROTON],
        lambda_=baryon_mu[Species.LAMBDA],
        xi_minus=baryon_mu[Species.XI_MINUS],
        electron=electron_mu,
        muon=muon_mu,
    )
    return (
        fields,
        composition,
        momenta,
        coupling,
        effective_masses,
        scalar_densities,
        rearrangement,
        chemical,
    )


def _physical_residuals_from_quantities(
    *,
    baryon_density: float,
    active_set: ActiveSet,
    masses: RMFMasses,
    fields: MeanFields,
    composition: Composition,
    coupling: CouplingSnapshot,
    scalar_densities: Mapping[Species, float],
    chemical: ChemicalPotentials,
) -> tuple[FloatArray, tuple[str, ...]]:
    """Assemble the independent physical audit from reconstructed quantities."""

    sigma_source = sum(
        coupling.values[species]["sigma"] * scalar_densities[species]
        for species in _BARYONS
    )
    omega_source = sum(
        coupling.values[species]["omega"]
        * _composition_density(composition, species)
        for species in _BARYONS
    )
    rho_source = sum(
        coupling.values[species]["rho"]
        * _ISOSPIN[species]
        * _composition_density(composition, species)
        for species in _BARYONS
    )
    phi_source = sum(
        coupling.values[species]["phi"]
        * _composition_density(composition, species)
        for species in _BARYONS
    )
    residuals = [
        masses.sigma**2 * fields.sigma - sigma_source,
        masses.omega**2 * fields.omega - omega_source,
        masses.rho**2 * fields.rho - rho_source,
        masses.phi**2 * fields.phi - phi_source,
        composition.baryon_density - baryon_density,
        composition.net_charge_density,
    ]
    names = [
        "sigma_field",
        "omega_field",
        "rho_field",
        "phi_field",
        "baryon_density",
        "charge_neutrality",
    ]
    if active_set.electron:
        residuals.append(
            chemical.neutron - chemical.proton - chemical.electron
        )
        names.append("beta_np")
    if active_set.lambda_:
        residuals.append(chemical.lambda_ - chemical.neutron)
        names.append("beta_lambda")
    if active_set.xi_minus:
        residuals.append(
            chemical.xi_minus - chemical.neutron - chemical.electron
        )
        names.append("beta_xi_minus")
    if active_set.muon:
        residuals.append(chemical.muon - chemical.electron)
        names.append("beta_muon")
    return np.asarray(residuals, dtype=np.float64), tuple(names)


def _physical_residuals(
    unknowns: ArrayLike,
    *,
    baryon_density: float,
    active_set: ActiveSet,
    parameters: DDBParameters,
    masses: RMFMasses,
    hyperons: HyperonCouplings,
) -> tuple[FloatArray, tuple[str, ...]]:
    """Return unscaled equations and stable diagnostic names."""

    (
        fields,
        composition,
        _,
        coupling,
        _,
        scalar_densities,
        _,
        chemical,
    ) = _state_quantities(
        unknowns,
        baryon_density=baryon_density,
        active_set=active_set,
        parameters=parameters,
        masses=masses,
        hyperons=hyperons,
    )
    return _physical_residuals_from_quantities(
        baryon_density=baryon_density,
        active_set=active_set,
        masses=masses,
        fields=fields,
        composition=composition,
        coupling=coupling,
        scalar_densities=scalar_densities,
        chemical=chemical,
    )


def equilibrium_residuals(
    unknowns: FloatArray,
    *,
    baryon_density: float,
    active_set: ActiveSet,
    parameters: DDBParameters,
    masses: RMFMasses,
    hyperons: HyperonCouplings,
    options: SolverOptions,
) -> FloatArray:
    """Return scaled field, conservation, and beta-equilibrium residuals.

    Unknowns are four mean fields followed by nonnegative Fermi momenta in
    ``active_set.fermions`` order.  For the fully active set the residual vector
    contains four field equations, total density, charge neutrality, nucleonic
    beta equilibrium, Lambda equilibrium, Xi-minus equilibrium, and lepton
    equilibrium.
    """

    if not np.isfinite(baryon_density) or baryon_density <= 0.0:
        raise ValueError("baryon_density must be finite and positive.")
    physical, _ = _physical_residuals(
        unknowns,
        baryon_density=baryon_density,
        active_set=active_set,
        parameters=parameters,
        masses=masses,
        hyperons=hyperons,
    )
    field_scale = max(baryon_density, 1.0e-6) * options.field_residual_scale
    density_scale = max(baryon_density, 1.0e-6) * options.density_residual_scale
    chemical_scale = max(masses.neutron, 1.0) * options.chemical_potential_residual_scale
    scales = np.asarray(
        [field_scale] * 4
        + [density_scale] * 2
        + [chemical_scale] * (physical.size - 6),
        dtype=np.float64,
    )
    if np.any(scales <= 0.0) or np.any(~np.isfinite(scales)):
        raise ValueError("All residual scales must be finite and positive.")
    return np.asarray(physical / scales, dtype=np.float64)


def _initial_unknowns(
    *,
    baryon_density: float,
    active_set: ActiveSet,
    previous_state: MatterState | None,
    parameters: DDBParameters,
    masses: RMFMasses,
) -> FloatArray:
    """Construct a continuation-aware initial vector for one active set."""

    if not np.isfinite(baryon_density) or baryon_density <= 0.0:
        raise ValueError("baryon_density must be finite and positive.")
    k_max = float(np.cbrt(3.0 * np.pi**2 * baryon_density))

    if previous_state is None:
        hyperon_fraction = 0.02 * int(active_set.lambda_) + 0.02 * int(
            active_set.xi_minus
        )
        proton_fraction = 0.10
        neutron_density = max(
            (1.0 - proton_fraction - hyperon_fraction) * baryon_density,
            1.0e-8 * baryon_density,
        )
        proton_density = proton_fraction * baryon_density
        trial_density = {
            Species.NEUTRON: neutron_density,
            Species.PROTON: proton_density,
            Species.LAMBDA: 0.02 * baryon_density if active_set.lambda_ else 0.0,
            Species.XI_MINUS: 0.02 * baryon_density if active_set.xi_minus else 0.0,
            Species.MUON: 0.20 * proton_density if active_set.muon else 0.0,
        }
        trial_density[Species.ELECTRON] = max(
            proton_density
            - trial_density[Species.MUON]
            - trial_density[Species.XI_MINUS],
            1.0e-10 * baryon_density,
        )
        x = baryon_density / parameters.saturation_density
        gamma_sigma = parameters.gamma_sigma0 * float(
            isoscalar_shape(x, parameters.a_sigma)
        )
        gamma_omega = parameters.gamma_omega0 * float(
            isoscalar_shape(x, parameters.a_omega)
        )
        gamma_rho = parameters.gamma_rho0 * float(
            isovector_shape(x, parameters.a_rho)
        )
        sigma = min(
            gamma_sigma * baryon_density / masses.sigma**2,
            0.8 * min(masses.neutron, masses.proton) / gamma_sigma,
        )
        omega = gamma_omega * baryon_density / masses.omega**2
        rho = (
            gamma_rho
            * (trial_density[Species.PROTON] - trial_density[Species.NEUTRON])
            / (2.0 * masses.rho**2)
        )
        fields = [sigma, omega, rho, 0.0]
    else:
        ratio = baryon_density / previous_state.baryon_density
        field_ratio = ratio
        fields = [
            previous_state.fields.sigma * field_ratio,
            previous_state.fields.omega * field_ratio,
            previous_state.fields.rho * field_ratio,
            previous_state.fields.phi * field_ratio,
        ]
        trial_density = {
            species: _composition_density(previous_state.composition, species) * ratio
            for species in Species
        }

        # A finite onset seed avoids a zero derivative with respect to k_F.
        if active_set.electron and trial_density[Species.ELECTRON] == 0.0:
            target = previous_state.chemical_potentials.electron
            k_seed = np.sqrt(max(target * target - masses.electron**2, 0.0))
            trial_density[Species.ELECTRON] = float(
                number_density_from_fermi_momentum(max(k_seed, 0.02 * k_max))
            )
        if active_set.muon and trial_density[Species.MUON] == 0.0:
            target = previous_state.chemical_potentials.electron
            k_seed = np.sqrt(max(target * target - masses.muon**2, 0.0))
            trial_density[Species.MUON] = float(
                number_density_from_fermi_momentum(max(k_seed, 0.02 * k_max))
            )
        if active_set.lambda_ and trial_density[Species.LAMBDA] == 0.0:
            target = previous_state.chemical_potentials.neutron
            mu_zero = previous_state.chemical_potentials.lambda_
            effective_mass = previous_state.effective_masses[Species.LAMBDA]
            kinetic_target = effective_mass + max(target - mu_zero, 0.0)
            k_seed = np.sqrt(max(kinetic_target**2 - effective_mass**2, 0.0))
            trial_density[Species.LAMBDA] = float(
                number_density_from_fermi_momentum(max(k_seed, 0.02 * k_max))
            )
        if active_set.xi_minus and trial_density[Species.XI_MINUS] == 0.0:
            target = (
                previous_state.chemical_potentials.neutron
                + previous_state.chemical_potentials.electron
            )
            mu_zero = previous_state.chemical_potentials.xi_minus
            effective_mass = previous_state.effective_masses[Species.XI_MINUS]
            kinetic_target = effective_mass + max(target - mu_zero, 0.0)
            k_seed = np.sqrt(max(kinetic_target**2 - effective_mass**2, 0.0))
            trial_density[Species.XI_MINUS] = float(
                number_density_from_fermi_momentum(max(k_seed, 0.02 * k_max))
            )

    momenta = [
        float(
            np.cbrt(
                3.0
                * np.pi**2
                * max(float(trial_density.get(species, 0.0)), 0.0)
            )
        )
        for species in active_set.fermions
    ]
    return np.asarray([*fields, *momenta], dtype=np.float64)


def _unknown_bounds(
    *,
    baryon_density: float,
    active_set: ActiveSet,
    parameters: DDBParameters,
    masses: RMFMasses,
    hyperons: HyperonCouplings,
) -> tuple[FloatArray, FloatArray]:
    """Return bounds enforcing nonnegative Fermi momenta and physical fields."""

    coupling = evaluate_couplings(baryon_density, parameters, hyperons)
    scalar_limits = [
        _species_mass(masses, species) / coupling.values[species]["sigma"]
        for species in _BARYONS
        if coupling.values[species]["sigma"] > 0.0
    ]
    sigma_upper = 0.999999 * min(scalar_limits)
    k_upper = float(np.cbrt(3.0 * np.pi**2 * baryon_density)) * (1.0 + 1.0e-10)
    lower = np.asarray(
        [0.0, 0.0, -5.0, -5.0] + [0.0] * len(active_set.fermions),
        dtype=np.float64,
    )
    upper = np.asarray(
        [sigma_upper, 5.0, 5.0, 5.0]
        + [k_upper] * len(active_set.fermions),
        dtype=np.float64,
    )
    return lower, upper


def _state_from_optimizer_result(
    result: OptimizeResult,
    *,
    baryon_density: float,
    active_set: ActiveSet,
    active_set_updates: int,
    parameters: DDBParameters,
    masses: RMFMasses,
    hyperons: HyperonCouplings,
    options: SolverOptions,
    continued_from_density: float | None,
) -> MatterState:
    """Reconstruct and independently audit a candidate optimizer solution."""

    audited_unknowns = np.asarray(result.x, dtype=np.float64).copy()
    if not active_set.lambda_ and not active_set.xi_minus:
        # The phi equation is then exactly homogeneous; remove optimizer-level
        # roundoff so the no-hyperon invariant is represented exactly.
        audited_unknowns[3] = 0.0
    (
        fields,
        composition,
        momenta,
        coupling,
        effective_masses,
        scalar_densities,
        rearrangement,
        chemical,
    ) = _state_quantities(
        audited_unknowns,
        baryon_density=baryon_density,
        active_set=active_set,
        parameters=parameters,
        masses=masses,
        hyperons=hyperons,
    )
    physical, names = _physical_residuals_from_quantities(
        baryon_density=baryon_density,
        active_set=active_set,
        masses=masses,
        fields=fields,
        composition=composition,
        coupling=coupling,
        scalar_densities=scalar_densities,
        chemical=chemical,
    )
    field_scale = max(baryon_density, 1.0e-6) * options.field_residual_scale
    density_scale = max(baryon_density, 1.0e-6) * options.density_residual_scale
    chemical_scale = (
        max(masses.neutron, 1.0) * options.chemical_potential_residual_scale
    )
    scales = np.asarray(
        [field_scale] * 4
        + [density_scale] * 2
        + [chemical_scale] * (physical.size - 6),
        dtype=np.float64,
    )
    scaled = np.asarray(physical / scales, dtype=np.float64)
    scaled_norm = float(np.linalg.norm(scaled, ord=np.inf))

    baryon_energy = sum(
        float(kinetic_energy_density(momenta[species], effective_masses[species]))
        for species in _BARYONS
    )
    lepton_energy = float(
        kinetic_energy_density(momenta[Species.ELECTRON], masses.electron)
        + kinetic_energy_density(momenta[Species.MUON], masses.muon)
    )
    scalar_term = 0.5 * (masses.sigma * fields.sigma) ** 2
    omega_term = 0.5 * (masses.omega * fields.omega) ** 2
    rho_term = 0.5 * (masses.rho * fields.rho) ** 2
    phi_term = 0.5 * (masses.phi * fields.phi) ** 2
    energy_density = float(
        baryon_energy
        + lepton_energy
        + scalar_term
        + omega_term
        + rho_term
        + phi_term
    )

    euler_pressure = float(
        chemical.neutron * composition.neutron
        + chemical.proton * composition.proton
        + chemical.lambda_ * composition.lambda_
        + chemical.xi_minus * composition.xi_minus
        + chemical.electron * composition.electron
        + chemical.muon * composition.muon
        - energy_density
    )
    kinetic_pressure_sum = sum(
        float(kinetic_pressure(momenta[species], effective_masses[species]))
        for species in _BARYONS
    ) + float(
        kinetic_pressure(momenta[Species.ELECTRON], masses.electron)
        + kinetic_pressure(momenta[Species.MUON], masses.muon)
    )
    explicit_pressure = float(
        kinetic_pressure_sum
        - scalar_term
        + omega_term
        + rho_term
        + phi_term
        + baryon_density * rearrangement
    )
    pressure_difference = euler_pressure - explicit_pressure
    physical_map = {name: float(value) for name, value in zip(names, physical, strict=True)}
    physical_map["pressure_euler_minus_explicit"] = pressure_difference

    residual_gate = max(
        1.0e-7,
        100.0 * options.relative_tolerance,
        100.0 * options.absolute_tolerance,
    )
    pressure_scale = max(abs(energy_density), abs(euler_pressure), 1.0e-12)
    pressure_gate = max(1.0e-9, 100.0 * options.relative_tolerance) * pressure_scale
    if not np.isfinite(scaled_norm) or scaled_norm > residual_gate:
        raise EquilibriumConvergenceError(
            f"Residual audit failed at n_B={baryon_density}: "
            f"||r_scaled||_inf={scaled_norm:.3e} > {residual_gate:.3e}."
        )
    if not np.isfinite(pressure_difference) or abs(pressure_difference) > pressure_gate:
        raise EquilibriumConvergenceError(
            f"Pressure audit failed at n_B={baryon_density}: "
            f"P_Euler-P_explicit={pressure_difference:.3e}."
        )
    if min(
        composition.neutron,
        composition.proton,
        composition.lambda_,
        composition.xi_minus,
        composition.electron,
        composition.muon,
    ) < 0.0:
        raise EquilibriumConvergenceError("A converged state has a negative density.")

    diagnostics = SolverDiagnostics(
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        function_evaluations=int(result.nfev),
        active_set_updates=active_set_updates,
        scaled_residual_norm=scaled_norm,
        physical_residual_norm=float(np.linalg.norm(physical, ord=np.inf)),
        continued_from_density=continued_from_density,
        physical_residuals=physical_map,
    )
    return MatterState(
        baryon_density=baryon_density,
        fields=fields,
        composition=composition,
        chemical_potentials=chemical,
        effective_masses=effective_masses,
        rearrangement_self_energy=rearrangement,
        energy_density=energy_density,
        pressure=euler_pressure,
        active_set=active_set,
        diagnostics=diagnostics,
    )


def _solve_active_set(
    *,
    baryon_density: float,
    active_set: ActiveSet,
    active_set_updates: int,
    previous_state: MatterState | None,
    parameters: DDBParameters,
    masses: RMFMasses,
    hyperons: HyperonCouplings,
    options: SolverOptions,
) -> MatterState:
    """Run one bounded SciPy least-squares solve for a fixed active set."""

    initial = _initial_unknowns(
        baryon_density=baryon_density,
        active_set=active_set,
        previous_state=previous_state,
        parameters=parameters,
        masses=masses,
    )
    lower, upper = _unknown_bounds(
        baryon_density=baryon_density,
        active_set=active_set,
        parameters=parameters,
        masses=masses,
        hyperons=hyperons,
    )
    interior = 1.0e-12 * np.maximum(1.0, upper - lower)
    initial = np.minimum(np.maximum(initial, lower + interior), upper - interior)
    if options.use_numba:
        kernel_inputs = _equilibrium_kernel_inputs(
            baryon_density=baryon_density,
            active_set=active_set,
            parameters=parameters,
            masses=masses,
            hyperons=hyperons,
            options=options,
        )
        result = least_squares(
            _equilibrium_residual_kernel,
            initial,
            jac=_equilibrium_jacobian_kernel,
            bounds=(lower, upper),
            args=kernel_inputs,
            xtol=options.relative_tolerance,
            ftol=options.relative_tolerance,
            gtol=options.relative_tolerance,
            max_nfev=options.maximum_function_evaluations,
            x_scale="jac",
        )
    else:
        # Retain the original Python/finite-difference path as an independent
        # numerical reference and as an explicitly selectable debug mode.
        result = least_squares(
            equilibrium_residuals,
            initial,
            bounds=(lower, upper),
            kwargs={
                "baryon_density": baryon_density,
                "active_set": active_set,
                "parameters": parameters,
                "masses": masses,
                "hyperons": hyperons,
                "options": options,
            },
            xtol=options.relative_tolerance,
            ftol=options.relative_tolerance,
            gtol=options.relative_tolerance,
            max_nfev=options.maximum_function_evaluations,
            x_scale="jac",
        )
    if not result.success:
        raise EquilibriumConvergenceError(
            f"Active-set solve failed at n_B={baryon_density}: {result.message}"
        )
    try:
        return _state_from_optimizer_result(
            result,
            baryon_density=baryon_density,
            active_set=active_set,
            active_set_updates=active_set_updates,
            parameters=parameters,
            masses=masses,
            hyperons=hyperons,
            options=options,
            continued_from_density=(
                None if previous_state is None else previous_state.baryon_density
            ),
        )
    except EquilibriumConvergenceError:
        # When an active species becomes thermodynamically disfavored again,
        # bounded least squares pins its Fermi momentum to zero but cannot
        # satisfy the corresponding active equality.  Retry on the neighboring
        # complementarity branch; the next audited solve and inactive gap test
        # decide whether that branch is admissible.
        boundary_active_set = _active_set_after_boundary_exit(
            result.x,
            active_set=active_set,
            options=options,
        )
        if boundary_active_set != active_set:
            raise _ActiveSetBoundaryTransition(boundary_active_set) from None
        raise


def _initial_active_set(
    *,
    baryon_density: float,
    previous_state: MatterState | None,
    hyperons: HyperonCouplings,
) -> ActiveSet:
    """Choose the continuation active set before checking threshold gaps."""

    _ = baryon_density
    if previous_state is None:
        return ActiveSet()
    return ActiveSet(
        electron=previous_state.active_set.electron,
        muon=(
            previous_state.active_set.muon
            and previous_state.active_set.electron
        ),
        lambda_=previous_state.active_set.lambda_ and hyperons.include_lambda,
        xi_minus=(
            previous_state.active_set.xi_minus and hyperons.include_xi_minus
        ),
    )


def _active_set_after_boundary_exit(
    unknowns: ArrayLike,
    *,
    active_set: ActiveSet,
    options: SolverOptions,
) -> ActiveSet:
    """Drop one active threshold species whose trial density reached zero."""

    _, composition, _ = _unpack_unknowns(unknowns, active_set)
    candidates = (
        ("electron", Species.ELECTRON),
        ("muon", Species.MUON),
        ("lambda_", Species.LAMBDA),
        ("xi_minus", Species.XI_MINUS),
    )
    for attribute, species in candidates:
        if (
            getattr(active_set, attribute)
            and _composition_density(composition, species)
            <= options.threshold_density_tolerance
        ):
            values = {
                "electron": active_set.electron,
                "muon": active_set.muon,
                "lambda_": active_set.lambda_,
                "xi_minus": active_set.xi_minus,
            }
            values[attribute] = False
            if attribute == "electron":
                # A charge chemical potential below m_e is necessarily also
                # below m_mu, so both leptons are inactive on this branch.
                values["muon"] = False
            return ActiveSet(**values)
    return active_set


def _updated_active_set(
    state: MatterState,
    hyperons: HyperonCouplings,
    masses: RMFMasses,
    options: SolverOptions,
) -> ActiveSet:
    """Apply complementarity inequalities to add or remove threshold species."""

    tolerance = max(
        1.0e-9,
        10.0
        * state.diagnostics.scaled_residual_norm
        * max(state.chemical_potentials.neutron, 1.0),
    )
    muon_gap = (
        state.chemical_potentials.muon - state.chemical_potentials.electron
    )
    electron_gap = masses.electron - state.chemical_potentials.electron
    lambda_gap = hyperon_threshold_gap(state, HyperonSpecies.LAMBDA)
    xi_gap = hyperon_threshold_gap(state, HyperonSpecies.XI_MINUS)
    # Exits are handled only when a bounded active equality becomes infeasible
    # and reaches zero density in ``_solve_active_set``.  Removing a successfully
    # converged, merely tiny onset density here can cause add/remove cycling.
    # Add one threshold species per nonlinear solve.  Simultaneously activating
    # every negative gap can be wrong because the first new charged species
    # changes the other gaps before their own threshold is reached.
    if not state.active_set.electron and electron_gap < -tolerance:
        return ActiveSet(
            electron=True,
            muon=False,
            lambda_=state.active_set.lambda_ and hyperons.include_lambda,
            xi_minus=state.active_set.xi_minus and hyperons.include_xi_minus,
        )
    if (
        state.active_set.electron
        and not state.active_set.muon
        and muon_gap < -tolerance
    ):
        return ActiveSet(
            electron=state.active_set.electron,
            muon=True,
            lambda_=state.active_set.lambda_ and hyperons.include_lambda,
            xi_minus=state.active_set.xi_minus and hyperons.include_xi_minus,
        )
    if (
        hyperons.include_lambda
        and not state.active_set.lambda_
        and lambda_gap < -tolerance
    ):
        return ActiveSet(
            electron=state.active_set.electron,
            muon=state.active_set.muon,
            lambda_=True,
            xi_minus=state.active_set.xi_minus and hyperons.include_xi_minus,
        )
    if (
        hyperons.include_xi_minus
        and not state.active_set.xi_minus
        and xi_gap < -tolerance
    ):
        return ActiveSet(
            electron=state.active_set.electron,
            muon=state.active_set.muon,
            lambda_=state.active_set.lambda_ and hyperons.include_lambda,
            xi_minus=True,
        )
    return ActiveSet(
        electron=state.active_set.electron,
        muon=state.active_set.muon,
        lambda_=state.active_set.lambda_ and hyperons.include_lambda,
        xi_minus=state.active_set.xi_minus and hyperons.include_xi_minus,
    )


def solve_beta_equilibrium_point(
    baryon_density: float,
    parameters: DDBParameters,
    masses: RMFMasses,
    hyperons: HyperonCouplings,
    *,
    options: SolverOptions | None = None,
    previous_state: MatterState | None = None,
    nucleonic_backend: NucleonicDDBBackend | None = None,
) -> Any:
    """Solve one cold, neutral beta-equilibrium state.

    The zero-coupling branch delegates before invoking any new numerical code.
    For nonzero couplings, the flow iterates bounded solves and active
    sets until all equality and complementarity conditions agree.
    """

    solver_options = options if options is not None else SolverOptions()

    if not np.isfinite(baryon_density) or baryon_density <= 0.0:
        raise ValueError("baryon_density must be finite and positive.")

    if hyperons.is_nucleonic_limit:
        if nucleonic_backend is None:
            raise NucleonicBackendRequiredError(
                "Exact nucleonic recovery requires an injected nucleonic DDB backend."
            )
        return nucleonic_backend.solve_point(
            baryon_density,
            parameters,
            options=solver_options,
        )

    active_set = _initial_active_set(
        baryon_density=baryon_density,
        previous_state=previous_state,
        hyperons=hyperons,
    )
    state: MatterState | None = None
    for update_index in range(solver_options.maximum_active_set_updates):
        try:
            state = _solve_active_set(
                baryon_density=baryon_density,
                active_set=active_set,
                active_set_updates=update_index,
                previous_state=previous_state if state is None else state,
                parameters=parameters,
                masses=masses,
                hyperons=hyperons,
                options=solver_options,
            )
        except _ActiveSetBoundaryTransition as transition:
            active_set = transition.active_set
            continue
        next_active_set = _updated_active_set(
            state,
            hyperons,
            masses,
            solver_options,
        )
        if next_active_set == active_set:
            return state
        active_set = next_active_set

    raise EquilibriumConvergenceError(
        f"Active set did not stabilize at n_B={baryon_density} after "
        f"{solver_options.maximum_active_set_updates} updates."
    )


def build_eos_table(
    baryon_densities: ArrayLike,
    parameters: DDBParameters,
    masses: RMFMasses,
    hyperons: HyperonCouplings,
    *,
    options: SolverOptions | None = None,
    nucleonic_backend: NucleonicDDBBackend | None = None,
) -> Any:
    """Trace beta-equilibrium matter over an increasing density grid.

    In the exact nucleonic limit, the full grid is delegated in one call so the
    established backend retains its own continuation, ordering, and output
    representation.  The hyperonic branch returns a tuple of
    ``MatterState`` objects.
    """

    solver_options = options if options is not None else SolverOptions()

    if hyperons.is_nucleonic_limit:
        if nucleonic_backend is None:
            raise NucleonicBackendRequiredError(
                "Exact nucleonic recovery requires an injected nucleonic DDB backend."
            )
        return nucleonic_backend.solve_grid(
            baryon_densities,
            parameters,
            options=solver_options,
        )

    densities = np.asarray(baryon_densities, dtype=np.float64)
    if densities.ndim != 1 or densities.size == 0:
        raise ValueError("baryon_densities must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(densities)) or np.any(densities <= 0.0):
        raise ValueError("All baryon densities must be finite and positive.")
    if np.any(np.diff(densities) <= 0.0):
        raise ValueError("baryon_densities must be strictly increasing for continuation.")

    states: list[MatterState] = []
    previous: MatterState | None = None
    for density in densities:
        state = solve_beta_equilibrium_point(
            float(density),
            parameters,
            masses,
            hyperons,
            options=solver_options,
            previous_state=previous,
            nucleonic_backend=nucleonic_backend,
        )
        if not isinstance(state, MatterState):  # Defensive in the hyperonic branch.
            raise TypeError("The hyperonic solver must return MatterState objects.")
        states.append(state)
        previous = state
    return tuple(states)


def hyperon_threshold_gap(state: MatterState, species: HyperonSpecies) -> float:
    """Return the inactive-species zero-momentum complementarity gap.

    Lambda uses ``mu_Lambda(k_F=0) - mu_n``.  Xi-minus uses
    ``mu_Xi(k_F=0) - (mu_n + mu_e)``.
    """

    selected = HyperonSpecies(species)
    if selected is HyperonSpecies.LAMBDA:
        density = state.composition.lambda_
        effective_mass = state.effective_masses[Species.LAMBDA]
        momentum = float(np.cbrt(3.0 * np.pi**2 * density))
        kinetic_shift = np.hypot(momentum, effective_mass) - effective_mass
        mu_zero = state.chemical_potentials.lambda_ - kinetic_shift
        target = state.chemical_potentials.neutron
    else:
        density = state.composition.xi_minus
        effective_mass = state.effective_masses[Species.XI_MINUS]
        momentum = float(np.cbrt(3.0 * np.pi**2 * density))
        kinetic_shift = np.hypot(momentum, effective_mass) - effective_mass
        mu_zero = state.chemical_potentials.xi_minus - kinetic_shift
        target = (
            state.chemical_potentials.neutron
            + state.chemical_potentials.electron
        )
    return float(mu_zero - target)


def locate_hyperon_threshold(
    species: HyperonSpecies,
    density_bracket: tuple[float, float],
    parameters: DDBParameters,
    masses: RMFMasses,
    hyperons: HyperonCouplings,
    *,
    options: SolverOptions | None = None,
    nucleonic_backend: NucleonicDDBBackend | None = None,
) -> HyperonThreshold:
    """Refine one sign-changing threshold bracket with a scalar root.

    Each scalar-function evaluation independently solves beta equilibrium.
    The complementarity gap remains well-defined immediately above onset by
    subtracting the active species' Fermi kinetic shift.
    """

    selected = HyperonSpecies(species)
    lower, upper = map(float, density_bracket)
    if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError("density_bracket must contain two finite positive densities.")
    if selected is HyperonSpecies.LAMBDA and not hyperons.include_lambda:
        raise ValueError("Cannot locate a Lambda threshold when Lambda is excluded.")
    if selected is HyperonSpecies.XI_MINUS and not hyperons.include_xi_minus:
        raise ValueError("Cannot locate a Xi-minus threshold when Xi-minus is excluded.")
    if hyperons.is_nucleonic_limit:
        raise ValueError("Hyperon thresholds are undefined in the nucleonic-limit branch.")

    solver_options = options if options is not None else SolverOptions()

    def gap(density: float) -> float:
        state = solve_beta_equilibrium_point(
            float(density),
            parameters,
            masses,
            hyperons,
            options=solver_options,
            nucleonic_backend=nucleonic_backend,
        )
        if not isinstance(state, MatterState):
            raise TypeError("Threshold evaluation requires a hyperonic MatterState.")
        return hyperon_threshold_gap(state, selected)

    gap_lower = gap(lower)
    gap_upper = gap(upper)
    if gap_lower == 0.0:
        root_density = lower
        converged = True
    elif gap_upper == 0.0:
        root_density = upper
        converged = True
    else:
        if np.signbit(gap_lower) == np.signbit(gap_upper):
            raise ValueError(
                f"Threshold bracket does not change sign: gaps are "
                f"{gap_lower:.6e} and {gap_upper:.6e}."
            )
        result = root_scalar(
            gap,
            bracket=(lower, upper),
            method="brentq",
            xtol=solver_options.threshold_density_tolerance,
            rtol=max(solver_options.relative_tolerance, 4.0 * np.finfo(float).eps),
        )
        root_density = float(result.root)
        converged = bool(result.converged)
    offset = max(
        solver_options.threshold_density_tolerance,
        10.0 * np.finfo(float).eps * root_density,
    )
    below_density = max(lower, root_density - offset)
    above_density = min(upper, root_density + offset)
    return HyperonThreshold(
        species=selected,
        baryon_density=root_density,
        gap_below=float(gap(below_density)),
        gap_above=float(gap(above_density)),
        bracket=(lower, upper),
        density_tolerance=solver_options.threshold_density_tolerance,
        converged=converged,
    )


def thermodynamic_consistency_residuals(state: MatterState) -> Mapping[str, float]:
    """Return Euler-pressure, Gibbs-Duhem, and conservation audit residuals."""

    composition = state.composition
    chemical = state.chemical_potentials
    euler_pressure = (
        chemical.neutron * composition.neutron
        + chemical.proton * composition.proton
        + chemical.lambda_ * composition.lambda_
        + chemical.xi_minus * composition.xi_minus
        + chemical.electron * composition.electron
        + chemical.muon * composition.muon
        - state.energy_density
    )
    return {
        "euler_pressure": float(state.pressure - euler_pressure),
        "explicit_pressure": float(
            state.diagnostics.physical_residuals.get(
                "pressure_euler_minus_explicit", np.nan
            )
        ),
        "gibbs_duhem": float(
            state.energy_density
            + state.pressure
            - chemical.neutron * state.baryon_density
        ),
        "baryon_density": float(
            composition.baryon_density - state.baryon_density
        ),
        "charge_neutrality": float(composition.net_charge_density),
        "beta_np": float(
            chemical.neutron - chemical.proton - chemical.electron
        ),
        "beta_muon": float(
            chemical.muon - chemical.electron if state.active_set.muon else 0.0
        ),
        "beta_lambda": float(
            chemical.lambda_ - chemical.neutron
            if state.active_set.lambda_
            else 0.0
        ),
        "beta_xi_minus": float(
            chemical.xi_minus - chemical.neutron - chemical.electron
            if state.active_set.xi_minus
            else 0.0
        ),
    }


class DDBHyperonEOS:
    """Configured interface for DDB-Lambda-Xi-minus core matter."""

    def __init__(
        self,
        parameters: DDBParameters,
        masses: RMFMasses,
        hyperons: HyperonCouplings,
        *,
        options: SolverOptions | None = None,
        nucleonic_backend: NucleonicDDBBackend | None = None,
    ) -> None:
        self.parameters = parameters
        self.masses = masses
        self.hyperons = hyperons
        self.options = options if options is not None else SolverOptions()
        self.nucleonic_backend = nucleonic_backend

    def solve_point(
        self,
        baryon_density: float,
        *,
        previous_state: MatterState | None = None,
    ) -> Any:
        """Solve one density, delegating unchanged in the nucleonic limit."""

        return solve_beta_equilibrium_point(
            baryon_density,
            self.parameters,
            self.masses,
            self.hyperons,
            options=self.options,
            previous_state=previous_state,
            nucleonic_backend=self.nucleonic_backend,
        )

    def build_table(self, baryon_densities: ArrayLike) -> Any:
        """Trace a density grid, delegating unchanged in the nucleonic limit."""

        return build_eos_table(
            baryon_densities,
            self.parameters,
            self.masses,
            self.hyperons,
            options=self.options,
            nucleonic_backend=self.nucleonic_backend,
        )


__all__ = [
    "APPROVED_LAMBDA_MASS_MEV",
    "APPROVED_PHI_MASS_MEV",
    "APPROVED_XI_MINUS_MASS_MEV",
    "ActiveSet",
    "ChemicalPotentials",
    "Composition",
    "CouplingSnapshot",
    "DDBHyperonEOS",
    "DDBParameters",
    "EquilibriumConvergenceError",
    "HyperonCouplings",
    "HyperonSpecies",
    "HyperonThreshold",
    "HBARC_MEV_FM",
    "MatterState",
    "MeanFields",
    "NUMBA_AVAILABLE",
    "NucleonicBackendRequiredError",
    "NucleonicDDBBackend",
    "RMFMasses",
    "SolverDiagnostics",
    "SolverOptions",
    "Species",
    "build_eos_table",
    "equilibrium_residuals",
    "evaluate_couplings",
    "hyperon_threshold_gap",
    "isoscalar_shape",
    "isoscalar_shape_derivative",
    "isovector_shape",
    "isovector_shape_derivative",
    "kinetic_energy_density",
    "kinetic_pressure",
    "locate_hyperon_threshold",
    "number_density_from_fermi_momentum",
    "rearrangement_self_energy",
    "scalar_density",
    "solve_beta_equilibrium_point",
    "thermodynamic_consistency_residuals",
]
