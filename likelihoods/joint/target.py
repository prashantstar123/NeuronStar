"""Single shared likelihood target for all inference engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from likelihoods.gw170817 import log_likelihood as gw_log_likelihood
from likelihoods.maximum_mass import log_likelihood as maximum_mass_log_likelihood
from likelihoods.nicer import log_likelihood as nicer_log_likelihood
from likelihoods.nuclear import (
    A1_OBSERVATION,
    A1_SIGMA,
    gaussian_log_likelihood,
    observable_vector,
)
from likelihoods.pqcd import log_likelihood as pqcd_log_likelihood


@dataclass(frozen=True)
class LikelihoodBreakdown:
    nuclear: float
    maximum_mass: float
    nicer: float
    gw170817: float
    pqcd: float

    @property
    def total(self) -> float:
        return float(
            self.nuclear
            + self.maximum_mass
            + self.nicer
            + self.gw170817
            + self.pqcd
        )


@dataclass(frozen=True)
class JointLikelihood:
    """Configuration and evaluation of the corrected full target.

    The class consumes already-computed EOS observables and stellar curves.  It
    contains no EOS or TOV implementation, allowing UltraNest, A-NET and TSNPE
    to call exactly the same target.
    """

    nuclear_observation: np.ndarray = field(
        default_factory=lambda: A1_OBSERVATION.copy()
    )
    nuclear_sigma: np.ndarray = field(default_factory=lambda: A1_SIGMA.copy())
    nicer_interpolators: Sequence[Any] = ()
    gw_kernel: Any | None = None
    gw_chirp_mass: float | None = None
    pqcd_enabled: bool = True
    pqcd_density: float = 1.2
    maximum_mass_threshold: float = 2.0
    maximum_mass_width: float = 0.05

    def evaluate(self, theta, nmp, density, energy, pressure, branch):
        if branch is None:
            return LikelihoodBreakdown(-np.inf, 0.0, 0.0, 0.0, 0.0)
        nuclear = float(
            gaussian_log_likelihood(
                observable_vector(theta, nmp),
                self.nuclear_observation,
                self.nuclear_sigma,
            )
        )
        maximum_mass = maximum_mass_log_likelihood(
            branch.maximum_mass,
            self.maximum_mass_threshold,
            self.maximum_mass_width,
        )
        nicer = (
            nicer_log_likelihood(
                branch.mass,
                branch.radius,
                branch.maximum_mass,
                self.nicer_interpolators,
            )
            if self.nicer_interpolators
            else 0.0
        )
        if (self.gw_kernel is None) != (self.gw_chirp_mass is None):
            raise ValueError("GW kernel and chirp mass must be supplied together")
        gw170817 = (
            gw_log_likelihood(
                branch.mass,
                branch.tidal_lambda,
                branch.maximum_mass,
                self.gw_kernel,
                self.gw_chirp_mass,
            )
            if self.gw_kernel is not None
            else 0.0
        )
        pqcd = (
            pqcd_log_likelihood(
                density, energy, pressure, rho_eval=self.pqcd_density
            )
            if self.pqcd_enabled
            else 0.0
        )
        values = LikelihoodBreakdown(
            nuclear=nuclear,
            maximum_mass=float(maximum_mass),
            nicer=nicer,
            gw170817=gw170817,
            pqcd=pqcd,
        )
        if not np.isfinite(values.total):
            return LikelihoodBreakdown(-np.inf, 0.0, 0.0, 0.0, 0.0)
        return values
