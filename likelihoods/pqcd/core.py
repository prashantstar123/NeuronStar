"""Numerical pQCD constraint of Komoltsev, Gorda and Kurkela (arXiv:2204.11877).

The formulas follow their QCD likelihood code, "QCD likelihood function", Zenodo record 7781233
(doi:10.5281/zenodo.7781233), licensed CC BY 4.0; the same code is used by the InferenceWorkflow
module of CompactObject.
"""

from __future__ import annotations

from numba import njit
from numpy import log, pi

GEV3_TO_FM3 = 1.0e3 / 1.9732705**3


@njit
def pnlo(alpha):
    return 1.0 - 0.637 * alpha


@njit
def pnnlo(alpha, scale):
    return -alpha**2 * (-1.831 + 0.304 * log(alpha)) + alpha**2 * (
        -2.706 - 0.912 * log(scale)
    )


@njit
def pn3lo(alpha):
    return 0.484816 * alpha**3


@njit
def alpha_s(mu, scale):
    numerator = 4 * pi * (
        1.0
        - (64.0 * log(log(0.777632 * mu**2 * scale**2)))
        / (81.0 * log(0.777632 * mu**2 * scale**2))
    )
    denominator = 9.0 * log(0.777632 * mu**2 * scale**2)
    return numerator / denominator


@njit
def das_dmu(mu, scale):
    numerator = (
        -2.20644
        - 2.79253 * log(0.777632 * mu**2 * scale**2)
        + 4.41288 * log(log(0.777632 * mu**2 * scale**2))
    )
    denominator = mu * (log(0.777632 * mu**2 * scale**2)) ** 3
    return numerator / denominator


@njit
def dp_das(alpha, scale):
    return -0.637 + alpha * (
        -2.054 - 0.608 * log(alpha) - 1.824 * log(scale)
    ) + 1.45445 * alpha**2


@njit
def pfd(mu):
    return mu**4 / (108 * pi**2)


@njit
def dpfd(mu):
    return mu**3 / (27 * pi**2)


@njit
def pressure(mu, scale):
    alpha = alpha_s(mu, scale)
    return (
        pnlo(alpha) + pnnlo(alpha, scale) + pn3lo(alpha)
    ) * pfd(mu) * GEV3_TO_FM3


@njit
def number_density(mu, scale):
    alpha = alpha_s(mu, scale)
    p_alpha = pnlo(alpha) + pnnlo(alpha, scale) + pn3lo(alpha)
    return (
        dp_das(alpha, scale) * das_dmu(mu, scale) * pfd(mu)
        + p_alpha * dpfd(mu)
    ) * GEV3_TO_FM3


@njit
def constraints(scale, energy0, pressure0, density0, mu_qcd=2.6, sound_speed2=1):
    mu0 = (energy0 + pressure0) / density0
    pressure_qcd = pressure(mu_qcd, scale)
    density_qcd = number_density(mu_qcd, scale)
    delta_pressure = pressure_qcd - pressure0
    pressure_min = (
        sound_speed2
        / (1.0 + sound_speed2)
        * (mu_qcd * (mu_qcd / mu0) ** (1.0 / sound_speed2) - mu0)
        * density0
    )
    pressure_max = (
        sound_speed2
        / (1.0 + sound_speed2)
        * (mu_qcd - mu0 * (mu0 / mu_qcd) ** (1.0 / sound_speed2))
        * density_qcd
    )
    density_max = density_qcd * (mu0 / mu_qcd) ** (1.0 / sound_speed2)
    return (pressure_min < delta_pressure < pressure_max) and (
        density0 < density_max
    )
