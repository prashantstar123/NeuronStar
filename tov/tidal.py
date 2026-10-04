"""Validated tidal-deformability extension of the NUMBA TOV solver."""

from __future__ import annotations

import numpy as np
from numba import njit

from . import solver


@njit(fastmath=True, error_model="numpy")
def tov_single_lambda(p_c_geom, p_tab, eps_tab, r_init, h, n_steps):
    PI = np.pi
    p_surface = max(1e-20, p_tab[0] * 1.001)
    eps_c = solver.eps_of_p_lin(p_c_geom, p_tab, eps_tab)
    p = p_c_geom - (2.0 * PI / 3.0) * (eps_c + p_c_geom) * (
        eps_c + 3.0 * p_c_geom
    ) * r_init**2
    m = (4.0 * PI / 3.0) * eps_c * r_init**3
    r = r_init
    y = 2.0
    R_surf = r_init
    M_surf = m
    y_surf = 2.0
    found = False
    for _ in range(n_steps):
        if found:
            break
        k1p, k1m, k1y = solver.tov_rhs(r, p, m, y, p_tab, eps_tab)
        k2p, k2m, k2y = solver.tov_rhs(
            r + h / 2,
            p + h / 2 * k1p,
            m + h / 2 * k1m,
            y + h / 2 * k1y,
            p_tab,
            eps_tab,
        )
        k3p, k3m, k3y = solver.tov_rhs(
            r + h / 2,
            p + h / 2 * k2p,
            m + h / 2 * k2m,
            y + h / 2 * k2y,
            p_tab,
            eps_tab,
        )
        k4p, k4m, k4y = solver.tov_rhs(
            r + h,
            p + h * k3p,
            m + h * k3m,
            y + h * k3y,
            p_tab,
            eps_tab,
        )
        p_new = p + h / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        m_new = m + h / 6.0 * (k1m + 2 * k2m + 2 * k3m + k4m)
        y_new = y + h / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
        if p_new < p_surface:
            fraction = (p - p_surface) / max(p - p_new, 1e-30)
            fraction = min(max(fraction, 0.0), 1.0)
            R_surf = r + h * fraction
            M_surf = m + fraction * (m_new - m)
            y_surf = y + fraction * (y_new - y)
            found = True
        else:
            r += h
            p = p_new
            m = m_new
            y = y_new
    if not found:
        return np.nan, np.nan, np.nan
    compactness = 2.0 * M_surf / R_surf
    if compactness > 0.99:
        compactness = 0.99
    f1 = 2.0 - y_surf + (y_surf - 1.0) * compactness
    numerator = (
        (1.0 / 20.0)
        * compactness**5
        * (1.0 - compactness) ** 2
        * f1
    )
    denominator = (
        compactness
        * (6.0 - 3.0 * y_surf + 1.5 * compactness * (5.0 * y_surf - 8.0))
        + 0.25
        * compactness**3
        * (
            26.0
            - 22.0 * y_surf
            + compactness * (3.0 * y_surf - 2.0)
            + compactness**2 * (1.0 + y_surf)
        )
        + 3.0 * (1.0 - compactness) ** 2 * f1 * np.log1p(-compactness)
    )
    tidal_lambda = (
        (2.0 / 3.0)
        * (numerator / denominator)
        * (R_surf / max(M_surf, 1e-10)) ** 5
    )
    return M_surf, R_surf * solver.KM_PER_MSOL, tidal_lambda


@njit(fastmath=True, error_model="numpy")
def mrl_curve(p_c_arr, p_tab, eps_tab, h):
    n = len(p_c_arr)
    mass = np.empty(n)
    radius = np.empty(n)
    tidal_lambda = np.empty(n)
    for index in range(n):
        mass[index], radius[index], tidal_lambda[index] = tov_single_lambda(
            p_c_arr[index], p_tab, eps_tab, 1e-4, h, solver.N_TOV_STEPS
        )
    return mass, radius, tidal_lambda


_mrl_curve = mrl_curve
