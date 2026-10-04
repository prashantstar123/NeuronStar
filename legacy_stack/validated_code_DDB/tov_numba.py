#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone numba TOV integrator — copied verbatim from
gw_eos_tmnre/scripts/numba_tov_lambda.py (the user's validated numba TOV,
<1% on Lambda_1.4 vs JAX), with the GW/MPA1/Lindblom + bilby coupling stripped.

Generic: give it an EOS table (eps, p in MeV/fm^3) and it returns the M-R-Lambda
curve via RK4 TOV at a grid of central pressures, then Mmax on the stable branch.
"""
import numpy as np
from numba import njit, prange

MEV_FM3_TO_GEOM = 2.88839519e-6     # matches jax_tov.constants
KM_PER_MSOL     = 1.476625
N_TOV_STEPS     = 8000
N_STARS         = 40
P_C_MEV  = np.geomspace(3.0, 600.0, N_STARS)
P_C_GEOM = np.ascontiguousarray(P_C_MEV * MEV_FM3_TO_GEOM)


@njit(fastmath=True, inline="always")
def eps_of_p_lin(p, p_tab, eps_tab):
    n = len(p_tab)
    log_p = np.log(max(p, 1e-40))
    log_pt0 = np.log(p_tab[0]); log_pt1 = np.log(p_tab[n-1])
    if log_p <= log_pt0:
        return eps_tab[0]
    if log_p >= log_pt1:
        return eps_tab[n-1]
    lo = 0; hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if np.log(p_tab[mid]) <= log_p:
            lo = mid
        else:
            hi = mid
    x0 = np.log(p_tab[lo]); x1 = np.log(p_tab[hi])
    y0 = np.log(eps_tab[lo]); y1 = np.log(eps_tab[hi])
    dx = x1 - x0
    if dx <= 0.0:                      # duplicate/non-monotonic pressure rung -> use lower eps
        return eps_tab[lo]
    t = (log_p - x0) / dx
    return np.exp(y0 + t * (y1 - y0))


@njit(fastmath=True, inline="always")
def dedp_of_p(p, p_tab, eps_tab):
    h = max(p * 1e-4, 1e-30)
    e_p = eps_of_p_lin(p + h, p_tab, eps_tab)
    e_m = eps_of_p_lin(max(p - h, p * 0.5), p_tab, eps_tab)
    return (e_p - e_m) / (2.0 * h)


@njit(fastmath=True)
def _tov_rhs(r, p, m, y, p_tab, eps_tab):
    PI = np.pi
    p_safe = max(p, 1e-40)
    eps = eps_of_p_lin(p_safe, p_tab, eps_tab)
    dedp = dedp_of_p(p_safe, p_tab, eps_tab)
    rr = max(r, 1e-12)
    safe_denom = max(rr - 2.0 * m, 1e-20)
    dp_dr = -(eps + p_safe) * (m + 4.0 * PI * rr**3 * p_safe) / (rr * safe_denom)
    dm_dr = 4.0 * PI * rr**2 * eps
    F_val = (rr - 4.0 * PI * rr**3 * (eps - p_safe)) / safe_denom
    Q_val = (4.0 * PI * rr * (5.0 * eps + 9.0 * p_safe + (eps + p_safe) * dedp
                              - 6.0 / (4.0 * PI * rr**2))) / safe_denom \
            - 4.0 * ((m + 4.0 * PI * rr**3 * p_safe) / (rr**2 * (1.0 - 2.0 * m / rr)))**2
    dy_dr = (-y * y - y * F_val - rr * rr * Q_val) / rr
    return dp_dr, dm_dr, dy_dr


@njit(fastmath=True)
def tov_single(p_c_geom, p_tab, eps_tab, r_init=1e-4, h=2e-3, n_steps=N_TOV_STEPS):
    PI = np.pi
    p_surface = max(1e-20, p_tab[0] * 1.001)
    eps_c = eps_of_p_lin(p_c_geom, p_tab, eps_tab)
    p_init = p_c_geom - (2.0 * PI / 3.0) * (eps_c + p_c_geom) * (eps_c + 3.0 * p_c_geom) * r_init**2
    m_init = (4.0 * PI / 3.0) * eps_c * r_init**3
    r = r_init; p = p_init; m = m_init; y = 2.0
    R_surf = r_init; M_surf = m_init; y_surf = 2.0
    found = False
    for _ in range(n_steps):
        if found:
            break
        k1p, k1m, k1y = _tov_rhs(r,       p,           m,           y,           p_tab, eps_tab)
        k2p, k2m, k2y = _tov_rhs(r + h/2, p + h/2*k1p, m + h/2*k1m, y + h/2*k1y, p_tab, eps_tab)
        k3p, k3m, k3y = _tov_rhs(r + h/2, p + h/2*k2p, m + h/2*k2m, y + h/2*k2y, p_tab, eps_tab)
        k4p, k4m, k4y = _tov_rhs(r + h,   p + h*k3p,   m + h*k3m,   y + h*k3y,   p_tab, eps_tab)
        p_new = p + h/6.0*(k1p + 2*k2p + 2*k3p + k4p)
        m_new = m + h/6.0*(k1m + 2*k2m + 2*k3m + k4m)
        y_new = y + h/6.0*(k1y + 2*k2y + 2*k3y + k4y)
        if p_new < p_surface:
            frac = (p - p_surface) / max(p - p_new, 1e-30)
            frac = min(max(frac, 0.0), 1.0)
            R_surf = r + h * frac
            M_surf = m + frac * (m_new - m)
            y_surf = y + frac * (y_new - y)
            found = True
        else:
            r += h; p = p_new; m = m_new; y = y_new
    if not found:
        return np.nan, np.nan
    return M_surf, R_surf * KM_PER_MSOL


@njit(fastmath=True)   # serial: n_pool gives process-level parallelism; parallel=True triggers a numba parfor bug here
def _mr_curve(p_c_arr, p_tab, eps_tab, h):
    n = len(p_c_arr)
    M = np.empty(n); R = np.empty(n)
    for i in range(n):
        m, r = tov_single(p_c_arr[i], p_tab, eps_tab, 1e-4, h)
        M[i] = m; R[i] = r
    return M, R


def mmax_from_eos(eps_mev, p_mev, h=2e-3):
    """EOS table (eps, p in MeV/fm^3) -> M_max on the stable branch (Msun).
    Central pressures are capped at the table max so the TOV never extrapolates
    into the (artificially stiff) edge-clamped region. Returns np.nan if degenerate.
    h : RK4 step size (geom units). 2e-3 = reference; coarser (e.g. 1.2e-2) is
    ~5x faster with Mmax error <1e-3 Msun (validated)."""
    eps_g = np.ascontiguousarray(eps_mev * MEV_FM3_TO_GEOM)
    p_g   = np.ascontiguousarray(p_mev   * MEV_FM3_TO_GEOM)
    pc = P_C_GEOM[P_C_GEOM < 0.999 * p_g[-1]]
    if len(pc) < 3:
        return np.nan
    pc = np.ascontiguousarray(pc)
    M, R = _mr_curve(pc, p_g, eps_g, h)
    valid = np.isfinite(M) & (M > 0)
    if valid.sum() < 3:
        return np.nan
    return float(np.max(M[valid]))
