#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GW170817 marginalized tidal likelihood along an EOS Lambda(M) curve.

Standalone module for the DDB live-TSNPE astro extension. Implements the
(Mc, q, Lambda1, Lambda2) KDE-marginalization likelihood used by Cartaxo 2026
and the user's feasibility paper (see
~/CompactObject/InferenceWorkflow/Likelihood.py:TidalLikihood_kernel for the
reference recipe).

JAX env MUST be pinned to CPU + x64 BEFORE importing jax / the validated
forward, otherwise JAX grabs the GPU and dies on a CuDNN mismatch. The pin is
done at import time below so callers do not have to remember it.

Public API
----------
    load_gw_kde(path, subsample=4000) -> (kernelGW, Mc_obs)
    logL_gw(M_curve, L_curve, Mmax, kernelGW, Mc_obs, nq=20, qmin=0.7) -> float

Optional (only needs the validated DDB forward, for end-to-end smoke):
    mrl_one(eps_mev, P_mev, h=2e-3) -> (M, R, Lambda, Mmax)
    mrl_from_theta(theta7) -> (M, R, Lambda, Mmax)
"""
import os
# --- JAX env pin: MUST happen before any jax import (see module docstring) ---
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("CUDA_ROOT", "/tmp")

import numpy as np
import h5py
from scipy.stats import gaussian_kde
try:
    from scipy.integrate import trapezoid as _trapz
except ImportError:  # older scipy
    _trapz = np.trapz

# GW170817 host-galaxy (NGC 4993) redshift used by LVC for source-frame masses.
Z_GW170817 = 0.0099

# HDF5 field names CONFIRMED by inspecting the file (structured/record array):
#   m1_detector_frame_Msun, m2_detector_frame_Msun, lambda1, lambda2
_DEFAULT_GW_PATH = (
    "/home/nucleartheory/gw_eos_tmnre/data/real_events/"
    "GW170817/GW170817_GWTC-1.hdf5"
)
_GW_DATASET = "IMRPhenomPv2NRT_lowSpin_posterior"


def load_gw_kde(path=_DEFAULT_GW_PATH, subsample=4000, z=Z_GW170817, seed=0):
    """Build the 4D GW170817 tidal KDE in source frame.

    Reads dataset 'IMRPhenomPv2NRT_lowSpin_posterior' (a structured/record
    array, ~8078 rows) with fields m1_detector_frame_Msun,
    m2_detector_frame_Msun, lambda1, lambda2. Masses are converted to the
    SOURCE frame via m_src = m_det / (1 + z), z = 0.0099.

    Conventions:
        m2 <= m1 (heavier = primary).  lambda1 pairs with m1, lambda2 with m2.
        Mc = (m1*m2)**0.6 / (m1+m2)**0.2   (chirp mass)
        q  = m2 / m1  in (0, 1]

    Returns
    -------
    kernelGW : scipy.stats.gaussian_kde
        KDE on vstack([Mc, q, Lambda1, Lambda2]) (source frame), optionally
        subsampled to `subsample` rows.
    Mc_obs : float
        Chirp mass at which gaussian_kde(Mc) is maximized (the KDE *mode*).
        GW measures Mc to ~0.1%, so this is effectively fixed (~1.186 Msun).
    """
    with h5py.File(path, "r") as f:
        d = f[_GW_DATASET][:]
        m1_det = np.asarray(d["m1_detector_frame_Msun"], dtype=np.float64)
        m2_det = np.asarray(d["m2_detector_frame_Msun"], dtype=np.float64)
        lam1 = np.asarray(d["lambda1"], dtype=np.float64)
        lam2 = np.asarray(d["lambda2"], dtype=np.float64)

    # source-frame masses
    m1 = m1_det / (1.0 + z)
    m2 = m2_det / (1.0 + z)

    # enforce m2 <= m1 (and carry the matching tidal deformabilities)
    swap = m2 > m1
    m1s = np.where(swap, m2, m1)
    m2s = np.where(swap, m1, m2)
    l1s = np.where(swap, lam2, lam1)
    l2s = np.where(swap, lam1, lam2)
    m1, m2, lam1, lam2 = m1s, m2s, l1s, l2s

    Mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    q = m2 / m1

    # subsample for speed (4D KDE eval cost scales with n_train)
    n = len(Mc)
    if subsample is not None and subsample < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=subsample, replace=False)
        Mc_k, q_k, lam1_k, lam2_k = Mc[idx], q[idx], lam1[idx], lam2[idx]
    else:
        Mc_k, q_k, lam1_k, lam2_k = Mc, q, lam1, lam2

    kernelGW = gaussian_kde(np.vstack([Mc_k, q_k, lam1_k, lam2_k]))

    # Mc_obs = mode of the 1D Mc KDE (GW measures Mc to ~0.1%).
    mc_kde = gaussian_kde(Mc)
    grid = np.linspace(Mc.min(), Mc.max(), 2001)
    Mc_obs = float(grid[np.argmax(mc_kde(grid))])

    return kernelGW, Mc_obs


def logL_gw(M_curve, L_curve, Mmax, kernelGW, Mc_obs, nq=20, qmin=0.7):
    """Marginalized GW170817 tidal log-likelihood for an EOS Lambda(M) curve.

    Fixes the chirp mass at Mc_obs (well measured), then integrates the GW KDE
    over the mass-ratio q. For each q on a fixed grid:
        m1 = Mc_obs * (1+q)**0.2 / q**0.6,   m2 = q * m1   (m2 <= m1)
    The EOS supplies Lambda(m1), Lambda(m2) via log-Lambda interpolation along
    the (M_curve, L_curve) stable branch. Masses must lie on the curve and be
    <= Mmax (TOV truncation, never hard-reject).

    Parameters
    ----------
    M_curve, L_curve : 1D arrays
        Stable-branch mass (Msun) and tidal deformability Lambda(M), sorted
        ascending in M, strictly positive L (for log interp).
    Mmax : float
        Maximum mass on the stable branch (Msun).
    kernelGW : gaussian_kde
        From load_gw_kde; trained on [Mc, q, Lambda1, Lambda2].
    Mc_obs : float
        Observed chirp mass (mode of Mc KDE).
    nq : int
        Number of mass-ratio grid points (fixed grid -> deterministic logL).
    qmin : float
        Lower q bound (q in [qmin, 1.0]).

    Returns
    -------
    logL : float
        log( trapz(density, q) + 1e-300 ), or -1e30 if no q is admissible.
    """
    M_curve = np.asarray(M_curve, dtype=np.float64)
    L_curve = np.asarray(L_curve, dtype=np.float64)

    qs = np.linspace(qmin, 1.0, nq)
    m1 = Mc_obs * (1.0 + qs) ** 0.2 / qs ** 0.6
    m2 = qs * m1

    Mmin = M_curve.min()
    ok = (m1 <= Mmax) & (m2 <= Mmax) & (m1 >= Mmin) & (m2 >= Mmin)
    if not ok.any():
        return -1e30

    logL_tab = np.log(L_curve)
    L1 = np.exp(np.interp(m1[ok], M_curve, logL_tab))
    L2 = np.exp(np.interp(m2[ok], M_curve, logL_tab))

    pts = np.vstack([np.full(ok.sum(), Mc_obs), qs[ok], L1, L2])
    dens = kernelGW(pts)

    integral = _trapz(dens, qs[ok])
    return float(np.log(integral + 1e-300))


# ---------------------------------------------------------------------------
# Optional: EOS forward -> (M, R, Lambda, Mmax) using the validated DDB code.
# Only imported lazily so the GW likelihood works without the forward present.
# ---------------------------------------------------------------------------
_FWD = None


def _load_forward():
    global _FWD
    if _FWD is not None:
        return _FWD
    import sys
    sys.path.insert(0, "/home/nucleartheory/Desktop/validated_code_DDB")
    import ddb_certified_forward as rn

    import numba
    from numba import njit

    T = rn.T  # tov_numba module

    # Tidal RK4 TOV returning (M, R_km, Lambda) -- copied verbatim from the
    # user's validated numba_tov_lambda.tov_single (the _tov_rhs already
    # integrates the y variable; the base forward just discards it).
    eps_of_p_lin = T.eps_of_p_lin
    _tov_rhs = T._tov_rhs
    KM_PER_MSOL = T.KM_PER_MSOL
    N_TOV_STEPS = T.N_TOV_STEPS

    @njit(fastmath=True)
    def tov_single_lambda(p_c_geom, p_tab, eps_tab, r_init=1e-4, h=2e-3,
                          n_steps=N_TOV_STEPS):
        PI = np.pi
        p_surface = max(1e-20, p_tab[0] * 1.001)
        eps_c = eps_of_p_lin(p_c_geom, p_tab, eps_tab)
        p_init = p_c_geom - (2.0 * PI / 3.0) * (eps_c + p_c_geom) * \
            (eps_c + 3.0 * p_c_geom) * r_init**2
        m_init = (4.0 * PI / 3.0) * eps_c * r_init**3
        r = r_init
        p = p_init; m = m_init; y = 2.0
        R_surf = r_init; M_surf = m_init; y_surf = 2.0
        found = False
        for _ in range(n_steps):
            if found:
                break
            k1p, k1m, k1y = _tov_rhs(r, p, m, y, p_tab, eps_tab)
            k2p, k2m, k2y = _tov_rhs(r + h/2, p + h/2*k1p, m + h/2*k1m, y + h/2*k1y, p_tab, eps_tab)
            k3p, k3m, k3y = _tov_rhs(r + h/2, p + h/2*k2p, m + h/2*k2m, y + h/2*k2y, p_tab, eps_tab)
            k4p, k4m, k4y = _tov_rhs(r + h, p + h*k3p, m + h*k3m, y + h*k3y, p_tab, eps_tab)
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
            return np.nan, np.nan, np.nan
        C = 2.0 * M_surf / R_surf
        if C > 0.99:
            C = 0.99
        fac1 = 2.0 - y_surf + (y_surf - 1.0) * C
        num = (1.0 / 20.0) * C**5 * (1.0 - C)**2 * fac1
        den = (C * (6.0 - 3.0 * y_surf + 1.5 * C * (5.0 * y_surf - 8.0))
               + 0.25 * C**3 * (26.0 - 22.0 * y_surf + C * (3.0 * y_surf - 2.0)
                                + C**2 * (1.0 + y_surf))
               + 3.0 * (1.0 - C)**2 * fac1 * np.log1p(-C))
        k2 = num / den
        Lambda = (2.0 / 3.0) * k2 * (R_surf / max(M_surf, 1e-10))**5
        return M_surf, R_surf * KM_PER_MSOL, Lambda

    @njit(fastmath=True)
    def _mrl_curve(p_c_arr, p_tab, eps_tab, h):
        n = len(p_c_arr)
        M = np.empty(n); R = np.empty(n); Lc = np.empty(n)
        for i in range(n):
            m, r, lam = tov_single_lambda(p_c_arr[i], p_tab, eps_tab, 1e-4, h)
            M[i] = m; R[i] = r; Lc[i] = lam
        return M, R, Lc

    _FWD = dict(rn=rn, T=T, _mrl_curve=_mrl_curve, H_TOV=rn.H_TOV, OFM=rn.OFM)
    return _FWD


def mrl_one(eps_mev, P_mev, h=None):
    """EOS table (eps, P in MeV/fm^3) -> stable-branch (M, R, Lambda, Mmax).

    Mirrors ddb_certified_forward.mmax_one (graft BPS crust -> geom -> curve over
    P_C_GEOM < 0.999*pg[-1]) but also returns Lambda. The branch is cut at
    argmax(M) and made monotone/unique in M (so np.interp works).
    """
    F = _load_forward()
    rn = F["rn"]; T = F["T"]; _mrl_curve = F["_mrl_curve"]
    H = F["H_TOV"] if h is None else h
    L = rn.L
    ok = np.isfinite(eps_mev) & np.isfinite(P_mev) & (eps_mev > 0) & (P_mev > 0)
    eps_mev, P_mev = eps_mev[ok], P_mev[ok]
    if len(eps_mev) < 5:
        return None
    o = np.argsort(eps_mev)
    eps_mev, P_mev = eps_mev[o], P_mev[o]
    ef, pf = L.graft_bps_crust(eps_mev, P_mev)
    eg = np.ascontiguousarray(ef * T.MEV_FM3_TO_GEOM)
    pg = np.ascontiguousarray(pf * T.MEV_FM3_TO_GEOM)
    pc = T.P_C_GEOM[T.P_C_GEOM < 0.999 * pg[-1]]
    if len(pc) < 4:
        return None
    M, R, Lam = _mrl_curve(np.ascontiguousarray(pc), pg, eg, H)
    v = np.isfinite(M) & (M > 0) & np.isfinite(Lam) & (Lam > 0)
    M, R, Lam = M[v], R[v], Lam[v]
    if len(M) < 4:
        return None
    Mmax = float(np.max(M))
    cut = np.argmax(M) + 1
    M, R, Lam = M[:cut], R[:cut], Lam[:cut]
    Mu, iu = np.unique(M, return_index=True)
    return Mu, R[iu], Lam[iu], Mmax


def mrl_from_theta(theta7):
    """Single DDB parameter vector [6 couplings, rho0] -> (M, R, Lambda, Mmax)."""
    F = _load_forward()
    rn = F["rn"]; OFM = F["OFM"]
    import jax.numpy as jnp
    th = np.asarray(theta7, dtype=np.float64).reshape(7)
    _, e, p = rn.eos_batch(jnp.asarray(th[:6][None, :]), jnp.asarray(th[6:7]))
    e = np.asarray(e)[0] * OFM
    p = np.asarray(p)[0] * OFM
    return mrl_one(e, p)


# ===========================================================================
# UNIT TEST
# ===========================================================================
if __name__ == "__main__":
    import time

    print("=" * 70)
    print("UNIT TEST: GW170817 marginalized tidal likelihood")
    print("=" * 70)

    # --- load KDE ---
    t0 = time.time()
    kernelGW, Mc_obs = load_gw_kde(subsample=4000)
    print(f"\nload_gw_kde: {time.time()-t0:.2f} s")
    print(f"  Mc_obs (mode of Mc KDE, source frame) = {Mc_obs:.4f} Msun "
          f"(expect ~1.186)")

    # report GW posterior medians (source frame, full sample)
    with h5py.File(_DEFAULT_GW_PATH, "r") as f:
        d = f[_GW_DATASET][:]
        m1 = np.asarray(d["m1_detector_frame_Msun"], float) / (1 + Z_GW170817)
        m2 = np.asarray(d["m2_detector_frame_Msun"], float) / (1 + Z_GW170817)
        lam1 = np.asarray(d["lambda1"], float)
        lam2 = np.asarray(d["lambda2"], float)
    swap = m2 > m1
    m1s = np.where(swap, m2, m1); m2s = np.where(swap, m1, m2)
    l1s = np.where(swap, lam2, lam1); l2s = np.where(swap, lam1, lam2)
    qmed = np.median(m2s / m1s)
    print(f"  median q                = {qmed:.4f}")
    print(f"  median Lambda1 (heavier)= {np.median(l1s):.1f}")
    print(f"  median Lambda2 (lighter)= {np.median(l2s):.1f}")
    print(f"  median Mc (source)      = "
          f"{np.median((m1s*m2s)**0.6/(m1s+m2s)**0.2):.4f}")

    # --- three synthetic Lambda(M) curves on a fixed M grid ---
    Mg = np.linspace(1.0, 2.0, 25)
    Mmax = 2.1

    # Lambda(M) ~ scale * (M/1.4)^(-6) is the standard steep falloff.
    def make_curve(lam14):
        return lam14 * (Mg / 1.4) ** (-6.0)

    curves = {
        "(a) GW-preferred  Lambda(1.4)~400 ": make_curve(400.0),
        "(b) too soft      Lambda(1.4)~50  ": make_curve(50.0),
        "(c) too stiff     Lambda(1.4)~1500": make_curve(1500.0),
    }

    print("\nSynthetic curve logL_gw (Mmax=2.1):")
    results = {}
    for name, Lc in curves.items():
        # timing over repeated evals
        n_rep = 50
        t0 = time.time()
        for _ in range(n_rep):
            ll = logL_gw(Mg, Lc, Mmax, kernelGW, Mc_obs)
        dt_ms = (time.time() - t0) / n_rep * 1e3
        results[name] = ll
        print(f"  {name}:  logL = {ll:12.4f}   ({dt_ms:.3f} ms/eval)")
        if dt_ms > 5.0:
            print(f"    [WARN] {dt_ms:.2f} ms/eval > 5ms -- 4D KDE may need "
                  f"a gridded interpolator replacement for production.")

    la = results["(a) GW-preferred  Lambda(1.4)~400 "]
    lb = results["(b) too soft      Lambda(1.4)~50  "]
    lc = results["(c) too stiff     Lambda(1.4)~1500"]
    print("\nAssertions:")
    print(f"  logL(a) > logL(b)?  {la > lb}   ({la:.3f} > {lb:.3f})")
    print(f"  logL(a) > logL(c)?  {la > lc}   ({la:.3f} > {lc:.3f})")
    assert la > lb, "GW-preferred should beat too-soft"
    assert la > lc, "GW-preferred should beat too-stiff"
    print("\nALL ASSERTIONS PASSED")
