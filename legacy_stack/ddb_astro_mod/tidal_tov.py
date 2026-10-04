#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone, unit-tested module adding tidal deformability Lambda(M) to the
validated DDB forward.

This computes, per parameter vector theta, the stable-branch mass-radius-tidal
curve (M, R, Lambda) and the maximum mass Mmax, by:
  core EOS (eos_batch, JAX) -> graft_bps_crust -> geometric units
  -> numba RK4 TOV+tidal integration (Love-k2, "1/20" form)
  -> stable branch up to argmax(M), sorted & uniquified in M.

The numba kernels `tov_single_lambda` and `_mrl_curve` are copied VERBATIM from
~/ddb_ks_posteriors.py (the Love k2 / "1/20" variant). They integrate M, R, and
the metric function y, then form Lambda = (2/3) k2 (R/M)^5 via the standard
Hinderer/Damour-Nagar formula.

IMPORTANT (env): set JAX_PLATFORMS=cpu, JAX_ENABLE_X64=True, CUDA_ROOT before any
JAX import, else JAX grabs the GPU and dies on a CuDNN mismatch.

Run with: /home/nucleartheory/anaconda3/envs/sbi_gpu/bin/python tidal_tov.py
"""
import os, sys

# --- MUST set these before importing jax / the validated forward ---
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # caller may set JAX_PLATFORMS=cuda for the GPU forward
os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("CUDA_ROOT", "/tmp")

import numpy as np
from numba import njit
from joblib import Parallel, delayed

# --- import the validated DDB forward ---
_DDB_DIR = os.path.expanduser("~/Desktop/validated_code_DDB")
sys.path.insert(0, _DDB_DIR)
import ddb_certified_forward as rn  # noqa: E402  (triggers jax import -> needs env above)

# Re-exported handles from the validated forward
T = rn.T                      # tov_numba module
OFM = rn.OFM                  # hbar*c factor: e,p in fm^-1 * OFM -> MeV/fm^3
H_TOV = rn.H_TOV              # RK4 step (geometric)
eos_batch = rn.eos_batch      # JAX: (theta6, rho0) -> (dens, e, p) [e,p in fm^-1]
graft_bps_crust = rn.L.graft_bps_crust
NW = 20                       # joblib workers (mirror rn.NW)


# ===================================================================
# numba tidal-TOV kernels -- copied VERBATIM from ~/ddb_ks_posteriors.py (L36-68)
# (Love k2, "1/20" form: num = (1/20) C^5 (1-C)^2 f1; Lambda = (2/3) k2 (R/M)^5)
# ===================================================================
@njit(fastmath=True, error_model="numpy")
def tov_single_lambda(p_c_geom, p_tab, eps_tab, r_init, h, n_steps):
    PI = np.pi; p_surface = max(1e-20, p_tab[0] * 1.001)
    eps_c = T.eps_of_p_lin(p_c_geom, p_tab, eps_tab)
    p = p_c_geom - (2.0 * PI / 3.0) * (eps_c + p_c_geom) * (eps_c + 3.0 * p_c_geom) * r_init**2
    m = (4.0 * PI / 3.0) * eps_c * r_init**3; r = r_init; y = 2.0
    R_surf = r_init; M_surf = m; y_surf = 2.0; found = False
    for _ in range(n_steps):
        if found: break
        k1p, k1m, k1y = T._tov_rhs(r,       p,           m,           y,           p_tab, eps_tab)
        k2p, k2m, k2y = T._tov_rhs(r + h/2, p + h/2*k1p,  m + h/2*k1m,  y + h/2*k1y,  p_tab, eps_tab)
        k3p, k3m, k3y = T._tov_rhs(r + h/2, p + h/2*k2p,  m + h/2*k2m,  y + h/2*k2y,  p_tab, eps_tab)
        k4p, k4m, k4y = T._tov_rhs(r + h,   p + h*k3p,    m + h*k3m,    y + h*k3y,    p_tab, eps_tab)
        p_new = p + h/6.0*(k1p + 2*k2p + 2*k3p + k4p); m_new = m + h/6.0*(k1m + 2*k2m + 2*k3m + k4m); y_new = y + h/6.0*(k1y + 2*k2y + 2*k3y + k4y)
        if p_new < p_surface:
            frac = (p - p_surface) / max(p - p_new, 1e-30); frac = min(max(frac, 0.0), 1.0)
            R_surf = r + h*frac; M_surf = m + frac*(m_new - m); y_surf = y + frac*(y_new - y); found = True
        else:
            r += h; p = p_new; m = m_new; y = y_new
    if not found: return np.nan, np.nan, np.nan
    C = 2.0 * M_surf / R_surf
    if C > 0.99: C = 0.99
    f1 = 2.0 - y_surf + (y_surf - 1.0) * C
    num = (1.0/20.0) * C**5 * (1.0 - C)**2 * f1
    den = (C*(6.0 - 3.0*y_surf + 1.5*C*(5.0*y_surf - 8.0)) + 0.25*C**3*(26.0 - 22.0*y_surf + C*(3.0*y_surf - 2.0) + C**2*(1.0 + y_surf)) + 3.0*(1.0 - C)**2*f1*np.log1p(-C))
    return M_surf, R_surf * T.KM_PER_MSOL, (2.0/3.0)*(num/den)*(R_surf/max(M_surf, 1e-10))**5

@njit(fastmath=True, error_model="numpy")
def _mrl_curve(p_c_arr, p_tab, eps_tab, h):
    n = len(p_c_arr); M = np.empty(n); R = np.empty(n); Lam = np.empty(n)
    for i in range(n):
        M[i], R[i], Lam[i] = tov_single_lambda(p_c_arr[i], p_tab, eps_tab, 1e-4, h, T.N_TOV_STEPS)
    return M, R, Lam


# ===================================================================
# Per-theta and batch drivers
# ===================================================================
def mrl_one(eps_mev, P_mev, h):
    """Core EOS (eps, P in MeV/fm^3) -> stable-branch (M, R, Lambda) arrays + Mmax.

    Returns (M_stable, R_stable, L_stable, Mmax) or None on failure.
      M_stable [Msun], R_stable [km], L_stable [dimensionless], sorted & unique in M.
    """
    eps = np.asarray(eps_mev, dtype=np.float64)
    P = np.asarray(P_mev, dtype=np.float64)
    ok = np.isfinite(eps) & np.isfinite(P) & (eps > 0) & (P > 0)
    eps, P = eps[ok], P[ok]
    if len(eps) < 5:
        return None
    o = np.argsort(eps); eps, P = eps[o], P[o]
    try:
        ef, pf = graft_bps_crust(eps, P)
        eg = np.ascontiguousarray(ef * T.MEV_FM3_TO_GEOM)
        pg = np.ascontiguousarray(pf * T.MEV_FM3_TO_GEOM)
        pc = T.P_C_GEOM[T.P_C_GEOM < 0.999 * pg[-1]]
        if len(pc) < 4:
            return None
        M, R, Lam = _mrl_curve(np.ascontiguousarray(pc), pg, eg, h)
        v = np.isfinite(M) & np.isfinite(R) & np.isfinite(Lam) & (M > 0) & (R > 0) & (Lam > 0)
        M, R, Lam = M[v], R[v], Lam[v]
        if len(M) < 4:
            return None
        mi = int(np.argmax(M)); Mmax = float(M[mi])
        Ms, Rs, Ls = M[:mi+1], R[:mi+1], Lam[:mi+1]
        oo = np.argsort(Ms); Ms, Rs, Ls = Ms[oo], Rs[oo], Ls[oo]
        u = np.unique(Ms, return_index=True)[1]; Ms, Rs, Ls = Ms[u], Rs[u], Ls[u]
        if len(Ms) < 4:
            return None
        return (Ms, Rs, Ls, Mmax)
    except Exception:
        return None


def mrl_batch(th6, rho0, h=H_TOV):
    """Batch of 6-couplings (th6) + rho0 -> list of (M, R, L, Mmax) or None per theta.

    Mirrors rn.mmax_batch: JAX eos_batch for the core EOS (e,p in fm^-1 -> *OFM for
    MeV/fm^3), then joblib Parallel over mrl_one. Returns a Python list of length
    len(th6); each entry is (M_arr, R_arr, L_arr, Mmax) or None.
    """
    import jax.numpy as jnp
    th6 = np.asarray(th6, dtype=np.float64)
    rho0 = np.asarray(rho0, dtype=np.float64)
    _, e, p = eos_batch(jnp.asarray(th6), jnp.asarray(rho0))
    e = np.asarray(e, dtype=np.float64) * OFM
    p = np.asarray(p, dtype=np.float64) * OFM
    out = Parallel(n_jobs=NW, batch_size=8)(
        delayed(mrl_one)(e[k], p[k], h) for k in range(len(th6))
    )
    return out


# ===================================================================
# UNIT TEST
# ===================================================================
def _interp_R_at(M, R, mtarget):
    if M.min() <= mtarget <= M.max():
        return float(np.interp(mtarget, M, R))
    return np.nan


def _interp_L_at(M, L, mtarget):
    if M.min() <= mtarget <= M.max():
        return float(10.0 ** np.interp(mtarget, M, np.log10(L)))
    return np.nan


def _run_unit_test():
    import time
    POST = os.path.expanduser(
        "~/sbi-ddb-paper/FINAL_DDB_sig10/live_tsnpe_ddb/posterior_FINAL.npy")
    post = np.load(POST).astype(np.float64)
    print(f"posterior_FINAL.npy: shape={post.shape} dtype={post.dtype}")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(post), size=30, replace=False)
    sample = post[idx]
    th6 = sample[:, :6]; rho0 = sample[:, 6]

    t0 = time.time()
    res = mrl_batch(th6, rho0)
    t_mrl = time.time() - t0
    t1 = time.time()
    mm_ref = rn.mmax_batch(th6, rho0)
    t_mm = time.time() - t1

    n_ok = sum(r is not None for r in res)
    print(f"mrl_batch: {n_ok}/30 valid in {t_mrl:.1f}s | mmax_batch: {t_mm:.1f}s")

    # (1) Mmax agreement
    max_dMmax = 0.0; n_cmp = 0
    R14s = []; L14s = []; nan_count = 0
    for r, mr in zip(res, mm_ref):
        if r is None:
            if np.isfinite(mr):
                nan_count += 1   # ref finite but we failed -> note
            continue
        M, R, L, Mmax = r
        if np.isfinite(mr):
            d = abs(Mmax - float(mr)); max_dMmax = max(max_dMmax, d); n_cmp += 1
        if not (np.all(np.isfinite(M)) and np.all(np.isfinite(R)) and np.all(np.isfinite(L))):
            nan_count += 1
        R14s.append(_interp_R_at(M, R, 1.4))
        L14s.append(_interp_L_at(M, L, 1.4))

    R14s = np.array(R14s); L14s = np.array(L14s)
    R14f = R14s[np.isfinite(R14s)]; L14f = L14s[np.isfinite(L14s)]
    med_R14 = float(np.median(R14f)); med_L14 = float(np.median(L14f))

    print(f"(1) max |Mmax_mrl - Mmax_ref| = {max_dMmax:.5f} Msun over {n_cmp} curves "
          f"(threshold 0.01) -> {'PASS' if max_dMmax < 0.01 else 'FAIL'}")
    print(f"(2) median R(1.4) = {med_R14:.3f} km (N={len(R14f)}) "
          f"-> {'PASS' if 12.0 <= med_R14 <= 13.0 else 'CHECK'}  "
          f"[min {R14f.min():.2f}, max {R14f.max():.2f}]")
    print(f"(3) median Lambda(1.4) = {med_L14:.1f} (N={len(L14f)}) "
          f"-> {'PASS (few hundred)' if 150 <= med_L14 <= 900 else 'CHECK'}  "
          f"[min {L14f.min():.1f}, max {L14f.max():.1f}]")
    print(f"(4) NaN/crash count among valid curves = {nan_count} "
          f"-> {'PASS' if nan_count == 0 else 'FAIL'}")
    return dict(max_dMmax=max_dMmax, med_R14=med_R14, med_L14=med_L14,
                n_ok=n_ok, nan_count=nan_count)


if __name__ == "__main__":
    _run_unit_test()
