"""
nicer_like.py -- NICER marginalized M-R likelihood along an EOS curve.

Standalone module implementing the field-standard "marginalized likelihood
along the EOS M-R curve" for NICER X-ray sources (J0030 Miller 2019, J0740
Dittmann 2024). For speed each source posterior is rendered ONCE onto a fixed
(M,R) grid via gaussian_kde, then wrapped in a RegularGridInterpolator returning
log-density. Per-curve evaluation is a cheap 40-point trapezoid over the curve
-- no KDE is ever called inside a per-theta loop.

Recipe sources:
  ~/CompactObject/InferenceWorkflow/Likelihood.py  (MRlikihood_kernel)
  ~/Desktop/Fmode_GR_Project/04_output_raw/nicer_reweight_R14_from_SBI_json.py
       (read_samples_generic, gaussian_kde build/eval)

NICER source priors are ~flat in (M,R), so no explicit prior division is done;
the constant is absorbed into the overall normalization of the combined
posterior (LOGL_REF in the pipeline). This module returns the raw marginalized
log-likelihood per source, summed over sources.

CONFIRMED column orders (verified by reading file headers):
  J0030_2spot_RM.txt        : col0 = R_km, col1 = M_Msun            (no weights)
  J0740_NICERXMM_full_mr.txt : col0 = R_km, col1 = M_Msun, col2 = weight
"""

import os
# MUST be set before any jax import (the validated forward imports jax).
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("CUDA_ROOT", "/tmp")

import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator

# ----------------------------------------------------------------------------
# NICER gridded likelihood
# ----------------------------------------------------------------------------

def build_nicer_grid(path, mcol, rcol, wcol_or_None,
                     nM=400, nR=400, bw=0.08,
                     Mlo=None, Mhi=None, Rlo=None, Rhi=None,
                     max_rows=60000, seed=0):
    """Render a NICER (M,R) posterior sample file onto a fixed grid of
    log-density and return a RegularGridInterpolator over (M, R).

    Args:
        path : whitespace-delimited text, '#' comments ignored.
        mcol, rcol : 0-based column indices for mass (Msun) and radius (km).
        wcol_or_None : 0-based weight column index, or None for unweighted.
        nM, nR : grid resolution in mass / radius.
        bw : gaussian_kde bw_method (Scott-scaled bandwidth factor).
        Mlo/Mhi/Rlo/Rhi : grid bounds; if None, taken from data 1-99 pct.
        max_rows : subsample cap before building the KDE.
        seed : RNG seed for subsampling reproducibility.

    Returns:
        RegularGridInterpolator((M_centers, R_centers), logP, method='linear',
        bounds_error=False, fill_value=log_floor). Querying it gives log-density;
        exp() it to recover (unnormalized) probability density.
    """
    raw = np.loadtxt(path, comments="#")
    raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
    M = raw[:, mcol].astype(np.float64)
    R = raw[:, rcol].astype(np.float64)
    if wcol_or_None is not None:
        w = raw[:, wcol_or_None].astype(np.float64)
    else:
        w = np.ones_like(M)

    # drop non-finite / non-positive-weight rows
    good = np.isfinite(M) & np.isfinite(R) & np.isfinite(w) & (w >= 0)
    M, R, w = M[good], R[good], w[good]

    # auto bounds from 1-99 percentile of the data
    if Mlo is None:
        Mlo = np.percentile(M, 1.0)
    if Mhi is None:
        Mhi = np.percentile(M, 99.0)
    if Rlo is None:
        Rlo = np.percentile(R, 1.0)
    if Rhi is None:
        Rhi = np.percentile(R, 99.0)

    # subsample to <= max_rows for a tractable KDE build (weight-respecting)
    n = len(M)
    if n > max_rows:
        rng = np.random.default_rng(seed)
        p = w / w.sum()
        idx = rng.choice(n, size=max_rows, replace=False, p=p)
        Ms, Rs, ws = M[idx], R[idx], w[idx]
    else:
        Ms, Rs, ws = M, R, w

    kde = gaussian_kde(np.vstack([Ms, Rs]), weights=ws, bw_method=bw)

    M_centers = np.linspace(Mlo, Mhi, nM)
    R_centers = np.linspace(Rlo, Rhi, nR)
    MM, RR = np.meshgrid(M_centers, R_centers, indexing="ij")
    pts = np.vstack([MM.ravel(), RR.ravel()])
    dens = kde(pts).reshape(nM, nR)          # ONE KDE evaluation over the mesh

    floor = dens.max() * 1e-6
    logP = np.log(np.maximum(dens, floor))
    log_floor = float(np.log(floor))

    return RegularGridInterpolator(
        (M_centers, R_centers), logP,
        method="linear", bounds_error=False, fill_value=log_floor,
    )


def logL_nicer_one(M_curve, R_curve, Mmax, interp):
    """Marginalized log-likelihood of one NICER source along an EOS M-R curve.

    Integrates the source 2D density along the stable branch parameterized by
    mass: L = (1/(Mhi-Mlo)) * int_{Mlo}^{Mhi} p_src(M, R(M)) dM, evaluated on a
    fixed 40-point mass grid (deterministic). Returns log(L + tiny).

    Args:
        M_curve, R_curve : stable-branch mass (Msun) and radius (km) arrays,
            sorted/unique in M (as produced by per_sample / _mrl_curve cuts).
        Mmax : maximum mass of the EOS (caps the integral; never hard-reject).
        interp : RegularGridInterpolator from build_nicer_grid (log-density).
    """
    M_curve = np.asarray(M_curve, dtype=np.float64)
    R_curve = np.asarray(R_curve, dtype=np.float64)
    Mlo = max(float(M_curve.min()), 1.0)
    Mhi = min(float(Mmax), float(M_curve.max()))
    if Mhi <= Mlo:
        return -1e30
    Mg = np.linspace(Mlo, Mhi, 40)
    Rg = np.interp(Mg, M_curve, R_curve)
    dens = np.exp(interp(np.column_stack([Mg, Rg])))
    integral = np.trapz(dens, Mg) / (Mhi - Mlo)
    return float(np.log(integral + 1e-300))


def logL_nicer(M_curve, R_curve, Mmax, interps):
    """Total NICER log-likelihood = sum over source interpolators.

    Args:
        interps : iterable of RegularGridInterpolator (one per source).
    """
    return float(sum(logL_nicer_one(M_curve, R_curve, Mmax, ip) for ip in interps))


# ----------------------------------------------------------------------------
# Unit test
# ----------------------------------------------------------------------------

def _selftest():
    import time, sys
    sys.path.insert(0, os.path.expanduser("~/Desktop/validated_code_DDB"))
    sys.path.insert(0, os.path.expanduser("~"))  # for ddb_ks_posteriors (per_sample)
    import ddb_certified_forward as rn
    OFM = rn.OFM

    J0030 = "/home/nucleartheory/ddb_obs_data/J0030_2spot_RM.txt"
    J0740 = "/home/nucleartheory/ddb_obs_data/J0740_NICERXMM_full_mr.txt"

    print("== building NICER grids ==")
    t0 = time.time()
    # nM=nR=200 is ample (KDE is smooth at bw=0.08); 400x400 quadruples build cost
    # for no accuracy gain. R,M ordering in files -> M=col1, R=col0.
    ip_j0030 = build_nicer_grid(J0030, mcol=1, rcol=0, wcol_or_None=None, nM=200, nR=200)
    ip_j0740 = build_nicer_grid(J0740, mcol=1, rcol=0, wcol_or_None=2, nM=200, nR=200)
    build_t = time.time() - t0
    interps = [ip_j0030, ip_j0740]
    print(f"build time (both grids, one-time): {build_t:.2f} s")

    # synthetic curves
    Mline = np.linspace(1.0, 2.1, 60)
    # (a) through the blobs: R~12.5 (both J0030 ~13 km and J0740 high-mass ~12.5 km)
    Ra = np.full_like(Mline, 12.5)
    # (b) far away: R~9 km, below the 1pct radius bound of both sources -> density floor
    Rb = np.full_like(Mline, 9.0)
    Mmax_syn = 2.2

    la = logL_nicer(Mline, Ra, Mmax_syn, interps)
    lb = logL_nicer(Mline, Rb, Mmax_syn, interps)
    la0 = logL_nicer_one(Mline, Ra, Mmax_syn, ip_j0030)
    la1 = logL_nicer_one(Mline, Ra, Mmax_syn, ip_j0740)
    lb0 = logL_nicer_one(Mline, Rb, Mmax_syn, ip_j0030)
    lb1 = logL_nicer_one(Mline, Rb, Mmax_syn, ip_j0740)
    print(f"synthetic (a) through blobs R=12.5: logL_total={la:.3f}  [J0030={la0:.3f}, J0740={la1:.3f}]", flush=True)
    print(f"synthetic (b) far away   R= 9.0 : logL_total={lb:.3f}  [J0030={lb0:.3f}, J0740={lb1:.3f}]", flush=True)
    print(f"contrast logL(a)-logL(b) = {la - lb:.3f}  (expect >> 0)", flush=True)
    assert la > lb + 5.0, "FAIL: through-blob curve not strongly preferred over far curve"

    # per-curve eval timing
    t0 = time.time()
    NIT = 2000
    for _ in range(NIT):
        logL_nicer(Mline, Ra, Mmax_syn, interps)
    per_eval_ms = (time.time() - t0) / NIT * 1e3
    print(f"per-curve eval time: {per_eval_ms*1e3:.1f} us  ({per_eval_ms:.4f} ms)", flush=True)
    assert per_eval_ms < 1.0, "FAIL: per-curve eval not sub-ms"

    # ---- real EOS curves from the posterior ----
    # Build the (M,R) stable branch from each coupling set via the validated
    # forward. KEY: call eos_batch ONCE on all thetas (a single JAX trace), and
    # warm up the numba TOV before timing -- otherwise per-curve "time" is
    # dominated by JAX retracing / numba JIT, not the physics.
    import ddb_ks_posteriors as ks
    import jax.numpy as jnp
    post = np.load(os.path.expanduser(
        "~/sbi-ddb-paper/FINAL_DDB_sig10/live_tsnpe_ddb/posterior_FINAL.npy")).astype(np.float64)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(post), size=20, replace=False)
    th6 = post[idx, :6]; rho0 = post[idx, 6]
    print("== evaluating logL_nicer on 20 real EOS curves ==", flush=True)

    def _branch(e, p):
        ok = np.isfinite(e) & np.isfinite(p) & (e > 0) & (p > 0)
        e, p = e[ok], p[ok]
        if len(e) < 5:
            return None
        o = np.argsort(e); e, p = e[o], p[o]
        ef, pf = rn.L.graft_bps_crust(e, p)
        eg = np.ascontiguousarray(ef * rn.T.MEV_FM3_TO_GEOM)
        pg = np.ascontiguousarray(pf * rn.T.MEV_FM3_TO_GEOM)
        pc = rn.T.P_C_GEOM[rn.T.P_C_GEOM < 0.999 * pg[-1]]
        if len(pc) < 4:
            return None
        M, R, Lam = ks._mrl_curve(np.ascontiguousarray(pc), pg, eg, rn.H_TOV)
        v = np.isfinite(M) & np.isfinite(R) & (M > 0) & (R > 0)
        M, R = M[v], R[v]
        if len(M) < 4:
            return None
        mi = int(np.argmax(M)); Mmx = M[mi]
        Ms, Rs = M[:mi+1], R[:mi+1]
        oo = np.argsort(Ms); Ms, Rs = Ms[oo], Rs[oo]
        u = np.unique(Ms, return_index=True)[1]; Ms, Rs = Ms[u], Rs[u]
        if len(Ms) < 4:
            return None
        return Ms, Rs, Mmx

    # single batched EOS call (JAX), then numba warmup on the first valid curve
    _, eb, pb = rn.eos_batch(jnp.asarray(th6), jnp.asarray(rho0))
    eb = np.asarray(eb) * OFM; pb = np.asarray(pb) * OFM
    _branch(eb[0], pb[0])   # warm up numba JIT (not timed)

    t0 = time.time()
    vals = []
    for k in range(len(idx)):
        br = _branch(eb[k], pb[k])
        if br is None:
            continue
        Ms, Rs, Mmx = br
        vals.append(logL_nicer(Ms, Rs, Mmx, interps))
    real_t = time.time() - t0
    vals = np.array(vals)
    print(f"real curves: {len(vals)}/20 valid, TOV+logL total {real_t:.2f} s "
          f"({real_t/max(len(vals),1)*1e3:.1f} ms/curve incl. numba TOV)", flush=True)
    print(f"real logL_nicer spread: min={vals.min():.2f} med={np.median(vals):.2f} "
          f"max={vals.max():.2f}  (mean={vals.mean():.2f}, std={vals.std():.2f})", flush=True)
    print("ALL ASSERTIONS PASSED", flush=True)
    return dict(build_t=build_t, per_eval_us=per_eval_ms*1e3,
                la=la, lb=lb, contrast=la-lb,
                real_min=float(vals.min()), real_med=float(np.median(vals)),
                real_max=float(vals.max()))


if __name__ == "__main__":
    _selftest()
