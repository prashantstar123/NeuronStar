"""Batched, arithmetic-identical evaluation of the joint likelihood for many rows.

Every component reproduces likelihoods/joint/target.py, likelihoods/nicer/single_weight.py,
likelihoods/gw170817/kde.py (scipy gaussian_kde.evaluate internals), likelihoods/pqcd and
likelihoods/nuclear term by term; only the Python-level per-row overhead is removed.
Checked against JointLikelihood.evaluate before use.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from scipy.linalg import solve_triangular
from likelihoods.nuclear import gaussian_log_likelihood, observable_vector
from likelihoods.maximum_mass import log_likelihood as maximum_mass_log_likelihood
from likelihoods.pqcd.core import constraints
from likelihoods.pqcd.likelihood import SCALE_GRID


@njit(cache=True)
def _pqcd_count(energy0, pressure0, rho_eval, scales):
    count = 0
    for i in range(scales.size):
        if constraints(scales[i], energy0, pressure0, rho_eval):
            count += 1
    return count


def pqcd_batch(density, energy, pressure, rho_eval=1.2):
    """density: shared (K,) grid; energy/pressure: (N, K)."""
    n = energy.shape[0]
    out = np.zeros(n, dtype=np.float64)
    scales = np.ascontiguousarray(SCALE_GRID, dtype=np.float64)
    dmin = density.min(); dmax = density.max()
    if rho_eval < dmin or rho_eval > dmax:
        return out
    for i in range(n):
        energy0 = float(np.interp(rho_eval, density, energy[i])) / 1000.0
        pressure0 = float(np.interp(rho_eval, density, pressure[i])) / 1000.0
        if not (np.isfinite(energy0) and np.isfinite(pressure0)) or pressure0 <= 0:
            out[i] = 0.0
            continue
        count = _pqcd_count(energy0, pressure0, rho_eval, scales)
        out[i] = float(np.log(count / len(SCALE_GRID) + 1e-30))
    return out


def nicer_batch(branches, interpolators, valid):
    """Per-source NICER terms (N, S), -1e30 where the curve cannot be marginalised."""
    n = len(branches); s = len(interpolators)
    out = np.zeros((n, s), dtype=np.float64)
    out[np.asarray(valid, dtype=bool)] = -1e30
    rows = [i for i in range(n) if valid[i]]
    if not rows:
        return out
    grids = np.empty((len(rows), 40)); radii = np.empty((len(rows), 40)); span = np.empty(len(rows)); ok = np.zeros(len(rows), dtype=bool)
    for k, i in enumerate(rows):
        b = branches[i]
        mass = np.asarray(b.mass, dtype=np.float64); radius = np.asarray(b.radius, dtype=np.float64)
        lower = max(float(mass.min()), 1.0); upper = min(float(b.maximum_mass), float(mass.max()))
        if upper <= lower:
            continue
        ok[k] = True
        grids[k] = np.linspace(lower, upper, 40)
        radii[k] = np.interp(grids[k], mass, radius)
        span[k] = upper - lower
    sel = np.nonzero(ok)[0]
    if sel.size == 0:
        return out
    pts = np.column_stack([grids[sel].ravel(), radii[sel].ravel()])
    for j, interpolator in enumerate(interpolators):
        density = np.exp(interpolator(pts)).reshape(sel.size, 40)
        integral = np.trapezoid(density, grids[sel], axis=1) / span[sel]
        vals = np.log(integral + 1e-300)
        for kk, k in enumerate(sel):
            out[rows[k], j] = float(vals[kk])
    return out


@njit(cache=True, parallel=True)
def _kde_estimate(data_w, xi_w, weights, norm):
    """Exact loop of scipy.stats._stats.gaussian_kernel_estimate (sequential over data per query)."""
    n = data_w.shape[0]; d = data_w.shape[1]; m = xi_w.shape[0]
    est = np.zeros(m, dtype=np.float64)
    for j in prange(m):
        acc = 0.0
        for i in range(n):
            arg = 0.0
            for k in range(d):
                residual = data_w[i, k] - xi_w[j, k]
                arg += residual * residual
            arg = math.exp(-arg / 2.0) * norm
            acc += weights[i] * arg
        est[j] = acc
    return est


class BatchedGWKernel:
    def __init__(self, kernel):
        self.kernel = kernel
        cho = np.asarray(kernel.cho_cov, dtype=np.float64)
        self.cho_cov = cho
        self.data_w = np.ascontiguousarray(solve_triangular(cho, kernel.dataset, lower=True).T, dtype=np.float64)
        self.weights = np.ascontiguousarray(kernel.weights, dtype=np.float64)
        d = kernel.d
        norm = math.pow(2.0 * math.pi, -d / 2.0)
        for i in range(d):
            norm /= cho[i, i]
        self.norm = norm

    def evaluate(self, points):
        """points: (d, M) exactly as gaussian_kde.evaluate takes them."""
        xi_w = np.ascontiguousarray(solve_triangular(self.cho_cov, points, lower=True).T, dtype=np.float64)
        return _kde_estimate(self.data_w, xi_w, self.weights, self.norm)


def gw_batch(branches, bkernel, observed_chirp_mass, valid, nq=20, qmin=0.7):
    n = len(branches)
    out = np.full(n, -1e30, dtype=np.float64)
    mass_ratio = np.linspace(qmin, 1.0, nq)
    m1 = observed_chirp_mass * (1.0 + mass_ratio) ** 0.2 / mass_ratio ** 0.6
    m2 = mass_ratio * m1
    pts = []; slices = []; qsel = []
    for i in range(n):
        if not valid[i]:
            continue
        b = branches[i]
        mass = np.asarray(b.mass, dtype=np.float64); tidal_lambda = np.asarray(b.tidal_lambda, dtype=np.float64)
        minimum_mass = mass.min()
        v = (m1 <= b.maximum_mass) & (m2 <= b.maximum_mass) & (m1 >= minimum_mass) & (m2 >= minimum_mass)
        if not v.any():
            continue
        log_lambda = np.log(tidal_lambda)
        lambda1 = np.exp(np.interp(m1[v], mass, log_lambda))
        lambda2 = np.exp(np.interp(m2[v], mass, log_lambda))
        p = np.vstack([np.full(v.sum(), observed_chirp_mass), mass_ratio[v], lambda1, lambda2])
        pts.append(p); slices.append(i); qsel.append(mass_ratio[v])
    if not pts:
        return out
    allp = np.concatenate(pts, axis=1)
    dens = bkernel.evaluate(allp)
    pos = 0
    for i, p, q in zip(slices, pts, qsel):
        k = p.shape[1]
        density = dens[pos:pos + k]; pos += k
        integral = np.trapezoid(density, q)
        out[i] = float(np.log(integral + 1e-300))
    return out


def evaluate_batch(target, theta, nmp, density, energy, pressure, branches, bkernel=None):
    """Return dict with per-row components identical to JointLikelihood.evaluate + per-source NICER.

    Rows with branch None get total -inf and components 0 (as the target does)."""
    n = len(branches)
    valid = np.array([b is not None for b in branches])
    nuclear = np.zeros(n); mmax = np.zeros(n); nicer_src = np.zeros((n, len(target.nicer_interpolators))); nicer = np.zeros(n); gw = np.zeros(n); pqcd = np.zeros(n)
    if valid.any():
        idx = np.nonzero(valid)[0]
        # nuclear term: identical per-row arithmetic (7-term sums)
        for i in idx:
            nuclear[i] = float(gaussian_log_likelihood(observable_vector(theta[i], nmp[i]), target.nuclear_observation, target.nuclear_sigma))
        mm = np.array([branches[i].maximum_mass for i in idx])
        mmax[idx] = maximum_mass_log_likelihood(mm, target.maximum_mass_threshold, target.maximum_mass_width)
        if target.nicer_interpolators:
            nicer_src = nicer_batch(branches, target.nicer_interpolators, valid)
            # target sums the per-source terms with Python sum() left to right
            for i in idx:
                acc = 0
                for j in range(nicer_src.shape[1]):
                    acc = acc + nicer_src[i, j]
                nicer[i] = float(acc)
        if target.gw_kernel is not None:
            if bkernel is None:
                bkernel = BatchedGWKernel(target.gw_kernel)
            gw = gw_batch(branches, bkernel, target.gw_chirp_mass, valid)
            gw[~valid] = 0.0
        if target.pqcd_enabled:
            pq_all = pqcd_batch(density, energy, pressure, target.pqcd_density)
            pqcd[idx] = pq_all[idx]
    total = np.full(n, -np.inf)
    nuclear[~valid] = -np.inf   # JointLikelihood.evaluate returns (-inf, 0, 0, 0, 0) for a missing branch
    for i in np.nonzero(valid)[0]:
        t = float(nuclear[i] + mmax[i] + nicer[i] + gw[i] + pqcd[i])
        total[i] = t
    nonfinite = valid & ~np.isfinite(total)
    # target returns LikelihoodBreakdown(-inf, 0, 0, 0, 0) when the total is not finite
    for i in np.nonzero(nonfinite)[0]:
        nuclear[i] = -np.inf; mmax[i] = nicer[i] = gw[i] = pqcd[i] = 0.0; total[i] = -np.inf
    return dict(nuclear=nuclear, maximum_mass=mmax, nicer=nicer, nicer_sources=nicer_src, gw170817=gw, pqcd=pqcd, total=total, valid=valid & ~nonfinite)
