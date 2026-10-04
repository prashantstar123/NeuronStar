"""pQCD (Komoltsev-Kurkela / Gorda) log-likelihood factor for a beta-eq EOS.

Reuses the verbatim njit core from CompactObject/InferenceWorkflow/pQCD.py (constraints()):
given (eps, P, n) at a density, it checks whether a stable causal interpolation to the
pQCD point (mu=2.6 GeV) exists, over the renormalization scale X in [1,4]. The log-likelihood
is log(fraction of X that pass). DETERMINISTIC X grid (log-uniform) so logL(theta) is a fixed
function of theta (only matters for reproducibility; mirrors NICER/GW determinism).

NOTE (verified for DDB): at NS-core central densities (~0.9-1.1 fm^-3) this is ~null for nucleonic
DDB EOS (pass-fraction 1.0 -> logL 0); it only constrains the very stiff tail above ~1.3 fm^-3.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/CompactObject/InferenceWorkflow"))
import pQCD as _PQ   # njit: constraints(X,e0,p0,n0,muQCD,cs2), pressure, number_density

_XGRID = np.linspace(1.0, 4.0, 400)   # deterministic LINEAR-uniform X in [1,4] (Cartaxo 2026 choice)

def logL_pqcd(dens_fm3, en_mev, pr_mev, rho_eval=1.2):
    """EOS arrays (dens fm^-3, energy & pressure in MeV/fm^3) -> pQCD log-likelihood.

    Evaluated at number density rho_eval (fm^-3). Cartaxo 2026 anchor = 1.2 fm^-3 (~7.5 n0),
    "the maximum density point of the calculated EOS" (Mmax central density).
    Returns log(fraction of X in [1,4] that satisfy the Komoltsev-Kurkela connection), or
    -1e30 if the EOS does not cover rho_eval.
    """
    dens_fm3 = np.asarray(dens_fm3, dtype=np.float64)
    if rho_eval < dens_fm3.min() or rho_eval > dens_fm3.max():
        return 0.0   # outside EOS grid -> treat as unconstrained (null), not a rejection
    e = float(np.interp(rho_eval, dens_fm3, np.asarray(en_mev, np.float64))) / 1000.0   # GeV/fm^3
    p = float(np.interp(rho_eval, dens_fm3, np.asarray(pr_mev, np.float64))) / 1000.0   # GeV/fm^3
    if not (np.isfinite(e) and np.isfinite(p)) or p <= 0:
        return 0.0
    cnt = 0
    for X in _XGRID:
        if _PQ.constraints(X, e, p, rho_eval):
            cnt += 1
    return float(np.log(cnt / len(_XGRID) + 1e-30))
