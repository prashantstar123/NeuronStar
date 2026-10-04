"""Combined forward: one eos_batch -> per-theta (M,R,Lambda,Mmax) AND pQCD logL, in one joblib
pass (no double eos_batch). Used by the astrophysical likelihood pipelines so NICER/GW (from the curve)
and pQCD (from the EOS) share a single EOS solve."""
import os, sys
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, os.path.expanduser("~/Desktop/validated_code_DDB"))
sys.path.insert(0, os.path.expanduser("~/ddb_astro_mod"))
import ddb_certified_forward as rn
import tidal_tov as TT
import pqcd_like as PQ
import jax.numpy as jnp

NW = 20

def _work(dens, e, p, h, do_pqcd, rho_eval):
    c = TT.mrl_one(e, p, h)                                   # (M,R,Lambda,Mmax) or None
    pq = PQ.logL_pqcd(dens, e, p, rho_eval) if do_pqcd else 0.0
    return c, float(pq)

def curve_pqcd_batch(th6, rho0, do_pqcd=True, rho_eval=1.1, h=None):
    """th6 (N,6), rho0 (N,) -> list of (curve_or_None, logL_pqcd). One eos_batch + parallel TOV+pQCD."""
    h = rn.H_TOV if h is None else h
    th6 = np.asarray(th6, np.float64); rho0 = np.asarray(rho0, np.float64)
    dens, e, p = rn.eos_batch(jnp.asarray(th6), jnp.asarray(rho0))
    dens = np.asarray(dens, np.float64); e = np.asarray(e, np.float64) * rn.OFM; p = np.asarray(p, np.float64) * rn.OFM
    return Parallel(n_jobs=NW, batch_size=8)(
        delayed(_work)(dens[k], e[k], p[k], h, do_pqcd, rho_eval) for k in range(len(th6)))
