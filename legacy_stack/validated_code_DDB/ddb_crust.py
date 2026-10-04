#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BPS outer crust + gamma=4/3 polytrope bridge graft (Malik22 prescription).
Standalone extract of likelihood_cpu.graft_bps_crust. Loads the local bps.dat
(BPS outer-crust EOS table, columns: eps, p in MeV/fm^3)."""
import os
import numpy as np

TOV_H      = 1.2e-2        # numba-TOV RK4 step (Mmax error <1e-5 Msun vs 2e-3; validated)
GAM_BRIDGE = 4.0 / 3.0     # gamma=4/3 polytrope bridge

_HERE = os.path.dirname(os.path.abspath(__file__))
_arr = np.loadtxt(os.path.join(_HERE, "bps.dat"))
_o = np.argsort(_arr[:, 0])
_EPS_BPS, _P_BPS = _arr[_o, 0], _arr[_o, 1]

def graft_bps_crust(eps_core, p_core, n_bridge=30):
    """Core (eps,p in MeV/fm^3) -> BPS crust + gamma=4/3 bridge + core, monotonic."""
    eps_bps, p_bps = _EPS_BPS, _P_BPS
    eps_lo = float(eps_bps[-1]); p_lo = float(p_bps[-1])
    eps_hi = float(eps_core[0]); p_hi = float(p_core[0])
    if eps_lo >= eps_hi:
        keep = eps_bps < eps_hi
        eps_bps = eps_bps[keep]; p_bps = p_bps[keep]
        if len(eps_bps) == 0:
            return eps_core, p_core
        eps_lo = float(eps_bps[-1]); p_lo = float(p_bps[-1])
    denom = eps_hi**GAM_BRIDGE - eps_lo**GAM_BRIDGE
    a1 = (eps_hi**GAM_BRIDGE * p_lo - eps_lo**GAM_BRIDGE * p_hi) / denom
    b1 = (p_hi - p_lo) / denom
    step = (eps_hi - eps_lo) / float(n_bridge)
    e_bridge = np.linspace(eps_lo + step, eps_hi - step, max(0, n_bridge - 1))
    p_bridge = a1 + b1 * e_bridge ** GAM_BRIDGE
    eps_full = np.concatenate([eps_bps, e_bridge, eps_core])
    p_full   = np.concatenate([p_bps,   p_bridge, p_core])
    keep = np.ones(len(eps_full), dtype=bool)
    for i in range(1, len(eps_full)):
        if eps_full[i] <= eps_full[i-1] or p_full[i] <= p_full[i-1]:
            keep[i] = False
    return eps_full[keep], p_full[keep]
