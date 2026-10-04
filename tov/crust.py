"""BPS outer crust plus the validated gamma=4/3 bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

TOV_H = 1.2e-2
GAM_BRIDGE = 4.0 / 3.0
DEFAULT_BPS_PATH = Path(__file__).resolve().parent / "data" / "bps.dat"


def load_bps_table(path: str | Path = DEFAULT_BPS_PATH):
    array = np.loadtxt(path)
    order = np.argsort(array[:, 0])
    return array[order, 0], array[order, 1]


def graft_bps_crust(eps_core, p_core, n_bridge=30, bps_path=DEFAULT_BPS_PATH):
    """Graft the BPS crust and validated polytropic bridge onto a core EOS."""

    eps_bps, p_bps = load_bps_table(bps_path)
    eps_lo = float(eps_bps[-1])
    p_lo = float(p_bps[-1])
    eps_hi = float(eps_core[0])
    p_hi = float(p_core[0])
    if eps_lo >= eps_hi:
        keep = eps_bps < eps_hi
        eps_bps = eps_bps[keep]
        p_bps = p_bps[keep]
        if len(eps_bps) == 0:
            return eps_core, p_core
        eps_lo = float(eps_bps[-1])
        p_lo = float(p_bps[-1])
    denom = eps_hi**GAM_BRIDGE - eps_lo**GAM_BRIDGE
    a1 = (eps_hi**GAM_BRIDGE * p_lo - eps_lo**GAM_BRIDGE * p_hi) / denom
    b1 = (p_hi - p_lo) / denom
    step = (eps_hi - eps_lo) / float(n_bridge)
    e_bridge = np.linspace(eps_lo + step, eps_hi - step, max(0, n_bridge - 1))
    p_bridge = a1 + b1 * e_bridge**GAM_BRIDGE
    eps_full = np.concatenate([eps_bps, e_bridge, eps_core])
    p_full = np.concatenate([p_bps, p_bridge, p_core])
    keep = np.ones(len(eps_full), dtype=bool)
    for index in range(1, len(eps_full)):
        if (
            eps_full[index] <= eps_full[index - 1]
            or p_full[index] <= p_full[index - 1]
        ):
            keep[index] = False
    return eps_full[keep], p_full[keep]
