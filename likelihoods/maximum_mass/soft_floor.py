"""Validated soft lower bound on the neutron-star maximum mass."""

from __future__ import annotations

import numpy as np


def log_likelihood(maximum_mass, threshold=2.0, width=0.05):
    """Return ``log(sigmoid((Mmax-threshold)/width))`` stably."""

    if width <= 0:
        raise ValueError("maximum-mass transition width must be positive")
    maximum_mass = np.asarray(maximum_mass, dtype=np.float64)
    result = -np.logaddexp(0.0, -(maximum_mass - threshold) / width)
    return float(result) if result.ndim == 0 else result
