"""Gaussian likelihood for the seven nuclear observables used in the paper."""

from __future__ import annotations

import numpy as np

OBSERVABLE_NAMES = ("rho0", "E0", "K0", "Jsym", "P08", "P12", "P16")
A1_OBSERVATION = np.array(
    [
        0.153,
        -16.1,
        230.0,
        32.5,
        0.505714285714279,
        1.24142857142857,
        2.4857142857143,
    ],
    dtype=np.float64,
)
A1_SIGMA = np.array(
    [
        0.005,
        0.2,
        40.0,
        1.8,
        0.19428571428571,
        0.608571428571428,
        1.382857142857144,
    ],
    dtype=np.float64,
)


def observable_vector(theta, nmp):
    """Compose ``[rho0, E0, K0, Jsym, P08, P12, P16]``.

    ``theta`` may be a single DDB-family parameter vector or an ``(N, D)``
    array.  The first seven coordinates retain the DDB ordering, so ``rho0``
    is coordinate six; extensions may append composition parameters. ``nmp``
    must carry the corresponding six derived observables.
    """

    theta = np.asarray(theta, dtype=np.float64)
    nmp = np.asarray(nmp, dtype=np.float64)
    if theta.ndim == 1:
        if theta.shape[0] < 7 or nmp.shape != (6,):
            raise ValueError(
                f"expected theta (D>=7,) and nmp (6,), got {theta.shape} and {nmp.shape}"
            )
        return np.concatenate([[theta[6]], nmp])
    if (
        theta.ndim != 2
        or theta.shape[1] < 7
        or nmp.shape != (len(theta), 6)
    ):
        raise ValueError(
            f"expected theta (N,D>=7) and nmp (N,6), got {theta.shape} and {nmp.shape}"
        )
    return np.column_stack([theta[:, 6], nmp])


def gaussian_log_likelihood(
    values,
    observation=A1_OBSERVATION,
    sigma=A1_SIGMA,
    *,
    normalized=False,
):
    """Evaluate the paper's independent-Gaussian nuclear factor.

    Production runs used the unnormalized form by default.  Keeping that
    convention explicit is essential for reproducing reported evidences.
    """

    values = np.asarray(values, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if observation.shape != (7,) or sigma.shape != (7,):
        raise ValueError("nuclear observation and sigma must each have shape (7,)")
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("nuclear sigma values must be finite and positive")
    if values.shape[-1:] != (7,):
        raise ValueError(f"last observable dimension must be seven, got {values.shape}")
    result = -0.5 * np.sum(((values - observation) / sigma) ** 2, axis=-1)
    if normalized:
        result = result - np.sum(np.log(sigma * np.sqrt(2.0 * np.pi)))
    return result
