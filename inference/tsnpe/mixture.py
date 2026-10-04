"""Density utilities for deterministic-mixture importance sampling.

The routines in this module operate in physical DDB parameter coordinates.
Every density is normalized on the prior box; no proportional-density
shortcuts are used in evidence calculations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


def _normalized_weights(log_weight: np.ndarray) -> np.ndarray:
    log_weight = np.asarray(log_weight, dtype=np.float64)
    if log_weight.ndim != 1 or not np.isfinite(log_weight).any():
        raise ValueError("log weights must be a 1D array with a finite entry")
    log_normalization = float(logsumexp(log_weight))
    if not np.isfinite(log_normalization):
        raise ValueError("importance weights cannot be normalized")
    return np.exp(log_weight - log_normalization)


def weighted_mean_covariance(
    values: np.ndarray, log_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the population mean and covariance under log weights."""

    values = np.asarray(values, dtype=np.float64)
    log_weight = np.asarray(log_weight, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != log_weight.shape[0]:
        raise ValueError("values must be (N,D) and match the log-weight length")
    weight = _normalized_weights(log_weight)
    mean = np.sum(weight[:, None] * values, axis=0)
    centered = values - mean
    covariance = np.sum(
        weight[:, None, None]
        * centered[:, :, None]
        * centered[:, None, :],
        axis=0,
    )
    return mean, covariance


@dataclass(frozen=True)
class TruncatedGaussian:
    """A multivariate Gaussian conditioned to a rectangular prior box."""

    mean: np.ndarray
    cholesky: np.ndarray
    low: np.ndarray
    high: np.ndarray
    normalization: float
    scale: float
    normalization_draws: int

    @classmethod
    def from_covariance(
        cls,
        mean: np.ndarray,
        covariance: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        *,
        scale: float,
        rng: np.random.Generator,
        normalization_draws: int = 2_000_000,
        diagonal_jitter: float = 1e-12,
    ) -> "TruncatedGaussian":
        mean = np.asarray(mean, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        dimension = len(mean)
        if covariance.shape != (dimension, dimension):
            raise ValueError("covariance has the wrong shape")
        if low.shape != mean.shape or high.shape != mean.shape:
            raise ValueError("bounds have the wrong shape")
        if not np.all(high > low):
            raise ValueError("every upper bound must exceed its lower bound")
        if scale <= 0.0 or normalization_draws < 1:
            raise ValueError("scale and normalization draw count must be positive")
        cholesky = np.linalg.cholesky(
            # Keep the multiplication order used to generate the frozen
            # proposal; this also permits a byte-for-byte replay gate.
            covariance * float(scale) * float(scale)
            + float(diagonal_jitter) * np.eye(dimension)
        )
        draws = (
            rng.standard_normal((int(normalization_draws), dimension))
            @ cholesky.T
            + mean
        )
        normalization = float(
            np.mean(((draws > low) & (draws < high)).all(axis=1))
        )
        if not 0.0 < normalization <= 1.0:
            raise ValueError("truncation normalization is zero or invalid")
        return cls(
            mean=mean,
            cholesky=cholesky,
            low=low,
            high=high,
            normalization=normalization,
            scale=float(scale),
            normalization_draws=int(normalization_draws),
        )

    def log_density(self, values: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        if values.shape[1] != len(self.mean):
            raise ValueError("values have the wrong parameter dimension")
        displacement = np.linalg.solve(
            self.cholesky, (values - self.mean).T
        ).T
        dimension = values.shape[1]
        result = (
            -0.5 * np.sum(displacement * displacement, axis=1)
            - np.sum(np.log(np.diag(self.cholesky)))
            - 0.5 * dimension * np.log(2.0 * np.pi)
            - np.log(self.normalization)
        )
        inside = ((values > self.low) & (values < self.high)).all(axis=1)
        return np.where(inside, result, -np.inf)

    def sample(
        self,
        size: int,
        *,
        rng: np.random.Generator,
        batch_size: int = 40_000,
    ) -> np.ndarray:
        """Draw exactly from the box-truncated Gaussian by rejection."""

        if size < 1 or batch_size < 1:
            raise ValueError("sample size and rejection batch size must be positive")
        accepted: list[np.ndarray] = []
        count = 0
        while count < size:
            draw = (
                rng.standard_normal((batch_size, len(self.mean)))
                @ self.cholesky.T
                + self.mean
            )
            draw = draw[((draw > self.low) & (draw < self.high)).all(axis=1)]
            if len(draw):
                accepted.append(draw)
                count += len(draw)
        return np.concatenate(accepted, axis=0)[:size]


def replace_mixture_component(
    assumed_log_density: np.ndarray,
    actual_component_log_density: np.ndarray,
    substituted_component_log_density: np.ndarray,
    component_fraction: float,
) -> np.ndarray:
    """Replace one erroneously substituted component in a mixture density.

    If ``q_assumed`` contains ``fraction * q_substituted``, this returns the
    log of ``q_assumed - fraction*q_substituted + fraction*q_actual``.  The
    signed operation is evaluated in a common exponential scale and refuses
    any non-positive corrected denominator.
    """

    assumed = np.asarray(assumed_log_density, dtype=np.float64)
    actual = np.asarray(actual_component_log_density, dtype=np.float64)
    substituted = np.asarray(
        substituted_component_log_density, dtype=np.float64
    )
    if assumed.shape != actual.shape or assumed.shape != substituted.shape:
        raise ValueError("mixture-density arrays must have matching shapes")
    if assumed.ndim != 1:
        raise ValueError("mixture-density arrays must be one-dimensional")
    if not 0.0 < component_fraction < 1.0:
        raise ValueError("component fraction must lie strictly between zero and one")
    log_fraction = np.log(float(component_fraction))
    scale = np.maximum.reduce(
        [assumed, log_fraction + actual, log_fraction + substituted]
    )
    scaled = (
        np.exp(assumed - scale)
        + np.exp(log_fraction + actual - scale)
        - np.exp(log_fraction + substituted - scale)
    )
    if not np.isfinite(scaled).all() or not np.all(scaled > 0.0):
        raise ValueError("corrected mixture density is not strictly positive")
    return scale + np.log(scaled)
