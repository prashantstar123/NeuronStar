"""Deterministic distribution features for jointly amortized EN v3.

The v2 conditional EN received raw point clouds through a learned DeepSets
mean.  Its training clouds were only affine deformations of three baseline
clouds, so correlation, skewness and modality were fixed by construction.
This module provides a data-independent, permutation-invariant description of
an arbitrary two-dimensional mass--radius cloud.  Multi-scale empirical
characteristic-function features retain distribution shape, while explicit
robust summaries stabilize location, scale and tail information.

The feature map is fixed before any held-out NICER source is evaluated.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.special import ndtr


FEATURE_SCHEMA = "conditional-en-cloud-features-v3"
SOURCE_SLOTS = 3
COORDINATES = 2
MASS_CENTRE = 1.55
MASS_SCALE = 0.65
RADIUS_CENTRE = 12.5
RADIUS_SCALE = 3.0
QUANTILE_LEVELS = np.asarray(
    [0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99], dtype=np.float64
)
RFF_LENGTH_SCALES = np.asarray(
    [0.05, 0.10, 0.20, 0.40, 0.80, 1.60], dtype=np.float64
)
RFF_DIRECTIONS_PER_SCALE = 16
RFF_SEED = 2026092103


def _frequency_matrix() -> np.ndarray:
    """Return the immutable multi-bandwidth Fourier frequency matrix."""

    rng = np.random.default_rng(RFF_SEED)
    blocks = []
    for length_scale in RFF_LENGTH_SCALES:
        direction = rng.normal(size=(RFF_DIRECTIONS_PER_SCALE, COORDINATES))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        # Random radial jitter avoids aliasing while retaining deterministic,
        # declared frequency coverage at every resolution.
        radial = rng.uniform(
            0.75 / length_scale,
            1.25 / length_scale,
            size=(RFF_DIRECTIONS_PER_SCALE, 1),
        )
        blocks.append(direction * radial)
    return np.concatenate(blocks, axis=0).astype(np.float64)


RFF_FREQUENCIES = _frequency_matrix()
SUMMARY_DIMENSION = 28
FEATURE_DIMENSION = SUMMARY_DIMENSION + 2 * len(RFF_FREQUENCIES)


def feature_specification() -> dict[str, object]:
    frequency_hash = hashlib.sha256(RFF_FREQUENCIES.tobytes()).hexdigest()
    return {
        "schema": FEATURE_SCHEMA,
        "physical_standardization": {
            "mass_centre": MASS_CENTRE,
            "mass_scale": MASS_SCALE,
            "radius_centre": RADIUS_CENTRE,
            "radius_scale": RADIUS_SCALE,
        },
        "quantile_levels": QUANTILE_LEVELS.tolist(),
        "rff_length_scales": RFF_LENGTH_SCALES.tolist(),
        "rff_directions_per_scale": RFF_DIRECTIONS_PER_SCALE,
        "rff_seed": RFF_SEED,
        "rff_frequency_sha256": frequency_hash,
        "summary_dimension": SUMMARY_DIMENSION,
        "feature_dimension_per_source": FEATURE_DIMENSION,
        "synthetic_gaussian_mixture_features": "analytic",
    }


def feature_specification_json() -> str:
    return json.dumps(feature_specification(), sort_keys=True)


def cloud_features(clouds: np.ndarray) -> np.ndarray:
    """Map ``(..., 3, points, 2)`` clouds to ``(..., 3, features)``.

    The operation is invariant to point order within each semantic source
    slot.  Coordinates are kept in physical units for the explicit summaries
    and put on a fixed, data-independent scale for the Fourier features.
    """

    value = np.asarray(clouds, dtype=np.float64)
    if value.ndim < 3 or value.shape[-3] != SOURCE_SLOTS:
        raise ValueError("clouds must have three semantic source slots")
    if value.shape[-1] != COORDINATES or value.shape[-2] < 8:
        raise ValueError("clouds must end in (points,2) with at least 8 points")
    if not np.isfinite(value).all():
        raise ValueError("clouds contain non-finite coordinates")

    mean = value.mean(axis=-2)
    centred = value - mean[..., None, :]
    standard_deviation = np.sqrt(
        np.maximum(np.mean(centred * centred, axis=-2), 1.0e-12)
    )
    standardized = centred / standard_deviation[..., None, :]
    quantiles = np.quantile(value, QUANTILE_LEVELS, axis=-2)
    # Move quantile level behind the source axis, then flatten coordinate and
    # level in a stable order.
    quantiles = np.moveaxis(quantiles, 0, -2).reshape(*value.shape[:-3], 3, -1)
    correlation = np.mean(
        standardized[..., 0] * standardized[..., 1], axis=-1
    )[..., None]
    skewness = np.mean(standardized**3, axis=-2)
    excess_kurtosis = np.mean(standardized**4, axis=-2) - 3.0
    cross_skewness = np.stack(
        [
            np.mean(standardized[..., 0] ** 2 * standardized[..., 1], axis=-1),
            np.mean(standardized[..., 0] * standardized[..., 1] ** 2, axis=-1),
        ],
        axis=-1,
    )
    leading_shape = value.shape[:-3]
    identity = np.broadcast_to(
        np.eye(SOURCE_SLOTS, dtype=np.float64),
        (*leading_shape, SOURCE_SLOTS, SOURCE_SLOTS),
    )
    summary = np.concatenate(
        [
            mean,
            standard_deviation,
            quantiles,
            correlation,
            skewness,
            excess_kurtosis,
            cross_skewness,
            identity,
        ],
        axis=-1,
    )
    if summary.shape[-1] != SUMMARY_DIMENSION:
        raise RuntimeError("internal cloud-summary dimension changed")

    scaled = np.empty_like(value)
    scaled[..., 0] = (value[..., 0] - MASS_CENTRE) / MASS_SCALE
    scaled[..., 1] = (value[..., 1] - RADIUS_CENTRE) / RADIUS_SCALE
    phase = np.einsum("...spd,fd->...spf", scaled, RFF_FREQUENCIES)
    characteristic = np.concatenate(
        [np.cos(phase).mean(axis=-2), np.sin(phase).mean(axis=-2)], axis=-1
    )
    result = np.concatenate([summary, characteristic], axis=-1)
    if result.shape[-1] != FEATURE_DIMENSION:
        raise RuntimeError("internal cloud-feature dimension changed")
    return result.astype(np.float32)


def _mixture_marginal_quantiles(
    weight: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> np.ndarray:
    """Numerically invert a normalized univariate Gaussian-mixture CDF."""

    lower = np.full(len(QUANTILE_LEVELS), np.min(mean - 9.0 * standard_deviation))
    upper = np.full(len(QUANTILE_LEVELS), np.max(mean + 9.0 * standard_deviation))
    for _ in range(56):
        midpoint = 0.5 * (lower + upper)
        cdf = np.sum(
            weight[None, :]
            * ndtr(
                (midpoint[:, None] - mean[None, :])
                / standard_deviation[None, :]
            ),
            axis=1,
        )
        lower = np.where(cdf < QUANTILE_LEVELS, midpoint, lower)
        upper = np.where(cdf < QUANTILE_LEVELS, upper, midpoint)
    return 0.5 * (lower + upper)


def gaussian_mixture_features(
    weight: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    source_slot: int,
) -> np.ndarray:
    """Return the exact v3 features of one normalized Gaussian mixture.

    This is the population counterpart of :func:`cloud_features`.  It removes
    finite-cloud noise from synthetic training data while using the identical
    physical scaling, Fourier frequencies and feature ordering as a real
    empirical NICER cloud.
    """

    weight = np.asarray(weight, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if weight.ndim != 1 or mean.shape != (len(weight), 2):
        raise ValueError("mixture weights and means have inconsistent shapes")
    if covariance.shape != (len(weight), 2, 2):
        raise ValueError("mixture covariance has the wrong shape")
    active = weight > 0
    weight = weight[active]
    mean = mean[active]
    covariance = covariance[active]
    weight = weight / weight.sum()
    if not 0 <= source_slot < SOURCE_SLOTS:
        raise ValueError("source slot must be 0, 1, or 2")

    population_mean = np.sum(weight[:, None] * mean, axis=0)
    offset = mean - population_mean
    total_covariance = np.sum(
        weight[:, None, None]
        * (
            covariance
            + np.einsum("ki,kj->kij", offset, offset)
        ),
        axis=0,
    )
    standard_deviation = np.sqrt(np.maximum(np.diag(total_covariance), 1.0e-12))
    correlation = total_covariance[0, 1] / (
        standard_deviation[0] * standard_deviation[1]
    )

    third = np.empty(2, dtype=np.float64)
    fourth = np.empty(2, dtype=np.float64)
    quantile = np.empty((len(QUANTILE_LEVELS), 2), dtype=np.float64)
    for coordinate in range(2):
        variance = covariance[:, coordinate, coordinate]
        delta = offset[:, coordinate]
        third[coordinate] = np.sum(
            weight * (delta**3 + 3.0 * delta * variance)
        )
        fourth[coordinate] = np.sum(
            weight
            * (delta**4 + 6.0 * delta**2 * variance + 3.0 * variance**2)
        )
        quantile[:, coordinate] = _mixture_marginal_quantiles(
            weight, mean[:, coordinate], np.sqrt(variance)
        )
    skewness = third / standard_deviation**3
    excess_kurtosis = fourth / standard_deviation**4 - 3.0
    cross_skewness = np.asarray(
        [
            np.sum(
                weight
                * (
                    offset[:, 0] ** 2 * offset[:, 1]
                    + offset[:, 1] * covariance[:, 0, 0]
                    + 2.0 * offset[:, 0] * covariance[:, 0, 1]
                )
            )
            / (standard_deviation[0] ** 2 * standard_deviation[1]),
            np.sum(
                weight
                * (
                    offset[:, 0] * offset[:, 1] ** 2
                    + offset[:, 0] * covariance[:, 1, 1]
                    + 2.0 * offset[:, 1] * covariance[:, 0, 1]
                )
            )
            / (standard_deviation[0] * standard_deviation[1] ** 2),
        ]
    )
    identity = np.eye(SOURCE_SLOTS, dtype=np.float64)[source_slot]
    summary = np.concatenate(
        [
            population_mean,
            standard_deviation,
            quantile.reshape(-1),
            [correlation],
            skewness,
            excess_kurtosis,
            cross_skewness,
            identity,
        ]
    )

    coordinate_scale = np.asarray([MASS_SCALE, RADIUS_SCALE])
    coordinate_centre = np.asarray([MASS_CENTRE, RADIUS_CENTRE])
    scaled_mean = (mean - coordinate_centre) / coordinate_scale
    scaled_covariance = covariance / (
        coordinate_scale[None, :, None] * coordinate_scale[None, None, :]
    )
    phase = scaled_mean @ RFF_FREQUENCIES.T
    attenuation = np.exp(
        -0.5
        * np.einsum(
            "fd,kde,fe->kf",
            RFF_FREQUENCIES,
            scaled_covariance,
            RFF_FREQUENCIES,
        )
    )
    characteristic_real = np.sum(
        weight[:, None] * attenuation * np.cos(phase), axis=0
    )
    characteristic_imaginary = np.sum(
        weight[:, None] * attenuation * np.sin(phase), axis=0
    )
    result = np.concatenate(
        [summary, characteristic_real, characteristic_imaginary]
    )
    if result.shape != (FEATURE_DIMENSION,) or not np.isfinite(result).all():
        raise RuntimeError("invalid analytic Gaussian-mixture features")
    return result.astype(np.float32)


__all__ = [
    "FEATURE_DIMENSION",
    "FEATURE_SCHEMA",
    "RFF_FREQUENCIES",
    "cloud_features",
    "feature_specification",
    "feature_specification_json",
    "gaussian_mixture_features",
]
