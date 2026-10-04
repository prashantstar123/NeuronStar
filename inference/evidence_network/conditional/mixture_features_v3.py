"""Canonical component features for conditional Evidence Network v3."""

from __future__ import annotations

import numpy as np


COMPONENT_SPECIFICATION = {
    "fields": [
        "weight",
        "standardized_mass_mean",
        "standardized_radius_mean",
        "log_mass_sigma_over_0.1",
        "log_radius_sigma_over_0.7",
        "atanh_correlation_over_2",
    ],
    "mass_centre": 1.55,
    "mass_scale": 0.65,
    "radius_centre_km": 12.5,
    "radius_scale_km": 3.0,
    "mass_sigma_reference": 0.1,
    "radius_sigma_reference_km": 0.7,
    "correlation_clip": 0.999,
    "component_order": "permutation_invariant",
}


def component_features(
    weights: np.ndarray, means: np.ndarray, covariance: np.ndarray
) -> np.ndarray:
    """Convert normalized Gaussian-mixture parameters to fixed physical units."""
    sigma_mass = np.sqrt(np.maximum(covariance[..., 0, 0], 1.0e-10))
    sigma_radius = np.sqrt(np.maximum(covariance[..., 1, 1], 1.0e-10))
    correlation = np.clip(
        covariance[..., 0, 1] / (sigma_mass * sigma_radius), -0.999, 0.999
    )
    return np.stack(
        [
            weights,
            (means[..., 0] - 1.55) / 0.65,
            (means[..., 1] - 12.5) / 3.0,
            np.log(sigma_mass / 0.1),
            np.log(sigma_radius / 0.7),
            np.arctanh(correlation) / 2.0,
        ],
        axis=-1,
    ).astype(np.float32)


__all__ = ["COMPONENT_SPECIFICATION", "component_features"]
