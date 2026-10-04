"""GPU-batched normalized mixture likelihoods for conditional EN v3 labels."""

from __future__ import annotations

import numpy as np
import torch


class BatchedMixtureNICER:
    """Evaluate normalized Gaussian-mixture source densities on M--R curves.

    Mixtures are generated independently of any reported NICER replacement.
    Each component is a normalized two-dimensional Gaussian and the component
    weights sum to one, so the density normalization required for absolute
    evidence is known analytically.
    """

    def __init__(self, *, relative_floor: float = 1.0e-6) -> None:
        if not 0.0 < relative_floor < 1.0:
            raise ValueError("relative density floor must lie in (0,1)")
        self.log_relative_floor = float(np.log(relative_floor))

    @torch.inference_mode()
    def evaluate_chunk(
        self,
        weights: torch.Tensor,
        means: torch.Tensor,
        covariance: torch.Tensor,
        quadrature,
    ) -> torch.Tensor:
        """Return ``(scenario,row)`` marginalized log likelihoods.

        ``weights``, ``means`` and ``covariance`` have shapes ``(S,K)``,
        ``(S,K,2)`` and ``(S,K,2,2)``.  Zero-weight padded components are
        allowed.  ``quadrature`` is produced by ``BatchedShiftedNICER`` and is
        reused so the synthetic and paper likelihoods use identical stellar
        curve integration.
        """

        if weights.ndim != 2:
            raise ValueError("mixture weights must have shape (scenario,component)")
        scenarios, components = weights.shape
        if means.shape != (scenarios, components, 2):
            raise ValueError("mixture means have the wrong shape")
        if covariance.shape != (scenarios, components, 2, 2):
            raise ValueError("mixture covariance has the wrong shape")
        if torch.any(weights < 0):
            raise ValueError("mixture weights must be non-negative")
        total = weights.sum(dim=1)
        if torch.any(torch.abs(total - 1.0) > 2.0e-5):
            raise ValueError("mixture weights must sum to one")

        mass, radius, valid, safe_lower, safe_upper = quadrature
        determinant = (
            covariance[:, :, 0, 0] * covariance[:, :, 1, 1]
            - covariance[:, :, 0, 1] * covariance[:, :, 1, 0]
        )
        if torch.any((determinant <= 0) & (weights > 0)):
            raise ValueError("active mixture covariance is not positive definite")
        safe_determinant = determinant.clamp_min(1.0e-20)
        precision00 = covariance[:, :, 1, 1] / safe_determinant
        precision11 = covariance[:, :, 0, 0] / safe_determinant
        precision01 = -covariance[:, :, 0, 1] / safe_determinant
        log_component = torch.where(
            weights > 0,
            torch.log(weights.clamp_min(1.0e-30))
            - np.log(2.0 * np.pi)
            - 0.5 * torch.log(safe_determinant),
            torch.full_like(weights, -torch.inf),
        )

        delta_mass = (
            mass[None, :, :, None] - means[:, None, None, :, 0]
        )
        delta_radius = (
            radius[None, :, :, None] - means[:, None, None, :, 1]
        )
        mahalanobis = (
            precision00[:, None, None, :] * delta_mass**2
            + 2.0
            * precision01[:, None, None, :]
            * delta_mass
            * delta_radius
            + precision11[:, None, None, :] * delta_radius**2
        )
        log_density = torch.logsumexp(
            log_component[:, None, None, :] - 0.5 * mahalanobis,
            dim=3,
        )
        # Match the certified NICER grid's relative 1e-6 floor without using
        # any target-source file.  The component-peak bound is conservative
        # for overlapping mixtures and changes only negligible far tails.
        peak_bound = torch.logsumexp(log_component, dim=1)
        log_floor = peak_bound[:, None, None] + self.log_relative_floor
        log_density = torch.maximum(log_density, log_floor)
        integral = torch.trapezoid(
            torch.exp(log_density), mass[None, :, :], dim=2
        ) / (safe_upper - safe_lower)[None, :]
        return torch.where(
            valid[None, :],
            torch.log(integral + 1.0e-30),
            torch.full_like(integral, -1.0e30),
        )


__all__ = ["BatchedMixtureNICER"]
