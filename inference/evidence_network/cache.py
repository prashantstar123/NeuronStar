"""Versioned cache and direct-Monte-Carlo target for Evidence Networks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

NUCLEAR_COORDINATE_NAMES = (
    "rho0",
    "E0",
    "K0",
    "Jsym0",
    "P_0p08",
    "P_0p12",
    "P_0p16",
)
DEFAULT_NUCLEAR_OBSERVATION = np.array(
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
DEFAULT_NUCLEAR_SIGMA = np.array(
    [
        0.005,
        0.2,
        40.0,
        1.8,
        0.194285714285714,
        0.608571428571429,
        1.38285714285714,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class DirectEvidence:
    log_evidence: float
    effective_sample_size: float
    monte_carlo_error: float
    rows: int


@dataclass(frozen=True)
class RestrictedTilt:
    index: np.ndarray
    normalized_weight: np.ndarray
    log_normalization: float
    effective_sample_size: float


@dataclass(frozen=True)
class EvidenceCache:
    """Importance-corrected joint bank for evidence amortization.

    ``log_prior_correction`` is ``log prior - log proposal``.  The denominator
    includes all proposed rows, including rows for which the stellar target was
    invalid; this is essential when multiple proposal streams are appended.
    """

    nuclear_prediction: np.ndarray
    log_astrophysical: np.ndarray
    log_prior_correction: np.ndarray
    log_proposal_denominator: float
    source: str | None = None

    def __post_init__(self) -> None:
        prediction = np.asarray(self.nuclear_prediction, dtype=np.float64)
        astro = np.asarray(self.log_astrophysical, dtype=np.float64)
        correction = np.asarray(self.log_prior_correction, dtype=np.float64)
        if prediction.ndim != 2 or prediction.shape[1] != 7:
            raise ValueError("nuclear predictions must have shape (N,7)")
        if astro.shape != (len(prediction),) or correction.shape != (len(prediction),):
            raise ValueError("cache likelihood/correction lengths do not match X")
        if not np.isfinite(prediction).all():
            raise ValueError("cache contains non-finite nuclear predictions")
        if np.isnan(astro).any() or np.isnan(correction).any():
            raise ValueError("cache contains NaN likelihood or proposal corrections")
        if not np.isfinite(self.log_proposal_denominator):
            raise ValueError("cache proposal denominator is non-finite")
        object.__setattr__(self, "nuclear_prediction", prediction)
        object.__setattr__(self, "log_astrophysical", astro)
        object.__setattr__(self, "log_prior_correction", correction)

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceCache":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            required = {"X", "LA", "LPC", "lpc_all_lse"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"evidence cache is missing arrays: {missing}")
            return cls(
                nuclear_prediction=np.asarray(archive["X"], dtype=np.float64),
                log_astrophysical=np.asarray(archive["LA"], dtype=np.float64),
                log_prior_correction=np.asarray(
                    archive["LPC"], dtype=np.float64
                ),
                log_proposal_denominator=float(archive["lpc_all_lse"]),
                source=str(path),
            )

    @property
    def rows(self) -> int:
        return len(self.nuclear_prediction)

    def log_nuclear_likelihood(
        self,
        observation: np.ndarray,
        sigma: np.ndarray = DEFAULT_NUCLEAR_SIGMA,
    ) -> np.ndarray:
        """Production unnormalized seven-factor Gaussian log likelihood."""

        observation = np.asarray(observation, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        if observation.shape != (7,) or sigma.shape != (7,) or np.any(sigma <= 0):
            raise ValueError("nuclear observation and sigma must be positive 7-vectors")
        standardized = (observation - self.nuclear_prediction) / sigma
        return -0.5 * np.sum(standardized * standardized, axis=1)

    def direct_evidence(
        self,
        observation: np.ndarray = DEFAULT_NUCLEAR_OBSERVATION,
        sigma: np.ndarray = DEFAULT_NUCLEAR_SIGMA,
        *,
        row_mask: np.ndarray | None = None,
    ) -> DirectEvidence:
        log_weight = (
            self.log_prior_correction
            + self.log_astrophysical
            + self.log_nuclear_likelihood(observation, sigma)
        )
        if row_mask is not None:
            row_mask = np.asarray(row_mask, dtype=bool)
            if row_mask.shape != (self.rows,):
                raise ValueError("evidence row mask has the wrong shape")
            log_weight = log_weight[row_mask]
        finite = np.isfinite(log_weight)
        if not finite.any():
            raise ValueError("no finite rows contribute to the evidence")
        selected = log_weight[finite]
        normalization = float(logsumexp(selected))
        weight = np.exp(selected - normalization)
        effective_sample_size = float(1.0 / np.sum(weight * weight))
        return DirectEvidence(
            log_evidence=normalization - self.log_proposal_denominator,
            effective_sample_size=effective_sample_size,
            monte_carlo_error=effective_sample_size**-0.5,
            rows=int(finite.sum()),
        )

    def restricted_tilt(
        self,
        observation: np.ndarray = DEFAULT_NUCLEAR_OBSERVATION,
        sigma: np.ndarray = DEFAULT_NUCLEAR_SIGMA,
        *,
        radius: float = 6.0,
    ) -> RestrictedTilt:
        observation = np.asarray(observation, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        if radius <= 0.0:
            raise ValueError("restriction radius must be positive")
        inside = np.max(
            np.abs((self.nuclear_prediction - observation) / sigma), axis=1
        ) <= radius
        log_tilt = self.log_prior_correction + self.log_astrophysical
        valid = inside & np.isfinite(log_tilt)
        index = np.flatnonzero(valid)
        if len(index) == 0:
            raise ValueError("restricted Evidence-Network tilt is empty")
        log_sum = float(logsumexp(log_tilt[index]))
        weight = np.exp(log_tilt[index] - log_sum)
        return RestrictedTilt(
            index=index,
            normalized_weight=weight,
            log_normalization=log_sum - self.log_proposal_denominator,
            effective_sample_size=float(1.0 / np.sum(weight * weight)),
        )

    def sample_smoothed_tilt(
        self,
        size: int,
        *,
        seed: int,
        observation: np.ndarray = DEFAULT_NUCLEAR_OBSERVATION,
        sigma: np.ndarray = DEFAULT_NUCLEAR_SIGMA,
        radius: float = 6.0,
    ) -> tuple[np.ndarray, RestrictedTilt]:
        if size < 1:
            raise ValueError("training sample size must be positive")
        sigma = np.asarray(sigma, dtype=np.float64)
        tilt = self.restricted_tilt(observation, sigma, radius=radius)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            tilt.index, size=int(size), replace=True, p=tilt.normalized_weight
        )
        targets = self.nuclear_prediction[chosen] + rng.normal(
            0.0, 1.0, (int(size), 7)
        ) * sigma[None, :]
        return targets, tilt
