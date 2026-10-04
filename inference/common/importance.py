"""Stable importance-weight, evidence, and ESS calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImportanceWeights:
    log_weight: np.ndarray
    normalized_weight: np.ndarray
    ess: float
    log_evidence: float
    log_evidence_standard_error: float
    finite_count: int

    def resample(self, values, size=None, seed=0):
        values = np.asarray(values)
        if len(values) != len(self.normalized_weight):
            raise ValueError("value and importance-weight lengths differ")
        size = len(values) if size is None else int(size)
        index = np.random.default_rng(seed).choice(
            len(values), size=size, replace=True, p=self.normalized_weight
        )
        return values[index], index


def compute_importance_weights(log_target, log_proposal):
    """Normalize ``target/proposal`` and estimate ESS and evidence.

    ``log_target`` must include the prior density.  ``log_proposal`` must be a
    normalized density in the same physical coordinates.
    """

    log_target = np.asarray(log_target, dtype=np.float64)
    log_proposal = np.asarray(log_proposal, dtype=np.float64)
    if log_target.shape != log_proposal.shape or log_target.ndim != 1:
        raise ValueError("target and proposal log densities must be matching 1D arrays")
    log_weight = log_target - log_proposal
    finite = np.isfinite(log_weight)
    finite_count = int(finite.sum())
    if finite_count == 0:
        raise ValueError("no finite importance weights")
    maximum = float(np.max(log_weight[finite]))
    scaled = np.zeros_like(log_weight)
    scaled[finite] = np.exp(log_weight[finite] - maximum)
    scaled_sum = float(scaled.sum())
    normalized = scaled / scaled_sum
    ess = float(1.0 / np.sum(normalized**2))
    log_evidence = maximum + np.log(scaled_sum) - np.log(len(log_weight))

    # Delta-method Monte Carlo error for log(mean weight), evaluated in the
    # shifted scale to avoid overflow.  The proposal draw count, including
    # zero-weight/out-of-support rows, is the denominator by construction.
    mean_scaled = float(np.mean(scaled))
    if len(scaled) > 1 and mean_scaled > 0:
        variance_of_mean = float(np.var(scaled, ddof=1) / len(scaled))
        log_evidence_standard_error = np.sqrt(variance_of_mean) / mean_scaled
    else:
        log_evidence_standard_error = float("nan")
    return ImportanceWeights(
        log_weight=log_weight,
        normalized_weight=normalized,
        ess=ess,
        log_evidence=float(log_evidence),
        log_evidence_standard_error=float(log_evidence_standard_error),
        finite_count=finite_count,
    )
