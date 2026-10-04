"""Named nuclear-observation configurations used by the paper."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

from likelihoods.nuclear import (
    A1_OBSERVATION,
    gaussian_log_likelihood,
    observable_vector,
)


NUCLEAR_REPLACEMENTS = MappingProxyType(
    {
        "A1": {},
        "K0_200": {2: 200.0},
        "K0_260": {2: 260.0},
        "Jsym_29": {3: 29.0},
        "Jsym_36": {3: 36.0},
    }
)
NUCLEAR_SCENARIO_NAMES = tuple(NUCLEAR_REPLACEMENTS)


def nuclear_observation(scenario: str) -> np.ndarray:
    """Return a defensive copy of one named seven-coordinate observation."""

    try:
        replacements = NUCLEAR_REPLACEMENTS[scenario]
    except KeyError as error:
        available = ", ".join(NUCLEAR_SCENARIO_NAMES)
        raise KeyError(
            f"unknown nuclear scenario {scenario!r}; available: {available}"
        ) from error
    observation = A1_OBSERVATION.copy()
    for column, value in replacements.items():
        observation[column] = value
    return observation


def shift_frozen_a1_log_likelihood(
    reference_log_likelihood: np.ndarray,
    theta: np.ndarray,
    nuclear_matter_observables: np.ndarray,
    target_observation: np.ndarray,
) -> np.ndarray:
    """Transform the frozen A1 target by only the declared nuclear factor.

    Astrophysical terms and the validity mask are invariant under a nuclear
    observation change. The shifted reference can therefore be derived exactly
    by adding the target-minus-A1 Gaussian nuclear log-likelihood.
    """

    reference = np.asarray(reference_log_likelihood, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    nmp = np.asarray(nuclear_matter_observables, dtype=np.float64)
    if reference.shape != (len(theta),):
        raise ValueError("reference log likelihood and theta rows do not match")
    values = observable_vector(theta, nmp)
    delta = gaussian_log_likelihood(
        values, observation=target_observation
    ) - gaussian_log_likelihood(values, observation=A1_OBSERVATION)
    shifted = reference.copy()
    valid = shifted > -1e50
    shifted[valid] += delta[valid]
    return shifted
