"""Public contract for equation-of-state plug-ins."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np


CoreEvaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
BatchCoreEvaluator = Callable[
    [np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
]
ObservableEvaluator = Callable[[np.ndarray], np.ndarray]


def _readonly_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class EOSPlugin:
    """A small, backend-independent EOS definition.

    The mandatory callable returns a core EOS table as density, energy
    density, and pressure. Energy density and pressure are in MeV/fm^3. A
    vectorized implementation and nuclear-observable maps are optional so a
    new EOS can be registered before backend-specific acceleration is added.
    """

    key: str
    display_name: str
    parameter_names: tuple[str, ...]
    prior_low: np.ndarray
    prior_high: np.ndarray
    core_eos_fn: CoreEvaluator = field(repr=False)
    core_eos_batch_fn: BatchCoreEvaluator | None = field(default=None, repr=False)
    nuclear_observables_fn: ObservableEvaluator | None = field(
        default=None, repr=False
    )
    nuclear_observables_batch_fn: ObservableEvaluator | None = field(
        default=None, repr=False
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = self.key.strip()
        if not key or any(character.isspace() for character in key):
            raise ValueError("EOS plug-in key must be non-empty and contain no spaces")
        display_name = self.display_name.strip()
        if not display_name:
            raise ValueError("EOS plug-in display name must be non-empty")
        names = tuple(str(name).strip() for name in self.parameter_names)
        if not names or any(not name for name in names):
            raise ValueError("EOS parameter names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("EOS parameter names must be unique")
        low = _readonly_vector(self.prior_low, "prior_low")
        high = _readonly_vector(self.prior_high, "prior_high")
        if len(names) != len(low) or len(low) != len(high):
            raise ValueError("parameter names and prior bounds must have equal length")
        if np.any(low >= high):
            raise ValueError("every lower prior bound must be below its upper bound")
        if not callable(self.core_eos_fn):
            raise TypeError("core_eos_fn must be callable")
        for name in (
            "core_eos_batch_fn",
            "nuclear_observables_fn",
            "nuclear_observables_batch_fn",
        ):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "prior_low", low)
        object.__setattr__(self, "prior_high", high)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def ndim(self) -> int:
        return len(self.parameter_names)

    @property
    def has_nuclear_observables(self) -> bool:
        return self.nuclear_observables_fn is not None

    def prior_transform(self, unit: np.ndarray) -> np.ndarray:
        unit = np.asarray(unit, dtype=np.float64)
        if unit.shape[-1:] != (self.ndim,):
            raise ValueError(
                f"expected final unit-cube dimension {self.ndim}, got {unit.shape}"
            )
        if np.any(~np.isfinite(unit)) or np.any((unit < 0.0) | (unit > 1.0)):
            raise ValueError("unit-cube coordinates must be finite and in [0, 1]")
        return self.prior_low + unit * (self.prior_high - self.prior_low)

    def core_eos(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.ndim,):
            raise ValueError(f"expected theta shape ({self.ndim},), got {theta.shape}")
        density, energy, pressure = self.core_eos_fn(theta)
        density = np.asarray(density, dtype=np.float64)
        energy = np.asarray(energy, dtype=np.float64)
        pressure = np.asarray(pressure, dtype=np.float64)
        if density.ndim != 1 or energy.shape != density.shape or pressure.shape != density.shape:
            raise ValueError("core EOS must return three equal-length 1D arrays")
        return density, energy, pressure

    def core_eos_batch(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 2 or theta.shape[1] != self.ndim:
            raise ValueError(
                f"expected theta shape (N,{self.ndim}), got {theta.shape}"
            )
        if self.core_eos_batch_fn is not None:
            density, energy, pressure = self.core_eos_batch_fn(theta)
            outputs = tuple(
                np.asarray(value, dtype=np.float64)
                for value in (density, energy, pressure)
            )
        else:
            rows = [self.core_eos(row) for row in theta]
            outputs = tuple(np.stack(items, axis=0) for items in zip(*rows))
        if any(value.ndim != 2 or value.shape[0] != len(theta) for value in outputs):
            raise ValueError("batched core EOS outputs must each have shape (N, grid)")
        if outputs[1].shape != outputs[0].shape or outputs[2].shape != outputs[0].shape:
            raise ValueError("batched core EOS outputs must have identical shapes")
        return outputs

    def nuclear_observables(self, theta: np.ndarray) -> np.ndarray:
        if self.nuclear_observables_fn is None:
            raise NotImplementedError(
                f"EOS plug-in {self.key!r} has no nuclear-observable map"
            )
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.ndim,):
            raise ValueError(f"expected theta shape ({self.ndim},), got {theta.shape}")
        return np.asarray(self.nuclear_observables_fn(theta), dtype=np.float64)

    def nuclear_observables_batch(self, theta: np.ndarray) -> np.ndarray:
        if self.nuclear_observables_batch_fn is not None:
            theta = np.asarray(theta, dtype=np.float64)
            if theta.ndim != 2 or theta.shape[1] != self.ndim:
                raise ValueError(
                    f"expected theta shape (N,{self.ndim}), got {theta.shape}"
                )
            return np.asarray(
                self.nuclear_observables_batch_fn(theta), dtype=np.float64
            )
        return np.stack([self.nuclear_observables(row) for row in theta], axis=0)
