"""Portable corrected target for registered DDB-family EOS models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from eos import get_eos
from likelihoods import JointLikelihood
from likelihoods.gw170817 import load_kde
from likelihoods.nicer import build_nicer_grid
from tov import solve_stable_branch
from workflows.data_gate import (
    validate_observational_data,
    validate_substituted_pulsar_data,
)
from workflows.nuclear_scenarios import (
    nuclear_observation,
    shift_frozen_a1_log_likelihood,
)
from workflows.source_scenarios import (
    nicer_source_paths,
    source_substitution,
)

BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
HYPERONIC_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 2500)
COMPONENT_NAMES = ("nuclear", "maximum_mass", "nicer", "gw170817", "pqcd")


def _solve_or_none(energy, pressure):
    try:
        return solve_stable_branch(energy, pressure)
    except (ValueError, FloatingPointError):
        return None


def _evaluate_model_row(model_key, parameters, nuclear_observables, target):
    """Evaluate one non-vectorized EOS row without backend-specific physics."""

    plugin = get_eos(model_key)
    try:
        density, energy, pressure = plugin.core_eos(parameters)
        branch = solve_stable_branch(
            energy,
            pressure,
            require_monotonic_graft=plugin.metadata.get(
                "require_monotonic_tov_graft", False
            ),
        )
    except (ValueError, FloatingPointError, RuntimeError):
        branch = None
        density = energy = pressure = np.empty(0, dtype=np.float64)
    return target.evaluate(
        parameters,
        nuclear_observables,
        density,
        energy,
        pressure,
        branch,
    )


@dataclass
class A1Problem:
    """Shared target for registered EOS and declared data substitutions."""

    target: JointLikelihood
    workers: int = 1
    chunk_size: int = 2500
    nuclear_scenario: str = "A1"
    source_scenario: str = "A1"
    model: str = "ddb"

    @property
    def eos_plugin(self):
        return get_eos(self.model)

    @property
    def parameter_names(self):
        return self.eos_plugin.parameter_names

    @property
    def prior_low(self):
        return self.eos_plugin.prior_low

    @property
    def prior_high(self):
        return self.eos_plugin.prior_high

    @property
    def _buckets(self):
        return HYPERONIC_BUCKETS if self.model == "ddb-hyperonic" else BUCKETS

    @classmethod
    def from_data_root(
        cls,
        data_root: str | Path,
        workers=1,
        verify_data=True,
        nuclear_scenario="A1",
        source_scenario="A1",
        source_data_root: str | Path | None = None,
        model="ddb",
    ):
        if workers < 1:
            raise ValueError("workers must be positive")
        plugin = get_eos(model)
        if not plugin.has_nuclear_observables:
            raise ValueError(
                f"EOS plug-in {model!r} cannot evaluate the paper's nuclear target"
            )
        paths = (
            validate_observational_data(data_root)
            if verify_data
            else {
                name: Path(data_root) / name
                for name in (
                    "J0030_2spot_RM.txt",
                    "J0740_NICERXMM_full_mr.txt",
                    "J0437_post_equal_weights.dat",
                    "GW170817_GWTC-1.hdf5",
                )
            }
        )
        replacement = source_substitution(source_scenario)
        substituted_paths = None
        if replacement is not None:
            substitution_root = Path(source_data_root or data_root)
            substituted_paths = (
                validate_substituted_pulsar_data(
                    substitution_root, replacement.filename
                )
                if verify_data
                else {replacement.filename: substitution_root / replacement.filename}
            )
        source_records = nicer_source_paths(
            paths, source_scenario, substituted_paths
        )
        common = dict(nM=150, nR=150, max_rows=20_000, bw=0.08, seed=0)
        nicer = tuple(
            build_nicer_grid(
                path,
                mcol=spec.mass_column,
                rcol=spec.radius_column,
                wcol_or_None=spec.weight_column,
                **common,
            )
            for path, spec in source_records
        )
        gw_kernel, gw_chirp_mass = load_kde(
            paths["GW170817_GWTC-1.hdf5"], subsample=4000, seed=0
        )
        return cls(
            target=JointLikelihood(
                nuclear_observation=nuclear_observation(nuclear_scenario),
                nicer_interpolators=nicer,
                gw_kernel=gw_kernel,
                gw_chirp_mass=gw_chirp_mass,
            ),
            workers=workers,
            nuclear_scenario=nuclear_scenario,
            source_scenario=source_scenario,
            model=model,
        )

    def prior_transform(self, unit):
        unit = np.asarray(unit, dtype=np.float64)
        return self.prior_low + unit * (self.prior_high - self.prior_low)

    def evaluate_batch(self, theta, return_components=False):
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        plugin = self.eos_plugin
        if theta.ndim != 2 or theta.shape[1] != plugin.ndim:
            raise ValueError(
                f"expected theta shape (N,{plugin.ndim}), got {theta.shape}"
            )
        total = np.full(len(theta), -1e100, dtype=np.float64)
        components = np.full(
            (len(theta), len(COMPONENT_NAMES)), np.nan, dtype=np.float64
        )
        for start in range(0, len(theta), self.chunk_size):
            stop = min(start + self.chunk_size, len(theta))
            local = theta[start:stop]
            count = len(local)
            try:
                bucket = next(size for size in self._buckets if size >= count)
            except StopIteration as error:
                raise ValueError(
                    f"chunk size {count} exceeds largest JAX bucket {self._buckets[-1]}"
                ) from error
            padded = (
                local
                if count == bucket
                else np.vstack(
                    [local, np.repeat(local[-1:], bucket - count, axis=0)]
                )
            )
            nmp = plugin.nuclear_observables_batch(padded)[:count]
            if plugin.metadata.get("row_parallel_forward", False):
                terms_by_row = Parallel(
                    n_jobs=min(self.workers, count), batch_size=1
                )(
                    delayed(_evaluate_model_row)(
                        plugin.key,
                        local[row],
                        nmp[row],
                        self.target,
                    )
                    for row in range(count)
                )
            else:
                density, energy, pressure = plugin.core_eos_batch(padded)
                density = density[:count]
                energy = energy[:count]
                pressure = pressure[:count]
                curves = Parallel(
                    n_jobs=min(self.workers, count), batch_size=8
                )(
                    delayed(_solve_or_none)(energy[row], pressure[row])
                    for row in range(count)
                )
                terms_by_row = Parallel(
                    n_jobs=min(self.workers, count),
                    prefer="threads",
                    batch_size=8,
                )(
                    delayed(self.target.evaluate)(
                        local[row],
                        nmp[row],
                        density[row],
                        energy[row],
                        pressure[row],
                        branch,
                    )
                    for row, branch in enumerate(curves)
                )
            for row, terms in enumerate(terms_by_row):
                if not np.isfinite(terms.total):
                    continue
                output_row = start + row
                total[output_row] = terms.total
                components[output_row] = (
                    terms.nuclear,
                    terms.maximum_mass,
                    terms.nicer,
                    terms.gw170817,
                    terms.pqcd,
                )
        return (total, components) if return_components else total

    def evaluate_evidence_cache_batch(self, theta):
        """Return nuclear predictions and the non-nuclear likelihood target.

        The Evidence Network amortizes over the seven nuclear-observation
        coordinates.  Its cache therefore stores the forward prediction
        ``(rho0, E0, K0, J, P0.08, P0.12, P0.16)`` separately from the
        astrophysical likelihood.  The latter is assembled from the same
        component values used by :meth:`evaluate_batch`; no Evidence-Network
        likelihood implementation exists.
        """

        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        plugin = self.eos_plugin
        if theta.ndim != 2 or theta.shape[1] != plugin.ndim:
            raise ValueError(
                f"expected theta shape (N,{plugin.ndim}), got {theta.shape}"
            )
        _total, components = self.evaluate_batch(theta, return_components=True)
        observation_prediction = np.empty((len(theta), 7), dtype=np.float64)
        try:
            rho0_column = plugin.parameter_names.index("rho0")
        except ValueError as error:
            raise ValueError(
                f"EOS plug-in {plugin.key!r} has no rho0 parameter"
            ) from error
        observation_prediction[:, 0] = theta[:, rho0_column]
        for start in range(0, len(theta), self.chunk_size):
            stop = min(start + self.chunk_size, len(theta))
            local = theta[start:stop]
            count = len(local)
            bucket = next(size for size in self._buckets if size >= count)
            padded = (
                local
                if count == bucket
                else np.vstack(
                    [local, np.repeat(local[-1:], bucket - count, axis=0)]
                )
            )
            observation_prediction[start:stop, 1:] = (
                plugin.nuclear_observables_batch(padded)[:count]
            )
        log_astrophysical = np.sum(components[:, 1:], axis=1)
        return observation_prediction, log_astrophysical, components[:, 1:]

    def certify(self, certificate: str | Path, tolerance=1e-4) -> dict[str, Any]:
        expected_key = (
            "fixed_logl"
            if self.source_scenario == "A1"
            else f"fixed_logl_{self.source_scenario}"
        )
        with np.load(certificate, allow_pickle=False) as saved:
            if expected_key not in saved.files:
                raise ValueError(
                    f"certificate does not contain target {expected_key!r} for "
                    f"source scenario {self.source_scenario}"
                )
            theta = np.asarray(saved["theta"], dtype=np.float64)
            expected = np.asarray(saved[expected_key], dtype=np.float64)
        if expected.shape != (len(theta),):
            raise ValueError("certificate theta and likelihood rows do not match")
        target_observation = np.asarray(
            self.target.nuclear_observation, dtype=np.float64
        )
        basis = f"frozen_{self.source_scenario}_full_target"
        if np.array_equal(target_observation, nuclear_observation("A1")):
            pass
        else:
            bucket = next(size for size in self._buckets if size >= len(theta))
            padded = np.vstack(
                [theta, np.repeat(theta[-1:], bucket - len(theta), axis=0)]
            )
            nmp = self.eos_plugin.nuclear_observables_batch(padded)[: len(theta)]
            expected = shift_frozen_a1_log_likelihood(
                expected,
                theta,
                nmp,
                target_observation,
            )
            basis += "_plus_exact_nuclear_delta"
        actual = self.evaluate_batch(theta)
        expected_valid = expected > -1e50
        actual_valid = actual > -1e50
        valid_mask_equal = bool(np.array_equal(expected_valid, actual_valid))
        mismatch_indices = np.flatnonzero(expected_valid != actual_valid)
        maximum_error = (
            float(np.max(np.abs(actual[expected_valid] - expected[expected_valid])))
            if valid_mask_equal and expected_valid.any()
            else float("inf")
        )
        return {
            "status": (
                "PASS"
                if valid_mask_equal and maximum_error <= tolerance
                else "FAIL"
            ),
            "rows": int(len(theta)),
            "expected_valid_rows": int(expected_valid.sum()),
            "actual_valid_rows": int(actual_valid.sum()),
            "valid_mask_equal": valid_mask_equal,
            "valid_mask_mismatch_indices": mismatch_indices.tolist(),
            "max_abs_log_likelihood_error": maximum_error,
            "tolerance": tolerance,
            "nuclear_scenario": self.nuclear_scenario,
            "nuclear_observation": target_observation.tolist(),
            "source_scenario": self.source_scenario,
            "model": self.model,
            "certification_basis": basis,
        }
