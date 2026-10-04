#!/usr/bin/env python3
"""Build broad-shape jointly amortized conditional-EN v3 scenarios.

This generator is deliberately independent of J0614, J1231 and J1614.  It
draws normalized one-to-four-component mass--radius densities over a declared
physical domain, including narrow, correlated, skewed, heavy-core/tail and
multimodal cases.  Exact labels are evaluated on the independent full-prior
physics bank; no posterior estimator or conventional sampler enters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from inference.evidence_network.cache import (  # noqa: E402
    DEFAULT_NUCLEAR_OBSERVATION,
    DEFAULT_NUCLEAR_SIGMA,
)
from inference.evidence_network.conditional.batched_mixture_v3 import (  # noqa: E402
    BatchedMixtureNICER,
)
from inference.evidence_network.conditional.batched_nicer import (  # noqa: E402
    BatchedShiftedNICER,
)
from inference.evidence_network.conditional.cloud_features_v3 import (  # noqa: E402
    cloud_features,
    feature_specification,
    gaussian_mixture_features,
)
from inference.evidence_network.conditional.build_dataset import (  # noqa: E402
    IDENTITY,
    SOURCE_NAMES,
    atomic_savez,
    sha256,
    weighted_source_table,
)
from likelihoods.nicer import build_nicer_grid  # noqa: E402
from workflows.nuclear_scenarios import (  # noqa: E402
    NUCLEAR_SCENARIO_NAMES,
    nuclear_observation,
)
from workflows.source_scenarios import BASE_NICER_SOURCES  # noqa: E402


HELD_OUT_FILENAMES = (
    "J0614_mrsamples.dat",
    "J1231_wmrsamples.txt",
    "J1614_STU_mrsamples_post_equal_weights.dat",
)
MAX_COMPONENTS = 4
FAMILY_NAMES = (
    "single_gaussian",
    "general_mixture",
    "core_tail_mixture",
    "asymmetric_or_bimodal_mixture",
)


def nuclear_design(
    count: int, seed: int, *, design: str = "legacy"
) -> tuple[np.ndarray, list[str]]:
    """Build a deterministic nuclear-data design containing every paper case."""
    if count < len(NUCLEAR_SCENARIO_NAMES):
        raise ValueError(
            f"at least {len(NUCLEAR_SCENARIO_NAMES)} nuclear designs are required"
        )
    observations = [
        np.asarray(nuclear_observation(name), dtype=np.float64)
        for name in NUCLEAR_SCENARIO_NAMES
    ]
    names = list(NUCLEAR_SCENARIO_NAMES)
    if design == "sobol":
        from scipy.stats import qmc

        remaining = count - len(observations)
        if remaining:
            power = int(np.ceil(np.log2(remaining)))
            unit = qmc.Sobol(
                d=7, scramble=True, seed=seed
            ).random_base2(power)[:remaining]
            offsets = qmc.scale(unit, np.full(7, -2.5), np.full(7, 2.5))
            for index, offset in enumerate(offsets, start=len(observations)):
                observations.append(
                    DEFAULT_NUCLEAR_OBSERVATION
                    + np.asarray(offset, dtype=np.float64) * DEFAULT_NUCLEAR_SIGMA
                )
                names.append(f"sobol_joint_{index:04d}")
        return np.asarray(observations, dtype=np.float64), names
    if design != "legacy":
        raise ValueError(f"unknown nuclear design: {design}")
    rng = np.random.default_rng(seed)
    while len(observations) < count:
        offset = np.zeros(7, dtype=np.float64)
        if len(observations) % 2:
            coordinate = int(rng.integers(7))
            offset[coordinate] = rng.uniform(-2.5, 2.5)
            names.append(f"random_single_{len(observations):03d}")
        else:
            offset = np.clip(rng.normal(0.0, 1.15, size=7), -2.5, 2.5)
            names.append(f"random_joint_{len(observations):03d}")
        observations.append(
            DEFAULT_NUCLEAR_OBSERVATION + offset * DEFAULT_NUCLEAR_SIGMA
        )
    return np.asarray(observations, dtype=np.float64), names


def stabilized_log_matrix_product(
    source_log_weight: torch.Tensor,
    nuclear_factor: torch.Tensor,
    nuclear_log_maximum: torch.Tensor,
) -> torch.Tensor:
    """Evaluate all source/nuclear log-sums with one stabilized matrix product."""
    source_log_maximum = torch.max(source_log_weight, dim=1).values
    source_factor = torch.exp(
        source_log_weight - source_log_maximum[:, None]
    )
    product = source_factor @ nuclear_factor.T
    return (
        torch.log(product)
        + source_log_maximum[:, None]
        + nuclear_log_maximum[None, :]
    )


class BroadDensityGenerator:
    """Draw data-independent normalized source densities and point clouds."""

    def __init__(
        self,
        data_root: Path,
        points: int,
        seed: int,
        *,
        single_components_only: bool = False,
        mass_centre_range: tuple[float, float] = (0.85, 2.25),
        radius_centre_range: tuple[float, float] = (8.5, 16.5),
        mass_sigma_range: tuple[float, float] = (0.015, 0.48),
        radius_sigma_range: tuple[float, float] = (0.12, 2.30),
        correlation_limit: float = 0.95,
    ) -> None:
        self.points = int(points)
        self.rng = np.random.default_rng(seed)
        self.single_components_only = bool(single_components_only)
        self.mass_centre_range = tuple(float(value) for value in mass_centre_range)
        self.radius_centre_range = tuple(float(value) for value in radius_centre_range)
        self.mass_sigma_range = tuple(float(value) for value in mass_sigma_range)
        self.radius_sigma_range = tuple(float(value) for value in radius_sigma_range)
        self.correlation_limit = float(correlation_limit)
        for name, bounds in (
            ("mass centre", self.mass_centre_range),
            ("radius centre", self.radius_centre_range),
            ("mass sigma", self.mass_sigma_range),
            ("radius sigma", self.radius_sigma_range),
        ):
            if len(bounds) != 2 or not 0 < bounds[0] < bounds[1]:
                raise ValueError(f"invalid {name} range")
        if not 0 < self.correlation_limit < 1:
            raise ValueError("correlation limit must lie in (0,1)")
        self.centres: dict[str, tuple[float, float]] = {}
        self.pools: dict[str, np.ndarray] = {}
        self.source_records: list[dict[str, object]] = []
        for name, source in zip(SOURCE_NAMES, BASE_NICER_SOURCES, strict=True):
            path = data_root / source.filename
            mass, radius, weight, centre = weighted_source_table(path, source)
            self.centres[name] = (float(centre[0]), float(centre[1]))
            if len(mass) > 20_000:
                selected = np.random.default_rng(0).choice(
                    len(mass), size=20_000, replace=False, p=weight
                )
            else:
                selected = np.arange(len(mass))
            self.pools[name] = np.column_stack([mass[selected], radius[selected]])
            self.source_records.append(
                {
                    "slot": name,
                    "filename": source.filename,
                    "sha256": sha256(path),
                    "rows": int(len(mass)),
                }
            )
        self.baseline_features = cloud_features(
            np.stack([self.pools[name] for name in SOURCE_NAMES])
        )

    def covariance(self, sigma_mass: float, sigma_radius: float, rho: float):
        return np.asarray(
            [
                [sigma_mass**2, rho * sigma_mass * sigma_radius],
                [rho * sigma_mass * sigma_radius, sigma_radius**2],
            ],
            dtype=np.float32,
        )

    def component_scale(self) -> tuple[float, float, float]:
        sigma_mass = float(np.exp(self.rng.uniform(
            np.log(self.mass_sigma_range[0]), np.log(self.mass_sigma_range[1])
        )))
        sigma_radius = float(np.exp(self.rng.uniform(
            np.log(self.radius_sigma_range[0]), np.log(self.radius_sigma_range[1])
        )))
        rho = float(self.rng.uniform(
            -self.correlation_limit, self.correlation_limit
        ))
        return sigma_mass, sigma_radius, rho

    def draw_mixture(self):
        family = (
            0 if self.single_components_only
            else int(self.rng.integers(len(FAMILY_NAMES)))
        )
        centre = np.asarray(
            [
                self.rng.uniform(*self.mass_centre_range),
                self.rng.uniform(*self.radius_centre_range),
            ],
            dtype=np.float64,
        )
        weight = np.zeros(MAX_COMPONENTS, dtype=np.float32)
        mean = np.zeros((MAX_COMPONENTS, 2), dtype=np.float32)
        covariance = np.repeat(
            np.eye(2, dtype=np.float32)[None, :, :], MAX_COMPONENTS, axis=0
        )

        if family == 0:
            components = 1
            weight[0] = 1.0
            mean[0] = centre
            covariance[0] = self.covariance(*self.component_scale())
        elif family == 1:
            components = int(self.rng.integers(2, MAX_COMPONENTS + 1))
            active_weight = self.rng.dirichlet(np.full(components, 0.8))
            base_mass = float(np.exp(self.rng.uniform(np.log(0.03), np.log(0.35))))
            base_radius = float(np.exp(self.rng.uniform(np.log(0.20), np.log(1.6))))
            offset = self.rng.normal(size=(components, 2)) * np.asarray(
                [base_mass, base_radius]
            )
            offset -= np.sum(active_weight[:, None] * offset, axis=0)
            for component in range(components):
                weight[component] = active_weight[component]
                mean[component] = centre + offset[component]
                covariance[component] = self.covariance(*self.component_scale())
        elif family == 2:
            components = 2
            core_fraction = float(self.rng.uniform(0.55, 0.92))
            weight[:2] = [core_fraction, 1.0 - core_fraction]
            mean[:2] = centre
            core_mass, core_radius, rho = self.component_scale()
            core_mass = min(core_mass, 0.18)
            core_radius = min(core_radius, 0.9)
            tail_factor = float(self.rng.uniform(2.5, 7.0))
            covariance[0] = self.covariance(core_mass, core_radius, rho)
            covariance[1] = self.covariance(
                min(core_mass * tail_factor, 0.65),
                min(core_radius * tail_factor, 3.0),
                float(self.rng.uniform(-0.85, 0.85)),
            )
        else:
            components = int(self.rng.integers(2, MAX_COMPONENTS + 1))
            active_weight = self.rng.dirichlet(
                np.linspace(0.35, 1.4, components)
            )
            angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
            direction = np.asarray([np.cos(angle), np.sin(angle)])
            separation = np.asarray(
                [self.rng.uniform(0.04, 0.45), self.rng.uniform(0.3, 2.8)]
            )
            signed = np.linspace(-1.0, 1.0, components)[:, None]
            offsets = signed * direction[None, :] * separation[None, :]
            offsets -= np.sum(active_weight[:, None] * offsets, axis=0)
            for component in range(components):
                weight[component] = active_weight[component]
                mean[component] = centre + offsets[component]
                covariance[component] = self.covariance(*self.component_scale())
        weight /= weight.sum()
        return family, weight, mean, covariance

    def sample_mixture(
        self, weight: np.ndarray, mean: np.ndarray, covariance: np.ndarray
    ) -> np.ndarray:
        active = np.flatnonzero(weight > 0)
        component = self.rng.choice(active, size=self.points, p=weight[active])
        cloud = np.empty((self.points, 2), dtype=np.float32)
        for index in active:
            selected = component == index
            if not np.any(selected):
                continue
            cloud[selected] = self.rng.multivariate_normal(
                mean[index], covariance[index], size=int(selected.sum())
            )
        return cloud

    def draw(
        self,
        count: int,
        *,
        identity_fraction: float,
        one_slot_fraction: float,
    ):
        weights = np.zeros((count, 3, MAX_COMPONENTS), dtype=np.float32)
        means = np.zeros((count, 3, MAX_COMPONENTS, 2), dtype=np.float32)
        covariance = np.repeat(
            np.eye(2, dtype=np.float32)[None, None, None, :, :],
            count * 3 * MAX_COMPONENTS,
            axis=0,
        ).reshape(count, 3, MAX_COMPONENTS, 2, 2)
        active_mask = np.zeros((count, 3), dtype=bool)
        family = np.full((count, 3), -1, dtype=np.int8)
        modes = np.empty(count, dtype=np.int8)
        features = np.empty(
            (count, 3, self.baseline_features.shape[-1]), dtype=np.float32
        )
        for scenario in range(count):
            selector = self.rng.random()
            if selector < identity_fraction:
                active: tuple[int, ...] = ()
                modes[scenario] = 0
            elif selector < identity_fraction + one_slot_fraction:
                active = (int(self.rng.integers(3)),)
                modes[scenario] = 1
            else:
                # Include two- and three-slot changes so the network learns
                # cross-source composition without letting this stress mode
                # dominate the reported one-source use case.
                number = int(self.rng.integers(2, 4))
                active = tuple(self.rng.choice(3, size=number, replace=False))
                modes[scenario] = number
            for slot, name in enumerate(SOURCE_NAMES):
                if slot in active:
                    item = self.draw_mixture()
                    family[scenario, slot] = item[0]
                    weights[scenario, slot] = item[1]
                    means[scenario, slot] = item[2]
                    covariance[scenario, slot] = item[3]
                    active_mask[scenario, slot] = True
                    features[scenario, slot] = gaussian_mixture_features(
                        item[1], item[2], item[3], slot
                    )
                else:
                    features[scenario, slot] = self.baseline_features[slot]
                    # Valid padded mixture parameters simplify serialization;
                    # they are never evaluated for inactive slots.
                    weights[scenario, slot, 0] = 1.0
                    means[scenario, slot, 0] = np.asarray(self.centres[name])
        return weights, means, covariance, active_mask, family, modes, features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=12_000)
    parser.add_argument("--validation-scenarios", type=int, default=1_200)
    parser.add_argument("--samples-per-scenario", type=int, default=1024)
    parser.add_argument("--cloud-points", type=int, default=1024)
    parser.add_argument("--scenario-batch", type=int, default=8)
    parser.add_argument("--row-chunk", type=int, default=25_000)
    parser.add_argument("--quadrature-points", type=int, default=40)
    parser.add_argument(
        "--target-mode",
        choices=("flow", "direct"),
        default="flow",
        help="flow samples or direct absolute-log-evidence targets",
    )
    parser.add_argument("--nuclear-designs", type=int, default=32)
    parser.add_argument(
        "--nuclear-design",
        choices=("legacy", "sobol"),
        default="legacy",
        help="low-discrepancy Sobol or random nuclear design",
    )
    parser.add_argument(
        "--single-components-only",
        action="store_true",
        help="draw one Gaussian per active source for component-basis training",
    )
    parser.add_argument("--mass-centre-min", type=float, default=0.85)
    parser.add_argument("--mass-centre-max", type=float, default=2.25)
    parser.add_argument("--radius-centre-min", type=float, default=8.5)
    parser.add_argument("--radius-centre-max", type=float, default=16.5)
    parser.add_argument("--mass-sigma-min", type=float, default=0.015)
    parser.add_argument("--mass-sigma-max", type=float, default=0.48)
    parser.add_argument("--radius-sigma-min", type=float, default=0.12)
    parser.add_argument("--radius-sigma-max", type=float, default=2.30)
    parser.add_argument("--correlation-limit", type=float, default=0.95)
    parser.add_argument("--restriction-radius", type=float, default=6.0)
    parser.add_argument("--minimum-tilt-ess", type=float, default=20.0)
    parser.add_argument("--identity-fraction", type=float, default=0.10)
    parser.add_argument("--one-slot-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260931)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--maximum-bank-rows",
        type=int,
        help="deterministic development-only bank thinning; forbidden in production",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists() and not arguments.force:
        raise FileExistsError(arguments.output)
    if not 0.0 <= arguments.identity_fraction <= 1.0:
        raise ValueError("invalid identity fraction")
    if not 0.0 <= arguments.one_slot_fraction <= 1.0:
        raise ValueError("invalid one-slot fraction")
    if arguments.identity_fraction + arguments.one_slot_fraction > 1.0:
        raise ValueError("invalid scenario fractions")
    if arguments.smoke:
        arguments.scenarios = min(arguments.scenarios, 48)
        arguments.validation_scenarios = min(arguments.validation_scenarios, 8)
        arguments.samples_per_scenario = min(arguments.samples_per_scenario, 128)
        arguments.cloud_points = min(arguments.cloud_points, 256)
        arguments.scenario_batch = min(arguments.scenario_batch, 4)
        arguments.row_chunk = min(arguments.row_chunk, 12_500)
        arguments.quadrature_points = min(arguments.quadrature_points, 16)
        arguments.minimum_tilt_ess = min(arguments.minimum_tilt_ess, 1.0)
        if arguments.maximum_bank_rows is None:
            arguments.maximum_bank_rows = 50_000
    if not 0 < arguments.validation_scenarios < arguments.scenarios:
        raise ValueError("invalid training/validation split")
    for filename in HELD_OUT_FILENAMES:
        if (arguments.data_root / filename).exists():
            raise RuntimeError(
                f"held-out NICER source is present in training root: {filename}"
            )

    started = time.perf_counter()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    with np.load(arguments.bank, allow_pickle=False) as bank:
        required = {
            "schema", "model", "X", "R", "MG", "MM", "LA", "LPC",
            "lpc_all_lse", "metadata",
        }
        missing = required - set(bank.files)
        if missing:
            raise RuntimeError(f"conditional physics bank lacks {sorted(missing)}")
        if str(bank["schema"].item()) != "conditional-en-independent-physics-v1":
            raise RuntimeError("unexpected conditional physics-bank schema")
        model_name = str(bank["model"].item())
        prediction = np.asarray(bank["X"], dtype=np.float32)
        radius = np.asarray(bank["R"], dtype=np.float32)
        mass_grid = np.asarray(bank["MG"], dtype=np.float32)
        maximum_mass = np.asarray(bank["MM"], dtype=np.float32)
        log_astro = np.asarray(bank["LA"], dtype=np.float32)
        correction = np.asarray(bank["LPC"], dtype=np.float32)
        denominator = float(bank["lpc_all_lse"])
        parent_metadata = json.loads(str(bank["metadata"].item()))
    if parent_metadata.get("forbidden_artifacts_used"):
        raise RuntimeError("conditional physics bank has forbidden ancestry")
    if parent_metadata.get("held_out_sources_used"):
        raise RuntimeError("conditional physics bank consumed a held-out source")

    original_rows = len(prediction)
    development_thinning = arguments.maximum_bank_rows is not None
    if development_thinning:
        keep_rows = int(arguments.maximum_bank_rows)
        if not arguments.smoke:
            raise RuntimeError("bank thinning is permitted only with --smoke")
        if keep_rows < 1:
            raise ValueError("maximum bank rows must be positive")
        if keep_rows < original_rows:
            selected = np.random.default_rng(2026092104).choice(
                original_rows, size=keep_rows, replace=False
            )
            prediction = prediction[selected]
            radius = radius[selected]
            maximum_mass = maximum_mass[selected]
            log_astro = log_astro[selected]
            correction = correction[selected]
    rows = len(prediction)

    scenario_generator = BroadDensityGenerator(
        arguments.data_root,
        arguments.cloud_points,
        arguments.seed,
        single_components_only=arguments.single_components_only,
        mass_centre_range=(arguments.mass_centre_min, arguments.mass_centre_max),
        radius_centre_range=(arguments.radius_centre_min, arguments.radius_centre_max),
        mass_sigma_range=(arguments.mass_sigma_min, arguments.mass_sigma_max),
        radius_sigma_range=(arguments.radius_sigma_min, arguments.radius_sigma_max),
        correlation_limit=arguments.correlation_limit,
    )
    common_grid = dict(nM=150, nR=150, max_rows=20_000, bw=0.08, seed=0)
    interpolators = {
        name: build_nicer_grid(
            arguments.data_root / source.filename,
            mcol=source.mass_column,
            rcol=source.radius_column,
            wcol_or_None=source.weight_column,
            **common_grid,
        )
        for name, source in zip(SOURCE_NAMES, BASE_NICER_SOURCES, strict=True)
    }
    covered = mass_grid >= 1.0
    curve_evaluator = BatchedShiftedNICER(
        interpolators,
        radius[:, covered],
        mass_grid[covered],
        maximum_mass,
        scenario_generator.centres,
        device=str(device),
        row_chunk=arguments.row_chunk,
        quadrature_points=arguments.quadrature_points,
    )
    mixture_evaluator = BatchedMixtureNICER(relative_floor=1.0e-6)
    quadrature_cache = [
        (start, stop, curve_evaluator.curve_quadrature(start, stop))
        for start, stop in curve_evaluator.chunks()
    ]
    source_base_log_weight = torch.as_tensor(
        correction + log_astro, dtype=torch.float32, device=device
    )
    prediction_tensor = torch.as_tensor(prediction, device=device)
    observation_tensor = torch.as_tensor(
        DEFAULT_NUCLEAR_OBSERVATION.astype(np.float32), device=device
    )
    sigma_tensor = torch.as_tensor(
        DEFAULT_NUCLEAR_SIGMA.astype(np.float32), device=device
    )
    support = torch.max(
        torch.abs((prediction_tensor - observation_tensor) / sigma_tensor), dim=1
    ).values <= arguments.restriction_radius
    if arguments.target_mode == "direct":
        nuclear_observations, nuclear_design_names = nuclear_design(
            arguments.nuclear_designs,
            arguments.seed + 503,
            design=arguments.nuclear_design,
        )
        nuclear_observation_tensor = torch.as_tensor(
            nuclear_observations.astype(np.float32), device=device
        )
        # Evaluate every Gaussian nuclear likelihood without materializing the
        # (design, bank-row, dimension) broadcast tensor.  The quadratic form
        # is algebraically identical and keeps a 1,024-point design within an
        # 8-GiB GPU.
        scaled_prediction = prediction_tensor / sigma_tensor[None, :]
        scaled_observation = nuclear_observation_tensor / sigma_tensor[None, :]
        nuclear_log_likelihood = (
            scaled_observation @ scaled_prediction.T
            - 0.5 * torch.sum(
                torch.square(scaled_observation), dim=1
            )[:, None]
            - 0.5 * torch.sum(
                torch.square(scaled_prediction), dim=1
            )[None, :]
        )
        nuclear_log_maximum = torch.max(nuclear_log_likelihood, dim=1).values
        nuclear_log_likelihood.sub_(nuclear_log_maximum[:, None]).exp_()
        nuclear_factor = nuclear_log_likelihood
        del scaled_prediction, scaled_observation
    else:
        nuclear_observations = None
        nuclear_design_names = None
        nuclear_log_maximum = None
        nuclear_factor = None

    identity = np.empty((3, rows), dtype=np.float32)
    identity_transform = torch.as_tensor(IDENTITY[None, :], device=device)
    identity_started = time.perf_counter()
    for start, stop, quadrature in quadrature_cache:
        for slot, name in enumerate(SOURCE_NAMES):
            identity[slot, start:stop] = (
                curve_evaluator.evaluate_chunk(
                    name, identity_transform, quadrature
                )[0]
                .cpu()
                .numpy()
            )
    identity_seconds = time.perf_counter() - identity_started

    accepted_features: list[np.ndarray] = []
    accepted_weights: list[np.ndarray] = []
    accepted_means: list[np.ndarray] = []
    accepted_covariance: list[np.ndarray] = []
    accepted_active: list[np.ndarray] = []
    accepted_family: list[np.ndarray] = []
    accepted_modes: list[int] = []
    accepted_logc: list[float] = []
    accepted_ess: list[float] = []
    accepted_samples: list[np.ndarray] = []
    accepted_logz: list[np.ndarray] = []
    attempts = 0
    next_progress = max(64, arguments.scenarios // 50)
    label_started = time.perf_counter()
    sampling_generator = torch.Generator(device=device).manual_seed(
        arguments.seed + 101
    )
    while len(accepted_features) < arguments.scenarios:
        requested = arguments.scenario_batch
        payload = scenario_generator.draw(
            requested,
            identity_fraction=arguments.identity_fraction,
            one_slot_fraction=arguments.one_slot_fraction,
        )
        weights, means, covariance, active_mask, family, modes, features = payload
        attempts += requested
        weight_tensor = torch.as_tensor(weights, device=device)
        mean_tensor = torch.as_tensor(means, device=device)
        covariance_tensor = torch.as_tensor(covariance, device=device)
        source_log_weight = source_base_log_weight[None, :].expand(
            requested, -1
        ).clone()
        for start, stop, quadrature in quadrature_cache:
            for slot in range(3):
                active = np.flatnonzero(active_mask[:, slot])
                if len(active) == 0:
                    continue
                active_tensor = torch.as_tensor(active, device=device)
                component_stop = 1 if arguments.single_components_only else None
                changed = mixture_evaluator.evaluate_chunk(
                    weight_tensor[active_tensor, slot, :component_stop],
                    mean_tensor[active_tensor, slot, :component_stop],
                    covariance_tensor[active_tensor, slot, :component_stop],
                    quadrature,
                )
                baseline = torch.as_tensor(
                    identity[slot, start:stop], device=device
                )[None, :]
                source_log_weight[active_tensor, start:stop] += changed - baseline
        restricted_log_weight = torch.where(
            support[None, :],
            source_log_weight,
            torch.full_like(source_log_weight, -torch.inf),
        )
        log_sum = torch.logsumexp(restricted_log_weight, dim=1)
        log_sum_two = torch.logsumexp(2.0 * restricted_log_weight, dim=1)
        ess = torch.exp(2.0 * log_sum - log_sum_two)
        # The fiducial A1 source configuration is a mandatory anchor.  Its
        # hyperonic restricted-tilt ESS is about 17 on the present clean bank,
        # which is sufficient for resampling but can fall below a deliberately
        # conservative synthetic-shape gate.  Never let that gate erase the
        # very configuration the network must reproduce.
        identity_case = torch.as_tensor(modes == 0, device=device)
        keep = torch.isfinite(log_sum) & (
            (ess >= arguments.minimum_tilt_ess) | identity_case
        )
        if keep.any():
            kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
            if arguments.target_mode == "flow":
                normalized = torch.exp(
                    restricted_log_weight[kept_indices]
                    - log_sum[kept_indices, None]
                )
                sampled = torch.multinomial(
                    normalized,
                    arguments.samples_per_scenario,
                    replacement=True,
                    generator=sampling_generator,
                )
                noisy = prediction_tensor[sampled] + torch.randn(
                    (*sampled.shape, 7),
                    device=device,
                    generator=sampling_generator,
                ) * sigma_tensor[None, None, :]
                noisy_cpu = noisy.cpu().numpy().astype(np.float32)
                direct_logz_cpu = None
            else:
                if nuclear_factor is None or nuclear_log_maximum is None:
                    raise RuntimeError("direct nuclear likelihood was not built")
                direct_logz_cpu = (
                    stabilized_log_matrix_product(
                        source_log_weight[kept_indices],
                        nuclear_factor,
                        nuclear_log_maximum,
                    )
                    .sub(denominator)
                    .cpu()
                    .numpy()
                )
                noisy_cpu = None
            kept_cpu = kept_indices.cpu().numpy()
            logc_cpu = (log_sum[kept_indices] - denominator).cpu().numpy()
            ess_cpu = ess[kept_indices].cpu().numpy()
            for local, original in enumerate(kept_cpu):
                if len(accepted_features) >= arguments.scenarios:
                    break
                accepted_features.append(features[original])
                accepted_weights.append(weights[original])
                accepted_means.append(means[original])
                accepted_covariance.append(covariance[original])
                accepted_active.append(active_mask[original])
                accepted_family.append(family[original])
                accepted_modes.append(int(modes[original]))
                accepted_logc.append(float(logc_cpu[local]))
                accepted_ess.append(float(ess_cpu[local]))
                if noisy_cpu is not None:
                    accepted_samples.append(noisy_cpu[local])
                if direct_logz_cpu is not None:
                    accepted_logz.append(direct_logz_cpu[local])
        if (
            len(accepted_features) >= next_progress
            or len(accepted_features) >= arguments.scenarios
        ):
            print(
                f"[conditional-en-v3-data] accepted={len(accepted_features)}/"
                f"{arguments.scenarios} attempted={attempts} "
                f"last-ESS(min/median/max)="
                f"{float(ess.min()):.1f}/{float(ess.median()):.1f}/"
                f"{float(ess.max()):.1f}",
                flush=True,
            )
            next_progress += max(64, arguments.scenarios // 50)
    label_seconds = time.perf_counter() - label_started

    metadata = {
        "schema_version": 3,
        "lineage_class": "independent_conditional_evidence_network",
        "model": model_name,
        "physics_bank": str(arguments.bank.resolve()),
        "physics_bank_sha256": sha256(arguments.bank),
        "baseline_sources": scenario_generator.source_records,
        "held_out_sources_absent": ["J0614", "J1231", "J1614"],
        "held_out_source_files_used": [],
        "forbidden_artifacts_used": [],
        "amortized_inputs": ["three_NICER_cloud_distributions", "nuclear_observation"],
        "source_density_family": {
            "families": list(FAMILY_NAMES),
            "maximum_components": MAX_COMPONENTS,
            "mass_centre_domain": list(scenario_generator.mass_centre_range),
            "radius_centre_domain_km": list(scenario_generator.radius_centre_range),
            "component_mass_sigma_domain": list(scenario_generator.mass_sigma_range),
            "component_radius_sigma_domain_km": list(scenario_generator.radius_sigma_range),
            "component_correlation_domain": [
                -scenario_generator.correlation_limit,
                scenario_generator.correlation_limit,
            ],
            "normalized_analytic_density": True,
            "relative_density_floor": 1.0e-6,
            "single_components_only": arguments.single_components_only,
        },
        "scenario_structure": {
            "identity_fraction": arguments.identity_fraction,
            "one_slot_fraction": arguments.one_slot_fraction,
            "multi_slot_fraction": 1.0
            - arguments.identity_fraction
            - arguments.one_slot_fraction,
            "identity_bypasses_synthetic_ess_gate": True,
        },
        "feature_specification": feature_specification(),
        "target_mode": arguments.target_mode,
        "restriction_radius_sigma": arguments.restriction_radius,
        "stellar_curve_quadrature_cached_once": True,
        "development_bank_thinning": {
            "enabled": development_thinning,
            "original_rows": original_rows,
            "used_rows": rows,
        },
        "query_contract": {
            "source_specific_cache": False,
            "source_specific_training": False,
            "new_likelihood_evaluations": 0,
            "new_TOV_evaluations": 0,
        },
    }
    common_payload = {
        "source_features": np.asarray(accepted_features, dtype=np.float32),
        "mixture_weight": np.asarray(accepted_weights, dtype=np.float32),
        "mixture_mean": np.asarray(accepted_means, dtype=np.float32),
        "mixture_covariance": np.asarray(accepted_covariance, dtype=np.float32),
        "active_mask": np.asarray(accepted_active, dtype=bool),
        "family": np.asarray(accepted_family, dtype=np.int8),
        "mode": np.asarray(accepted_modes, dtype=np.int8),
        "tilt_ess": np.asarray(accepted_ess, dtype=np.float64),
        "ntr": np.int64(arguments.scenarios - arguments.validation_scenarios),
        "OBS": DEFAULT_NUCLEAR_OBSERVATION,
        "SIG": DEFAULT_NUCLEAR_SIGMA,
    }
    if arguments.target_mode == "flow":
        atomic_savez(
            arguments.output,
            schema=np.asarray("conditional-en-broad-shape-scenarios-v3"),
            y=np.asarray(accepted_samples, dtype=np.float32),
            logC=np.asarray(accepted_logc, dtype=np.float64),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            **common_payload,
        )
    else:
        if nuclear_observations is None or nuclear_design_names is None:
            raise RuntimeError("direct nuclear design was not built")
        direct_metadata = {
            "schema_version": 3,
            "lineage_class": "independent_conditional_evidence_network",
            "physics_bank": str(arguments.bank.resolve()),
            "physics_bank_sha256": sha256(arguments.bank),
            "held_out_sources_absent": ["J0614", "J1231", "J1614"],
            "held_out_source_files_used": [],
            "forbidden_artifacts_used": [],
            "target": "absolute_log_evidence",
            "nuclear_design_names": nuclear_design_names,
            "nuclear_design_domain_sigma": [-2.5, 2.5],
            "nuclear_design": arguments.nuclear_design,
            "query_contract": metadata["query_contract"],
            "scenario_metadata": metadata,
            "integrated_single_pass_labels": True,
        }
        atomic_savez(
            arguments.output,
            schema=np.asarray("conditional-en-direct-regression-v3"),
            nuclear_observations=nuclear_observations,
            logZ=np.asarray(accepted_logz, dtype=np.float64),
            metadata=np.asarray(json.dumps(direct_metadata, sort_keys=True)),
            **common_payload,
        )
    report = {
        "status": "SMOKE_COMPLETED" if arguments.smoke else "PASS",
        "stage": "broad-shape joint NICER+nuclear conditional EN construction",
        "model": model_name,
        "scenarios": arguments.scenarios,
        "training_scenarios": arguments.scenarios
        - arguments.validation_scenarios,
        "attempted_scenarios": attempts,
        "samples_per_scenario": arguments.samples_per_scenario,
        "target_mode": arguments.target_mode,
        "nuclear_designs": (
            arguments.nuclear_designs if arguments.target_mode == "direct" else 0
        ),
        "minimum_tilt_ess": float(np.min(accepted_ess)),
        "median_tilt_ess": float(np.median(accepted_ess)),
        "maximum_tilt_ess": float(np.max(accepted_ess)),
        "identity_likelihood_seconds": identity_seconds,
        "scenario_label_seconds": label_seconds,
        "construction_wall_seconds": time.perf_counter() - started,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "metadata": metadata,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
