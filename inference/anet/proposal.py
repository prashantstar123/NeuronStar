"""Frozen fmpe9c amortized proposal used by the paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from workflows.data_gate import (
    validate_observational_data,
    validate_substituted_pulsar_data,
)
from workflows.source_scenarios import (
    BASE_NICER_SOURCES,
    source_substitution,
)


class DeepSets(nn.Module):
    def __init__(self, hidden_dimension, embedding_dimension):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(5, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, embedding_dimension),
        )
        self.register_buffer("event_identity", torch.eye(3), persistent=False)

    def forward(self, cloud):
        batch, sources, points, _ = cloud.shape
        identity = self.event_identity[None, :, None, :].expand(
            batch, sources, points, 3
        )
        augmented = torch.cat([cloud, identity], dim=-1)
        return self.rho(self.phi(augmented).mean(dim=2)).reshape(batch, -1)


class VectorField(nn.Module):
    def __init__(self, dimension, context_dimension, hidden, layers, time_features=16):
        super().__init__()
        network = [
            nn.Linear(dimension + time_features + context_dimension, hidden),
            nn.GELU(),
        ]
        for _ in range(layers - 1):
            network.extend([nn.Linear(hidden, hidden), nn.GELU()])
        network.append(nn.Linear(hidden, dimension))
        self.net = nn.Sequential(*network)
        frequency = torch.exp(torch.linspace(0, 5, time_features // 2))[None, :]
        self.register_buffer("frequency", frequency, persistent=False)

    def time_embedding(self, time):
        return torch.cat(
            [torch.sin(self.frequency * time), torch.cos(self.frequency * time)],
            dim=1,
        )

    def forward(self, value, time, context):
        return self.net(
            torch.cat([value, self.time_embedding(time), context], dim=1)
        )


def _sample_cloud(path, mass_column, radius_column, weight_column, points, rng):
    array = np.loadtxt(path)
    if weight_column is None:
        weight = np.ones(len(array), dtype=np.float64)
    else:
        weight = np.asarray(array[:, weight_column], dtype=np.float64)
    weight /= weight.sum()
    index = rng.choice(len(array), points, p=weight)
    return np.column_stack([array[index, mass_column], array[index, radius_column]])


def mass_radius_clouds(
    data_root,
    points=256,
    seed=0,
    *,
    source_scenario="A1",
    source_data_root=None,
    verify_data=True,
):
    """Construct the exact ordered A-NET clouds for one source scenario.

    All three A1 clouds are drawn first from one RNG stream.  A declared
    replacement cloud is then drawn from the continued stream and installed
    in its target slot, matching the unseen-source pipeline of the paper exactly.
    """

    base_root = Path(data_root)
    paths = (
        validate_observational_data(base_root)
        if verify_data
        else {spec.filename: base_root / spec.filename for spec in BASE_NICER_SOURCES}
    )
    rng = np.random.default_rng(seed)
    clouds = [
        _sample_cloud(
            paths[spec.filename],
            spec.mass_column,
            spec.radius_column,
            spec.weight_column,
            points,
            rng,
        )
        for spec in BASE_NICER_SOURCES
    ]
    replacement = source_substitution(source_scenario)
    if replacement is not None:
        replacement_root = Path(source_data_root or data_root)
        replacement_paths = (
            validate_substituted_pulsar_data(
                replacement_root, replacement.filename
            )
            if verify_data
            else {replacement.filename: replacement_root / replacement.filename}
        )
        clouds[replacement.slot] = _sample_cloud(
            replacement_paths[replacement.filename],
            replacement.mass_column,
            replacement.radius_column,
            replacement.weight_column,
            points,
            rng,
        )
    return np.stack(clouds)


def baseline_mass_radius_clouds(data_root, points=256, seed=0):
    """Backward-compatible name for the exact A1 conditioning clouds."""

    return mass_radius_clouds(data_root, points=points, seed=seed)


class FMPEProposal:
    """Load, condition, sample and density-evaluate the frozen A-NET proposal."""

    def __init__(
        self,
        checkpoint,
        standardization,
        *,
        device=None,
        hidden=512,
        layers=4,
        steps=64,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.steps = int(steps)
        if self.steps < 1:
            raise ValueError("Heun step count must be positive")
        with np.load(standardization) as saved:
            self.standardization = {name: saved[name] for name in saved.files}
        self.theta_mean = np.asarray(
            self.standardization["tmth"], dtype=np.float64
        )
        self.theta_scale = np.asarray(
            self.standardization["tsth"], dtype=np.float64
        )
        dsd = int(self.standardization["dsd"])
        dse = int(self.standardization["dse"])
        self.cloud_points = int(self.standardization["kpts"])
        self.deep_sets = DeepSets(dsd, dse).to(self.device)
        context_dimension = 3 * dse + 7 + 2
        self.vector_field = VectorField(
            7, context_dimension, hidden, layers
        ).to(self.device)
        try:
            state = torch.load(
                checkpoint, map_location=self.device, weights_only=True
            )
        except TypeError:  # torch versions before weights_only
            state = torch.load(checkpoint, map_location=self.device)
        self.deep_sets.load_state_dict(state["ds"])
        self.vector_field.load_state_dict(state["vf"])
        self.deep_sets.eval()
        self.vector_field.eval()

    def _context(self, clouds, observation, gamma_low, rho_eval, rows):
        saved = self.standardization
        cloud = torch.tensor(
            (np.asarray(clouds, dtype=np.float64) - saved["cmean"])
            / saved["cstd"],
            dtype=torch.float32,
            device=self.device,
        )[None]
        nuclear = torch.tensor(
            (np.asarray(observation, dtype=np.float64) - saved["xm"])
            / saved["xs"],
            dtype=torch.float32,
            device=self.device,
        )[None]
        dial = torch.tensor(
            [
                [
                    (gamma_low - float(saved["GL_M"])) / float(saved["GL_S"]),
                    (rho_eval - float(saved["RH_M"])) / float(saved["RH_S"]),
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            context = torch.cat([self.deep_sets(cloud), nuclear, dial], dim=1)
        return context.expand(rows, -1)

    def _divergence(self, value, time, context):
        divergence = torch.zeros(len(value), device=self.device)
        for dimension in range(7):
            with torch.enable_grad():
                differentiable = value.detach().requires_grad_(True)
                component = self.vector_field(
                    differentiable, time, context
                )[:, dimension]
                gradient = torch.autograd.grad(component.sum(), differentiable)[0]
                divergence = divergence + gradient[:, dimension]
        return divergence.detach()

    def sample_with_log_density(
        self,
        clouds,
        observation,
        *,
        size,
        seed=0,
        gamma_low=0.0,
        rho_eval=1.2,
    ):
        """Heun-sample theta and integrate the exact 7D divergence for log q."""

        if np.asarray(clouds).shape != (3, self.cloud_points, 2):
            raise ValueError(
                f"expected cloud shape (3,{self.cloud_points},2), "
                f"got {np.asarray(clouds).shape}"
            )
        size = int(size)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        context = self._context(
            clouds, observation, gamma_low, rho_eval, size
        )
        value = torch.randn(
            size, 7, device=self.device, generator=generator
        )
        log_density = -0.5 * 7 * np.log(2 * np.pi) - 0.5 * (value**2).sum(1)
        step = 1.0 / self.steps
        for index in range(self.steps):
            time = torch.full(
                (size, 1), index * step, device=self.device
            )
            with torch.no_grad():
                velocity1 = self.vector_field(value, time, context)
            divergence1 = self._divergence(value, time, context)
            predictor = value + step * velocity1
            with torch.no_grad():
                velocity2 = self.vector_field(predictor, time + step, context)
            divergence2 = self._divergence(predictor, time + step, context)
            value = (
                value + step * 0.5 * (velocity1 + velocity2)
            ).detach()
            log_density = log_density - step * 0.5 * (
                divergence1 + divergence2
            )
        standardized = value.cpu().numpy().astype(np.float64)
        theta = standardized * self.theta_scale + self.theta_mean
        # Change from standardized theta coordinates to physical coordinates.
        physical_log_density = (
            log_density.cpu().numpy().astype(np.float64)
            - np.sum(np.log(self.theta_scale))
        )
        return theta, physical_log_density
