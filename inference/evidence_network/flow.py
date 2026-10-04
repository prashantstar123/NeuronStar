"""Continuous-normalizing-flow Evidence Network."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from inference.evidence_network.cache import (
    DEFAULT_NUCLEAR_OBSERVATION,
    DEFAULT_NUCLEAR_SIGMA,
    EvidenceCache,
)


class VectorField(nn.Module):
    """FMPE-style GELU vector field from seven coordinates plus time."""

    def __init__(self, width: int = 256, dimension: int = 7):
        super().__init__()
        self.dimension = int(dimension)
        self.width = int(width)
        self.net = nn.Sequential(
            nn.Linear(self.dimension + 1, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.dimension),
        )

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, time], dim=1))


@dataclass(frozen=True)
class FlowTrainingConfig:
    width: int = 256
    epochs: int = 40
    training_rows: int = 400_000
    reference_rows: int = 100_000
    batch_size: int = 8192
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    heun_steps: int = 64
    restriction_radius: float = 6.0


@dataclass
class EvidenceFlow:
    vector_field: VectorField
    mean: np.ndarray
    standard_deviation: np.ndarray
    log_restricted_normalization: float
    nuclear_sigma: np.ndarray
    heun_steps: int = 64
    restriction_radius: float = 6.0
    device: str = "cpu"

    @property
    def dimension(self) -> int:
        return len(self.mean)

    @property
    def log_gaussian_normalization(self) -> float:
        return float(
            np.sum(0.5 * np.log(2.0 * np.pi) + np.log(self.nuclear_sigma))
        )

    def log_density(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if points.shape[1] != self.dimension:
            raise ValueError("Evidence-Network query has the wrong dimension")
        model = self.vector_field.to(self.device).eval()
        state = torch.as_tensor(
            ((points - self.mean) / self.standard_deviation).astype(np.float32),
            device=self.device,
        )
        step = 1.0 / self.heun_steps
        integrated_divergence = torch.zeros(len(state), device=self.device)

        def velocity_divergence(z: torch.Tensor, t: torch.Tensor):
            z = z.detach().requires_grad_(True)
            velocity = model(z, t)
            trace = torch.zeros(len(z), device=z.device)
            for column in range(self.dimension):
                gradient = torch.autograd.grad(
                    velocity[:, column].sum(),
                    z,
                    retain_graph=column < self.dimension - 1,
                )[0]
                trace = trace + gradient[:, column]
            return velocity.detach(), trace.detach()

        for iteration in range(self.heun_steps, 0, -1):
            high_time = torch.full(
                (len(state), 1), iteration * step, device=self.device
            )
            low_time = torch.full(
                (len(state), 1), (iteration - 1) * step, device=self.device
            )
            high_velocity, high_divergence = velocity_divergence(
                state, high_time
            )
            predictor = state - step * high_velocity
            low_velocity, low_divergence = velocity_divergence(
                predictor, low_time
            )
            state = state - 0.5 * step * (high_velocity + low_velocity)
            integrated_divergence = integrated_divergence + 0.5 * step * (
                high_divergence + low_divergence
            )
        base = (
            -0.5 * torch.sum(state * state, dim=1)
            - 0.5 * self.dimension * np.log(2.0 * np.pi)
        )
        log_jacobian_standardization = -float(
            np.sum(np.log(self.standard_deviation))
        )
        return (
            base - integrated_divergence + log_jacobian_standardization
        ).detach().cpu().numpy().astype(np.float64)

    def log_evidence(self, observation: np.ndarray) -> np.ndarray:
        density = self.log_density(observation)
        return (
            self.log_restricted_normalization
            + density
            + self.log_gaussian_normalization
        )

    def save(self, path: str | Path, config: FlowTrainingConfig | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "state_dict": self.vector_field.state_dict(),
                "width": self.vector_field.width,
                "dimension": self.dimension,
                "mean": self.mean,
                "standard_deviation": self.standard_deviation,
                "log_restricted_normalization": self.log_restricted_normalization,
                "nuclear_sigma": self.nuclear_sigma,
                "heun_steps": self.heun_steps,
                "restriction_radius": self.restriction_radius,
                "training_config": asdict(config) if config is not None else None,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "EvidenceFlow":
        payload = torch.load(path, map_location=device, weights_only=False)
        model = VectorField(
            width=int(payload["width"]), dimension=int(payload["dimension"])
        )
        model.load_state_dict(payload["state_dict"])
        return cls(
            vector_field=model,
            mean=np.asarray(payload["mean"], dtype=np.float64),
            standard_deviation=np.asarray(
                payload["standard_deviation"], dtype=np.float64
            ),
            log_restricted_normalization=float(
                payload["log_restricted_normalization"]
            ),
            nuclear_sigma=np.asarray(payload["nuclear_sigma"], dtype=np.float64),
            heun_steps=int(payload["heun_steps"]),
            restriction_radius=float(payload["restriction_radius"]),
            device=device,
        )


def train_evidence_flow(
    cache: EvidenceCache,
    *,
    seed: int,
    config: FlowTrainingConfig = FlowTrainingConfig(),
    observation: np.ndarray = DEFAULT_NUCLEAR_OBSERVATION,
    sigma: np.ndarray = DEFAULT_NUCLEAR_SIGMA,
    device: str | None = None,
    standardization: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[EvidenceFlow, list[float]]:
    """Train one exact-density flow on the cache's restricted astro tilt."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sigma = np.asarray(sigma, dtype=np.float64)
    tilt = cache.restricted_tilt(
        observation, sigma, radius=config.restriction_radius
    )
    if standardization is None:
        reference, _ = cache.sample_smoothed_tilt(
            config.reference_rows,
            seed=seed + 1_000_003,
            observation=observation,
            sigma=sigma,
            radius=config.restriction_radius,
        )
        mean = reference.mean(axis=0)
        standard_deviation = reference.std(axis=0)
    else:
        mean = np.asarray(standardization[0], dtype=np.float64)
        standard_deviation = np.asarray(standardization[1], dtype=np.float64)
        if mean.shape != (7,) or standard_deviation.shape != (7,):
            raise ValueError("Evidence-Network standardization must be two 7-vectors")
    if np.any(standard_deviation <= 0.0):
        raise RuntimeError("Evidence-Network reference scale is degenerate")
    target, _ = cache.sample_smoothed_tilt(
        config.training_rows,
        seed=seed * 7919 + 13,
        observation=observation,
        sigma=sigma,
        radius=config.restriction_radius,
    )
    standardized = ((target - mean) / standard_deviation).astype(np.float32)
    target_tensor = torch.from_numpy(standardized)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = VectorField(width=config.width).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.epochs
    )
    losses: list[float] = []
    for _epoch in range(config.epochs):
        permutation = torch.randperm(config.training_rows)
        running = 0.0
        batches = 0
        model.train()
        for start in range(0, config.training_rows, config.batch_size):
            index = permutation[start : start + config.batch_size]
            endpoint = target_tensor[index].to(device)
            base = torch.randn_like(endpoint)
            time = torch.rand(len(endpoint), 1, device=device)
            path = (1.0 - time) * base + time * endpoint
            prediction = model(path, time)
            loss = torch.mean((prediction - (endpoint - base)) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        losses.append(running / batches)
    model.eval()
    return (
        EvidenceFlow(
            vector_field=model,
            mean=mean,
            standard_deviation=standard_deviation,
            log_restricted_normalization=tilt.log_normalization,
            nuclear_sigma=sigma,
            heun_steps=config.heun_steps,
            restriction_radius=config.restriction_radius,
            device=device,
        ),
        losses,
    )
