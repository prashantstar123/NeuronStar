"""Permutation-aware conditional Evidence Network.

The network consumes three set-valued NICER mass--radius clouds and one
seven-dimensional nuclear observation.  It has no EOS-parameter, posterior,
or conventional-sampler input at query time.
"""

from __future__ import annotations

import torch
from torch import nn


class NICERSetEncoder(nn.Module):
    """Encode three fixed semantic NICER slots, invariant within each cloud.

    A learned DeepSets mean embedding is augmented by robust distribution
    summaries.  The latter prevent a finite cloud resample from hiding the
    location, scale, covariance, or tail displacement of a new source.
    """

    def __init__(self, hidden: int = 128, embedding: int = 64) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.embedding = int(embedding)
        self.phi = nn.Sequential(
            nn.Linear(5, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.embedding),
        )
        # mean(2), standard deviation(2), five quantiles per coordinate(10),
        # covariance(1), skewness(2), and the semantic slot one-hot(3).
        self.summary_features = 20
        self.summary_rho = nn.Sequential(
            nn.Linear(self.summary_features, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.embedding),
        )
        self.register_buffer("event_identity", torch.eye(3), persistent=False)
        self.register_buffer(
            "quantile_levels",
            torch.tensor([0.05, 0.16, 0.50, 0.84, 0.95]),
            persistent=False,
        )

    @property
    def output_dimension(self) -> int:
        return 3 * 2 * self.embedding

    def forward(self, clouds: torch.Tensor) -> torch.Tensor:
        if clouds.ndim != 4 or clouds.shape[1] != 3 or clouds.shape[-1] != 2:
            raise ValueError("NICER clouds must have shape (batch,3,points,2)")
        batch, sources, points, _ = clouds.shape
        identity = self.event_identity[None, :, None, :].expand(
            batch, sources, points, 3
        )
        augmented = torch.cat([clouds, identity], dim=-1)
        learned = self.rho(self.phi(augmented).mean(dim=2))
        mean = clouds.mean(dim=2)
        centred = clouds - mean[:, :, None, :]
        standard_deviation = torch.sqrt(
            torch.mean(centred * centred, dim=2).clamp_min(1.0e-8)
        )
        covariance = torch.mean(
            centred[:, :, :, 0] * centred[:, :, :, 1], dim=2
        )[:, :, None]
        skewness = torch.mean(
            (centred / standard_deviation[:, :, None, :]) ** 3,
            dim=2,
        )
        quantiles = torch.quantile(
            clouds.float(), self.quantile_levels, dim=2
        ).permute(1, 2, 0, 3).reshape(batch, sources, 10)
        summary = torch.cat(
            [
                mean,
                standard_deviation,
                quantiles,
                covariance,
                skewness,
                self.event_identity[None, :, :].expand(batch, -1, -1),
            ],
            dim=2,
        )
        summarized = self.summary_rho(summary)
        return torch.cat([learned, summarized], dim=2).reshape(
            batch, self.output_dimension
        )


class ConditionalEvidenceNet(nn.Module):
    """Map NICER clouds and nuclear observations to absolute log evidence."""

    def __init__(
        self,
        *,
        set_hidden: int = 128,
        set_embedding: int = 64,
        hidden: int = 256,
        layers: int = 4,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("at least one evidence head layer is required")
        self.encoder = NICERSetEncoder(set_hidden, set_embedding)
        input_dimension = self.encoder.output_dimension + 7
        blocks: list[nn.Module] = [nn.Linear(input_dimension, hidden), nn.GELU()]
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(hidden, hidden), nn.GELU()])
        blocks.append(nn.Linear(hidden, 1))
        self.head = nn.Sequential(*blocks)

    def forward(
        self, clouds: torch.Tensor, nuclear_observation: torch.Tensor
    ) -> torch.Tensor:
        if nuclear_observation.ndim != 2 or nuclear_observation.shape[1] != 7:
            raise ValueError("nuclear observations must have shape (batch,7)")
        if len(clouds) != len(nuclear_observation):
            raise ValueError("NICER and nuclear batches have different lengths")
        context = torch.cat(
            [self.encoder(clouds), nuclear_observation], dim=1
        )
        return self.head(context).squeeze(1)


class ConditionalVectorField(nn.Module):
    """Flow-matching vector field conditioned on a NICER set embedding."""

    def __init__(
        self,
        context_dimension: int,
        *,
        dimension: int = 7,
        hidden: int = 384,
        layers: int = 4,
        time_features: int = 16,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.context_dimension = int(context_dimension)
        blocks: list[nn.Module] = [
            nn.Linear(
                self.dimension + time_features + self.context_dimension,
                hidden,
            ),
            nn.GELU(),
        ]
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(hidden, hidden), nn.GELU()])
        blocks.append(nn.Linear(hidden, self.dimension))
        self.net = nn.Sequential(*blocks)
        frequency = torch.exp(torch.linspace(0, 5, time_features // 2))[None, :]
        self.register_buffer("frequency", frequency, persistent=False)

    def time_embedding(self, time: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [torch.sin(self.frequency * time), torch.cos(self.frequency * time)],
            dim=1,
        )

    def forward(
        self,
        value: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            torch.cat([value, self.time_embedding(time), context], dim=1)
        )


class ConditionalEvidenceFlowNet(nn.Module):
    """Conditional normalized flow plus its astrophysical normalization head."""

    def __init__(
        self,
        *,
        set_hidden: int = 128,
        set_embedding: int = 64,
        hidden: int = 384,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = NICERSetEncoder(set_hidden, set_embedding)
        context_dimension = self.encoder.output_dimension
        self.vector_field = ConditionalVectorField(
            context_dimension, hidden=hidden, layers=layers
        )
        self.normalization_head = nn.Sequential(
            nn.Linear(context_dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def context(self, clouds: torch.Tensor) -> torch.Tensor:
        return self.encoder(clouds)

    def normalization(self, context: torch.Tensor) -> torch.Tensor:
        return self.normalization_head(context).squeeze(1)
