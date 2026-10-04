"""Shape-aware jointly amortized conditional Evidence Network v3."""

from __future__ import annotations

import torch
from torch import nn

from .cloud_features_v3 import FEATURE_DIMENSION


class SourceFeatureEncoder(nn.Module):
    """Encode three semantic, permutation-invariant source descriptors."""

    def __init__(
        self,
        *,
        feature_dimension: int = FEATURE_DIMENSION,
        hidden: int = 256,
        embedding: int = 96,
    ) -> None:
        super().__init__()
        self.feature_dimension = int(feature_dimension)
        self.hidden = int(hidden)
        self.embedding = int(embedding)
        self.source = nn.Sequential(
            nn.Linear(self.feature_dimension, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.embedding),
        )
        self.cross_source = nn.Sequential(
            nn.Linear(3 * self.embedding, 3 * self.embedding),
            nn.GELU(),
            nn.Linear(3 * self.embedding, 3 * self.embedding),
        )

    @property
    def output_dimension(self) -> int:
        return 3 * self.embedding

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 3
            or features.shape[1] != 3
            or features.shape[2] != self.feature_dimension
        ):
            raise ValueError(
                "source features must have shape (batch,3,feature_dimension)"
            )
        encoded = self.source(features).reshape(len(features), -1)
        # Residual cross-slot mixing lets the evidence respond to combinations
        # of source densities while preserving their fixed semantic ordering.
        return encoded + self.cross_source(encoded)


class ConditionalVectorFieldV3(nn.Module):
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


class ConditionalEvidenceFlowNetV3(nn.Module):
    """Normalized nuclear-observable flow conditioned on source distributions."""

    def __init__(
        self,
        *,
        feature_dimension: int = FEATURE_DIMENSION,
        source_hidden: int = 256,
        source_embedding: int = 96,
        hidden: int = 384,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = SourceFeatureEncoder(
            feature_dimension=feature_dimension,
            hidden=source_hidden,
            embedding=source_embedding,
        )
        context_dimension = self.encoder.output_dimension
        self.vector_field = ConditionalVectorFieldV3(
            context_dimension, hidden=hidden, layers=layers
        )
        self.normalization_head = nn.Sequential(
            nn.Linear(context_dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def context(self, source_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(source_features)

    def normalization(self, context: torch.Tensor) -> torch.Tensor:
        return self.normalization_head(context).squeeze(1)


class DirectConditionalEvidenceNetV3(nn.Module):
    """Direct absolute-evidence head over source and nuclear data.

    This removes probability-flow integration at query time.  Its targets are
    exact log-sums over the same independent, prior-corrected physics bank used
    to train the flow-based estimator.
    """

    def __init__(
        self,
        *,
        feature_dimension: int = FEATURE_DIMENSION,
        source_hidden: int = 256,
        source_embedding: int = 96,
        hidden: int = 512,
        layers: int = 4,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("at least one evidence-head layer is required")
        self.encoder = SourceFeatureEncoder(
            feature_dimension=feature_dimension,
            hidden=source_hidden,
            embedding=source_embedding,
        )
        self.nuclear_encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        blocks: list[nn.Module] = [
            nn.Linear(self.encoder.output_dimension + 128, hidden),
            nn.GELU(),
        ]
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(hidden, hidden), nn.GELU()])
        blocks.append(nn.Linear(hidden, 1))
        self.head = nn.Sequential(*blocks)

    def forward(
        self,
        source_features: torch.Tensor,
        nuclear_observation: torch.Tensor,
    ) -> torch.Tensor:
        if nuclear_observation.ndim != 2 or nuclear_observation.shape[1] != 7:
            raise ValueError("nuclear observations must have shape (batch,7)")
        if len(source_features) != len(nuclear_observation):
            raise ValueError("source and nuclear batches differ")
        context = torch.cat(
            [
                self.encoder(source_features),
                self.nuclear_encoder(nuclear_observation),
            ],
            dim=1,
        )
        return self.head(context).squeeze(1)


class MixtureSourceEncoderV3(nn.Module):
    """Permutation-invariant encoder for normalized Gaussian-mixture sources."""

    def __init__(self, hidden: int = 192, embedding: int = 96) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.embedding = int(embedding)
        self.component = nn.Sequential(
            nn.Linear(9, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.embedding),
        )
        self.register_buffer("slot_identity", torch.eye(3), persistent=False)
        self.baseline = nn.Parameter(torch.zeros(3, 2 * self.embedding))
        nn.init.normal_(self.baseline, mean=0.0, std=0.02)
        self.cross_source = nn.Sequential(
            nn.Linear(3 * 2 * self.embedding, 3 * 2 * self.embedding),
            nn.GELU(),
            nn.Linear(3 * 2 * self.embedding, 3 * 2 * self.embedding),
        )

    @property
    def output_dimension(self) -> int:
        return 3 * 2 * self.embedding

    def forward(
        self,
        component_features: torch.Tensor,
        component_weights: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if component_features.ndim != 4 or component_features.shape[1:3] != (3, 4):
            raise ValueError("component features must have shape (batch,3,4,6)")
        if component_features.shape[-1] != 6:
            raise ValueError("each component requires six physical features")
        if component_weights.shape != component_features.shape[:3]:
            raise ValueError("component weights have the wrong shape")
        if active_mask.shape != component_features.shape[:2]:
            raise ValueError("active source mask has the wrong shape")
        batch = len(component_features)
        identity = self.slot_identity[None, :, None, :].expand(batch, 3, 4, 3)
        encoded = self.component(torch.cat([component_features, identity], dim=-1))
        valid = component_weights > 0
        weighted = torch.sum(encoded * component_weights[..., None], dim=2)
        maximum = encoded.masked_fill(~valid[..., None], -torch.inf).max(dim=2).values
        active_encoded = torch.cat([weighted, maximum], dim=-1)
        baseline = self.baseline[None, :, :].expand(batch, -1, -1)
        source = torch.where(active_mask[..., None], active_encoded, baseline)
        flattened = source.reshape(batch, -1)
        return flattened + self.cross_source(flattened)


class DirectMixtureEvidenceNetV3(nn.Module):
    """Absolute-evidence regressor over mixture-set and nuclear inputs."""

    def __init__(
        self,
        *,
        component_hidden: int = 192,
        component_embedding: int = 96,
        hidden: int = 512,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = MixtureSourceEncoderV3(component_hidden, component_embedding)
        self.nuclear_encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        blocks: list[nn.Module] = [
            nn.Linear(self.encoder.output_dimension + 128, hidden),
            nn.GELU(),
        ]
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(hidden, hidden), nn.GELU()])
        blocks.append(nn.Linear(hidden, 1))
        self.head = nn.Sequential(*blocks)

    def forward(
        self,
        component_features: torch.Tensor,
        component_weights: torch.Tensor,
        active_mask: torch.Tensor,
        nuclear_observation: torch.Tensor,
    ) -> torch.Tensor:
        context = torch.cat(
            [
                self.encoder(component_features, component_weights, active_mask),
                self.nuclear_encoder(nuclear_observation),
            ],
            dim=1,
        )
        return self.head(context).squeeze(1)


class ConditionalMixtureEvidenceFlowNetV3(nn.Module):
    """Normalized nuclear-observable flow conditioned on component sets."""

    def __init__(
        self,
        *,
        component_hidden: int = 192,
        component_embedding: int = 96,
        hidden: int = 384,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = MixtureSourceEncoderV3(component_hidden, component_embedding)
        context_dimension = self.encoder.output_dimension
        self.vector_field = ConditionalVectorFieldV3(
            context_dimension, hidden=hidden, layers=layers
        )
        self.normalization_head = nn.Sequential(
            nn.Linear(context_dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def context(
        self,
        component_features: torch.Tensor,
        component_weights: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(component_features, component_weights, active_mask)

    def normalization(self, context: torch.Tensor) -> torch.Tensor:
        return self.normalization_head(context).squeeze(1)


class NuclearFourierEncoderV4(nn.Module):
    """Smooth multiscale encoding of seven standardized nuclear observables."""

    def __init__(self, embedding: int = 256) -> None:
        super().__init__()
        frequencies = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=True)
        input_dimension = 7 * (1 + 2 * len(frequencies))
        self.net = nn.Sequential(
            nn.Linear(input_dimension, embedding),
            nn.SiLU(),
            nn.Linear(embedding, embedding),
            nn.SiLU(),
            nn.LayerNorm(embedding),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != 7:
            raise ValueError("nuclear observations must have shape (batch,7)")
        phase = (
            torch.pi
            * observation[:, :, None]
            * self.frequencies[None, None, :]
        )
        features = torch.cat(
            [
                observation,
                torch.sin(phase).reshape(len(observation), -1),
                torch.cos(phase).reshape(len(observation), -1),
            ],
            dim=1,
        )
        return self.net(features)


class ResidualEvidenceBlockV4(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class DirectMixtureEvidenceNetV4(nn.Module):
    """Residual direct log-evidence surrogate over source sets and nuclear data."""

    def __init__(
        self,
        *,
        component_hidden: int = 256,
        component_embedding: int = 128,
        nuclear_embedding: int = 256,
        hidden: int = 640,
        layers: int = 6,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("at least one residual block is required")
        self.encoder = MixtureSourceEncoderV3(
            component_hidden, component_embedding
        )
        self.nuclear_encoder = NuclearFourierEncoderV4(nuclear_embedding)
        self.source_projection = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dimension),
            nn.Linear(self.encoder.output_dimension, nuclear_embedding),
            nn.SiLU(),
        )
        self.source_normalization_head = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dimension),
            nn.Linear(self.encoder.output_dimension, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        joint_dimension = self.encoder.output_dimension + 3 * nuclear_embedding
        self.input = nn.Sequential(
            nn.Linear(joint_dimension, hidden),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualEvidenceBlockV4(hidden) for _ in range(layers)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        component_features: torch.Tensor,
        component_weights: torch.Tensor,
        active_mask: torch.Tensor,
        nuclear_observation: torch.Tensor,
    ) -> torch.Tensor:
        source = self.encoder(
            component_features, component_weights, active_mask
        )
        nuclear = self.nuclear_encoder(nuclear_observation)
        source_nuclear = self.source_projection(source)
        joint = torch.cat(
            [source, nuclear, source_nuclear, source_nuclear * nuclear], dim=1
        )
        return self.output(self.blocks(self.input(joint))).squeeze(1)

    def source_normalization(
        self,
        component_features: torch.Tensor,
        component_weights: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        source = self.encoder(
            component_features, component_weights, active_mask
        )
        return self.source_normalization_head(source).squeeze(1)


class SVDSourceBranchV4(nn.Module):
    """Map a component-set source description to low-rank evidence factors."""

    def __init__(
        self,
        *,
        output_dimension: int,
        component_hidden: int = 256,
        component_embedding: int = 128,
        hidden: int = 512,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = MixtureSourceEncoderV3(
            component_hidden, component_embedding
        )
        self.input = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dimension),
            nn.Linear(self.encoder.output_dimension, hidden),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualEvidenceBlockV4(hidden) for _ in range(layers)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, output_dimension),
        )

    def forward(
        self,
        component_features: torch.Tensor,
        component_weights: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encoder(
            component_features, component_weights, active_mask
        )
        return self.output(self.blocks(self.input(encoded)))


class SVDNuclearBranchV4(nn.Module):
    """Map a nuclear observation to low-rank evidence factors."""

    def __init__(
        self,
        *,
        output_dimension: int,
        embedding: int = 256,
        hidden: int = 512,
        layers: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = NuclearFourierEncoderV4(embedding)
        self.input = nn.Sequential(
            nn.Linear(embedding, hidden), nn.SiLU()
        )
        self.blocks = nn.Sequential(
            *[ResidualEvidenceBlockV4(hidden) for _ in range(layers)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, output_dimension),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.output(
            self.blocks(self.input(self.encoder(observation)))
        )


__all__ = [
    "ConditionalEvidenceFlowNetV3",
    "ConditionalMixtureEvidenceFlowNetV3",
    "ConditionalVectorFieldV3",
    "DirectConditionalEvidenceNetV3",
    "DirectMixtureEvidenceNetV3",
    "DirectMixtureEvidenceNetV4",
    "MixtureSourceEncoderV3",
    "NuclearFourierEncoderV4",
    "SVDNuclearBranchV4",
    "SVDSourceBranchV4",
]
