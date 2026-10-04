"""Stable paths to immutable resources shipped in source and wheel builds."""

from __future__ import annotations

from pathlib import Path


DEFAULT_A1_CERTIFICATE = (
    Path(__file__).resolve().parent
    / "data"
    / "a1_fixed_target_certification.npz"
)
DEFAULT_SOURCE_TARGET_CERTIFICATE = (
    Path(__file__).resolve().parent
    / "data"
    / "source_target_certification.npz"
)
DEFAULT_HYPERONIC_TARGET_CERTIFICATE = (
    Path(__file__).resolve().parent
    / "data"
    / "hyperonic_target_certification.npz"
)


def default_target_certificate(
    source_scenario: str = "A1",
    model: str = "ddb",
) -> Path:
    """Select the immutable target certificate for one model/configuration."""

    if model == "ddb-hyperonic":
        return DEFAULT_HYPERONIC_TARGET_CERTIFICATE
    if model != "ddb":
        raise KeyError(f"no target certificate is registered for model {model!r}")
    return (
        DEFAULT_A1_CERTIFICATE
        if source_scenario == "A1"
        else DEFAULT_SOURCE_TARGET_CERTIFICATE
    )
