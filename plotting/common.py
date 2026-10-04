"""Shared, side-effect-free helpers for deterministic post-processing."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXED_PDF_TIME = dt.datetime(2026, 8, 28, tzinfo=dt.timezone.utc)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_weights(weight: np.ndarray) -> np.ndarray:
    values = np.asarray(weight, dtype=np.float64)
    require(values.ndim == 1, "weights must be one-dimensional")
    require(np.isfinite(values).all(), "weights contain non-finite values")
    require(np.all(values >= 0.0), "weights contain negative values")
    total = float(values.sum())
    require(total > 0.0, "weights have zero total")
    values = values / total
    require(abs(float(values.sum()) - 1.0) <= 5e-15, "weights do not normalize")
    return values


def effective_sample_size(weight: np.ndarray) -> float:
    values = normalize_weights(weight)
    return float(1.0 / np.sum(values**2))


def weighted_quantile(
    values: np.ndarray,
    weight: np.ndarray,
    probabilities: Iterable[float],
) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    weights = normalize_weights(weight)
    probability = np.asarray(tuple(probabilities), dtype=np.float64)
    require(samples.ndim == 1 and len(samples) == len(weights), "bad quantile arrays")
    require(np.isfinite(samples).all(), "quantile samples contain non-finite values")
    require(np.all((probability >= 0.0) & (probability <= 1.0)), "bad quantile")
    order = np.argsort(samples, kind="mergesort")
    sorted_values = samples[order]
    sorted_weight = weights[order]
    position = np.cumsum(sorted_weight) - 0.5 * sorted_weight
    position /= sorted_weight.sum()
    return np.interp(
        probability,
        position,
        sorted_values,
        left=sorted_values[0],
        right=sorted_values[-1],
    )


def save_figure(figure: Any, output_stem: Path, *, dpi: int = 180) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = output_stem.with_suffix(".pdf")
    png = output_stem.with_suffix(".png")
    pdf_metadata = {
        "Title": output_stem.name,
        "Author": "Prashant Thakur",
        "Creator": "ddb-amortized-inference-paper",
        "CreationDate": FIXED_PDF_TIME,
        "ModDate": FIXED_PDF_TIME,
    }
    figure.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata=pdf_metadata,
    )
    figure.savefig(
        png,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "ddb-amortized-inference-paper"},
    )
    return pdf, png


def write_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def output_record(pdf: Path, png: Path) -> dict[str, Any]:
    return {
        "pdf": pdf.name,
        "pdf_sha256": sha256(pdf),
        "png": png.name,
        "png_sha256": sha256(png),
    }
