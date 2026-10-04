#!/usr/bin/env python3
"""Helpers of the Evidence Network trainer (green_en/solution/s2_green_en.py): exact on-the-fly nuclear labels.

Every optimizer step draws fresh continuous nuclear observations with draw_designs (1/3 single-axis U(-2.5, 2.5);
1/3 joint N(0, 1.15^2) clipped to +-2.5; 1/3 uniform box [-2.5, 2.5]^7, in sigma units about the default
observation), and ExactLabels computes their exact bank log-evidence from the cached source log-weights with one
full-precision matrix product.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

FAMILIES = ("axis", "gauss", "box")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def draw_designs(family: str, count: int, generator: torch.Generator, device) -> torch.Tensor:
    if family == "axis":
        shift = torch.zeros(count, 7, device=device)
        column = torch.randint(7, (count,), generator=generator, device=device)
        shift[torch.arange(count, device=device), column] = (
            torch.rand(count, generator=generator, device=device) * 5.0 - 2.5
        )
        return shift
    if family == "gauss":
        return (torch.randn(count, 7, generator=generator, device=device) * 1.15).clamp(-2.5, 2.5)
    if family == "box":
        return torch.rand(count, 7, generator=generator, device=device) * 5.0 - 2.5
    raise ValueError(family)


class ExactLabels:
    """log Z(s,t) = log sum_i exp(SW[s,i] - |t - z_i|^2/2) - denominator, in full fp32."""

    def __init__(self, source_log_weight: torch.Tensor, z_rows: torch.Tensor, denominator: float):
        self.sw, self.z, self.den = source_log_weight, z_rows, float(denominator)
        self.z2 = (z_rows * z_rows).sum(1)

    @torch.no_grad()
    def __call__(self, scenario_index: torch.Tensor, shift: torch.Tensor, *, with_ess: bool = False):
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            sw = self.sw[scenario_index]
            smax = sw.max(1, keepdim=True).values
            sf = torch.exp(sw - smax)
            nl = -0.5 * ((shift * shift).sum(1)[:, None] + self.z2[None, :] - 2.0 * shift @ self.z.T)
            nmax = nl.max(1, keepdim=True).values
            nf = torch.exp(nl - nmax)
            total = sf @ nf.T
            log_z = torch.log(total) + smax + nmax.T - self.den
            if not with_ess:
                return log_z
            return log_z, total * total / ((sf * sf) @ (nf * nf).T)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous


def summarize(residual: np.ndarray) -> dict:
    a = np.abs(residual)
    return {"values": int(residual.size), "bias": float(residual.mean()), "mae": float(a.mean()),
            "rmse": float(np.sqrt((residual ** 2).mean())), "p90_absolute": float(np.quantile(a, 0.9)),
            "maximum_absolute": float(a.max())}
