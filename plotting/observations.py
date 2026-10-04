"""Checksum-gated observational overlays shared by mass--radius figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import seaborn as sns

from plotting.common import ROOT, require, sha256


MANIFEST = ROOT / "data/plotting-manifest.json"


def records(group: str) -> list[dict]:
    manifest = json.loads(MANIFEST.read_text())
    return manifest["roots"][group]


def verify_root(root: Path, group: str) -> dict[str, str]:
    verified = {}
    for record in records(group):
        path = root / record["filename"]
        require(path.is_file(), f"missing plotting datum: {path}")
        require(path.stat().st_size == record["bytes"], f"size mismatch: {path}")
        digest = sha256(path)
        require(digest == record["sha256"], f"hash mismatch: {path}")
        verified[record["filename"]] = digest
    return verified


def load_mass_radius_overlays(root: Path) -> tuple[dict, dict[str, str]]:
    verified = verify_root(root, "mass_radius_overlays")
    values = {
        "gw50": np.loadtxt(root / "GW170817_50.csv", delimiter=","),
        "gw90": np.loadtxt(root / "GW170817_90.csv", delimiter=","),
        "miller0030": np.loadtxt(root / "Miller_68_3_J0030_0451.csv", delimiter=","),
        "riley0030": np.loadtxt(root / "Riley_68_3_J0030_0451.csv", delimiter=","),
        "riley0740": np.loadtxt(root / "Riley_68_J0740_6620.csv", delimiter=","),
        "miller0740_low": np.loadtxt(
            root / "Miler_6620_lower1.png.dat", usecols=(0, 1)
        ),
        "miller0740_high": np.loadtxt(
            root / "Miler_6620_upper1.png.dat", usecols=(0, 1)
        ),
        "j0437": np.loadtxt(
            root / "J0437_4715_posterior_RM.dat", usecols=(0, 1), unpack=True
        ),
        "dittmann0740": np.loadtxt(root / "Dittmann_68_J0740_6620.csv", delimiter=","),
    }
    return values, verified


def common_nucleonic(axis, observation: dict) -> None:
    axis.fill(
        observation["gw90"][:, 0],
        observation["gw90"][:, 1],
        color="steelblue",
        linewidth=1.5,
        alpha=0.25,
        edgecolor="steelblue",
        zorder=1,
    )
    axis.fill(
        observation["gw50"][:, 0],
        observation["gw50"][:, 1],
        color="steelblue",
        linestyle="dashed",
        linewidth=1.5,
        alpha=0.25,
        edgecolor="steelblue",
        zorder=1,
    )
    axis.fill(
        observation["dittmann0740"][:, 0],
        observation["dittmann0740"][:, 1],
        color="orangered",
        linewidth=1.5,
        alpha=0.30,
        linestyle="dotted",
        zorder=1,
    )


def draw_j0030(axis, observation: dict, *, alpha: float = 0.30) -> None:
    axis.fill(
        observation["miller0030"][:, 0],
        observation["miller0030"][:, 1],
        color="c",
        hatch="x",
        linewidth=1.5,
        alpha=alpha,
        linestyle="dashed",
        zorder=1,
    )
    axis.fill(
        observation["riley0030"][:, 0],
        observation["riley0030"][:, 1],
        color="goldenrod",
        linewidth=1.5,
        alpha=alpha,
        linestyle="dashed",
        zorder=1,
    )


def draw_j0437(
    axis,
    observation: dict,
    *,
    alpha: float = 0.30,
    seed: int = 3,
    sample: np.ndarray | None = None,
) -> None:
    values = observation["j0437"]
    if sample is None:
        picked = np.random.default_rng(seed).choice(values.shape[1], 8000, replace=False)
        sample = values[:, picked]
    sns.kdeplot(
        ax=axis,
        x=sample[0],
        y=sample[1],
        fill=True,
        color="#b45ec6",
        alpha=alpha,
        zorder=2,
    )


def draw_nuclear_panel(axis, observation: dict) -> None:
    axis.fill(
        observation["miller0030"][:, 0],
        observation["miller0030"][:, 1],
        color="c",
        hatch="x",
        lw=1.2,
        alpha=0.20,
        zorder=1,
    )
    axis.fill(
        observation["riley0030"][:, 0],
        observation["riley0030"][:, 1],
        color="goldenrod",
        lw=1.2,
        alpha=0.20,
        zorder=1,
    )
    axis.fill(
        observation["gw90"][:, 0],
        observation["gw90"][:, 1],
        color="steelblue",
        lw=1.2,
        alpha=0.20,
        zorder=1,
    )
    axis.fill(
        observation["gw50"][:, 0],
        observation["gw50"][:, 1],
        color="steelblue",
        lw=1.2,
        alpha=0.20,
        zorder=1,
    )
    axis.fill(
        observation["dittmann0740"][:, 0],
        observation["dittmann0740"][:, 1],
        color="orangered",
        lw=1.2,
        alpha=0.20,
        zorder=1,
    )
    values = observation["j0437"]
    picked = np.random.default_rng(3).choice(values.shape[1], 8000, replace=False)
    sns.kdeplot(
        ax=axis,
        x=values[0, picked],
        y=values[1, picked],
        fill=True,
        color="#b45ec6",
        alpha=0.20,
        levels=5,
        thresh=0.08,
        zorder=2,
    )
