#!/usr/bin/env python3
"""Figure 6: nucleonic DDB with PSR J1614-2230 replacing J0740 (A-NET+IS vs UltraNest), plus its comparison metrics.

The paper figure is J1614_mass_radius_paper_figure.pdf (copied as FINAL_J1614_methods_with_observations.pdf).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

CODE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = CODE_ROOT
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from plotting.mass_radius_style import draw_support_aware_band  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def interval_record(values: np.ndarray) -> dict[str, float]:
    low, median, high = map(float, values)
    return {
        "q05": low,
        "median": median,
        "q95": high,
        "minus": median - low,
        "plus": high - median,
    }


def configure_mass_radius_axis(axis) -> None:
    axis.set_xlim(9.0, 15.0)
    axis.set_ylim(0.7, 2.5)
    axis.set_xlabel(r"R [km]", fontsize=14)
    axis.set_ylabel(r"M [M$_\odot$]", fontsize=14)
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.tick_params(
        which="both", direction="in", top=True, right=True, labelsize=12
    )
    axis.tick_params(which="major", length=6)
    axis.tick_params(which="minor", length=3)
    axis.grid(alpha=0.30)


def draw_pointwise_band(
    axis,
    mass: np.ndarray,
    band: np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    linestyle,
    linewidth: float,
    zorder: int = 5,
) -> None:
    """Draw the two open edges of the pointwise 90% R(M) band."""

    axis.plot(
        band[0, mask], mass[mask], color=color, ls=linestyle,
        lw=linewidth, zorder=zorder,
    )
    axis.plot(
        band[2, mask], mass[mask], color=color, ls=linestyle,
        lw=linewidth, zorder=zorder,
    )


def draw_support_faded_tail(
    axis,
    mass: np.ndarray,
    band: np.ndarray,
    support_weight: np.ndarray,
    support_rows: np.ndarray,
    support_ess: np.ndarray,
    body_mask: np.ndarray,
    *,
    color: str,
    linestyle,
    linewidth: float,
    zorder: int = 5,
) -> None:
    """Continue the same pointwise band while posterior support fades away."""

    valid = (
        np.isfinite(band).all(axis=0)
        & (support_rows >= 2)
        & (support_ess >= 1.2)
    )
    body_last = int(np.flatnonzero(body_mask)[-1])
    for index in range(body_last, len(mass) - 1):
        if not (valid[index] and valid[index + 1]):
            continue
        relative_support = float(
            np.sqrt(np.mean(support_weight[index:index + 2]) / 0.05)
        )
        alpha = float(np.clip(relative_support, 0.035, 1.0))
        for edge in (0, 2):
            axis.plot(
                band[edge, index:index + 2],
                mass[index:index + 2],
                color=color,
                ls=linestyle,
                lw=linewidth,
                alpha=alpha,
                zorder=zorder,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anet", type=Path, default=PACKAGE_ROOT / "results/nucleonic/j1614_anet_mass_radius_curves.npz")
    parser.add_argument("--ultranest", type=Path, default=PACKAGE_ROOT / "results/nucleonic/j1614_ultranest_mass_radius_curves.npz")
    parser.add_argument("--j1614", type=Path, default=PACKAGE_ROOT / "data/observations/J1614_STU_mrsamples_post_equal_weights.dat")
    parser.add_argument("--overlay-root", type=Path, default=PACKAGE_ROOT / "data/plotting_overlays")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "build/figures/J1614")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(arguments.anet, allow_pickle=False) as saved:
        a = {name: np.asarray(saved[name]) for name in saved.files}
    with np.load(arguments.ultranest, allow_pickle=False) as saved:
        u = {name: np.asarray(saved[name]) for name in saved.files}
    mass = np.asarray(a["mass_grid"], dtype=np.float64)
    require(np.array_equal(mass, u["mass_grid"]), "mass-grid mismatch")
    require(a["band"].shape == (3, len(mass)), "bad A-NET band")
    require(u["band"].shape == (3, len(mass)), "bad UltraNest band")
    a_composition = str(a["composition"].item()) if "composition" in a else "nucleonic DDB"
    u_composition = str(u["composition"].item()) if "composition" in u else "nucleonic DDB"
    require(a_composition == u_composition, "composition mismatch")
    require(a_composition == "nucleonic DDB", "non-nucleonic input rejected")
    shaded = False

    # A 90% band is displayed only while at least 5% of each posterior supports
    # that stellar mass.  This is exactly the upper 90% Mmax boundary and avoids
    # comparing a conditional band built from a vanishing high-mass tail.
    common = (
        np.isfinite(a["band"]).all(axis=0)
        & np.isfinite(u["band"]).all(axis=0)
        & (a["support_weight"] >= 0.05)
        & (u["support_weight"] >= 0.05)
        & (a["support_ess"] >= 100.0)
        & (u["support_ess"] >= 100.0)
    )
    require(common.any(), "no common certified band range")
    a_band = np.asarray(a["band"], dtype=np.float64)
    u_band = np.asarray(u["band"], dtype=np.float64)
    difference = a_band - u_band
    edge_abs = np.abs(difference[[0, 2]][:, common])
    median_abs = np.abs(difference[1, common])
    intersection = np.maximum(
        0.0,
        np.minimum(a_band[2, common], u_band[2, common])
        - np.maximum(a_band[0, common], u_band[0, common]),
    )
    union = (
        np.maximum(a_band[2, common], u_band[2, common])
        - np.minimum(a_band[0, common], u_band[0, common])
    )
    overlap = intersection / union
    overlap_boolean = (
        np.maximum(a_band[0, common], u_band[0, common])
        <= np.minimum(a_band[2, common], u_band[2, common])
    )

    scalar = {}
    scalar_keys = {
        "maximum_mass_msun": "maximum_mass",
        "radius_at_maximum_mass_km": "radius_at_maximum_mass",
        "radius_1p4_km": "radius_1p4",
    }
    for output_name, key in scalar_keys.items():
        av = np.asarray(a[key], dtype=np.float64)
        uv = np.asarray(u[key], dtype=np.float64)
        aw = np.asarray(a["weight"], dtype=np.float64)
        uw = np.asarray(u["weight"], dtype=np.float64)

        def quantile(value, weight):
            valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
            value, weight = value[valid], weight[valid]
            order = np.argsort(value, kind="mergesort")
            value, weight = value[order], weight[order]
            position = (np.cumsum(weight) - 0.5 * weight) / weight.sum()
            return np.interp([0.05, 0.50, 0.95], position, value)

        aq = quantile(av, aw)
        uq = quantile(uv, uw)
        scalar[output_name] = {
            "A-NET+IS": interval_record(aq),
            "UltraNest": interval_record(uq),
            "median_difference_A-NET_minus_UltraNest": float(aq[1] - uq[1]),
            "q05_difference_A-NET_minus_UltraNest": float(aq[0] - uq[0]),
            "q95_difference_A-NET_minus_UltraNest": float(aq[2] - uq[2]),
        }

    metrics = {
        "status": "PASS",
        "target": "A1 with J1614 replacing J0740",
        "composition": a_composition,
        "credible_level_percent": 90,
        "display_convention": {
            "shared_renderer": "plotting.mass_radius_style.draw_support_aware_band",
            "certified_body_posterior_support": 0.05,
            "forced_high_mass_closure": False,
            "high_mass_tail": "width and opacity taper with posterior support",
            "shaded": shaded,
        },
        "posterior_rows": {
            "A-NET+IS_positive_weight": int(len(a["weight"])),
            "UltraNest": int(len(u["weight"])),
        },
        "posterior_ess": {
            "A-NET+IS": float(1.0 / np.sum(a["weight"] ** 2)),
            "UltraNest": float(1.0 / np.sum(u["weight"] ** 2)),
        },
        "comparison_range": {
            "definition": (
                "both pointwise bands finite, posterior support >=0.05, "
                "and conditional ESS >=100"
            ),
            "mass_min_msun": float(mass[common].min()),
            "mass_max_msun": float(mass[common].max()),
            "mass_points": int(common.sum()),
        },
        "displayed_pointwise_band": {
            "definition": (
                "open 5th and 95th weighted pointwise R(M) quantile edges in "
                "the common certified range, matching the manuscript's "
                "existing nucleonic mass-radius figure convention; the same "
                "conditional interval narrows with surviving posterior weight "
                "and fades with the square root of that support in the sparse "
                "high-mass tail"
            ),
            "mass_min_msun": float(mass[common].min()),
            "mass_max_msun": float(mass[common].max()),
            "forced_closure": False,
        },
        "band_agreement": {
            "mean_absolute_90pct_edge_difference_km": float(edge_abs.mean()),
            "rms_90pct_edge_difference_km": float(np.sqrt(np.mean(edge_abs**2))),
            "maximum_absolute_90pct_edge_difference_km": float(edge_abs.max()),
            "mean_absolute_median_curve_difference_km": float(median_abs.mean()),
            "maximum_absolute_median_curve_difference_km": float(median_abs.max()),
            "fraction_of_mass_points_with_overlapping_90pct_intervals": float(
                overlap_boolean.mean()
            ),
            "mean_interval_intersection_over_union": float(overlap.mean()),
            "minimum_interval_intersection_over_union": float(overlap.min()),
        },
        "scalar_intervals": scalar,
        "source_hashes": {
            str(arguments.anet): sha256(arguments.anet),
            str(arguments.ultranest): sha256(arguments.ultranest),
            str(arguments.j1614): sha256(arguments.j1614),
        },
    }

    # One-to-one diagnostic (not in the paper): overlap at left, edge residuals at right.
    figure, (axis, residual) = plt.subplots(
        1,
        2,
        figsize=(12.8, 6.0),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    configure_mass_radius_axis(axis)
    draw_support_aware_band(
        axis, mass, u_band, u["support_weight"], u["support_ess"],
        u["support_rows"], common, color="#2c7a3f", linestyle="-",
        linewidth=2.4, label="UltraNest", shaded=shaded, zorder=5,
    )
    draw_support_aware_band(
        axis, mass, a_band, a["support_weight"], a["support_ess"],
        a["support_rows"], common, color="#08519c", linestyle=(0, (6, 2)),
        linewidth=2.4, label="A-NET+IS", shaded=shaded, zorder=6,
    )
    axis.legend(
        handles=[
            Line2D([], [], color="#2c7a3f", lw=2.4, label="UltraNest"),
            Line2D(
                [],
                [],
                color="#08519c",
                lw=2.4,
                ls=(0, (6, 2)),
                label="A-NET+IS",
            ),
        ],
        loc="upper right",
        fontsize=12,
        frameon=False,
    )
    axis.set_title("J1614 replacement: pointwise 90% mass–radius bands", fontsize=13)

    residual.axvline(0.0, color="0.25", lw=1.2)
    residual.plot(
        difference[0, common],
        mass[common],
        color="#6a51a3",
        lw=2.0,
        label="lower edge",
    )
    residual.plot(
        difference[2, common],
        mass[common],
        color="#d95f0e",
        lw=2.0,
        ls="--",
        label="upper edge",
    )
    residual.plot(
        difference[1, common],
        mass[common],
        color="0.25",
        lw=1.5,
        ls=":",
        label="median",
    )
    maximum = max(0.05, 1.15 * float(np.max(np.abs(difference[:, common]))))
    residual.set_xlim(-maximum, maximum)
    residual.set_ylim(0.7, 2.5)
    residual.set_xlabel(r"A-NET+IS $-$ UltraNest radius [km]", fontsize=13)
    residual.set_ylabel(r"M [M$_\odot$]", fontsize=13)
    residual.xaxis.set_minor_locator(AutoMinorLocator())
    residual.yaxis.set_minor_locator(AutoMinorLocator())
    residual.tick_params(
        which="both", direction="in", top=True, right=True, labelsize=11
    )
    residual.legend(loc="best", fontsize=10.5, frameon=False)
    residual.grid(alpha=0.20)
    residual.set_title("Direct edge differences", fontsize=13)
    figure.tight_layout()
    diagnostic_stem = arguments.output_dir / "J1614_mass_radius_one_to_one"
    figure.savefig(diagnostic_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(diagnostic_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)

    # The paper figure (Figure 6): both bands over the observational constraints.
    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    root = arguments.overlay_root
    gw90 = np.loadtxt(root / "GW170817_90.csv", delimiter=",")
    gw50 = np.loadtxt(root / "GW170817_50.csv", delimiter=",")
    miller0030 = np.loadtxt(root / "Miller_68_3_J0030_0451.csv", delimiter=",")
    riley0030 = np.loadtxt(root / "Riley_68_3_J0030_0451.csv", delimiter=",")
    j0437 = np.loadtxt(
        root / "J0437_4715_posterior_RM.dat", usecols=(0, 1), unpack=True
    )
    axis.fill(gw90[:, 0], gw90[:, 1], color="steelblue", alpha=0.25, zorder=1)
    axis.fill(gw50[:, 0], gw50[:, 1], color="steelblue", alpha=0.25, zorder=1)
    axis.fill(
        miller0030[:, 0],
        miller0030[:, 1],
        color="c",
        hatch="x",
        alpha=0.30,
        zorder=1,
    )
    axis.fill(
        riley0030[:, 0],
        riley0030[:, 1],
        color="goldenrod",
        alpha=0.30,
        zorder=1,
    )
    picked = np.random.default_rng(3).choice(j0437.shape[1], 8000, replace=False)
    sns.kdeplot(
        ax=axis,
        x=j0437[0, picked],
        y=j0437[1, picked],
        fill=True,
        color="#b45ec6",
        alpha=0.30,
        zorder=2,
    )
    j1614 = np.loadtxt(arguments.j1614)
    require(j1614.ndim == 2 and j1614.shape[1] >= 2, "bad J1614 sample file")
    picked = np.random.default_rng(5).choice(
        len(j1614), min(8000, len(j1614)), replace=False
    )
    sns.kdeplot(
        ax=axis,
        x=j1614[picked, 1],
        y=j1614[picked, 0],
        fill=True,
        color="#8c3bab",
        alpha=0.35,
        zorder=2,
    )
    draw_support_aware_band(
        axis, mass, u_band, u["support_weight"], u["support_ess"],
        u["support_rows"], common, color="#2c7a3f", linestyle="-",
        linewidth=2.2, label="UltraNest", shaded=shaded, zorder=5,
    )
    draw_support_aware_band(
        axis, mass, a_band, a["support_weight"], a["support_ess"],
        a["support_rows"], common, color="#08519c", linestyle=(0, (6, 2)),
        linewidth=2.4, label="A-NET+IS", shaded=shaded, zorder=6,
    )
    configure_mass_radius_axis(axis)
    axis.legend(
        handles=[
            Line2D([], [], color="#2c7a3f", lw=2.2, label="UltraNest"),
            Line2D(
                [],
                [],
                color="#08519c",
                lw=2.4,
                ls=(0, (6, 2)),
                label="A-NET+IS",
            ),
        ],
        loc="upper right",
        fontsize=11,
        frameon=False,
    )
    figure.tight_layout()
    paper_stem = arguments.output_dir / "J1614_mass_radius_paper_figure"
    figure.savefig(paper_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(paper_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)

    metrics["outputs"] = {
        "diagnostic_pdf": str(diagnostic_stem.with_suffix(".pdf")),
        "diagnostic_png": str(diagnostic_stem.with_suffix(".png")),
        "paper_figure_pdf": str(paper_stem.with_suffix(".pdf")),
        "paper_figure_png": str(paper_stem.with_suffix(".png")),
    }
    for path in metrics["outputs"].values():
        metrics["source_hashes"][path] = sha256(Path(path))
    report = arguments.output_dir / "J1614_mass_radius_comparison.json"
    report.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
