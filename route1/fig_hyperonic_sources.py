#!/usr/bin/env python3
"""Figures 8 and 9: frozen hyperonic A-NET+IS vs independent UltraNest for an unseen NICER source
(J0614 or J1231 replacing J0030; J1614 replacing J0740). The drawing code is the code that
produced the published panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resampled_band(
    radius: np.ndarray,
    maximum_mass: np.ndarray,
    counts: np.ndarray,
    mass_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same equal-weight-resample convention as the hyperonic A1 figure (Figure 7)."""

    band = np.full((3, len(mass_grid)), np.nan, dtype=np.float64)
    rows = np.zeros(len(mass_grid), dtype=np.float64)
    for column, mass in enumerate(mass_grid):
        supported = np.isfinite(radius[:, column]) & (maximum_mass >= mass)
        if not supported.any():
            continue
        rows[column] = float(counts[supported].sum())
        values = np.repeat(radius[supported, column], counts[supported])
        band[:, column] = np.percentile(values, [5.0, 50.0, 95.0])
    return band, rows / float(counts.sum()), rows.copy(), rows.copy()


def load_ultranest(
    path: Path, prefix: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if prefix:
            key = lambda name: f"{prefix}_{name}"
        else:
            key = lambda name: name
        return (
            np.asarray(saved["mass_grid"], dtype=np.float64),
            np.asarray(saved[key("band")], dtype=np.float64),
            np.asarray(saved[key("support_weight")], dtype=np.float64),
            np.asarray(saved[key("support_ess")], dtype=np.float64),
            np.asarray(saved[key("support_rows")], dtype=np.float64),
        )


def draw_j1614_background(axis, observation: dict, j1614_path: Path) -> None:
    """Draw the baseline overlays with J1614 replacing J0740."""

    alpha = 0.4
    axis.fill(
        observation["miller0030"][:, 0],
        observation["miller0030"][:, 1],
        color="c",
        hatch="x",
        linewidth=2.0,
        alpha=alpha,
        linestyle="dashed",
        zorder=1,
    )
    axis.fill(
        observation["riley0030"][:, 0],
        observation["riley0030"][:, 1],
        color="goldenrod",
        linewidth=2.0,
        alpha=alpha,
        linestyle="dashed",
        zorder=1,
    )
    axis.fill(
        observation["gw90"][:, 0],
        observation["gw90"][:, 1],
        color="steelblue",
        linestyle="-",
        linewidth=2.0,
        alpha=alpha,
        edgecolor="steelblue",
        zorder=1,
    )
    axis.fill(
        observation["gw50"][:, 0],
        observation["gw50"][:, 1],
        color="steelblue",
        linestyle="dashed",
        linewidth=2.0,
        alpha=alpha,
        edgecolor="steelblue",
        zorder=1,
    )
    j0437 = observation["j0437"]
    selected = np.random.default_rng(3).choice(
        j0437.shape[1], min(8000, j0437.shape[1]), replace=False
    )
    sns.kdeplot(
        ax=axis,
        x=j0437[0, selected],
        y=j0437[1, selected],
        fill=True,
        color="#b45ec6",
        alpha=alpha,
        zorder=2,
    )
    values = np.loadtxt(j1614_path)
    require(values.ndim == 2 and values.shape[1] >= 2, "bad J1614 sample file")
    selected = np.random.default_rng(5).choice(
        len(values), min(8000, len(values)), replace=False
    )
    sns.kdeplot(
        ax=axis,
        x=values[selected, 1],
        y=values[selected, 0],
        fill=True,
        color="#8c3bab",
        alpha=0.35,
        zorder=2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("J0614", "J1231", "J1614"), required=True)
    parser.add_argument("--paper-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--resample", type=Path, required=True)
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--ultranest", type=Path, required=True)
    parser.add_argument("--ultranest-prefix", default="")
    parser.add_argument("--mr-data", type=Path, required=True)
    parser.add_argument("--pulsar-data", type=Path, required=True)
    parser.add_argument("--j1614-data", type=Path)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--minimum-ess", type=float, default=1000.0)
    parser.add_argument("--maximum-weight", type=float, default=0.01)
    arguments = parser.parse_args()

    sys.path.insert(0, str(arguments.paper_root.resolve()))
    from plotting.mass_radius import (
        hyperonic_overlays,
        hyperonic_pulsar_density,
        new_hyperonic_axis,
    )
    from plotting.mass_radius_style import (
        common_certified_body,
        draw_support_aware_band,
    )
    from plotting.observations import load_mass_radius_overlays

    with np.load(arguments.certificate, allow_pickle=False) as saved:
        require(bool(saved["gate_pass"]), "certificate did not pass its internal gate")
        source_ess = float(saved["posterior_ess"])
        maximum_weight = float(saved["maximum_normalized_weight"])
    with np.load(arguments.resample, allow_pickle=False) as saved:
        theta = np.asarray(saved["theta"], dtype=np.float64)
        counts = np.asarray(saved["counts"], dtype=np.int64)
        posterior_draws = int(saved["posterior_draws"])
        seed = int(saved["seed"])
        certified_hash = str(saved["source_sha256"].item())
    require(
        certified_hash == sha256(arguments.certificate),
        "resample does not reference the supplied certificate",
    )
    with np.load(arguments.curves, allow_pickle=False) as saved:
        curve_theta = np.asarray(saved["theta"], dtype=np.float64)
        mass = np.asarray(saved["MG"], dtype=np.float64)
        radius = np.asarray(saved["Rg"], dtype=np.float64)
        maximum_mass = np.asarray(saved["MM"], dtype=np.float64)
        curve_seconds = float(saved["total_seconds"])
        curve_workers = int(saved["workers"])
    require(np.array_equal(theta, curve_theta), "curve and resample rows differ")
    require(counts.sum() == posterior_draws, "resample count mismatch")

    anet_band, anet_weight, anet_ess, anet_rows = resampled_band(
        radius, maximum_mass, counts, mass
    )
    (
        ultranest_mass,
        ultranest_band,
        ultranest_weight,
        ultranest_ess,
        ultranest_rows,
    ) = load_ultranest(arguments.ultranest, arguments.ultranest_prefix)
    require(np.array_equal(mass, ultranest_mass), "mass-grid mismatch")

    common = common_certified_body(
        [ultranest_band, anet_band],
        [ultranest_weight, anet_weight],
        [ultranest_ess, anet_ess],
    )
    edge_difference = np.abs(
        np.concatenate(
            (
                anet_band[0, common] - ultranest_band[0, common],
                anet_band[2, common] - ultranest_band[2, common],
            )
        )
    )
    median_difference = np.abs(anet_band[1, common] - ultranest_band[1, common])
    intersection = np.maximum(
        0.0,
        np.minimum(anet_band[2, common], ultranest_band[2, common])
        - np.maximum(anet_band[0, common], ultranest_band[0, common]),
    )
    union = (
        np.maximum(anet_band[2, common], ultranest_band[2, common])
        - np.minimum(anet_band[0, common], ultranest_band[0, common])
    )

    observation, overlay_hashes = load_mass_radius_overlays(arguments.mr_data)
    figure, axis = new_hyperonic_axis()
    if arguments.scenario == "J0614":
        hyperonic_overlays(axis, observation, skip=("j0437",))
        hyperonic_pulsar_density(
            axis,
            arguments.pulsar_data / "J0614_mrsamples.dat",
            0,
            1,
            None,
            "#b45ec6",
        )
    elif arguments.scenario == "J1231":
        hyperonic_overlays(axis, observation, skip=("j0030",))
        hyperonic_pulsar_density(
            axis,
            arguments.pulsar_data / "J1231_wmrsamples.txt",
            1,
            2,
            0,
            "teal",
        )
    else:
        require(arguments.j1614_data is not None, "--j1614-data is required")
        draw_j1614_background(axis, observation, arguments.j1614_data)

    draw_support_aware_band(
        axis,
        mass,
        ultranest_band,
        ultranest_weight,
        ultranest_ess,
        ultranest_rows,
        common,
        color="forestgreen",
        linestyle="-",
        linewidth=1.8,
        label="UltraNest",
        shaded=True,
        zorder=4,
    )
    draw_support_aware_band(
        axis,
        mass,
        anet_band,
        anet_weight,
        anet_ess,
        anet_rows,
        common,
        color="royalblue",
        linestyle="--",
        linewidth=1.8,
        label="A-NET+IS",
        shaded=True,
        zorder=5,
    )
    axis.legend(
        handles=(
            Line2D([], [], color="forestgreen", lw=1.8, label="UltraNest"),
            Line2D([], [], color="royalblue", lw=1.8, ls="--", label="A-NET+IS"),
        ),
        loc="upper right",
        fontsize=11,
        frameon=False,
    )
    figure.tight_layout()
    arguments.output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = arguments.output_stem.with_suffix(".pdf")
    png = arguments.output_stem.with_suffix(".png")
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(figure)

    data_path = arguments.output_stem.with_name(arguments.output_stem.name + "_data.npz")
    np.savez_compressed(
        data_path,
        mass_grid=mass,
        ultranest_band=ultranest_band,
        ultranest_support_weight=ultranest_weight,
        ultranest_support_ess=ultranest_ess,
        ultranest_support_rows=ultranest_rows,
        anet_band=anet_band,
        anet_support_weight=anet_weight,
        anet_support_ess=anet_ess,
        anet_support_rows=anet_rows,
        common_body_mask=common,
    )
    passed = source_ess >= arguments.minimum_ess and maximum_weight <= arguments.maximum_weight
    record = {
        "status": "PASS" if passed else "FAIL",
        "scenario": arguments.scenario,
        "method": "frozen dual-head full-prior 9D A-NET with fresh exact IS",
        "retraining_for_query": False,
        "source_posterior_ess": source_ess,
        "maximum_normalized_weight": maximum_weight,
        "acceptance_gate": {
            "minimum_ess": arguments.minimum_ess,
            "maximum_normalized_weight": arguments.maximum_weight,
        },
        "posterior_resample": {
            "draws": posterior_draws,
            "unique_curves": int(len(theta)),
            "seed": seed,
        },
        "curve_replay": {"workers": curve_workers, "seconds": curve_seconds},
        "common_certified_mass_range_msun": [
            float(mass[common][0]),
            float(mass[common][-1]),
        ],
        "band_agreement": {
            "mean_absolute_90pct_edge_difference_km": float(edge_difference.mean()),
            "median_absolute_90pct_edge_difference_km": float(np.median(edge_difference)),
            "p95_absolute_90pct_edge_difference_km": float(np.percentile(edge_difference, 95.0)),
            "maximum_absolute_90pct_edge_difference_km": float(edge_difference.max()),
            "mean_absolute_median_difference_km": float(median_difference.mean()),
            "maximum_absolute_median_difference_km": float(median_difference.max()),
            "fraction_of_mass_points_with_overlapping_90pct_intervals": float(
                np.mean(intersection > 0.0)
            ),
            "mean_interval_intersection_over_union": float(np.mean(intersection / union)),
            "minimum_interval_intersection_over_union": float(np.min(intersection / union)),
        },
        "inputs_sha256": {
            str(arguments.certificate): sha256(arguments.certificate),
            str(arguments.resample): sha256(arguments.resample),
            str(arguments.curves): sha256(arguments.curves),
            str(arguments.ultranest): sha256(arguments.ultranest),
        },
        "observational_overlay_sha256": overlay_hashes,
        "outputs_sha256": {
            str(pdf): sha256(pdf),
            str(png): sha256(png),
            str(data_path): sha256(data_path),
        },
        "manuscript_modified": False,
    }
    record_path = arguments.output_stem.with_suffix(".json")
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
