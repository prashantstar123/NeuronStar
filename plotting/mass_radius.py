#!/usr/bin/env python3
"""Nucleonic mass--radius figures and the shared hyperonic drawing helpers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from plotting.common import ROOT, output_record, require, save_figure, sha256, write_record
from plotting.mass_radius_style import (
    common_certified_body,
    draw_support_aware_band,
    support_from_curve_counts,
)
from plotting.observations import (
    common_nucleonic,
    draw_j0030,
    draw_j0437,
    load_mass_radius_overlays,
    verify_root,
)


def nucleonic_axis(axis) -> None:
    axis.set_xlim(9, 15)
    axis.set_ylim(0.7, 2.5)
    axis.grid(alpha=0.3)
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.tick_params(which="major", length=6)
    axis.tick_params(which="minor", length=3)
    axis.set_xlabel("R [km]", fontsize=13)
    axis.set_ylabel(r"M [M$_\odot$]", fontsize=13)


def draw_nucleonic_comparison(
    axis,
    mass: np.ndarray,
    entries: list[tuple[np.ndarray, str]],
) -> None:
    settings = {
        "UltraNest": ("#2c7a3f", "-", 2.2, 6),
        "TSNPE": ("#6a51a3", (0, (1, 1)), 2.2, 7),
        "ANET": ("#08519c", (0, (6, 2)), 2.4, 8),
    }
    support = [support_from_curve_counts(band) for band, _ in entries]
    common = common_certified_body(
        [band[:3] for band, _ in entries],
        [item[0] for item in support],
        [item[1] for item in support],
    )
    labels = {"UltraNest": "UltraNest", "TSNPE": "TSNPE+MIS", "ANET": "A-NET+IS"}
    for (band, method), (weight, ess, rows) in zip(entries, support, strict=True):
        color, linestyle, width, zorder = settings[method]
        draw_support_aware_band(
            axis,
            mass,
            band[:3],
            weight,
            ess,
            rows,
            common,
            color=color,
            linestyle=linestyle,
            linewidth=width,
            label=labels[method],
            zorder=zorder,
        )


def save(figure, output_dir: Path, name: str) -> tuple[Path, Path]:
    paths = save_figure(figure, output_dir / name, dpi=150)
    plt.close(figure)
    return paths


def render_nucleonic(
    input_path: Path,
    observation: dict,
    pulsar_root: Path,
    output_dir: Path,
) -> dict[str, dict]:
    data = np.load(input_path, allow_pickle=False)
    mass = np.asarray(data["mass_grid"], dtype=np.float64)
    outputs = {}

    figure, axis = plt.subplots(figsize=(8, 6))
    draw_j0030(axis, observation)
    common_nucleonic(axis, observation)
    draw_j0437(axis, observation)
    draw_nucleonic_comparison(
        axis,
        mass,
        [
            (data["fixed_A1_UltraNest"], "UltraNest"),
            (data["fixed_A1_TSNPE"], "TSNPE"),
            (data["fixed_A1_ANET"], "ANET"),
        ],
    )
    handles = [
        Line2D([], [], color="#2c7a3f", lw=2.2, label="UltraNest"),
        Line2D([], [], color="#6a51a3", lw=2.2, ls=(0, (1, 1)), label="TSNPE+MIS"),
        Line2D([], [], color="#08519c", lw=2.4, ls=(0, (6, 2)), label="A-NET+IS"),
    ]
    nucleonic_axis(axis)
    axis.legend(handles=handles, fontsize=11, loc="upper right", frameon=False)
    pdf, png = save(figure, output_dir, "FINAL_A1_methods_with_observations")
    outputs["A1"] = output_record(pdf, png)

    rng = np.random.default_rng(3)
    j0437 = observation["j0437"]
    selected = rng.choice(j0437.shape[1], 8000, replace=False)
    j0437_swap = j0437[:, selected]
    j0614 = np.loadtxt(pulsar_root / "J0614_mrsamples.dat")
    selected = rng.choice(len(j0614), min(8000, len(j0614)), replace=False)
    sample_j0614 = j0614[selected]
    j1231 = np.loadtxt(pulsar_root / "J1231_wmrsamples.txt")
    probability = j1231[:, 0] / j1231[:, 0].sum()
    sample_j1231 = j1231[rng.choice(len(j1231), 8000, p=probability)]

    for case in ("J0614", "J1231"):
        figure, axis = plt.subplots(figsize=(8, 6))
        common_nucleonic(axis, observation)
        if case == "J0614":
            draw_j0030(axis, observation)
            sns.kdeplot(
                ax=axis,
                x=sample_j0614[:, 1],
                y=sample_j0614[:, 0],
                fill=True,
                color="#8c3bab",
                alpha=0.35,
                zorder=2,
            )
        else:
            draw_j0437(axis, observation, sample=j0437_swap)
            sns.kdeplot(
                ax=axis,
                x=sample_j1231[:, 2],
                y=sample_j1231[:, 1],
                fill=True,
                color="#2b8a8a",
                alpha=0.35,
                zorder=2,
            )
        draw_nucleonic_comparison(
            axis,
            mass,
            [
                (data[f"fixed_{case}_UltraNest"], "UltraNest"),
                (data[f"fixed_{case}_ANET"], "ANET"),
            ],
        )
        handles = [
            Line2D([], [], color="#2c7a3f", lw=2.2, label="UltraNest"),
            Line2D([], [], color="#08519c", lw=2.4, ls=(0, (6, 2)), label="A-NET+IS"),
        ]
        nucleonic_axis(axis)
        axis.legend(handles=handles, fontsize=11, loc="upper right", frameon=False)
        pdf, png = save(figure, output_dir, f"FINAL_{case}_methods_with_observations")
        outputs[case] = output_record(pdf, png)
    return outputs


def new_hyperonic_axis():
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.set_xlabel(r"R [km]", fontsize=13)
    axis.set_ylabel(r"M [M$_\odot$]", fontsize=13)
    axis.set_xlim(9, 15)
    axis.set_ylim(0.7, 2.5)
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.tick_params(which="both", direction="in", top=True, right=True, labelsize=12)
    axis.tick_params(which="major", length=6)
    axis.tick_params(which="minor", length=3)
    return figure, axis


def hyperonic_overlays(axis, observation: dict, *, skip=(), alpha=0.4) -> None:
    if "j0030" not in skip:
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
        observation["gw90"][:, 0], observation["gw90"][:, 1],
        color="steelblue", ls="-", lw=2, alpha=alpha, edgecolor="steelblue", zorder=1,
    )
    axis.fill(
        observation["gw50"][:, 0], observation["gw50"][:, 1],
        color="steelblue", ls="dashed", lw=2, alpha=alpha, edgecolor="steelblue", zorder=1,
    )
    axis.fill(
        observation["dittmann0740"][:, 0], observation["dittmann0740"][:, 1],
        color="orangered", lw=2.0, alpha=alpha, linestyle="dotted", zorder=1,
    )
    if "j0437" not in skip:
        draw_j0437(axis, observation, alpha=alpha)


def hyperonic_pulsar_density(
    axis, path: Path, mass_column: int, radius_column: int, weight_column: int | None, color: str
) -> None:
    values = np.loadtxt(path)
    weight = values[:, weight_column] if weight_column is not None else np.ones(len(values))
    weight = weight / weight.sum()
    selected = np.random.default_rng(5).choice(
        len(values), min(8000, len(values)), p=weight
    )
    sns.kdeplot(
        ax=axis,
        x=values[selected, radius_column],
        y=values[selected, mass_column],
        fill=True,
        color=color,
        alpha=0.4,
        zorder=2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the nucleonic mass--radius figures (Figs. 3a and 5).")
    parser.add_argument("--mr-data-root", type=Path, default=ROOT / "data/plotting_overlays")
    parser.add_argument("--pulsar-data-root", type=Path, default=ROOT / "data/observations")
    parser.add_argument(
        "--nucleonic-input",
        type=Path,
        default=ROOT / "results/nucleonic/nucleonic_mass_radius_bands.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/figures")
    arguments = parser.parse_args()

    observation, overlay_hashes = load_mass_radius_overlays(arguments.mr_data_root)
    pulsar_hashes = verify_root(arguments.pulsar_data_root, "substituted_pulsars")
    outputs = {
        "nucleonic": render_nucleonic(
            arguments.nucleonic_input,
            observation,
            arguments.pulsar_data_root,
            arguments.output_dir,
        )
    }
    record = {
        "status": "PASS",
        "credible_level_percent": 90,
        "display_convention": {
            "shared_renderer": "plotting.mass_radius_style.draw_support_aware_band",
            "certified_body_posterior_support": 0.05,
            "forced_high_mass_closure": False,
            "high_mass_tail": "width and opacity taper with posterior support",
        },
        "compact_sources": {arguments.nucleonic_input.name: sha256(arguments.nucleonic_input)},
        "observational_overlay_hashes": overlay_hashes,
        "substituted_pulsar_hashes": pulsar_hashes,
        "outputs": outputs,
    }
    write_record(arguments.output_dir / "mass_radius_figures.json", record)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
