#!/usr/bin/env python3
"""Render the four nuclear-observation amortization comparisons (Figures 4 and 10)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from plotting.common import ROOT, output_record, require, save_figure, sha256, write_record
from plotting.mass_radius_style import (
    common_certified_body,
    draw_support_aware_band,
    support_from_curve_counts,
)
from plotting.observations import draw_nuclear_panel, load_mass_radius_overlays


CASES = ("K0_200", "K0_260", "Jsym_29", "Jsym_36")
TITLES = {
    "K0_200": r"$K_0=200\ \mathrm{MeV}$",
    "K0_260": r"$K_0=260\ \mathrm{MeV}$",
    "Jsym_29": r"$J_{\mathrm{sym}}=29\ \mathrm{MeV}$",
    "Jsym_36": r"$J_{\mathrm{sym}}=36\ \mathrm{MeV}$",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results/nucleonic/nuclear_amortization_bands.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/nucleonic/nuclear_amortization_verification.json",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "build/figures/FINAL_nuclear_amortization_four_scenarios",
    )
    parser.add_argument("--mr-data-root", type=Path, required=True)
    arguments = parser.parse_args()

    source = np.load(arguments.input, allow_pickle=False)
    report = json.loads(arguments.report.read_text())
    observation, overlay_hashes = load_mass_radius_overlays(arguments.mr_data_root)
    mass = np.asarray(source["mass_grid"], dtype=np.float64)
    minimum = int(report["display_minimum_curve_count"])
    require(np.array_equal(mass, np.asarray(report["mass_grid_msun"])), "grid mismatch")

    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.4), sharex=True, sharey=True)
    for axis, case in zip(axes.ravel(), CASES):
        draw_nuclear_panel(axis, observation)
        specifications = (
            (f"{case}_ultranest", "#2c7a3f", "-", 2.0),
            (f"{case}_anet", "#08519c", (0, (6, 2)), 2.2),
        )
        bands = []
        supports = []
        for key, _, _, _ in specifications:
            band = np.asarray(source[key], dtype=np.float64)
            require(band.shape == (4, len(mass)), f"bad band: {key}")
            bands.append(band)
            supports.append(support_from_curve_counts(band))
        common = common_certified_body(
            [band[:3] for band in bands],
            [item[0] for item in supports],
            [item[1] for item in supports],
        )
        for (key, color, linestyle, width), band, support in zip(
            specifications, bands, supports, strict=True
        ):
            weight, ess, rows = support
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
                label="UltraNest" if key.endswith("ultranest") else "A-NET+IS",
                zorder=6,
            )
        axis.set_title(TITLES[case], fontsize=13)
        axis.set_xlim(9, 15)
        axis.set_ylim(0.7, 2.5)
        axis.grid(alpha=0.25, lw=0.6)
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
        axis.tick_params(
            which="both", direction="in", top=True, right=True, labelsize=11
        )
        axis.tick_params(which="major", length=6)
        axis.tick_params(which="minor", length=3)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$M$ [$M_\odot$]", fontsize=13)
    for axis in axes[1, :]:
        axis.set_xlabel(r"$R$ [km]", fontsize=13)

    handles = [
        Line2D([], [], color="#2c7a3f", lw=2.0, label="UltraNest"),
        Line2D(
            [],
            [],
            color="#08519c",
            lw=2.2,
            ls=(0, (6, 2)),
            label="A-NET+IS",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        fontsize=12,
        frameon=False,
        handlelength=3.0,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.9, w_pad=0.7)
    pdf, png = save_figure(figure, arguments.output_stem)
    plt.close(figure)
    record = {
        "status": "PASS",
        "figure": "four shifted nuclear-observation mass--radius comparisons",
        "credible_level_percent": 90,
        "display_convention": {
            "shared_renderer": "plotting.mass_radius_style.draw_support_aware_band",
            "certified_body_posterior_support": 0.05,
            "forced_high_mass_closure": False,
            "high_mass_tail": "width and opacity taper with posterior support",
        },
        "source": arguments.input.name,
        "source_sha256": sha256(arguments.input),
        "verification_report": arguments.report.name,
        "verification_report_sha256": sha256(arguments.report),
        "cases": list(CASES),
        "observational_overlay_hashes": overlay_hashes,
        "outputs": output_record(pdf, png),
    }
    write_record(arguments.output_stem.with_suffix(".json"), record)
    print(json.dumps(record["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
