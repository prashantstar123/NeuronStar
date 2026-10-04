#!/usr/bin/env python3
"""Render the corrected A1 mass--tidal 90% credible bands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import (
    AutoMinorLocator,
    FormatStrFormatter,
    LogLocator,
    MultipleLocator,
    NullFormatter,
)

from plotting.common import ROOT, output_record, require, save_figure, sha256, write_record


METHODS = {
    "UltraNest": {
        "key": "ultranest_pooled",
        "color": "#4C78A8",
        "linestyle": (0, (1.5, 2.2)),
        "alpha": 0.20,
        "zorder": 2,
    },
    "TSNPE+MIS": {
        "key": "tsnpe_mis",
        "color": "#F58518",
        "linestyle": (0, (6.0, 2.2)),
        "alpha": 0.18,
        "zorder": 3,
    },
    "A-NET+IS": {
        "key": "anet_is_direct_gmm_320k",
        "color": "#118A9B",
        "linestyle": "-",
        "alpha": 0.22,
        "zorder": 4,
    },
}


def render(
    source_path: Path,
    output_stem: Path,
    minimum_conditional_ess: float,
    full_opacity_support: float,
) -> dict:
    require(source_path.is_file(), f"missing compact product: {source_path}")
    source = np.load(source_path, allow_pickle=False)
    mass = np.asarray(source["mass_grid"], dtype=np.float64)
    require(mass.ndim == 1 and np.all(np.diff(mass) > 0), "invalid mass grid")

    rendered: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    endpoints: dict[str, dict] = {}
    for name, specification in METHODS.items():
        key = specification["key"]
        quantile = np.asarray(source[f"band_{key}"], dtype=np.float64)
        support = np.asarray(source[f"support_{key}"], dtype=np.float64)
        conditional_ess = np.asarray(
            source[f"conditional_ess_{key}"], dtype=np.float64
        )
        require(quantile.shape == (3, len(mass)), f"{name}: bad band shape")
        require(support.shape == mass.shape, f"{name}: bad support shape")
        require(conditional_ess.shape == mass.shape, f"{name}: bad ESS shape")
        mask = np.all(np.isfinite(quantile), axis=0) & (
            conditional_ess >= minimum_conditional_ess
        )
        index = np.flatnonzero(mask)
        require(len(index) > 0, f"{name}: no reliable mass--tidal interval")
        require(
            np.array_equal(index, np.arange(index[0], index[-1] + 1)),
            f"{name}: reliable interval is not contiguous",
        )
        last = int(index[-1])
        rendered[name] = quantile, support, mask
        endpoints[name] = {
            "last_reliable_mass_msun": float(mass[last]),
            "lambda_q05_q50_q95": quantile[:, last].tolist(),
            "posterior_support": float(support[last]),
            "conditional_ess": float(conditional_ess[last]),
        }

    figure, axis = plt.subplots(figsize=(8.2, 6.2))
    for name, specification in METHODS.items():
        quantile, support, mask = rendered[name]
        x = mass[mask]
        low, _median, high = quantile[:, mask]
        local_support = support[mask]
        terminal_support = float(local_support[-1])
        require(terminal_support > 0.0, f"{name}: non-positive endpoint support")
        if terminal_support < full_opacity_support:
            opacity = np.clip(
                (np.log(local_support) - np.log(terminal_support))
                / (np.log(full_opacity_support) - np.log(terminal_support)),
                0.0,
                1.0,
            )
            opacity[local_support >= full_opacity_support] = 1.0
            opacity[-1] = 0.0
        else:
            opacity = np.ones_like(local_support)

        for index in range(len(x) - 1):
            local_opacity = float(0.5 * (opacity[index] + opacity[index + 1]))
            if local_opacity <= 0.0:
                continue
            segment = slice(index, index + 2)
            axis.fill_between(
                x[segment],
                low[segment],
                high[segment],
                color=specification["color"],
                linewidth=0,
                alpha=specification["alpha"] * local_opacity,
                zorder=specification["zorder"],
            )
            for edge in (low, high):
                axis.plot(
                    x[segment],
                    edge[segment],
                    color=specification["color"],
                    linestyle=specification["linestyle"],
                    linewidth=1.65,
                    alpha=0.90 * local_opacity,
                    zorder=specification["zorder"] + 4,
                )

    axis.set_xlim(0.5, 2.5)
    axis.set_ylim(4.5, 1.4e5)
    axis.set_yscale("log")
    axis.set_xlabel(r"$M\,[M_{\odot}]$", fontsize=17)
    axis.set_ylabel(r"$\Lambda$", fontsize=18)
    axis.xaxis.set_major_locator(MultipleLocator(0.25))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.xaxis.set_minor_locator(AutoMinorLocator(5))
    axis.yaxis.set_major_locator(LogLocator(base=10.0))
    axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(which="major", color="#B8B8B8", alpha=0.42, linewidth=0.65)
    axis.grid(which="minor", color="#D0D0D0", alpha=0.30, linewidth=0.45)
    axis.tick_params(
        which="both", direction="in", top=True, right=True, labelsize=12.5
    )
    axis.tick_params(which="major", length=6.0, width=0.9)
    axis.tick_params(which="minor", length=3.0, width=0.7)
    handles = [
        Patch(
            facecolor=specification["color"],
            edgecolor=specification["color"],
            linestyle=specification["linestyle"],
            linewidth=1.65,
            alpha=0.28,
            label=f"{name} (90% CI)",
        )
        for name, specification in METHODS.items()
    ]
    axis.legend(
        handles=handles,
        loc="lower left",
        fontsize=11.5,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#CCCCCC",
    )
    figure.tight_layout()
    pdf, png = save_figure(figure, output_stem, dpi=220)
    plt.close(figure)

    record = {
        "status": "PASS",
        "figure": "A1 mass--tidal 90% credible bands",
        "source": source_path.name,
        "source_sha256": sha256(source_path),
        "minimum_conditional_ess": minimum_conditional_ess,
        "full_opacity_support": full_opacity_support,
        "posterior_arrays_changed": False,
        "method_endpoints": endpoints,
        "outputs": output_record(pdf, png),
    }
    write_record(output_stem.with_suffix(".json"), record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results/nucleonic/a1_mass_tidal_bands.npz",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "build/figures/FINAL_A1_mass_tidal_three_methods",
    )
    parser.add_argument("--minimum-conditional-ess", type=float, default=50.0)
    parser.add_argument("--full-opacity-support", type=float, default=0.02)
    arguments = parser.parse_args()
    record = render(
        arguments.input,
        arguments.output_stem,
        arguments.minimum_conditional_ess,
        arguments.full_opacity_support,
    )
    print(json.dumps(record["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
