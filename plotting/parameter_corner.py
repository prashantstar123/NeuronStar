#!/usr/bin/env python3
"""Render smooth 68%/90% marginal projections of the corrected A1 posterior."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import label
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from plotting.common import (
    ROOT,
    effective_sample_size,
    normalize_weights,
    output_record,
    require,
    save_figure,
    sha256,
    weighted_quantile,
    write_record,
)


PARAMETER_KEYS = (
    "a_sigma",
    "a_omega",
    "a_rho",
    "Gamma_sigma0",
    "Gamma_omega0",
    "Gamma_rho0",
    "rho0",
)
PARAMETER_LABELS = (
    r"$a_{\sigma}$",
    r"$a_{\omega}$",
    r"$a_{\rho}$",
    r"$\Gamma_{\sigma,0}$",
    r"$\Gamma_{\omega,0}$",
    r"$\Gamma_{\rho,0}$",
    r"$\rho_0\,[\mathrm{fm}^{-3}]$",
)
PRIOR_LOW = np.array([0.0, 0.0, 0.0, 6.5, 7.5, 5.0, 0.14])
PRIOR_HIGH = np.array([0.3, 0.298446, 1.3, 13.4895, 14.5, 12.5471, 0.17])
METHOD_STYLE = {
    "UltraNest": {"color": "#17365D", "linestyle": ":", "linewidth": 1.7},
    "TSNPE+MIS": {"color": "#D55E00", "linestyle": "--", "linewidth": 1.8},
    "A-NET+IS": {"color": "#007C91", "linestyle": "-", "linewidth": 2.0},
}
DISPLAY_TAIL = 0.0005
DISPLAY_PADDING = 0.035


def load_posterior(path: Path) -> tuple[dict, dict]:
    require(path.is_file(), f"missing compact posterior: {path}")
    saved = np.load(path, allow_pickle=False)
    require(tuple(saved["parameter_names"].tolist()) == PARAMETER_KEYS, "parameter order changed")
    posterior = {}
    for name, key in (
        ("UltraNest", "ultranest"),
        ("TSNPE+MIS", "tsnpe_mis"),
        ("A-NET+IS", "anet_is"),
    ):
        theta = np.asarray(saved[f"theta_{key}"], dtype=np.float64)
        weight = normalize_weights(saved[f"weight_{key}"])
        require(theta.ndim == 2 and theta.shape[1] == 7, f"{name}: bad theta")
        require(len(theta) == len(weight), f"{name}: row mismatch")
        require(np.isfinite(theta).all(), f"{name}: non-finite theta")
        require(np.all(theta >= PRIOR_LOW), f"{name}: theta below prior")
        require(np.all(theta <= PRIOR_HIGH), f"{name}: theta above prior")
        posterior[name] = {"theta": theta, "weight": weight}
    provenance = json.loads(str(saved["source_hashes"].item()))
    return posterior, provenance


def systematic_coreset(
    theta: np.ndarray, weight: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    if size <= 0 or size >= len(theta):
        return theta, weight
    cumulative = np.cumsum(weight)
    positions = (np.arange(size, dtype=np.float64) + 0.5) / size
    index = np.searchsorted(cumulative, positions, side="left")
    require(index.min() >= 0 and index.max() < len(theta), "bad coreset index")
    return theta[index], np.full(size, 1.0 / size, dtype=np.float64)


def make_kde(values: np.ndarray, weight: np.ndarray, multiplier: float):
    density = gaussian_kde(values, weights=weight)
    density.set_bandwidth(density.factor * multiplier)
    return density


def probability_thresholds(
    density: np.ndarray, probabilities: tuple[float, ...]
) -> dict[float, float]:
    ordered = np.sort(np.asarray(density, dtype=np.float64).ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    thresholds = {}
    for probability in probabilities:
        index = min(int(np.searchsorted(cumulative, probability)), len(ordered) - 1)
        thresholds[probability] = float(ordered[index])
    return thresholds


def connected_component_count(mask: np.ndarray) -> int:
    _components, count = label(mask, structure=np.ones((3, 3), dtype=int))
    return int(count)


def display_ranges(posterior: dict) -> list[tuple[float, float]]:
    ranges = []
    for column in range(7):
        low = min(
            weighted_quantile(
                values["theta"][:, column], values["weight"], (DISPLAY_TAIL,)
            )[0]
            for values in posterior.values()
        )
        high = max(
            weighted_quantile(
                values["theta"][:, column],
                values["weight"],
                (1.0 - DISPLAY_TAIL,),
            )[0]
            for values in posterior.values()
        )
        pad = DISPLAY_PADDING * (high - low)
        ranges.append(
            (
                max(float(PRIOR_LOW[column]), float(low - pad)),
                min(float(PRIOR_HIGH[column]), float(high + pad)),
            )
        )
    return ranges


def interval_summary(posterior: dict) -> dict:
    record = {}
    for method, values in posterior.items():
        record[method] = {}
        for column, key in enumerate(PARAMETER_KEYS):
            q05, median, q95 = weighted_quantile(
                values["theta"][:, column], values["weight"], (0.05, 0.50, 0.95)
            )
            record[method][key] = {
                "q05": float(q05),
                "median": float(median),
                "q95": float(q95),
                "minus": float(median - q05),
                "plus": float(q95 - median),
            }
    return record


def render(
    posterior: dict,
    ranges: list[tuple[float, float]],
    intervals: dict,
    output_stem: Path,
    *,
    bandwidth_multiplier: float,
    probabilities: tuple[float, ...],
    grid_1d: int,
    grid_2d: int,
    tsnpe_coreset_size: int,
) -> tuple[dict, tuple[Path, Path]]:
    kde_posterior = dict(posterior)
    theta, weight = systematic_coreset(
        posterior["TSNPE+MIS"]["theta"],
        posterior["TSNPE+MIS"]["weight"],
        tsnpe_coreset_size,
    )
    kde_posterior["TSNPE+MIS"] = {"theta": theta, "weight": weight}

    figure, axes = plt.subplots(7, 7, figsize=(16.8, 16.8), squeeze=False)
    teal = to_rgba(METHOD_STYLE["A-NET+IS"]["color"])[:3]
    gates = {
        method: {"one_dimensional_peaks": {}, "two_dimensional_components": {}}
        for method in posterior
    }
    density_cache = {}

    for row in range(7):
        for column in range(7):
            axis = axes[row, column]
            if column > row:
                axis.set_visible(False)
                continue
            axis.set_xlim(ranges[column])
            axis.xaxis.set_major_locator(MaxNLocator(3, prune=None))
            axis.yaxis.set_major_locator(MaxNLocator(3, prune=None))
            axis.tick_params(
                which="both",
                direction="in",
                top=True,
                right=True,
                labelsize=11.5,
                length=4.2,
                width=0.85,
            )
            for spine in axis.spines.values():
                spine.set_linewidth(0.75)

            if row == column:
                maximum = 0.0
                grid = np.linspace(*ranges[column], grid_1d)
                for method, values in kde_posterior.items():
                    density = make_kde(
                        values["theta"][:, column],
                        values["weight"],
                        bandwidth_multiplier,
                    )(grid)
                    require(np.isfinite(density).all(), f"{method}: bad 1D KDE")
                    peaks, _ = find_peaks(density, prominence=0.04 * density.max())
                    require(
                        len(peaks) == 1,
                        f"{method} {PARAMETER_KEYS[column]}: {len(peaks)} KDE peaks",
                    )
                    gates[method]["one_dimensional_peaks"][PARAMETER_KEYS[column]] = 1
                    density_cache[(method, column)] = density
                    maximum = max(maximum, float(density.max()))

                for method in ("A-NET+IS", "TSNPE+MIS", "UltraNest"):
                    density = density_cache[(method, column)]
                    style = METHOD_STYLE[method]
                    if method == "A-NET+IS":
                        axis.fill_between(
                            grid, 0, density, color=style["color"], alpha=0.09, linewidth=0
                        )
                    axis.plot(
                        grid,
                        density,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"] * 1.15,
                    )
                summary = intervals["A-NET+IS"][PARAMETER_KEYS[column]]
                style = METHOD_STYLE["A-NET+IS"]
                axis.axvline(summary["q05"], color=style["color"], ls="--", lw=1.2)
                axis.axvline(summary["median"], color=style["color"], ls="-", lw=1.3)
                axis.axvline(summary["q95"], color=style["color"], ls="--", lw=1.2)
                precision = 4 if column == 6 else 3
                axis.set_title(
                    rf"${summary['median']:.{precision}f}"
                    rf"_{{-{summary['minus']:.{precision}f}}}"
                    rf"^{{+{summary['plus']:.{precision}f}}}$",
                    fontsize=17,
                    color=style["color"],
                    pad=5,
                )
                axis.set_ylim(0.0, 1.10 * maximum)
                axis.set_yticks([])
            else:
                grid_x = np.linspace(*ranges[column], grid_2d)
                grid_y = np.linspace(*ranges[row], grid_2d)
                mesh_x, mesh_y = np.meshgrid(grid_x, grid_y, indexing="xy")
                points = np.vstack((mesh_x.ravel(), mesh_y.ravel()))
                panel_density = {}
                panel_levels = {}
                for method, values in kde_posterior.items():
                    density = make_kde(
                        np.vstack(
                            (values["theta"][:, column], values["theta"][:, row])
                        ),
                        values["weight"],
                        bandwidth_multiplier,
                    )(points).reshape(mesh_x.shape)
                    require(np.isfinite(density).all(), f"{method}: bad 2D KDE")
                    thresholds = probability_thresholds(density, probabilities)
                    levels = sorted(set(thresholds.values()))
                    require(len(levels) == len(probabilities), "degenerate KDE levels")
                    panel_density[method] = density
                    panel_levels[method] = levels
                    counts = {
                        f"{probability:.2f}": connected_component_count(
                            density >= thresholds[probability]
                        )
                        for probability in probabilities
                    }
                    require(
                        all(count == 1 for count in counts.values()),
                        f"{method} {PARAMETER_KEYS[row]}--{PARAMETER_KEYS[column]} "
                        f"has detached KDE components: {counts}",
                    )
                    gates[method]["two_dimensional_components"][
                        f"{PARAMETER_KEYS[row]}__{PARAMETER_KEYS[column]}"
                    ] = counts

                anet_density = panel_density["A-NET+IS"]
                anet_levels = panel_levels["A-NET+IS"]
                fill_alphas = np.linspace(0.12, 0.25, len(anet_levels))
                axis.contourf(
                    mesh_x,
                    mesh_y,
                    anet_density,
                    levels=[*anet_levels, float(anet_density.max()) * 1.001],
                    colors=[(*teal, float(alpha)) for alpha in fill_alphas],
                    antialiased=True,
                )
                for method in ("A-NET+IS", "TSNPE+MIS", "UltraNest"):
                    style = METHOD_STYLE[method]
                    axis.contour(
                        mesh_x,
                        mesh_y,
                        panel_density[method],
                        levels=panel_levels[method],
                        colors=[style["color"]],
                        linestyles=[style["linestyle"]],
                        linewidths=style["linewidth"] * 1.15,
                    )
                axis.set_ylim(ranges[row])

            if row < 6:
                axis.set_xticklabels([])
            else:
                axis.set_xlabel(PARAMETER_LABELS[column], fontsize=24, labelpad=14)
                for tick in axis.get_xticklabels():
                    tick.set_rotation(30)
            if column > 0:
                axis.set_yticklabels([])
            elif row > 0:
                axis.set_ylabel(PARAMETER_LABELS[row], fontsize=24, labelpad=14)

    handles = [
        Line2D(
            [],
            [],
            color=METHOD_STYLE[method]["color"],
            ls=METHOD_STYLE[method]["linestyle"],
            lw=2.4,
            label=method,
        )
        for method in ("UltraNest", "TSNPE+MIS", "A-NET+IS")
    ]
    figure.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.94, 0.94),
        frameon=True,
        fontsize=21,
        handlelength=3.2,
        borderpad=0.7,
        labelspacing=0.5,
    )
    figure.subplots_adjust(
        left=0.09, bottom=0.09, right=0.98, top=0.98, wspace=0.055, hspace=0.055
    )
    outputs = save_figure(figure, output_stem)
    plt.close(figure)
    return gates, outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results/nucleonic/a1_parameter_posteriors.npz",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "build/figures/FINAL_A1_parameter_corner_methods",
    )
    parser.add_argument("--bandwidth-multiplier", type=float, default=2.3)
    parser.add_argument("--probabilities", nargs="+", type=float, default=(0.68, 0.90))
    parser.add_argument("--grid-1d", type=int, default=400)
    parser.add_argument("--grid-2d", type=int, default=105)
    parser.add_argument("--tsnpe-coreset-size", type=int, default=24_000)
    arguments = parser.parse_args()

    probabilities = tuple(arguments.probabilities)
    require(1.0 <= arguments.bandwidth_multiplier <= 2.5, "unsafe KDE bandwidth")
    require(
        probabilities == tuple(sorted(set(probabilities)))
        and all(0.0 < value < 1.0 for value in probabilities),
        "probabilities must be unique, increasing, and strictly between zero and one",
    )
    require(len(probabilities) == 2, "this figure is gated for exactly two contours")
    require(arguments.grid_1d >= 80 and arguments.grid_2d >= 35, "KDE grid too small")

    posterior, provenance = load_posterior(arguments.input)
    ranges = display_ranges(posterior)
    intervals = interval_summary(posterior)
    gates, (pdf, png) = render(
        posterior,
        ranges,
        intervals,
        arguments.output_stem,
        bandwidth_multiplier=arguments.bandwidth_multiplier,
        probabilities=probabilities,
        grid_1d=arguments.grid_1d,
        grid_2d=arguments.grid_2d,
        tsnpe_coreset_size=arguments.tsnpe_coreset_size,
    )
    record = {
        "status": "PASS",
        "figure": "one- and two-dimensional marginal projections of the A1 posterior",
        "source": arguments.input.name,
        "source_sha256": sha256(arguments.input),
        "source_artifact_hashes": provenance,
        "methods": list(posterior),
        "native_rows": {name: len(values["theta"]) for name, values in posterior.items()},
        "effective_sample_size": {
            name: effective_sample_size(values["weight"])
            for name, values in posterior.items()
        },
        "estimator": "continuous weighted Gaussian KDE",
        "bandwidth_multiplier": arguments.bandwidth_multiplier,
        "probability_levels": list(probabilities),
        "grid_1d": arguments.grid_1d,
        "grid_2d": arguments.grid_2d,
        "tsnpe_deterministic_systematic_kde_coreset": arguments.tsnpe_coreset_size,
        "central_90_percent_intervals": intervals,
        "display_ranges": {
            key: list(value) for key, value in zip(PARAMETER_KEYS, ranges)
        },
        "visual_gates": gates,
        "outputs": output_record(pdf, png),
    }
    write_record(arguments.output_stem.with_suffix(".json"), record)
    print(json.dumps(record["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
