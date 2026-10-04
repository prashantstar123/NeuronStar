#!/usr/bin/env python3
"""Figure 11: marginal P-P calibration of the frozen A-NET proposals on held-out simulated data.

(a) nucleonic A-NET (1,000 held-out cases), (b) hyperonic A-NET, nine-parameter network (500 cases).
Input: results/appendix/pp_calibration_ranks.npz, the per-parameter ranks u = P_q(theta_j < theta*_j)
of the true parameters among the network draws (nucleonic: proposal restricted to the prior box, as in
the importance-sampling step). Grey bands: 68/95/99.7% binomial ranges for a perfectly calibrated
posterior (bilby make_pp_plot convention). Legend values: Kolmogorov-Smirnov p-values.
The drawing code is the code that produced the published figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binom, kstest  # noqa: E402

from plotting.common import ROOT  # noqa: E402

NUC_LAB = [r"$a_\sigma$", r"$a_\omega$", r"$a_\rho$", r"$\Gamma_{\sigma,0}$", r"$\Gamma_{\omega,0}$",
           r"$\Gamma_{\rho,0}$", r"$\rho_0$"]
HYP_LAB = NUC_LAB + [r"$x_{\sigma\Lambda}$", r"$x_{\sigma\Xi}$"]
COLORS = plt.get_cmap("tab10").colors
GRID = np.linspace(0, 1, 1001)


def band(ax, n):
    for ci in (0.68, 0.95, 0.997):
        edge = (1.0 - ci) / 2.0
        hi = binom.ppf(1 - edge, n, GRID) / n
        lo = binom.ppf(edge, n, GRID) / n
        lo[0] = hi[0] = 0.0
        ax.fill_between(GRID, lo, hi, color="k", alpha=0.1, lw=0, zorder=1)
    ax.plot([0, 1], [0, 1], color="0.35", lw=0.9, ls="--", zorder=2)


def panel(ax, ranks, labels, title, ylabel):
    n = len(ranks)
    band(ax, n)
    pvals = []
    for j, lab in enumerate(labels):
        u = ranks[:, j]
        ecdf = np.array([np.mean(u < x) for x in GRID])
        p = float(kstest(u, "uniform").pvalue)
        pvals.append(p)
        ax.plot(GRID, ecdf, color=COLORS[j % 10], lw=1.4, label=f"{lab} ({p:.2f})", zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.tick_params(direction="in", top=True, right=True, labelsize=9.5)
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("Credible interval", fontsize=11)
    if ylabel:
        ax.set_ylabel("Fraction of held-out cases in interval", fontsize=11)
    ax.legend(fontsize=8.6, ncol=2 if len(labels) > 7 else 1, frameon=False, loc="upper left",
              title=f"$N={n}$ (KS $p$)", title_fontsize=8.6, handlelength=1.2, columnspacing=0.6,
              labelspacing=0.3, borderaxespad=0.3)
    return pvals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/appendix/pp_calibration_ranks.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/figures")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    stem = arguments.output_dir / "FINAL_anet_calibration_pp_heldout_v2_20260924"
    with np.load(arguments.input) as ranks:
        nucleonic, hyperonic = ranks["nucleonic_ranks"], ranks["hyperonic_ranks"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.95))
    p_a = panel(axes[0], nucleonic, NUC_LAB, "(a) Nucleonic A-NET", True)
    p_b = panel(axes[1], hyperonic, HYP_LAB, "(b) Hyperonic A-NET", False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    (arguments.output_dir / "figure11_ks_p_values.json").write_text(
        json.dumps({"nucleonic": p_a, "hyperonic": p_b}, indent=1) + "\n")
    print("[figure 10] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
