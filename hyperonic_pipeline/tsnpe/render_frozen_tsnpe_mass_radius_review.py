#!/usr/bin/env python3
"""Render certified clean hyperonic TSNPE against locked comparison bands."""

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
from matplotlib.lines import Line2D


PAPER = Path("/home/nucleartheory/ddb-amortized-inference-paper")
sys.path.insert(0, str(PAPER))

from plotting.mass_radius import hyperonic_overlays, new_hyperonic_axis  # noqa: E402
from plotting.mass_radius_style import (  # noqa: E402
    common_certified_body,
    draw_support_aware_band,
)
from plotting.observations import load_mass_radius_overlays  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resampled_band(radius, maximum_mass, counts, mass_grid):
    band = np.full((3, len(mass_grid)), np.nan, dtype=np.float64)
    rows = np.zeros(len(mass_grid), dtype=np.float64)
    for column, mass in enumerate(mass_grid):
        supported = np.isfinite(radius[:, column]) & (maximum_mass >= mass)
        if not supported.any():
            continue
        rows[column] = float(counts[supported].sum())
        values = np.repeat(radius[supported, column], counts[supported])
        band[:, column] = np.percentile(values, [5.0, 50.0, 95.0])
    weight = rows / float(counts.sum())
    return band, weight, rows.copy(), rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--resample", type=Path, required=True)
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--locked-tail", type=Path, required=True)
    parser.add_argument("--mr-data-root", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    arguments = parser.parse_args()
    for suffix in (".pdf", ".png", ".json"):
        if arguments.output_stem.with_suffix(suffix).exists():
            raise FileExistsError(
                f"refusing to replace {arguments.output_stem.with_suffix(suffix)}"
            )

    posterior_report = json.loads(arguments.posterior.with_suffix(".json").read_text())
    posterior_hash = sha256(arguments.posterior)
    if (
        posterior_report.get("status") != "PASS"
        or not posterior_report.get("scientific_posterior_certified")
        or posterior_report.get("lineage_class") != "independent_tsnpe"
        or posterior_report.get("target_gate", {}).get("status") != "PASS"
        or posterior_report.get("forbidden_artifacts_used")
        or posterior_report.get("output_sha256") != posterior_hash
    ):
        raise RuntimeError("TSNPE posterior failed review gates")
    resample_report = json.loads(arguments.resample.with_suffix(".json").read_text())
    if (
        resample_report.get("status") != "PASS"
        or resample_report.get("source_sha256") != posterior_hash
        or resample_report.get("forbidden_artifacts_used")
        or resample_report.get("output_sha256") != sha256(arguments.resample)
    ):
        raise RuntimeError("curve resample failed review gates")

    with np.load(arguments.resample, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64)
        counts = np.asarray(archive["counts"], dtype=np.int64)
        posterior_draws = int(np.asarray(archive["posterior_draws"]).item())
    with np.load(arguments.curves, allow_pickle=False) as archive:
        curve_theta = np.asarray(archive["theta"], dtype=np.float64)
        mass = np.asarray(archive["MG"], dtype=np.float64)
        radius = np.asarray(archive["Rg"], dtype=np.float64)
        maximum_mass = np.asarray(archive["MM"], dtype=np.float64)
        curve_seconds = float(np.asarray(archive["total_seconds"]).item())
        curve_workers = int(np.asarray(archive["workers"]).item())
    if not np.array_equal(theta, curve_theta) or counts.sum() != posterior_draws:
        raise RuntimeError("curve replay does not match the certified resample")
    tsnpe_band, tsnpe_weight, tsnpe_ess, tsnpe_rows = resampled_band(
        radius, maximum_mass, counts, mass
    )

    with np.load(arguments.locked_tail, allow_pickle=False) as archive:
        locked_mass = np.asarray(archive["mass_grid"], dtype=np.float64)
        un_band = np.asarray(archive["UltraNest_band"], dtype=np.float64)
        un_weight = np.asarray(archive["UltraNest_support_weight"], dtype=np.float64)
        un_ess = np.asarray(archive["UltraNest_support_ess"], dtype=np.float64)
        un_rows = np.asarray(archive["UltraNest_support_rows"], dtype=np.float64)
        anet_band = np.asarray(archive["ANET_band"], dtype=np.float64)
        anet_weight = np.asarray(archive["ANET_support_weight"], dtype=np.float64)
        anet_ess = np.asarray(archive["ANET_support_ess"], dtype=np.float64)
        anet_rows = np.asarray(archive["ANET_support_rows"], dtype=np.float64)
    if not np.array_equal(mass, locked_mass):
        raise RuntimeError("TSNPE and locked comparison mass grids differ")
    common = common_certified_body(
        [un_band, tsnpe_band, anet_band],
        [un_weight, tsnpe_weight, anet_weight],
        [un_ess, tsnpe_ess, anet_ess],
    )

    observation, overlay_hashes = load_mass_radius_overlays(arguments.mr_data_root)
    figure, axis = new_hyperonic_axis()
    hyperonic_overlays(axis, observation)
    payload = [
        (un_band, un_weight, un_ess, un_rows, "forestgreen", "-", "UltraNest", 4),
        (tsnpe_band, tsnpe_weight, tsnpe_ess, tsnpe_rows, "crimson", "-.", "TSNPE+MIS", 5),
        (anet_band, anet_weight, anet_ess, anet_rows, "royalblue", "--", "A-NET+IS", 6),
    ]
    for band, weight, ess, rows, color, linestyle, label, zorder in payload:
        draw_support_aware_band(
            axis,
            mass,
            band,
            weight,
            ess,
            rows,
            common,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
            shaded=True,
            zorder=zorder,
        )
    axis.legend(
        handles=[
            Line2D([], [], color="forestgreen", lw=1.8, label="UltraNest"),
            Line2D([], [], color="crimson", lw=1.8, ls="-.", label="TSNPE+MIS"),
            Line2D([], [], color="royalblue", lw=1.8, ls="--", label="A-NET+IS"),
        ],
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

    differences = {}
    for label, band in (("TSNPE+MIS", tsnpe_band), ("A-NET+IS", anet_band)):
        edge = np.abs(
            np.concatenate(
                ([un_band[0, common] - band[0, common], un_band[2, common] - band[2, common]])
            )
        )
        differences[label] = {
            "median": float(np.median(edge)),
            "p95": float(np.percentile(edge, 95.0)),
            "maximum": float(np.max(edge)),
        }
    data = arguments.output_stem.with_name(arguments.output_stem.name + "_data.npz")
    np.savez_compressed(
        data,
        mass_grid=mass,
        ultranest_band=un_band,
        ultranest_support_weight=un_weight,
        tsnpe_band=tsnpe_band,
        tsnpe_support_weight=tsnpe_weight,
        tsnpe_support_ess=tsnpe_ess,
        tsnpe_support_rows=tsnpe_rows,
        anet_band=anet_band,
        anet_support_weight=anet_weight,
        common_body_mask=common,
    )
    receipt = {
        "status": "CLEAN_GATE_PASSED_REVIEW",
        "method": "clean full-prior 9D TSNPE with exact deterministic MIS",
        "source_posterior_ess": posterior_report["ess"],
        "maximum_normalized_weight": posterior_report["maximum_normalized_weight"],
        "log_evidence": posterior_report["log_evidence"],
        "log_evidence_standard_error": posterior_report[
            "log_evidence_standard_error"
        ],
        "posterior_resample_draws": posterior_draws,
        "posterior_resample_unique_curves": int(len(theta)),
        "curve_replay": {"workers": curve_workers, "seconds": curve_seconds},
        "common_certified_mass_range_msun": [
            float(mass[common][0]),
            float(mass[common][-1]),
        ],
        "absolute_90pct_edge_difference_from_UltraNest_km": differences,
        "inputs_sha256": {
            str(arguments.posterior): posterior_hash,
            str(arguments.resample): sha256(arguments.resample),
            str(arguments.curves): sha256(arguments.curves),
            str(arguments.locked_tail): sha256(arguments.locked_tail),
        },
        "observational_overlay_sha256": overlay_hashes,
        "outputs_sha256": {
            str(pdf): sha256(pdf),
            str(png): sha256(png),
            str(data): sha256(data),
        },
        "forbidden_artifacts_used": [],
        "manuscript_modified": False,
    }
    arguments.output_stem.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
