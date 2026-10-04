#!/usr/bin/env python3
"""Figure 7: hyperonic DDB-Lambda-Xi-minus A1, (a) mass--radius and (b) mass--tidal bands.

Panel (a) recomputes the TSNPE+MIS and A-NET+IS bands from the saved 6,000-draw posterior
selections and their stellar curves, and requires them to equal the published bands
(A1/anet_comparison_bands.npz, A1/tsnpe_comparison_bands.npz) bit for bit. The UltraNest band is
the saved published band. Panel (b) draws the saved mass--tidal bands.
The drawing code is the code that produced the published figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import plotting.mass_tidal as mass_tidal  # noqa: E402
from plotting.common import ROOT, save_figure  # noqa: E402
from plotting.mass_radius import hyperonic_overlays, new_hyperonic_axis  # noqa: E402
from plotting.mass_radius_style import common_certified_body, draw_support_aware_band  # noqa: E402
from plotting.observations import load_mass_radius_overlays  # noqa: E402

R = ROOT / "results/hyperonic"
INPUTS = {
    "ultranest_mr_band": (R / "hyperonic_mass_radius_tail_support.npz",
                          "11b6778fe3d15df6059c408692d6482a6cad12533b36d7030b77e6f508bf0cb0"),
    "anet_comparison_bands": (R / "A1/anet_comparison_bands.npz",
                              "0cfbd7ab0119a955e234587c905a3bc9cbff82c5e366122e3935fdb736ccee4e"),
    "anet_curves": (R / "A1/anet_curves.npz",
                    "fb1acb105dc9a2ebd6806ee703c111ad34d7f640735bfb4f1e161c5aaa029b14"),
    "anet_resample": (R / "A1/anet_resample6000.npz",
                      "63af9c88c1b004f6aaba3515e37faa7eb7a32ac80fdadaf8d3fb513c4deaaa87"),
    "anet_certificate": (R / "A1/anet_certificate.npz",
                         "df04ef1fd5c8cc293a33ea44039072bda61aa976c136348fb94d82307924f1f8"),
    "tsnpe_comparison_bands": (R / "A1/tsnpe_comparison_bands.npz",
                               "70efa82c12b39e22a4d23427b3c477a314a4ce4df0ec91090b4dd75f04a601a1"),
    "tsnpe_curves": (R / "A1/tsnpe_curves.npz",
                     "9089d0cd6fa029bdf6effb711abe9375c7708a9e37009e6ecbad93a7934e1399"),
    "tsnpe_resample": (R / "A1/tsnpe_resample6000.npz",
                       "d3afca0e31fdc88277b90b16ad405219bd0da4e9f45bdece06741242de479a63"),
    "mass_tidal_bands": (R / "A1/mass_tidal_bands.npz",
                         "68eb29ecf309e3adc4899954f6cd647316320dab5b04283d6aee85cef80372db"),
}
MINIMUM_CONDITIONAL_ESS = 50.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str) -> dict[str, np.ndarray]:
    path, _ = INPUTS[name]
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def resampled_band(radius, maximum_mass, counts, mass_grid):
    """M-R band convention: percentiles of the repeated 6,000 draws."""

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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/figures")
    parser.add_argument("--mr-data-root", type=Path, default=ROOT / "data/plotting_overlays")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    panel_a = arguments.output_dir / "FINAL_hyp_A1_methods_with_observations"
    panel_b = arguments.output_dir / "FINAL_hyp_A1_mass_tidal_three_methods"

    input_hashes = {}
    for name, (path, expected) in INPUTS.items():
        actual = sha256(path)
        require(expected is None or actual == expected, f"{name}: hash {actual} != recorded {expected}")
        input_hashes[name] = {"path": str(path.relative_to(ROOT)), "sha256": actual}
    checks: dict[str, object] = {"all_input_hashes_match_recorded_values": True}

    # ---- UltraNest M-R band: one saved array, identical in both comparison files.
    tail = load("ultranest_mr_band")
    anet_published = load("anet_comparison_bands")
    tsnpe_published = load("tsnpe_comparison_bands")
    mass = np.asarray(tail["mass_grid"], dtype=np.float64)
    un_band = np.asarray(tail["UltraNest_band"], dtype=np.float64)
    un_weight = np.asarray(tail["UltraNest_support_weight"], dtype=np.float64)
    un_ess = np.asarray(tail["UltraNest_support_ess"], dtype=np.float64)
    un_rows = np.asarray(tail["UltraNest_support_rows"], dtype=np.float64)
    for label, published in (("A-NET comparison bands", anet_published), ("TSNPE comparison bands", tsnpe_published)):
        require(np.array_equal(published["mass_grid"], mass), f"{label}: mass grid differs")
        require(
            np.array_equal(published["ultranest_band"], un_band, equal_nan=True),
            f"{label}: UltraNest band differs from the published UltraNest band",
        )
    checks["ultranest_mr_band_identical_in_both_comparison_files"] = True

    # ---- A-NET: dual-head certificate -> resample -> curves -> band.
    with np.load(INPUTS["anet_certificate"][0], allow_pickle=False) as certificate:
        require(bool(certificate["gate_pass"]), "A-NET certificate did not pass")
    anet_resample = load("anet_resample")
    anet_curves = load("anet_curves")
    require(
        str(anet_resample["source_sha256"]) == INPUTS["anet_certificate"][1],
        "A-NET resample does not reference the supplied certificate",
    )
    anet_counts = np.asarray(anet_resample["counts"], dtype=np.int64)
    require(int(anet_counts.sum()) == 6000, "A-NET resample is not 6,000 draws")
    require(np.array_equal(anet_resample["theta"], anet_curves["theta"]), "A-NET curve rows differ")
    require(np.array_equal(anet_curves["MG"], mass), "A-NET mass grid differs")
    anet_band, anet_weight, anet_ess, anet_rows = resampled_band(
        anet_curves["Rg"], anet_curves["MM"], anet_counts, mass
    )
    require(
        np.array_equal(anet_band, anet_published["anet_band"], equal_nan=True)
        and np.array_equal(anet_weight, anet_published["anet_support_weight"])
        and np.array_equal(anet_rows, anet_published["anet_support_rows"]),
        "recomputed A-NET band differs from the published A-NET band",
    )
    checks["anet_mr_band_recomputed_bit_identical_to_published"] = True

    # ---- TSNPE: TSNPE resample -> curves -> band.
    tsnpe_resample = load("tsnpe_resample")
    tsnpe_curves = load("tsnpe_curves")
    tsnpe_counts = np.asarray(tsnpe_resample["counts"], dtype=np.int64)
    require(int(tsnpe_counts.sum()) == 6000, "TSNPE resample is not 6,000 draws")
    require(
        np.array_equal(tsnpe_resample["theta"], tsnpe_curves["theta"]), "TSNPE curve rows differ"
    )
    require(np.array_equal(tsnpe_curves["MG"], mass), "TSNPE mass grid differs")
    require(tsnpe_resample["forbidden_artifacts_used"].size == 0, "TSNPE lineage flag")
    tsnpe_band, tsnpe_weight, tsnpe_ess, tsnpe_rows = resampled_band(
        tsnpe_curves["Rg"], tsnpe_curves["MM"], tsnpe_counts, mass
    )
    require(
        np.array_equal(tsnpe_band, tsnpe_published["tsnpe_band"], equal_nan=True)
        and np.array_equal(tsnpe_weight, tsnpe_published["tsnpe_support_weight"])
        and np.array_equal(tsnpe_rows, tsnpe_published["tsnpe_support_rows"]),
        "recomputed TSNPE band differs from the published TSNPE band",
    )
    checks["tsnpe_mr_band_recomputed_bit_identical_to_published"] = True

    # ---- Panel (a).
    common = common_certified_body(
        [un_band, tsnpe_band, anet_band],
        [un_weight, tsnpe_weight, anet_weight],
        [un_ess, tsnpe_ess, anet_ess],
    )
    observation, overlay_hashes = load_mass_radius_overlays(arguments.mr_data_root)
    figure, axis = new_hyperonic_axis()
    hyperonic_overlays(axis, observation)
    for band, weight, ess, rows, color, linestyle, label, zorder in (
        (un_band, un_weight, un_ess, un_rows, "forestgreen", "-", "UltraNest", 4),
        (tsnpe_band, tsnpe_weight, tsnpe_ess, tsnpe_rows, "crimson", "-.", "TSNPE+MIS", 5),
        (anet_band, anet_weight, anet_ess, anet_rows, "royalblue", "--", "A-NET+IS", 6),
    ):
        draw_support_aware_band(
            axis, mass, band, weight, ess, rows, common,
            color=color, linestyle=linestyle, linewidth=1.8, label=label,
            shaded=True, zorder=zorder,
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
    save_figure(figure, panel_a, dpi=180)
    plt.close(figure)

    # ---- Panel (b): the saved mass--tidal bands (same posterior selections and stellar curves).
    for name, key in (("UltraNest", "ultranest"), ("TSNPE+MIS", "tsnpe"), ("A-NET+IS", "anet")):
        mass_tidal.METHODS[name]["key"] = key
    mass_tidal.render(INPUTS["mass_tidal_bands"][0], panel_b, MINIMUM_CONDITIONAL_ESS, 0.02)

    record = {"status": "PASS", "figure": "Figure 7 (hyperonic A1)", "inputs": input_hashes,
              "observational_overlay_sha256": overlay_hashes, "checks": checks}
    (arguments.output_dir / "figure7_record.json").write_text(json.dumps(record, indent=1) + "\n")
    print("[figure 7] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
