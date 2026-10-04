#!/usr/bin/env python3
"""Compare 90% mass-radius bands from two disjoint posterior halves."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def band_from_resample(resample: Path, curves: Path):
    with np.load(resample, allow_pickle=False) as saved:
        theta = np.asarray(saved["theta"], dtype=np.float64)
        counts = np.asarray(saved["counts"], dtype=np.int64)
        draws = int(saved["posterior_draws"])
        half_ess = float(saved["half_posterior_ess"])
    with np.load(curves, allow_pickle=False) as saved:
        curve_theta = np.asarray(saved["theta"], dtype=np.float64)
        mass = np.asarray(saved["MG"], dtype=np.float64)
        radius = np.asarray(saved["Rg"], dtype=np.float64)
        maximum_mass = np.asarray(saved["MM"], dtype=np.float64)
    if not np.array_equal(theta, curve_theta) or counts.sum() != draws:
        raise RuntimeError(f"curve replay does not match resample {resample}")
    band = np.full((3, len(mass)), np.nan, dtype=np.float64)
    rows = np.zeros(len(mass), dtype=np.float64)
    for column, value in enumerate(mass):
        supported = np.isfinite(radius[:, column]) & (maximum_mass >= value)
        if not supported.any():
            continue
        rows[column] = float(counts[supported].sum())
        repeated = np.repeat(radius[supported, column], counts[supported])
        band[:, column] = np.percentile(repeated, [5.0, 50.0, 95.0])
    return mass, band, rows / draws, rows.copy(), half_ess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--resample", type=Path, action="append", required=True)
    parser.add_argument("--curves", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-median-edge-difference-km", type=float, default=0.05)
    parser.add_argument("--maximum-p95-edge-difference-km", type=float, default=0.10)
    parser.add_argument("--maximum-edge-difference-km", type=float, default=0.25)
    arguments = parser.parse_args()
    if len(arguments.resample) != 2 or len(arguments.curves) != 2:
        raise ValueError("exactly two --resample and two --curves files are required")

    sys.path.insert(0, str(arguments.paper_root.resolve()))
    from plotting.mass_radius_style import common_certified_body

    first = band_from_resample(arguments.resample[0], arguments.curves[0])
    second = band_from_resample(arguments.resample[1], arguments.curves[1])
    if not np.array_equal(first[0], second[0]):
        raise RuntimeError("split-half curve mass grids differ")
    mass = first[0]
    bands = [first[1], second[1]]
    weights = [first[2], second[2]]
    effective = [first[3], second[3]]
    common = common_certified_body(bands, weights, effective)
    if not common.any():
        raise RuntimeError("split halves have no common certified mass range")
    differences = np.abs(
        np.concatenate(
            [
                bands[0][0, common] - bands[1][0, common],
                bands[0][2, common] - bands[1][2, common],
            ]
        )
    )
    median = float(np.median(differences))
    p95 = float(np.percentile(differences, 95.0))
    maximum = float(differences.max())
    passed = (
        median <= arguments.maximum_median_edge_difference_km
        and p95 <= arguments.maximum_p95_edge_difference_km
        and maximum <= arguments.maximum_edge_difference_km
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "half_posterior_ess": [first[4], second[4]],
        "common_certified_mass_range_msun": [
            float(mass[common][0]),
            float(mass[common][-1]),
        ],
        "absolute_90pct_edge_difference_km": {
            "median": median,
            "p95": p95,
            "maximum": maximum,
        },
        "gates_km": {
            "median": arguments.maximum_median_edge_difference_km,
            "p95": arguments.maximum_p95_edge_difference_km,
            "maximum": arguments.maximum_edge_difference_km,
        },
        "inputs_sha256": {
            str(path): sha256(path)
            for path in [*arguments.resample, *arguments.curves]
        },
        "ultranest_used": False,
        "reference_present": False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
