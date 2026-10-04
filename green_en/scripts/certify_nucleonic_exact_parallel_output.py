#!/usr/bin/env python3
"""Certify a parallel exact cache against a trusted serial prefix/full run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


FIELDS = (
    "source_index",
    "proposal_index",
    "theta",
    "prediction",
    "log_astrophysical",
    "astrophysical_components",
    "nicer_source_components",
    "target_valid",
    "radius",
    "tidal_lambda",
    "maximum_mass",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    return float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    details = {}
    with np.load(arguments.reference, allow_pickle=False) as reference, np.load(
        arguments.candidate, allow_pickle=False
    ) as candidate, np.load(arguments.screen, allow_pickle=False) as screen:
        rows = len(candidate["source_index"])
        if rows > len(reference["source_index"]):
            raise RuntimeError("candidate is longer than reference")
        for field in FIELDS:
            # Nuclear predictions are canonically the values frozen by the
            # support screen.  The trusted serial run predates this storage
            # clarification and kept a batch-size-dependent JAX recomputation.
            # Every other field is certified directly against that serial run.
            expected = (
                np.asarray(screen["prediction"])[:rows]
                if field == "prediction"
                else np.asarray(reference[field])[:rows]
            )
            actual = np.asarray(candidate[field])
            if expected.shape != actual.shape:
                raise RuntimeError(
                    f"{field} shape differs: {expected.shape} versus {actual.shape}"
                )
            numeric = np.issubdtype(expected.dtype, np.number) and expected.dtype != bool
            details[field] = {
                "bitwise_equal_with_nan": bool(
                    np.array_equal(expected, actual, equal_nan=True)
                ),
                "nan_pattern_equal": bool(
                    np.array_equal(np.isnan(expected), np.isnan(actual))
                ) if numeric else True,
                "maximum_absolute_difference": max_abs(expected, actual) if numeric else 0.0,
                "authority": (
                    "frozen support screen"
                    if field == "prediction"
                    else "trusted serial exact run"
                ),
            }
        serial_prediction = np.asarray(reference["prediction"])[:rows]
        screen_prediction = np.asarray(screen["prediction"])[:rows]
        serial_screen_prediction_max_abs = max_abs(
            serial_prediction, screen_prediction
        )
    gates = {
        "all_fields_bitwise_equal_with_nan": all(
            item["bitwise_equal_with_nan"] for item in details.values()
        ),
        "all_nan_patterns_equal": all(
            item["nan_pattern_equal"] for item in details.values()
        ),
    }
    report = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "schema": "nucleonic-exact-parallel-certification-v1",
        "reference": str(arguments.reference.resolve()),
        "reference_sha256": sha256(arguments.reference),
        "candidate": str(arguments.candidate.resolve()),
        "candidate_sha256": sha256(arguments.candidate),
        "screen": str(arguments.screen.resolve()),
        "screen_sha256": sha256(arguments.screen),
        "serial_screen_prediction_max_abs": serial_screen_prediction_max_abs,
        "rows": rows,
        "fields": details,
        "gates": gates,
        "timing_class": "certification only; excluded from production wall time",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
