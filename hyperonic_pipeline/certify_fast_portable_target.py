#!/usr/bin/env python3
"""Compare a separate fast portable target replay with its frozen certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eos import get_eos
from workflows.nuclear_scenarios import nuclear_observation, shift_frozen_a1_log_likelihood


BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 2500)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--likelihood", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    arguments = parser.parse_args()

    with np.load(arguments.likelihood, allow_pickle=False) as stored:
        theta = np.asarray(stored["theta"], dtype=np.float64)
        actual = np.asarray(stored["logl"], dtype=np.float64)
        nuclear_scenario = str(stored["nuclear_scenario"].item())
        source_scenario = str(stored["source_scenario"].item())
        reference_present = bool(stored["reference_present"])
    if reference_present:
        raise RuntimeError("production likelihood unexpectedly contains references")
    with np.load(arguments.certificate, allow_pickle=False) as certificate:
        certificate_theta = np.asarray(certificate["theta"], dtype=np.float64)
        if source_scenario == "J1614":
            expected = np.asarray(certificate["logl"], dtype=np.float64)
        else:
            key = "fixed_logl" if source_scenario == "A1" else f"fixed_logl_{source_scenario}"
            expected = np.asarray(certificate[key], dtype=np.float64)
    if not np.array_equal(theta, certificate_theta):
        raise RuntimeError("likelihood rows differ from the certificate parameters")
    if nuclear_scenario != "A1":
        plugin = get_eos("ddb-hyperonic")
        nmp = np.empty((len(theta), 6), dtype=np.float64)
        for start in range(0, len(theta), 2500):
            stop = min(start + 2500, len(theta))
            local = theta[start:stop]
            count = len(local)
            bucket = next(size for size in BUCKETS if size >= count)
            padded = (
                local
                if count == bucket
                else np.vstack([local, np.repeat(local[-1:], bucket - count, axis=0)])
            )
            nmp[start:stop] = plugin.nuclear_observables_batch(padded)[:count]
        expected = shift_frozen_a1_log_likelihood(
            expected, theta, nmp, nuclear_observation(nuclear_scenario)
        )

    expected_valid = expected > -1.0e50
    actual_valid = actual > -1.0e50
    mask_equal = bool(np.array_equal(expected_valid, actual_valid))
    maximum_error = (
        float(np.max(np.abs(actual[expected_valid] - expected[expected_valid])))
        if mask_equal and expected_valid.any()
        else float("inf")
    )
    passed = mask_equal and maximum_error <= arguments.tolerance
    report = {
        "status": "PASS" if passed else "FAIL",
        "nuclear_scenario": nuclear_scenario,
        "source_scenario": source_scenario,
        "rows": len(theta),
        "expected_valid_rows": int(expected_valid.sum()),
        "actual_valid_rows": int(actual_valid.sum()),
        "valid_mask_equal": mask_equal,
        "max_abs_log_likelihood_error": maximum_error,
        "tolerance": arguments.tolerance,
        "likelihood_sha256": sha256(arguments.likelihood),
        "certificate_sha256": sha256(arguments.certificate),
        "reference_present": False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
