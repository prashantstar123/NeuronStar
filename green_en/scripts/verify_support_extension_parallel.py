#!/usr/bin/env python3
"""Certify serial/parallel identity of a support-extension screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


CORE_FIELDS = (
    "schema",
    "eos",
    "support_name",
    "theta",
    "prediction",
    "proposal_index",
    "retained_box_membership",
    "parameter_names",
    "prior_low",
    "prior_high",
    "observation",
    "sigma",
    "centres",
    "centre_names",
    "box_radius",
    "seed",
    "rng_substream",
    "total_proposals",
    "block_size",
    "block_accepted",
    "reference_present",
    "forbidden_artifacts_used",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--parallel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite report: {arguments.output}")

    checks = {}
    with np.load(arguments.serial, allow_pickle=False) as serial, np.load(
        arguments.parallel, allow_pickle=False
    ) as parallel:
        for field in CORE_FIELDS:
            if field not in serial.files or field not in parallel.files:
                raise RuntimeError(f"missing certification field {field}")
            checks[field] = bool(
                serial[field].dtype == parallel[field].dtype
                and serial[field].shape == parallel[field].shape
                and np.array_equal(serial[field], parallel[field])
            )
    if not all(checks.values()):
        failed = [field for field, passed in checks.items() if not passed]
        raise RuntimeError(f"serial/parallel identity failure: {failed}")
    report = {
        "schema": "support-extension-parallel-identity-v1",
        "status": "PASS",
        "serial": str(arguments.serial.resolve()),
        "serial_sha256": _sha256(arguments.serial),
        "parallel": str(arguments.parallel.resolve()),
        "parallel_sha256": _sha256(arguments.parallel),
        "comparison": "bitwise array equality with identical dtype and shape",
        "fields": checks,
        "timing_class": "certification only; excluded from production wall time",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
