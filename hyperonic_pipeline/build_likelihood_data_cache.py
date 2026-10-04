#!/usr/bin/env python3
"""Build a provenance-bearing cache of fixed NICER/GW/nuclear likelihood data."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import joblib
import numpy as np

import ddb_ultranest_hyp as likelihood


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = likelihood.SWAP or "A1"
    sources = [
        *(Path(row[0]) for row in likelihood.NICER_SOURCES),
        likelihood.GW_FILE,
        likelihood.NUCLEAR_BANK,
    ]
    started = time.perf_counter()
    data = likelihood.build_a1_likelihood_data(verbose=False)
    build_seconds = time.perf_counter() - started
    payload = {
        "schema": "ddb-hyperonic-likelihood-cache-v1",
        "configuration": configuration,
        "source_paths": tuple(str(path.resolve()) for path in sources),
        "source_sizes": np.asarray([path.stat().st_size for path in sources]),
        "source_sha256": tuple(_sha256(path) for path in sources),
        "data": data,
        "build_seconds": build_seconds,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, arguments.output, compress=0)
    loaded = joblib.load(arguments.output)
    if loaded["configuration"] != configuration:
        raise RuntimeError("cache round-trip changed its configuration")
    print(
        f"configuration={configuration} build_s={build_seconds:.3f} "
        f"cache_bytes={arguments.output.stat().st_size} saved={arguments.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
