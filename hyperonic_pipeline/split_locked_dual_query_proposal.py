#!/usr/bin/env python3
"""Split a locked dual-head query proposal while preserving target metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    arguments = parser.parse_args()
    if len(arguments.output) < 2:
        raise ValueError("at least two output shards are required")

    with np.load(arguments.proposal, allow_pickle=False) as saved:
        if str(saved["schema"].item()) != "ddb-hyperonic-clean-dual-parity-anet-mixture-v1":
            raise RuntimeError("input is not a locked dual-head query proposal")
        theta = np.asarray(saved["theta"], dtype=np.float64)
        nuclear = str(saved["nuclear_scenario"].item())
        source = str(saved["source_scenario"].item())
        proposal_seed = int(saved["proposal_seed"])
        reference_present = bool(saved["reference_present"])
        forbidden = np.asarray(saved["forbidden_artifacts_used"])
    if theta.ndim != 2 or theta.shape[1] != 9:
        raise RuntimeError("proposal theta is not (N,9)")
    if reference_present or forbidden.size:
        raise RuntimeError("proposal declares forbidden ancestry")

    partitions = np.array_split(np.arange(len(theta), dtype=np.int64), len(arguments.output))
    records = []
    for path, index in zip(arguments.output, partitions, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                theta=theta[index],
                index=index,
                nuclear_scenario=np.asarray(nuclear),
                source_scenario=np.asarray(source),
                proposal_seed=np.int64(proposal_seed),
                proposal_sha256=np.asarray(sha256(arguments.proposal)),
                reference_present=np.bool_(False),
            )
        os.replace(temporary, path)
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "rows": int(len(index)),
                "index_min": int(index[0]),
                "index_max": int(index[-1]),
            }
        )

    covered = np.concatenate(partitions)
    if not np.array_equal(covered, np.arange(len(theta), dtype=np.int64)):
        raise RuntimeError("shards do not exactly cover the proposal")
    report = {
        "status": "PASS",
        "proposal": str(arguments.proposal.resolve()),
        "proposal_sha256": sha256(arguments.proposal),
        "nuclear_scenario": nuclear,
        "source_scenario": source,
        "rows": int(len(theta)),
        "shards": records,
    }
    manifest = arguments.proposal.with_name(arguments.proposal.stem + "_shards.json")
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
