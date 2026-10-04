#!/usr/bin/env python3
"""Split a clean proposal into contiguous, indexed likelihood shards."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument(
        "--rows",
        type=int,
        action="append",
        help=(
            "Optional row count for each --output, in the same order. "
            "Counts must be positive and sum to the proposal size."
        ),
    )
    arguments = parser.parse_args()
    if not arguments.output:
        raise ValueError("at least one output shard is required")

    with np.load(arguments.proposal, allow_pickle=False) as proposal:
        schema = str(proposal["schema"].item())
        accepted_schemas = {
            "ddb-hyperonic-clean-proposal-v1",
            "ddb-hyperonic-clean-anet-proposal-v1",
            "ddb-hyperonic-clean-anet-mixture-proposal-v1",
            "ddb-hyperonic-clean-parity-anet-mixture-proposal-v1",
            "ddb-hyperonic-clean-dual-parity-anet-mixture-v1",
        }
        if schema not in accepted_schemas:
            raise RuntimeError(f"unrecognized clean proposal schema: {schema}")
        theta = np.asarray(proposal["theta"], dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != 9:
        raise RuntimeError("proposal theta must have shape (N, 9)")

    if arguments.rows is None:
        partitions = np.array_split(
            np.arange(len(theta), dtype=np.int64), len(arguments.output)
        )
    else:
        if len(arguments.rows) != len(arguments.output):
            raise ValueError("one --rows value is required for each --output")
        if any(rows < 1 for rows in arguments.rows):
            raise ValueError("all --rows values must be positive")
        if sum(arguments.rows) != len(theta):
            raise ValueError(
                f"--rows values cover {sum(arguments.rows)} rows, "
                f"but the proposal contains {len(theta)}"
            )
        stop = np.cumsum(arguments.rows, dtype=np.int64)
        start = np.concatenate([np.asarray([0], dtype=np.int64), stop[:-1]])
        partitions = [
            np.arange(first, last, dtype=np.int64)
            for first, last in zip(start, stop, strict=True)
        ]
    records: list[dict[str, object]] = []
    for path, index in zip(arguments.output, partitions, strict=True):
        if not len(index):
            raise RuntimeError("an output shard would be empty")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, theta=theta[index], index=index)
        os.replace(temporary, path)
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "rows": len(index),
                "index_min": int(index[0]),
                "index_max": int(index[-1]),
            }
        )

    covered = np.concatenate(partitions)
    if not np.array_equal(covered, np.arange(len(theta), dtype=np.int64)):
        raise RuntimeError("shards do not exactly cover the proposal")
    report = {
        "status": "PASS",
        "proposal_schema": schema,
        "proposal": str(arguments.proposal.resolve()),
        "proposal_sha256": sha256(arguments.proposal),
        "rows": len(theta),
        "shards": records,
    }
    manifest = arguments.proposal.with_name(arguments.proposal.stem + "_shards.json")
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
