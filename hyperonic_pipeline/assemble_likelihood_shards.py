#!/usr/bin/env python3
"""Assemble certified exact-likelihood shards in proposal-row order."""

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
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    with np.load(arguments.proposal, allow_pickle=False) as proposal:
        theta = np.asarray(proposal["theta"], dtype=np.float64)
    rows = len(theta)
    loglike = np.full(rows, np.nan, dtype=np.float64)
    filled = np.zeros(rows, dtype=bool)
    shard_reports: list[dict[str, object]] = []
    nuclear_scenario = None
    source_scenario = None
    for path in arguments.shard:
        with np.load(path, allow_pickle=False) as shard:
            local_theta = np.asarray(shard["theta"], dtype=np.float64)
            index = np.asarray(shard["index"], dtype=np.int64)
            local_loglike = np.asarray(shard["logl"], dtype=np.float64)
            reference_present = bool(
                shard["reference_present"] if "reference_present" in shard.files else False
            )
            local_nuclear = str(shard["nuclear_scenario"].item())
            local_source = str(shard["source_scenario"].item())
        if reference_present:
            raise RuntimeError(f"reference values present in shard {path}")
        if nuclear_scenario is None:
            nuclear_scenario = local_nuclear
            source_scenario = local_source
        elif (local_nuclear, local_source) != (nuclear_scenario, source_scenario):
            raise RuntimeError(f"likelihood scenario differs in shard {path}")
        if index.shape != (len(local_theta),) or local_loglike.shape != (len(local_theta),):
            raise RuntimeError(f"invalid shard shape: {path}")
        if np.any(index < 0) or np.any(index >= rows) or len(np.unique(index)) != len(index):
            raise RuntimeError(f"invalid shard indices: {path}")
        if filled[index].any():
            raise RuntimeError(f"overlapping shard indices: {path}")
        if not np.array_equal(theta[index], local_theta):
            raise RuntimeError(f"shard parameter rows differ from proposal: {path}")
        filled[index] = True
        loglike[index] = local_loglike
        shard_reports.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "rows": len(index),
                "index_min": int(index.min()),
                "index_max": int(index.max()),
            }
        )
    if not filled.all() or not np.isfinite(loglike).all():
        raise RuntimeError(f"incomplete likelihood assembly: {filled.sum()}/{rows}")
    assert nuclear_scenario is not None and source_scenario is not None
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            theta=theta,
            index=np.arange(rows, dtype=np.int64),
            logl=loglike,
            nuclear_scenario=np.asarray(nuclear_scenario),
            source_scenario=np.asarray(source_scenario),
            reference_present=np.bool_(False),
            proposal_sha256=np.asarray(sha256(arguments.proposal)),
            shard_manifest_json=np.asarray(json.dumps(shard_reports, sort_keys=True)),
        )
    os.replace(temporary, arguments.output)
    report = {
        "status": "PASS",
        "rows": rows,
        "valid_rows": int(np.sum(loglike > -1.0e50)),
        "nuclear_scenario": nuclear_scenario,
        "source_scenario": source_scenario,
        "proposal_sha256": sha256(arguments.proposal),
        "shards": shard_reports,
        "output_sha256": sha256(arguments.output),
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
