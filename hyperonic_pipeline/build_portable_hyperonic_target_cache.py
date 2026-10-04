#!/usr/bin/env python3
"""Freeze one corrected portable hyperonic likelihood target for fast replay."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np

from workflows.a1_problem import A1Problem
from workflows.nuclear_scenarios import NUCLEAR_SCENARIO_NAMES, nuclear_observation
from workflows.source_scenarios import SOURCE_SCENARIO_NAMES, source_substitution


BASE_FILES = (
    "J0030_2spot_RM.txt",
    "J0740_NICERXMM_full_mr.txt",
    "J0437_post_equal_weights.dat",
    "GW170817_GWTC-1.hdf5",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path)
    parser.add_argument(
        "--nuclear-scenario", choices=NUCLEAR_SCENARIO_NAMES, default="A1"
    )
    parser.add_argument(
        "--source-scenario", choices=SOURCE_SCENARIO_NAMES, default="A1"
    )
    arguments = parser.parse_args()
    if arguments.nuclear_scenario != "A1" and arguments.source_scenario != "A1":
        raise ValueError(
            "change either the NICER source or the nuclear observation, not both"
        )

    if arguments.base_cache is not None:
        if arguments.source_scenario != "A1":
            raise ValueError("--base-cache is only valid for baseline NICER sources")
        base = joblib.load(arguments.base_cache)
        if (
            base.get("schema") != "ddb-hyperonic-portable-target-cache-v1"
            or base.get("nuclear_scenario") != "A1"
            or base.get("source_scenario") != "A1"
        ):
            raise RuntimeError("--base-cache is not the corrected baseline A1 target")
        target = dataclasses.replace(
            base["target"],
            nuclear_observation=nuclear_observation(arguments.nuclear_scenario),
        )
        input_hashes = dict(base["input_sha256"])
        base_cache_sha256 = sha256(arguments.base_cache)
    else:
        problem = A1Problem.from_data_root(
            arguments.data_root,
            source_data_root=arguments.source_data_root,
            workers=1,
            verify_data=True,
            nuclear_scenario=arguments.nuclear_scenario,
            source_scenario=arguments.source_scenario,
            model="ddb-hyperonic",
        )
        target = problem.target
        source_root = arguments.source_data_root or arguments.data_root
        paths = [arguments.data_root / name for name in BASE_FILES]
        replacement = source_substitution(arguments.source_scenario)
        if replacement is not None:
            paths.append(source_root / replacement.filename)
        input_hashes = {str(path.resolve()): sha256(path) for path in paths}
        base_cache_sha256 = ""

    payload = {
        "schema": "ddb-hyperonic-portable-target-cache-v1",
        "model": "ddb-hyperonic",
        "nuclear_scenario": arguments.nuclear_scenario,
        "source_scenario": arguments.source_scenario,
        "nuclear_observation": np.asarray(
            target.nuclear_observation, dtype=np.float64
        ),
        "input_sha256": input_hashes,
        "base_cache_sha256": base_cache_sha256,
        "target": target,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    joblib.dump(payload, temporary, compress=3)
    os.replace(temporary, arguments.output)
    report = {
        key: value for key, value in payload.items() if key != "target"
    }
    report.update(
        {
            "status": "PASS",
            "output_sha256": sha256(arguments.output),
        }
    )
    report["nuclear_observation"] = report["nuclear_observation"].tolist()
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
