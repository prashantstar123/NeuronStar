#!/usr/bin/env python3
"""Run ordered nucleonic exact-row shards concurrently and assemble them."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def chunk_aligned_ranges(
    start: int, stop: int, shards: int, chunk_size: int
) -> list[tuple[int, int]]:
    """Balance whole evaluator chunks without changing numerical batches."""
    if stop <= start:
        raise ValueError("stop must exceed start")
    boundaries = list(range(start, stop, chunk_size)) + [stop]
    chunks = len(boundaries) - 1
    if shards > chunks:
        raise ValueError(f"{shards} shards exceed {chunks} evaluator chunks")
    quotient, remainder = divmod(chunks, shards)
    result = []
    chunk_cursor = 0
    for index in range(shards):
        width = quotient + (index < remainder)
        next_cursor = chunk_cursor + width
        result.append((boundaries[chunk_cursor], boundaries[next_cursor]))
        chunk_cursor = next_cursor
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--workers-per-shard", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=2048)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite output: {arguments.output}")
    if arguments.shards < 1 or arguments.workers_per_shard < 1:
        raise ValueError("shards and workers-per-shard must be positive")
    arguments.work_dir.mkdir(parents=True, exist_ok=False)
    ranges = chunk_aligned_ranges(
        arguments.start, arguments.stop, arguments.shards, arguments.chunk_size
    )
    commands = []
    processes = []
    logs = []
    launched_unix = time.time()
    for index, (start, stop) in enumerate(ranges):
        shard = arguments.work_dir / f"exact_shard_{index:02d}_{start}_{stop}.npz"
        log_path = arguments.work_dir / f"exact_shard_{index:02d}.log"
        command = [
            sys.executable,
            "scripts/evaluate_nucleonic_support_extension.py",
            "--screen", str(arguments.screen),
            "--template", str(arguments.template),
            "--data-root", str(arguments.data_root),
            "--output", str(shard),
            "--workers", str(arguments.workers_per_shard),
            "--start", str(start),
            "--stop", str(stop),
            "--chunk-size", str(arguments.chunk_size),
        ]
        environment = os.environ.copy()
        environment["NUMBA_NUM_THREADS"] = str(arguments.workers_per_shard)
        environment["NUMBA_CACHE_DIR"] = str(
            arguments.work_dir / f"numba_cache_{index:02d}"
        )
        environment["JOBLIB_TEMP_FOLDER"] = str(
            arguments.work_dir / f"joblib_{index:02d}"
        )
        Path(environment["NUMBA_CACHE_DIR"]).mkdir()
        Path(environment["JOBLIB_TEMP_FOLDER"]).mkdir()
        log = log_path.open("wb")
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        commands.append({"index": index, "start": start, "stop": stop,
                         "output": str(shard), "log": str(log_path),
                         "command": command})
        processes.append(process)
        logs.append(log)
    returncodes = []
    for process, log in zip(processes, logs, strict=True):
        returncodes.append(process.wait())
        log.close()
    if any(code != 0 for code in returncodes):
        raise RuntimeError(f"parallel exact shard failure: {returncodes}; {commands}")

    shard_paths = [Path(item["output"]) for item in commands]
    aggregate_command = [
        sys.executable,
        "scripts/assemble_nucleonic_exact_shards.py",
        "--screen", str(arguments.screen),
        "--template", str(arguments.template),
        "--output", str(arguments.output),
        "--start", str(arguments.start),
        "--stop", str(arguments.stop),
    ]
    for path in shard_paths:
        aggregate_command.extend(["--shard", str(path)])
    subprocess.run(aggregate_command, check=True)
    finished_unix = time.time()
    report = {
        "status": "PASS",
        "schema": "nucleonic-exact-parallel-stage-v1",
        "launched_unix": launched_unix,
        "finished_unix": finished_unix,
        "wall_seconds": finished_unix - launched_unix,
        "shards": commands,
        "returncodes": returncodes,
        "aggregate_command": aggregate_command,
        "output": str(arguments.output.resolve()),
    }
    report_path = arguments.output.with_name(arguments.output.stem + "_parallel_stage.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
