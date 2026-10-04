#!/usr/bin/env python3
"""Run one production command and write its wall-time record (the timing rule of Tables VII and VIII)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path


def _stamp(timestamp: float, offset_hours: int) -> str:
    zone = dt.timezone(dt.timedelta(hours=offset_hours))
    return dt.datetime.fromtimestamp(timestamp, tz=zone).isoformat(
        timespec="microseconds"
    )


def _hash_if_file(value: str):
    path = Path(value)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-output", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")
    if arguments.timing_output.exists():
        raise FileExistsError(f"refusing to overwrite the timing record: {arguments.timing_output}")

    started_unix = time.time()
    started_monotonic = time.monotonic()
    completed = subprocess.run(command, check=False)
    finished_unix = time.time()
    wall_seconds = time.monotonic() - started_monotonic
    record = {
        "schema": "clean-production-stage-timing-v1",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "stage": arguments.stage,
        "command": command,
        "command_files": [
            item for item in (_hash_if_file(value) for value in command) if item
        ],
        "cwd": str(Path.cwd().resolve()),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_unix": started_unix,
        "started_utc": _stamp(started_unix, 0),
        "started_kst": _stamp(started_unix, 9),
        "finished_unix": finished_unix,
        "finished_utc": _stamp(finished_unix, 0),
        "finished_kst": _stamp(finished_unix, 9),
        "wall_seconds": wall_seconds,
        "returncode": completed.returncode,
        "timing_scope": {
            "included": [
                "target interpreter and library startup",
                "JIT or model loading inside the target command",
                "input reading",
                "full production calculation and gates",
                "target output and timing-record writing",
            ],
            "excluded": [
                "development, profiling, certification, and pilot commands",
                "failed or repeated runs",
                "queue and resource waiting",
                "inter-stage idle and human delay",
                "transfer, staging, and archival",
            ],
        },
    }
    arguments.timing_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.timing_output.with_name(
        f".{arguments.timing_output.name}.tmp-{os.getpid()}"
    )
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, arguments.timing_output)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
