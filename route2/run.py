#!/usr/bin/env python3
"""Run the Route 2 checks or smoke runs and print one summary.

  ./reproduce.sh route2-check   downloads the large check inputs (once), then runs every replay check
  ./reproduce.sh route2-smoke   runs the tiny end-to-end smoke runs

Each check replays one stage of the paper's pipeline from frozen inputs and compares with the stored reference
results (see docs/route2/ and MANUAL_ROUTE2.md). Everything runs on CPU unless --device cuda is given.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time

from route2.common import ROOT

CHECKS = [
    ("evidence", "Evidence Network: 48 member values and the 16 Table VI evidences"),
    ("hyperonic_anet", "Hyperonic A-NET: A1 and the seven queries"),
    ("nucleonic_anet", "Nucleonic A-NET: A1, J0614, J1231, four nuclear shifts, J1614"),
    ("tsnpe_nucleonic", "Nucleonic TSNPE+MIS"),
    ("tsnpe_hyperonic", "Hyperonic TSNPE+MIS"),
    ("ultranest", "UltraNest evidences and corrected posteriors"),
    ("banks", "Prior banks: stored rows recomputed"),
]
SMOKE = [
    ("en_smoke", "Evidence Network: tiny label build, training and query"),
    ("anet_nucleonic_smoke", "Nucleonic A-NET: tiny bank, training, query and IS"),
    ("anet_hyperonic_smoke", "Hyperonic A-NET: tiny cache, training, dual sampler and IS"),
    ("tsnpe_smoke", "TSNPE: tiny round and mixture IS"),
]


def available(package: str, name: str) -> bool:
    return importlib.util.find_spec(f"route2.{package}.{name}") is not None


def run(package: str, items: list[tuple[str, str]], device: str | None) -> int:
    rows = []
    for name, description in items:
        if not available(package, name):
            print(f"\n== {description}: module route2.{package}.{name} is missing", flush=True)
            rows.append((name, "FAIL", 0.0))
            continue
        command = [sys.executable, "-m", f"route2.{package}.{name}"]
        if device and package == "checks":
            command += ["--device", device]
        print(f"\n== {description}  ({' '.join(command[1:])})", flush=True)
        started = time.time()
        code = subprocess.call(command, cwd=ROOT)
        rows.append((name, "PASS" if code == 0 else "FAIL", time.time() - started))
    print("\n== Route 2 summary")
    for name, status, seconds in rows:
        print(f"  {status}  {name:24s} {seconds / 60:5.1f} min")
    failed = [name for name, status, _ in rows if status != "PASS"]
    total = sum(seconds for _, _, seconds in rows)
    print(f"\nROUTE 2 {'PASSED' if not failed else 'FAILED'}: {len(rows) - len(failed)}/{len(rows)} "
          f"{'checks' if package == 'checks' else 'smoke runs'} in {total / 60:.1f} min"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("checks", "smoke"))
    parser.add_argument("--device", default=None, help="cpu (default) or cuda, passed to every check")
    arguments = parser.parse_args()
    if arguments.what == "checks":
        if subprocess.call([sys.executable, "-m", "route2.fetch"], cwd=ROOT) != 0:
            print("FAIL: the large check inputs could not be downloaded and verified")
            return 1
        return run("checks", CHECKS, arguments.device)
    return run("smoke", SMOKE, None)


if __name__ == "__main__":
    sys.exit(main())
