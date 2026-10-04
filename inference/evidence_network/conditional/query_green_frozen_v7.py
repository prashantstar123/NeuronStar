#!/usr/bin/env python3
"""Run the v5 query behind the support-extension freeze gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from inference.evidence_network.conditional import query_green_frozen_v5
from inference.evidence_network.conditional.support_extension_freeze_manifest import (
    DEFAULT_MANIFEST,
    authorize_checkpoint_hash,
    sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sector", choices=("hyperonic", "nucleonic"), required=True
    )
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args()

    checkpoint_hash = sha256(arguments.checkpoint)
    authorization = authorize_checkpoint_hash(
        checkpoint_hash, arguments.sector, arguments.freeze_manifest
    )

    original_argv = sys.argv
    sys.argv = [
        original_argv[0],
        "--checkpoint", str(arguments.checkpoint),
        "--heldout-root", str(arguments.heldout_root),
        "--output", str(arguments.output),
        "--device", str(arguments.device),
        "--sector", str(arguments.sector),
    ]
    try:
        result = query_green_frozen_v5.main()
    finally:
        sys.argv = original_argv
    if result != 0:
        return int(result)

    report = json.loads(arguments.output.read_text(encoding="utf-8"))
    if report["checkpoint_sha256_before"] != checkpoint_hash:
        raise RuntimeError("v5 query output used a different checkpoint hash")
    if report["checkpoint_sha256_after"] != checkpoint_hash:
        raise RuntimeError("checkpoint changed during the authorized query")
    report["freeze_authorization"] = authorization
    report["legacy_payload_query_ready_policy"] = (
        "VESTIGIAL_AND_IGNORED; authorization is exact external manifest membership"
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"freeze_authorization": authorization}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
