#!/usr/bin/env python3
"""Hyperonic query gate.

Runs the query ``query_green_frozen_v5.py`` behind an external, hash-pinned freeze manifest that lists exactly
the three hyperonic members trained on parent bank 02357702....

Differences from ``query_green_frozen_v7.py``:

* authorization uses the hyperonic manifest, whose SHA-256 must be supplied on the command line (it is recorded
  in the separate freeze file written before any held-out source is staged);
* the hard-coded comparison constants that v5 copies into its report
  (``ultranest_log_evidence``, ``ultranest_sigma``, ``delta``,
  ``separation_ultranest_sigma``) are removed from the saved EN output, and
  v5's console echo is suppressed, so no sampler value appears in any EN
  output.  They never entered the EN value itself: v5 computes
  ``log_evidence`` from the frozen checkpoint and the held-out density only;
* the checkpoint deserialization time is recorded separately.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from inference.evidence_network.conditional import query_green_frozen_v5  # noqa: E402

PARENT_BANK_SHA256 = "02357702e85e86d8154d1641e7d4fb818c66145d0364e127ad520986c2fe1fce"
MANIFEST_SCHEMA = "green-en-hyperonic-checkpoint-freeze-manifest-v1"
EXPECTED_MEMBERS = {("hyperonic", 9311), ("hyperonic", 9312), ("hyperonic", 9313)}
COMPARISON_FIELDS = (
    "ultranest_log_evidence",
    "ultranest_sigma",
    "delta",
    "separation_ultranest_sigma",
)


class FreezeAuthorizationError(RuntimeError):
    """Raised when a checkpoint is not in the hyperonic freeze manifest."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def authorize(checkpoint_sha256: str, manifest_path: Path, manifest_sha256: str) -> dict:
    actual = sha256(manifest_path)
    if actual != manifest_sha256:
        raise FreezeAuthorizationError(
            f"freeze manifest SHA-256 {actual} does not match the pinned {manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise FreezeAuthorizationError("unexpected hyperonic manifest schema")
    if manifest.get("parent_bank_sha256") != PARENT_BANK_SHA256:
        raise FreezeAuthorizationError("manifest does not name parent bank 02357702")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != 3:
        raise FreezeAuthorizationError("manifest must list exactly three members")
    identities = {(item.get("sector"), item.get("member")) for item in members}
    hashes = {item.get("checkpoint_sha256") for item in members}
    if identities != EXPECTED_MEMBERS or len(hashes) != 3:
        raise FreezeAuthorizationError("unexpected member set or duplicate checkpoint hash")
    for item in members:
        if not (
            item.get("checkpoint_frozen") is True
            and item.get("query_authorized") is True
            and item.get("evidence_space_weight") == 1.0 / 3.0
        ):
            raise FreezeAuthorizationError("manifest contains an invalid member policy")
    if manifest.get("heldout_query_performed_before_freeze") is not False:
        raise FreezeAuthorizationError("manifest chronology is invalid")
    if manifest.get("checkpoint_mutation") is not False:
        raise FreezeAuthorizationError("manifest permits checkpoint mutation")
    matches = [item for item in members if item["checkpoint_sha256"] == checkpoint_sha256]
    if len(matches) != 1:
        raise FreezeAuthorizationError("checkpoint hash is not in the hyperonic manifest")
    member = matches[0]
    return {
        "status": "AUTHORIZED_BY_HYPERONIC_FREEZE_MANIFEST",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "sector": "hyperonic",
        "member": int(member["member"]),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_frozen": True,
        "query_authorized": True,
        "evidence_space_weight": float(member["evidence_space_weight"]),
        "parent_bank_sha256": PARENT_BANK_SHA256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--freeze-manifest-sha256", required=True)
    arguments = parser.parse_args()

    checkpoint_hash = sha256(arguments.checkpoint)
    authorization = authorize(
        checkpoint_hash, arguments.freeze_manifest, arguments.freeze_manifest_sha256
    )

    load_seconds: list[float] = []
    original_load = torch.load

    def timed_load(*args, **kwargs):
        started = time.perf_counter()
        result = original_load(*args, **kwargs)
        load_seconds.append(time.perf_counter() - started)
        return result

    original_argv = sys.argv
    sys.argv = [
        original_argv[0],
        "--checkpoint", str(arguments.checkpoint),
        "--heldout-root", str(arguments.heldout_root),
        "--output", str(arguments.output),
        "--device", str(arguments.device),
        "--sector", "hyperonic",
    ]
    torch.load = timed_load
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = query_green_frozen_v5.main()
    finally:
        torch.load = original_load
        sys.argv = original_argv
    if result != 0:
        return int(result)

    report = json.loads(arguments.output.read_text(encoding="utf-8"))
    if report["checkpoint_sha256_before"] != checkpoint_hash:
        raise RuntimeError("v5 query output used a different checkpoint hash")
    if report["checkpoint_sha256_after"] != checkpoint_hash:
        raise RuntimeError("checkpoint changed during the authorized query")
    removed = []
    for configuration, row in report["results"].items():
        for field in COMPARISON_FIELDS:
            if field in row:
                del row[field]
                removed.append(f"{configuration}.{field}")
    report["comparison_fields_removed"] = {
        "fields": list(COMPARISON_FIELDS),
        "count": len(removed),
        "reason": (
            "v5 hard-codes sampler comparison constants; they never enter "
            "log_evidence and are stripped so the EN output contains no sampler value"
        ),
    }
    report["checkpoint_deserialization_seconds"] = load_seconds
    report["freeze_authorization"] = authorization
    report["query_wrapper"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256(Path(__file__).resolve()),
        "underlying_query": str(Path(query_green_frozen_v5.__file__).resolve()),
        "underlying_query_sha256": sha256(Path(query_green_frozen_v5.__file__).resolve()),
    }
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "member": authorization["member"],
        "log_evidence": {
            configuration: row["log_evidence"]
            for configuration, row in report["results"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
