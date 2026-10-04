"""Hash-pinned authorization for the 2026-09-23 support-extension checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(__file__).with_name(
    "green_en_support_extension_freeze_manifest_20260923.json"
)
MANIFEST_SHA256 = (
    "a657bae4a430f4478004499af6645e69cf8c9b96bd793bc856d845ea5b5c8410"
)


class FreezeAuthorizationError(RuntimeError):
    """Raised when a checkpoint is not in the support-extension manifest."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    if sha256(path) != MANIFEST_SHA256:
        raise FreezeAuthorizationError(
            "support-extension manifest SHA-256 does not match the pinned value"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != (
        "green-en-support-extension-checkpoint-freeze-manifest-v1"
    ):
        raise FreezeAuthorizationError("unexpected support-extension manifest schema")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != 6:
        raise FreezeAuthorizationError("freeze manifest must contain exactly six members")
    identities = [(item.get("sector"), item.get("member")) for item in members]
    hashes = [item.get("checkpoint_sha256") for item in members]
    if len(set(identities)) != 6 or len(set(hashes)) != 6:
        raise FreezeAuthorizationError("duplicate member identity or checkpoint hash")
    if set(identities) != {
        ("nucleonic", 9211), ("nucleonic", 9212), ("nucleonic", 9213),
        ("hyperonic", 9311), ("hyperonic", 9312), ("hyperonic", 9313),
    }:
        raise FreezeAuthorizationError("unexpected support-extension member set")
    for item in members:
        if not (
            item.get("checkpoint_frozen") is True
            and item.get("query_authorized") is True
            and item.get("blind_to_first_heldout_query") is True
            and item.get("evidence_space_weight") == 1.0 / 3.0
        ):
            raise FreezeAuthorizationError("manifest contains an invalid member policy")
    if manifest.get("heldout_query_performed_before_freeze") is not False:
        raise FreezeAuthorizationError("manifest chronology does not preserve blindness")
    if manifest.get("checkpoint_mutation") is not False:
        raise FreezeAuthorizationError("manifest permits checkpoint mutation")
    return manifest


def authorize_checkpoint_hash(
    checkpoint_sha256: str,
    sector: str,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Authorize a frozen checkpoint by sector and exact external hash only."""
    manifest = load_manifest(manifest_path)
    matches = [
        item
        for item in manifest["members"]
        if item["checkpoint_sha256"] == checkpoint_sha256
        and item["sector"] == sector
    ]
    if len(matches) != 1:
        raise FreezeAuthorizationError(
            f"checkpoint hash is not authorized for sector {sector!r}"
        )
    member = matches[0]
    return {
        "status": "AUTHORIZED_BY_SUPPORT_EXTENSION_MANIFEST",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": MANIFEST_SHA256,
        "sector": sector,
        "member": int(member["member"]),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_frozen": True,
        "query_authorized": True,
        "blind_to_first_heldout_query": True,
        "evidence_space_weight": float(member["evidence_space_weight"]),
        "legacy_payload_query_ready_used": False,
    }
