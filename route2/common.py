"""Helpers shared by the Route 2 checks."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# The checks run on CPU unless --device cuda is given; JAX (EOS/TOV) always runs on CPU, as in production.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = ROOT / "data/observations"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_observations() -> None:
    """Unpack the compressed observation files once (the same step as './reproduce.sh check')."""
    from route1.prepare import unpack

    for entry in json.loads((ROOT / "data/MANIFEST.json").read_text())["files"]:
        if "stored_as" in entry:
            unpack(entry)


def use_hyperonic_pipeline() -> None:
    """Make the hyperonic pipeline scripts importable, with the legacy modules shipped in legacy_stack/.

    Must run before any hyperonic pipeline module is imported: those modules read these settings at import.
    """
    os.environ.setdefault("CERTIFIED_DDB_DIR", str(ROOT / "legacy_stack/validated_code_DDB"))
    os.environ.setdefault("CERTIFIED_ASTRO_DIR", str(ROOT / "legacy_stack/ddb_astro_mod"))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    pipeline = str(ROOT / "hyperonic_pipeline")
    if pipeline not in sys.path:
        sys.path.insert(0, pipeline)
