#!/usr/bin/env python3
"""Extend a frozen clean TSNPE bridge proposal without refitting its density."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np

from build_tsnpe_astrophysical_bridge import (
    atomic_savez,
    ensemble_log_density,
    from_unconstrained,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--extra-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.extra_count < 1:
        raise ValueError("extra count must be positive")
    if arguments.output.exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")

    parent_report = json.loads(arguments.proposal.with_suffix(".json").read_text())
    if parent_report.get("lineage_class") != "independent_tsnpe":
        raise RuntimeError("parent proposal lacks independent TSNPE lineage")
    if parent_report.get("forbidden_artifacts_used"):
        raise RuntimeError("parent proposal declares forbidden ancestry")
    if parent_report.get("output_sha256") != sha256(arguments.proposal):
        raise RuntimeError("parent proposal hash does not match its report")
    if parent_report.get("checkpoint_sha256") != sha256(arguments.checkpoint):
        raise RuntimeError("proposal and bridge checkpoint hashes do not match")

    with np.load(arguments.proposal, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64)
        logq = np.asarray(archive["logq"], dtype=np.float64)
        component = np.asarray(archive["proposal_component"])
        low = np.asarray(archive["prior_low"], dtype=np.float64)
        high = np.asarray(archive["prior_high"], dtype=np.float64)
        pilot_rows = int(archive["pilot_rows"])
        metadata = json.loads(str(archive["metadata"].item()))
        forbidden = np.asarray(archive["forbidden_artifacts_used"])
    if forbidden.size or metadata.get("forbidden_artifacts_used"):
        raise RuntimeError("proposal archive declares forbidden ancestry")
    if metadata.get("proposal_mode") != "normalized_defensive_astrophysical_bridge":
        raise RuntimeError("proposal is not a normalized defensive bridge")

    checkpoint = joblib.load(arguments.checkpoint)
    if checkpoint.get("lineage_class") != "independent_tsnpe":
        raise RuntimeError("bridge checkpoint lacks independent TSNPE lineage")
    if checkpoint.get("forbidden_artifacts_used"):
        raise RuntimeError("bridge checkpoint declares forbidden ancestry")
    if not (
        np.array_equal(low, checkpoint["prior_low"])
        and np.array_equal(high, checkpoint["prior_high"])
    ):
        raise RuntimeError("proposal and checkpoint priors differ")
    models = list(checkpoint["models"])
    defensive_fraction = float(checkpoint["defensive_fraction"])

    defensive_rows = int(round(arguments.extra_count * defensive_fraction))
    learned_rows = arguments.extra_count - defensive_rows
    counts = np.full(len(models), learned_rows // len(models), dtype=int)
    counts[: learned_rows % len(models)] += 1
    pieces = []
    components = []
    for member, (specification, count) in enumerate(zip(models, counts, strict=True)):
        model = specification["model"]
        model.random_state = arguments.seed + 10_000 + member
        standardized, _ = model.sample(int(count))
        transformed = (
            standardized * np.asarray(specification["scale"])[None, :]
            + np.asarray(specification["mean"])[None, :]
        )
        pieces.append(from_unconstrained(transformed, low, high))
        components.append(np.full(int(count), member, dtype=np.uint8))
    rng = np.random.default_rng(arguments.seed + 20_000)
    pieces.append(rng.uniform(low, high, size=(defensive_rows, 9)))
    components.append(np.full(defensive_rows, len(models), dtype=np.uint8))
    extra_theta = np.vstack(pieces)
    extra_component = np.concatenate(components)
    permutation = rng.permutation(arguments.extra_count)
    extra_theta = extra_theta[permutation]
    extra_component = extra_component[permutation]
    learned_logq = ensemble_log_density(models, extra_theta, low, high)
    prior_logq = -float(np.log(high - low).sum())
    extra_logq = np.logaddexp(
        math.log1p(-defensive_fraction) + learned_logq,
        math.log(defensive_fraction) + prior_logq,
    )
    if not np.isfinite(extra_logq).all():
        raise RuntimeError("proposal extension has a non-finite normalized density")

    output_metadata = {
        **metadata,
        "training_candidate_rows": int(
            metadata["training_candidate_rows"] + arguments.extra_count
        ),
        "extension_parent": {
            "path": str(arguments.proposal.resolve()),
            "sha256": sha256(arguments.proposal),
        },
        "extension_rows": arguments.extra_count,
        "extension_seed": arguments.seed,
        "density_refitted_during_extension": False,
        "forbidden_artifacts_used": [],
    }
    atomic_savez(
        arguments.output,
        theta=np.vstack([theta, extra_theta]),
        logq=np.concatenate([logq, extra_logq]),
        proposal_component=np.concatenate([component, extra_component]),
        prior_low=low,
        prior_high=high,
        pilot_rows=np.asarray(pilot_rows, dtype=np.int64),
        metadata=np.asarray(json.dumps(output_metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    report = {
        "status": "PASS",
        **output_metadata,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
