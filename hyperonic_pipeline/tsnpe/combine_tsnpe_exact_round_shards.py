#!/usr/bin/env python3
"""Combine independently evaluated shards of one later clean TSNPE round."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARRAY_KEYS = (
    "theta",
    "prediction",
    "astrophysical_components",
    "x",
    "valid",
    "accepted",
    "training_mask",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if len(arguments.input) < 2:
        raise ValueError("at least two exact shards are required")
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")

    reference = json.loads(arguments.reference_report.read_text())
    if (
        reference.get("lineage_class") != "independent_tsnpe"
        or reference.get("round_index") != 0
        or reference.get("forbidden_artifacts_used")
    ):
        raise RuntimeError("acceptance reference lacks clean round-zero lineage")
    reference_hash = sha256(arguments.reference_report)

    arrays = {key: [] for key in ARRAY_KEYS}
    parents: list[dict[str, object]] = []
    prior_low = None
    prior_high = None
    round_index = None
    log_reference = None
    restricted_reference = None
    accept_seeds: set[int] = set()
    input_hashes: set[str] = set()
    proposal_hashes: set[str] = set()
    target_gate = None

    for path in arguments.input:
        report_path = path.with_suffix(".json")
        report = json.loads(report_path.read_text())
        if report.get("status") != "PASS":
            raise RuntimeError(f"{path} did not pass exact evaluation")
        if report.get("lineage_class") != "independent_tsnpe":
            raise RuntimeError(f"{path} lacks independent TSNPE lineage")
        if report.get("role") != "tsnpe_exact_round":
            raise RuntimeError(f"{path} has the wrong artifact role")
        if report.get("proposal_mode") != "restricted_uniform_prior":
            raise RuntimeError(f"{path} is not a later restricted-prior round")
        local_round = int(report["round_index"])
        if local_round < 1:
            raise RuntimeError(f"{path} is not a later TSNPE round")
        if round_index is None:
            round_index = local_round
        elif local_round != round_index:
            raise RuntimeError("exact shards come from different rounds")
        if int(report.get("pilot_rows", -1)) != 0:
            raise RuntimeError(f"{path} unexpectedly contains pilot rows")
        if report.get("forbidden_artifacts_used"):
            raise RuntimeError(f"{path} declares forbidden ancestry")
        if int(report.get("reference_exceedance_rows", -1)) != 0:
            raise RuntimeError(f"{path} failed the frozen-reference envelope gate")
        if (report.get("target_gate") or {}).get("status") != "PASS":
            raise RuntimeError(f"{path} failed the exact target certificate")
        parent = report.get("reference_parent") or {}
        if parent.get("sha256") != reference_hash:
            raise RuntimeError(f"{path} used a different acceptance reference")

        input_hash = sha256(path)
        if report.get("output_sha256") != input_hash:
            raise RuntimeError(f"{path} hash does not match its report")
        if input_hash in input_hashes:
            raise RuntimeError(f"duplicate exact shard supplied: {path}")
        input_hashes.add(input_hash)
        proposal_hash = str(report["proposal_sha256"])
        if proposal_hash in proposal_hashes:
            raise RuntimeError(f"duplicate proposal shard supplied: {path}")
        proposal_hashes.add(proposal_hash)
        accept_seed = int(report["accept_seed"])
        if accept_seed in accept_seeds:
            raise RuntimeError(f"duplicate accept/reject seed supplied: {accept_seed}")
        accept_seeds.add(accept_seed)

        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("proposal_sha256") != proposal_hash:
                raise RuntimeError(f"{path} archive/report proposal lineage differs")
            if metadata.get("reference_parent") != report.get("reference_parent"):
                raise RuntimeError(f"{path} archive/report reference lineage differs")
            if np.asarray(archive["forbidden_artifacts_used"]).size:
                raise RuntimeError(f"{path} archive declares forbidden ancestry")
            local_log_reference = float(np.asarray(archive["log_reference"]).item())
            local_restricted = float(
                np.asarray(archive["restricted_prior_log_reference"]).item()
            )
            if log_reference is None:
                log_reference = local_log_reference
                restricted_reference = local_restricted
            elif not (
                local_log_reference == log_reference
                and local_restricted == restricted_reference
            ):
                raise RuntimeError("exact shards used different frozen references")
            local_low = np.asarray(archive["prior_low"], dtype=np.float64)
            local_high = np.asarray(archive["prior_high"], dtype=np.float64)
            if prior_low is None:
                prior_low, prior_high = local_low, local_high
            elif not (
                np.array_equal(prior_low, local_low)
                and np.array_equal(prior_high, local_high)
            ):
                raise RuntimeError("exact shards do not share full-prior bounds")
            local_rows = len(np.asarray(archive["theta"]))
            for key in ARRAY_KEYS:
                value = np.asarray(archive[key])
                if len(value) != local_rows:
                    raise RuntimeError(f"{path} has inconsistent array lengths")
                arrays[key].append(value)
        target_gate = report["target_gate"]
        parents.append(
            {
                "path": str(path.resolve()),
                "sha256": input_hash,
                "proposal_sha256": proposal_hash,
            }
        )

    combined = {key: np.concatenate(parts) for key, parts in arrays.items()}
    metadata = {
        "schema_version": 1,
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_exact_round",
        "model": "ddb-hyperonic",
        "round_index": round_index,
        "proposal_mode": "restricted_uniform_prior",
        "full_prior_dimension": 9,
        "reference_parent": {
            "path": str(arguments.reference_report.resolve()),
            "sha256": reference_hash,
        },
        "parents": parents,
        "accept_seed": sorted(accept_seeds),
        "reference_margin": float(reference["reference_margin"]),
        "restricted_prior_log_reference": float(restricted_reference),
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        **combined,
        log_reference=np.asarray(float(log_reference)),
        restricted_prior_log_reference=np.asarray(float(restricted_reference)),
        prior_low=prior_low,
        prior_high=prior_high,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    report = {
        "status": "PASS",
        **metadata,
        "candidate_rows": int(len(combined["theta"])),
        "pilot_rows": 0,
        "valid_rows": int(combined["valid"].sum()),
        "accepted_rows_including_pilot": int(combined["accepted"].sum()),
        "training_rows": int(combined["training_mask"].sum()),
        "acceptance_fraction": float(combined["training_mask"].mean()),
        "log_reference": float(log_reference),
        "reference_exceedance_rows": 0,
        "target_gate": target_gate,
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
