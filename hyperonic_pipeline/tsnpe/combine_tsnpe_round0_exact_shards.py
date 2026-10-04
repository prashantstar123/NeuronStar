#!/usr/bin/env python3
"""Combine clean TSNPE round-zero continuation shards after exact rejection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")
    pilot = json.loads(arguments.pilot_report.read_text())
    if (
        pilot.get("lineage_class") != "independent_tsnpe"
        or pilot.get("round_index") != 0
        or pilot.get("forbidden_artifacts_used")
    ):
        raise RuntimeError("pilot report lacks clean round-zero lineage")

    concatenated: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "theta",
            "prediction",
            "astrophysical_components",
            "x",
            "valid",
            "accepted",
            "training_mask",
        )
    }
    parents = []
    prior_low = None
    prior_high = None
    reference_parent_hash = sha256(arguments.pilot_report)
    accept_seeds: set[int] = set()
    input_hashes: set[str] = set()
    for path in arguments.input:
        report = json.loads(path.with_suffix(".json").read_text())
        if report.get("lineage_class") != "independent_tsnpe":
            raise RuntimeError(f"{path} lacks independent TSNPE lineage")
        if report.get("role") != "tsnpe_exact_round":
            raise RuntimeError(f"{path} has the wrong artifact role")
        if report.get("round_index") != 0 or report.get("pilot_rows") != 0:
            raise RuntimeError(f"{path} is not a round-zero continuation shard")
        if report.get("proposal_mode") != "normalized_defensive_astrophysical_bridge":
            raise RuntimeError(f"{path} has the wrong round-zero proposal mode")
        if report.get("forbidden_artifacts_used"):
            raise RuntimeError(f"{path} declares forbidden ancestry")
        input_hash = sha256(path)
        if report.get("output_sha256") != input_hash:
            raise RuntimeError(f"{path} hash does not match its report")
        if input_hash in input_hashes:
            raise RuntimeError(f"duplicate exact shard supplied: {path}")
        input_hashes.add(input_hash)
        parent = report.get("reference_parent") or {}
        if parent.get("sha256") != reference_parent_hash:
            raise RuntimeError(f"{path} used a different pilot reference")
        if not np.isclose(
            float(report["log_reference"]),
            float(pilot["log_reference"]),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(f"{path} used a different frozen log reference")
        if int(report.get("reference_exceedance_rows", -1)) != 0:
            raise RuntimeError(f"{path} failed the frozen-reference envelope gate")
        if (report.get("target_gate") or {}).get("status") != "PASS":
            raise RuntimeError(f"{path} failed the shared target certificate")
        accept_seed = int(report["accept_seed"])
        if accept_seed in accept_seeds:
            raise RuntimeError(f"duplicate accept/reject seed supplied: {accept_seed}")
        accept_seeds.add(accept_seed)
        with np.load(path, allow_pickle=False) as archive:
            archive_metadata = json.loads(str(archive["metadata"].item()))
            if archive_metadata.get("proposal_sha256") != report.get("proposal_sha256"):
                raise RuntimeError(f"{path} archive/report proposal lineage differs")
            if archive_metadata.get("reference_parent") != report.get("reference_parent"):
                raise RuntimeError(f"{path} archive/report reference lineage differs")
            if np.asarray(archive["forbidden_artifacts_used"]).size:
                raise RuntimeError(f"{path} archive declares forbidden ancestry")
            if not np.isclose(
                float(np.asarray(archive["log_reference"]).item()),
                float(pilot["log_reference"]),
                rtol=0.0,
                atol=0.0,
            ):
                raise RuntimeError(f"{path} archive has a different frozen reference")
            for key in concatenated:
                concatenated[key].append(np.asarray(archive[key]))
            local_low = np.asarray(archive["prior_low"], dtype=np.float64)
            local_high = np.asarray(archive["prior_high"], dtype=np.float64)
            if prior_low is None:
                prior_low, prior_high = local_low, local_high
            elif not (
                np.array_equal(prior_low, local_low)
                and np.array_equal(prior_high, local_high)
            ):
                raise RuntimeError("exact shards do not share the same prior")
        parents.append(
            {"path": str(path.resolve()), "sha256": input_hash}
        )

    arrays = {key: np.concatenate(value) for key, value in concatenated.items()}
    metadata = {
        "schema_version": 1,
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_exact_round",
        "model": "ddb-hyperonic",
        "round_index": 0,
        "proposal_mode": "normalized_defensive_astrophysical_bridge",
        "full_prior_dimension": 9,
        "reference_parent": {
            "path": str(arguments.pilot_report.resolve()),
            "sha256": reference_parent_hash,
        },
        "parents": parents,
        "accept_seed": sorted(accept_seeds),
        "reference_margin": float(pilot["reference_margin"]),
        "restricted_prior_log_reference": float(
            pilot["restricted_prior_log_reference"]
        ),
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        **arrays,
        log_reference=np.asarray(float(pilot["log_reference"])),
        restricted_prior_log_reference=np.asarray(
            float(pilot["restricted_prior_log_reference"])
        ),
        prior_low=prior_low,
        prior_high=prior_high,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    report = {
        "status": "PASS",
        **metadata,
        "candidate_rows": int(len(arrays["theta"])),
        "pilot_rows": 0,
        "valid_rows": int(arrays["valid"].sum()),
        "accepted_rows_including_pilot": int(arrays["accepted"].sum()),
        "training_rows": int(arrays["training_mask"].sum()),
        "acceptance_fraction": float(arrays["training_mask"].mean()),
        "log_reference": float(pilot["log_reference"]),
        "reference_exceedance_rows": 0,
        "target_gate": pilot["target_gate"],
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
