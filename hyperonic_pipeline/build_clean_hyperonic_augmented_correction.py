#!/usr/bin/env python3
"""Build exact DM-MIS corrections for uniform, support, and Student rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp

from clean_student_t_ensemble import checkpoint_density
from likelihoods.nuclear import A1_OBSERVATION, A1_SIGMA


SCHEMAS = {
    "uniform_base": "ddb-hyperonic-clean-uniform-training-cache-v1",
    "nuclear_support_enrichment": "ddb-hyperonic-clean-support-training-cache-v1",
    "student_proposal": "ddb-hyperonic-clean-proposal-training-cache-v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: Path, role: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema",
            "theta",
            "prediction",
            "log_astrophysical",
            "target_valid",
            "prior_low",
            "prior_high",
            "reference_present",
            "forbidden_artifacts_used",
        }
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"{path} is missing {sorted(missing)}")
        if str(archive["schema"].item()) != SCHEMAS[role]:
            raise RuntimeError(f"unexpected {role} cache schema in {path}")
        if bool(archive["reference_present"]):
            raise RuntimeError(f"reference contamination declared by {path}")
        if archive["forbidden_artifacts_used"].size:
            raise RuntimeError(f"forbidden ancestry declared by {path}")
        result = {
            "theta": np.asarray(archive["theta"], dtype=np.float64),
            "prediction": np.asarray(archive["prediction"], dtype=np.float64),
            "log_astrophysical": np.asarray(
                archive["log_astrophysical"], dtype=np.float64
            ),
            "target_valid": np.asarray(archive["target_valid"], dtype=bool),
            "prior_low": np.asarray(archive["prior_low"], dtype=np.float64),
            "prior_high": np.asarray(archive["prior_high"], dtype=np.float64),
        }
        for key in (
            "total_proposals",
            "cbox",
            "screen_sha256",
            "source_index",
            "proposal_logq",
            "proposal_sha256",
            "checkpoint_sha256",
        ):
            if key in archive.files:
                result[key] = np.asarray(archive[key])
    rows = len(result["theta"])
    if (
        result["theta"].shape != (rows, 9)
        or result["prediction"].shape != (rows, 7)
        or result["log_astrophysical"].shape != (rows,)
        or result["target_valid"].shape != (rows,)
    ):
        raise RuntimeError(f"invalid cache shapes in {path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, action="append", required=True)
    parser.add_argument("--support-cache", type=Path, action="append", required=True)
    parser.add_argument("--proposal-cache", type=Path, action="append", required=True)
    parser.add_argument("--proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--base-proposals", type=int, default=600_000)
    parser.add_argument("--conditional-ess-gate", type=float, default=2000.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists() and not arguments.force:
        raise FileExistsError(f"counting correction exists: {arguments.output}")
    if arguments.base_proposals < 1 or arguments.conditional_ess_gate <= 0:
        raise ValueError("proposal count and ESS gate must be positive")

    started = time.time()
    paths = (
        list(arguments.base_cache)
        + list(arguments.support_cache)
        + list(arguments.proposal_cache)
    )
    roles = (
        ["uniform_base"] * len(arguments.base_cache)
        + ["nuclear_support_enrichment"] * len(arguments.support_cache)
        + ["student_proposal"] * len(arguments.proposal_cache)
    )
    loaded: list[dict[str, np.ndarray]] = []
    records: list[dict[str, object]] = []
    support_total_proposals = None
    support_cbox = None
    support_screen_sha256 = None
    proposal_sha256 = None
    proposal_checkpoint_sha256 = sha256(arguments.proposal_checkpoint)
    proposal_indices: list[np.ndarray] = []
    for path, role in zip(paths, roles, strict=True):
        data = load_cache(path, role)
        if loaded and not (
            np.array_equal(data["prior_low"], loaded[0]["prior_low"])
            and np.array_equal(data["prior_high"], loaded[0]["prior_high"])
        ):
            raise RuntimeError("cache prior bounds differ")
        if role == "nuclear_support_enrichment":
            local_total = int(data["total_proposals"])
            local_cbox = float(data["cbox"])
            local_screen = str(data["screen_sha256"].item())
            if support_total_proposals is None:
                support_total_proposals = local_total
                support_cbox = local_cbox
                support_screen_sha256 = local_screen
            elif (
                support_total_proposals != local_total
                or support_cbox != local_cbox
                or support_screen_sha256 != local_screen
            ):
                raise RuntimeError("support-cache shards do not share one screen")
        if role == "student_proposal":
            local_proposal = str(data["proposal_sha256"].item())
            local_checkpoint = str(data["checkpoint_sha256"].item())
            if local_checkpoint != proposal_checkpoint_sha256:
                raise RuntimeError("proposal-cache checkpoint hash differs")
            if proposal_sha256 is None:
                proposal_sha256 = local_proposal
            elif proposal_sha256 != local_proposal:
                raise RuntimeError("proposal-cache shards do not share one proposal")
            index = np.asarray(data["source_index"], dtype=np.int64)
            if index.shape != (len(data["theta"]),):
                raise RuntimeError("proposal-cache source indices have wrong shape")
            proposal_indices.append(index)
        loaded.append(data)
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "role": role,
                "rows": int(len(data["theta"])),
                "target_valid_rows": int(data["target_valid"].sum()),
            }
        )
    if support_total_proposals is None or support_cbox is None:
        raise RuntimeError("no support proposal count was found")
    if proposal_sha256 is None or not proposal_indices:
        raise RuntimeError("no Student proposal cache was found")
    covered_proposal_index = np.concatenate(proposal_indices)
    if not np.array_equal(
        covered_proposal_index,
        np.arange(len(covered_proposal_index), dtype=np.int64),
    ):
        raise RuntimeError("proposal-cache shards are not an exact ordered partition")
    proposal_total_rows = len(covered_proposal_index)

    theta = np.concatenate([data["theta"] for data in loaded], axis=0)
    prediction = np.concatenate([data["prediction"] for data in loaded], axis=0)
    logastro = np.concatenate(
        [data["log_astrophysical"] for data in loaded], axis=0
    )
    valid = np.concatenate([data["target_valid"] for data in loaded], axis=0)
    prior_low = loaded[0]["prior_low"]
    prior_high = loaded[0]["prior_high"]
    if not np.all((theta >= prior_low) & (theta <= prior_high)):
        raise RuntimeError("combined cache contains an out-of-prior row")
    support = np.all(
        np.abs((prediction - A1_OBSERVATION) / A1_SIGMA) <= support_cbox,
        axis=1,
    )
    support_start = sum(
        record["rows"] for record in records[: len(arguments.base_cache)]
    )
    support_stop = support_start + sum(
        record["rows"]
        for record in records[
            len(arguments.base_cache) : len(arguments.base_cache)
            + len(arguments.support_cache)
        ]
    )
    if not support[support_start:support_stop].all():
        raise RuntimeError("an enrichment row lies outside its declared support")

    checkpoint = joblib.load(arguments.proposal_checkpoint)
    prior_logq = -float(np.log(prior_high - prior_low).sum())
    proposal_logq = checkpoint_density(checkpoint, theta, prior_logq)
    if not np.isfinite(proposal_logq).all():
        raise RuntimeError("Student proposal density is non-finite")
    own_start = support_stop
    own_stored_logq = np.concatenate(
        [data["proposal_logq"] for data in loaded if "proposal_logq" in data]
    )
    own_replay_error = float(
        np.max(np.abs(proposal_logq[own_start:] - own_stored_logq))
    )
    if own_replay_error > 1.0e-10:
        raise RuntimeError(
            f"Student proposal-density replay failed: {own_replay_error:.3e}"
        )

    log_denominator_ratio = np.full(
        len(theta), math.log(arguments.base_proposals), dtype=np.float64
    )
    log_denominator_ratio = np.logaddexp(
        log_denominator_ratio,
        np.where(
            support,
            math.log(support_total_proposals),
            -np.inf,
        ),
    )
    log_denominator_ratio = np.logaddexp(
        log_denominator_ratio,
        math.log(proposal_total_rows) + proposal_logq - prior_logq,
    )
    log_correction = math.log(arguments.base_proposals) - log_denominator_ratio

    standardized = (prediction - A1_OBSERVATION) / A1_SIGMA
    log_kernel = -0.5 * np.sum(np.square(standardized), axis=1)
    log_weight = np.where(valid, log_correction + logastro + log_kernel, -np.inf)
    normalization = logsumexp(log_weight)
    normalized = np.exp(log_weight - normalization)
    conditional_ess = float(1.0 / np.sum(np.square(normalized)))
    maximum_normalized_weight = float(normalized.max())
    if conditional_ess < arguments.conditional_ess_gate:
        raise RuntimeError(
            f"conditional bank ESS {conditional_ess:.1f} is below "
            f"{arguments.conditional_ess_gate:.1f}"
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(
                "ddb-hyperonic-clean-augmented-counting-correction-v1"
            ),
            cache_sha256=np.asarray([record["sha256"] for record in records]),
            cache_rows=np.asarray(
                [record["rows"] for record in records], dtype=np.int64
            ),
            cache_roles=np.asarray([record["role"] for record in records]),
            log_prior_correction=log_correction,
            support_indicator=support,
            base_total_proposals=np.int64(arguments.base_proposals),
            support_total_proposals=np.int64(support_total_proposals),
            support_cbox=np.float64(support_cbox),
            support_screen_sha256=np.asarray(support_screen_sha256),
            proposal_total_rows=np.int64(proposal_total_rows),
            proposal_sha256=np.asarray(proposal_sha256),
            proposal_checkpoint_sha256=np.asarray(proposal_checkpoint_sha256),
            maximum_proposal_logq_replay_error=np.float64(own_replay_error),
            conditional_ess=np.float64(conditional_ess),
            conditional_ess_gate=np.float64(arguments.conditional_ess_gate),
            maximum_normalized_weight=np.float64(maximum_normalized_weight),
            reference_present=np.bool_(False),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
    os.replace(temporary, arguments.output)
    report = {
        "status": "PASS",
        "stage": "exact uniform-plus-support-plus-Student DM-MIS correction",
        "schema": "ddb-hyperonic-clean-augmented-counting-correction-v1",
        "caches": records,
        "rows": int(len(theta)),
        "base_total_proposals": arguments.base_proposals,
        "support_total_proposals": support_total_proposals,
        "support_cbox": support_cbox,
        "support_rows": int(support.sum()),
        "proposal_total_rows": proposal_total_rows,
        "proposal_sha256": proposal_sha256,
        "proposal_checkpoint_sha256": proposal_checkpoint_sha256,
        "maximum_proposal_logq_replay_error": own_replay_error,
        "conditional_ess_at_A1": conditional_ess,
        "conditional_ess_gate": arguments.conditional_ess_gate,
        "maximum_normalized_weight": maximum_normalized_weight,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "reference_present": False,
        "forbidden_artifacts_used": [],
        "wall_seconds": time.time() - started,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
