#!/usr/bin/env python3
"""Combine independent draws from one frozen normalized proposal.

This utility deliberately refuses to combine evaluations unless their prior,
tempering beta, and proposal checkpoint are identical.  Under that condition
all rows are iid draws from the same normalized density and may be
concatenated without multiple-proposal reweighting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SCHEMA = "ddb-hyperonic-clean-eval-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_ess(logweight: np.ndarray) -> float:
    finite = np.isfinite(logweight)
    if not finite.any():
        return 0.0
    shifted = logweight[finite] - np.max(logweight[finite])
    weight = np.exp(shifted)
    weight /= weight.sum()
    return float(1.0 / np.sum(weight * weight))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if len(arguments.input) < 2:
        raise ValueError("at least two --input evaluations are required")

    theta_parts: list[np.ndarray] = []
    logq_parts: list[np.ndarray] = []
    loglike_parts: list[np.ndarray] = []
    manifest: list[dict[str, object]] = []
    reference: dict[str, object] | None = None

    for path in arguments.input:
        with np.load(path, allow_pickle=False) as source:
            if str(source["schema"].item()) != SCHEMA:
                raise RuntimeError(f"unrecognized clean evaluation: {path}")
            theta = np.asarray(source["theta"], dtype=np.float64)
            logq = np.asarray(source["logq"], dtype=np.float64)
            loglike = np.asarray(source["logl"], dtype=np.float64)
            lower = np.asarray(source["prior_low"], dtype=np.float64)
            upper = np.asarray(source["prior_high"], dtype=np.float64)
            beta = float(source["proposal_beta"])
            checkpoint = str(source["checkpoint_sha256"].item())
            proposal_sha = str(source["proposal_sha256"].item())
            likelihood_sha = str(source["likelihood_sha256"].item())
        if theta.ndim != 2 or logq.shape != (len(theta),) or loglike.shape != (len(theta),):
            raise RuntimeError(f"invalid evaluation arrays: {path}")
        identity: dict[str, object] = {
            "prior_low": lower,
            "prior_high": upper,
            "proposal_beta": beta,
            "checkpoint_sha256": checkpoint,
        }
        if reference is None:
            reference = identity
        else:
            if beta != reference["proposal_beta"]:
                raise RuntimeError("proposal betas differ; MIS would be required")
            if checkpoint != reference["checkpoint_sha256"]:
                raise RuntimeError("proposal checkpoints differ; MIS would be required")
            if not np.array_equal(lower, reference["prior_low"]):
                raise RuntimeError("prior lower bounds differ")
            if not np.array_equal(upper, reference["prior_high"]):
                raise RuntimeError("prior upper bounds differ")
        theta_parts.append(theta)
        logq_parts.append(logq)
        loglike_parts.append(loglike)
        manifest.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "rows": int(len(theta)),
                "proposal_sha256": proposal_sha,
                "likelihood_sha256": likelihood_sha,
            }
        )

    assert reference is not None
    theta = np.concatenate(theta_parts)
    logq = np.concatenate(logq_parts)
    loglike = np.concatenate(loglike_parts)
    lower = np.asarray(reference["prior_low"], dtype=np.float64)
    upper = np.asarray(reference["prior_high"], dtype=np.float64)
    beta = float(reference["proposal_beta"])
    checkpoint = str(reference["checkpoint_sha256"])
    valid = loglike > -1.0e50
    logprior = -float(np.log(upper - lower).sum())
    current_ess = normalized_ess(logprior + beta * loglike[valid] - logq[valid])
    posterior_ess = normalized_ess(logprior + loglike[valid] - logq[valid])
    manifest_json = json.dumps(manifest, sort_keys=True)
    manifest_sha = hashlib.sha256(manifest_json.encode()).hexdigest()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(SCHEMA),
            theta=theta,
            logq=logq,
            logl=loglike,
            prior_low=lower,
            prior_high=upper,
            proposal_beta=np.float64(beta),
            checkpoint_sha256=np.asarray(checkpoint),
            proposal_sha256=np.asarray(f"combined-manifest:{manifest_sha}"),
            likelihood_sha256=np.asarray(f"combined-manifest:{manifest_sha}"),
            valid_rows=np.int64(valid.sum()),
            posterior_ess=np.float64(posterior_ess),
            source_manifest_json=np.asarray(manifest_json),
        )
    os.replace(temporary, arguments.output)

    report = {
        "status": "PASS",
        "rows": int(len(theta)),
        "valid_rows": int(valid.sum()),
        "proposal_beta": beta,
        "current_temperature_ess": current_ess,
        "posterior_ess": posterior_ess,
        "checkpoint_sha256": checkpoint,
        "source_manifest_sha256": manifest_sha,
        "output_sha256": sha256(arguments.output),
        "sources": manifest,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
