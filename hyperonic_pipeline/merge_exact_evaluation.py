#!/usr/bin/env python3
"""Merge one clean proposal with its fresh complete-likelihood evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--likelihood", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    with np.load(arguments.proposal, allow_pickle=False) as proposal:
        if str(proposal["schema"].item()) != "ddb-hyperonic-clean-proposal-v1":
            raise RuntimeError("unrecognized clean proposal")
        theta = np.asarray(proposal["theta"], dtype=np.float64)
        logq = np.asarray(proposal["logq"], dtype=np.float64)
        lower = np.asarray(proposal["prior_low"], dtype=np.float64)
        upper = np.asarray(proposal["prior_high"], dtype=np.float64)
        beta = float(proposal["beta"])
        checkpoint_sha = str(proposal["checkpoint_sha256"].item())
    with np.load(arguments.likelihood, allow_pickle=False) as evaluated:
        theta_evaluated = np.asarray(evaluated["theta"], dtype=np.float64)
        loglike = np.asarray(evaluated["logl"], dtype=np.float64)
        reference_present = bool(evaluated.get("reference_present", np.asarray(False)))
    if reference_present:
        raise RuntimeError("clean likelihood file unexpectedly contains reference values")
    if not np.array_equal(theta, theta_evaluated):
        raise RuntimeError("proposal and exact-likelihood rows differ")
    if logq.shape != (len(theta),) or loglike.shape != (len(theta),):
        raise RuntimeError("invalid evaluation-array shapes")
    valid = loglike > -1.0e50
    if not valid.any():
        raise RuntimeError("complete likelihood rejected every proposal row")
    logprior = -float(np.log(upper - lower).sum())
    final_logweight = logprior + loglike[valid] - logq[valid]
    shifted = final_logweight - np.max(final_logweight)
    weight = np.exp(shifted)
    weight /= weight.sum()
    posterior_ess = float(1.0 / np.sum(weight * weight))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray("ddb-hyperonic-clean-eval-v1"),
            theta=theta,
            logq=logq,
            logl=loglike,
            prior_low=lower,
            prior_high=upper,
            proposal_beta=np.float64(beta),
            checkpoint_sha256=np.asarray(checkpoint_sha),
            proposal_sha256=np.asarray(sha256(arguments.proposal)),
            likelihood_sha256=np.asarray(sha256(arguments.likelihood)),
            valid_rows=np.int64(valid.sum()),
            posterior_ess=np.float64(posterior_ess),
        )
    os.replace(temporary, arguments.output)
    report = {
        "status": "PASS",
        "rows": len(theta),
        "valid_rows": int(valid.sum()),
        "proposal_beta": beta,
        "posterior_ess": posterior_ess,
        "proposal_sha256": sha256(arguments.proposal),
        "likelihood_sha256": sha256(arguments.likelihood),
        "output_sha256": sha256(arguments.output),
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
