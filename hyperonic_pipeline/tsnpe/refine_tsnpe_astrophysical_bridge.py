#!/usr/bin/env python3
"""Add TSNPE-owned residual components to a clean astrophysical bridge."""

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
from sklearn.mixture import GaussianMixture

from build_tsnpe_astrophysical_bridge import (
    atomic_joblib,
    atomic_savez,
    ensemble_log_density,
    from_unconstrained,
    to_unconstrained,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, probabilities: tuple[float, ...]
) -> list[float]:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return [
        float(sorted_values[min(np.searchsorted(cumulative, q), len(values) - 1)])
        for q in probabilities
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=int, default=20_000)
    parser.add_argument("--count", type=int, default=200_000)
    parser.add_argument("--residual-members", type=int, default=3)
    parser.add_argument("--residual-components", type=int, default=16)
    parser.add_argument("--residual-training-rows", type=int, default=160_000)
    parser.add_argument("--residual-cap-quantile", type=float, default=0.999)
    parser.add_argument("--regularization", type=float, default=2.0e-3)
    parser.add_argument("--jitter", type=float, default=0.025)
    parser.add_argument("--maximum-iterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260924)
    arguments = parser.parse_args()
    if arguments.pilot < 1 or arguments.count < 1:
        raise ValueError("pilot and training candidate counts must be positive")
    if arguments.residual_members < 1 or arguments.residual_components < 1:
        raise ValueError("residual mixture configuration must be positive")
    if not 0.9 <= arguments.residual_cap_quantile < 1.0:
        raise ValueError("residual cap quantile must lie in [0.9,1)")
    for path in (arguments.checkpoint, arguments.output):
        if path.exists():
            raise FileExistsError(f"refusing to replace {path}")

    started = time.time()
    base = joblib.load(arguments.base_checkpoint)
    if base.get("lineage_class") != "independent_tsnpe":
        raise RuntimeError("base bridge lacks independent TSNPE lineage")
    if base.get("forbidden_artifacts_used"):
        raise RuntimeError("base bridge declares forbidden ancestry")
    if base.get("source_cache_sha256") != sha256(arguments.source_cache):
        raise RuntimeError("base bridge and source cache hashes do not match")

    with np.load(arguments.source_cache, allow_pickle=False) as archive:
        required = {
            "theta",
            "log_astrophysical",
            "log_proposal_correction",
            "prior_low",
            "prior_high",
            "metadata",
            "forbidden_artifacts_used",
        }
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"bridge source lacks {sorted(missing)}")
        theta = np.asarray(archive["theta"], dtype=np.float64)
        log_astrophysical = np.asarray(
            archive["log_astrophysical"], dtype=np.float64
        )
        correction = np.asarray(
            archive["log_proposal_correction"], dtype=np.float64
        )
        low = np.asarray(archive["prior_low"], dtype=np.float64)
        high = np.asarray(archive["prior_high"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata"].item()))
        forbidden = np.asarray(archive["forbidden_artifacts_used"])
    if metadata.get("lineage_class") != "independent_tsnpe" or forbidden.size:
        raise RuntimeError("bridge source lacks clean TSNPE lineage")
    if metadata.get("nuclear_observation_used_in_bridge_target") is not False:
        raise RuntimeError("bridge source target unexpectedly contains nuclear data")
    if theta.shape[1:] != (9,) or low.shape != (9,) or high.shape != (9,):
        raise RuntimeError("bridge source arrays have invalid dimensions")

    log_weight = correction + log_astrophysical
    finite = np.isfinite(log_weight)
    normalized = np.zeros(len(theta), dtype=np.float64)
    normalized[finite] = np.exp(
        log_weight[finite] - logsumexp(log_weight[finite])
    )
    source_ess = float(1.0 / np.sum(normalized * normalized))
    log_evidence = float(logsumexp(log_weight[finite]) - math.log(600_000.0))
    log_prior = -float(np.log(high - low).sum())
    defensive_fraction = float(base["defensive_fraction"])
    transformed_all = to_unconstrained(theta, low, high)
    models = list(base["models"])
    diagnostic_probabilities = (0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999)
    diagnostics = []

    for residual_index in range(arguments.residual_members):
        learned = ensemble_log_density(models, theta, low, high)
        logq = np.logaddexp(
            math.log1p(-defensive_fraction) + learned,
            math.log(defensive_fraction) + log_prior,
        )
        log_ratio = log_prior + log_astrophysical - log_evidence - logq
        diagnostic = weighted_quantiles(
            log_ratio[finite], normalized[finite], diagnostic_probabilities
        )
        cap = float(np.quantile(log_ratio[finite], arguments.residual_cap_quantile))
        residual_log_weight = np.full(len(theta), -np.inf, dtype=np.float64)
        residual_log_weight[finite] = (
            log_weight[finite]
            - logsumexp(log_weight[finite])
            + np.minimum(log_ratio[finite], cap)
        )
        residual_log_weight[finite] -= logsumexp(residual_log_weight[finite])
        residual_weight = np.zeros(len(theta), dtype=np.float64)
        residual_weight[finite] = np.exp(residual_log_weight[finite])
        rng = np.random.default_rng(arguments.seed + 1000 * residual_index)
        index = rng.choice(
            len(theta),
            size=arguments.residual_training_rows,
            replace=True,
            p=residual_weight,
        )
        transformed = transformed_all[index]
        mean = transformed.mean(axis=0)
        scale = transformed.std(axis=0)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 1.0e-10):
            raise RuntimeError("residual bridge standardization is degenerate")
        standardized = (transformed - mean) / scale
        standardized += rng.normal(0.0, arguments.jitter, standardized.shape)
        model = GaussianMixture(
            n_components=arguments.residual_components,
            covariance_type="full",
            reg_covar=arguments.regularization,
            max_iter=arguments.maximum_iterations,
            n_init=2,
            init_params="k-means++",
            random_state=arguments.seed + 1000 * residual_index,
        )
        model.fit(standardized)
        if not model.converged_:
            raise RuntimeError(f"residual bridge member {residual_index} did not converge")
        models.append(
            {
                "model": model,
                "mean": mean,
                "scale": scale,
                "iterations": int(model.n_iter_),
            }
        )
        diagnostics.append(
            {
                "stage": residual_index,
                "pre_fit_target_weighted_log_ratio_quantiles": dict(
                    zip((str(q) for q in diagnostic_probabilities), diagnostic)
                ),
                "residual_log_ratio_cap": cap,
                "fit_iterations": int(model.n_iter_),
            }
        )

    learned = ensemble_log_density(models, theta, low, high)
    source_logq = np.logaddexp(
        math.log1p(-defensive_fraction) + learned,
        math.log(defensive_fraction) + log_prior,
    )
    final_log_ratio = (
        log_prior + log_astrophysical - log_evidence - source_logq
    )
    final_quantiles = weighted_quantiles(
        final_log_ratio[finite], normalized[finite], diagnostic_probabilities
    )
    checkpoint = {
        "schema": "ddb-hyperonic-independent-tsnpe-residual-astro-bridge-v1",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_owned_normalized_residual_astrophysical_bridge",
        "models": models,
        "base_members": len(base["models"]),
        "residual_members": arguments.residual_members,
        "prior_low": low,
        "prior_high": high,
        "defensive_fraction": defensive_fraction,
        "source_cache_sha256": sha256(arguments.source_cache),
        "base_checkpoint_sha256": sha256(arguments.base_checkpoint),
        "seed": arguments.seed,
        "forbidden_artifacts_used": [],
    }
    atomic_joblib(arguments.checkpoint, checkpoint)

    total = arguments.pilot + arguments.count
    defensive_rows = int(round(total * defensive_fraction))
    learned_rows = total - defensive_rows
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
    proposal = np.vstack(pieces)
    proposal_component = np.concatenate(components)
    permutation = rng.permutation(total)
    proposal = proposal[permutation]
    proposal_component = proposal_component[permutation]
    learned_logq = ensemble_log_density(models, proposal, low, high)
    logq = np.logaddexp(
        math.log1p(-defensive_fraction) + learned_logq,
        math.log(defensive_fraction) + log_prior,
    )
    if not np.isfinite(logq).all():
        raise RuntimeError("refined TSNPE proposal has a non-finite density")

    proposal_metadata = {
        "schema_version": 1,
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_round_proposal",
        "proposal_mode": "normalized_defensive_astrophysical_bridge",
        "model": "ddb-hyperonic",
        "round_index": 0,
        "pilot_rows": arguments.pilot,
        "training_candidate_rows": arguments.count,
        "full_prior_dimension": 9,
        "uniform_prior_weight": defensive_fraction,
        "bridge_target_includes_nuclear_observation": False,
        "source_cache": str(arguments.source_cache.resolve()),
        "source_cache_sha256": sha256(arguments.source_cache),
        "base_checkpoint": str(arguments.base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha256(arguments.base_checkpoint),
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": sha256(arguments.checkpoint),
        "seed": arguments.seed,
        "forbidden_artifacts_used": [],
    }
    atomic_savez(
        arguments.output,
        theta=proposal,
        logq=logq,
        proposal_component=proposal_component,
        prior_low=low,
        prior_high=high,
        pilot_rows=np.asarray(arguments.pilot, dtype=np.int64),
        metadata=np.asarray(json.dumps(proposal_metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    report = {
        "status": "PASS",
        **proposal_metadata,
        "source_rows": int(len(theta)),
        "source_valid_rows": int(finite.sum()),
        "source_astrophysical_ess": source_ess,
        "base_members": len(base["models"]),
        "residual_members": arguments.residual_members,
        "total_learned_members": len(models),
        "residual_diagnostics": diagnostics,
        "final_target_weighted_log_ratio_quantiles": dict(
            zip((str(q) for q in diagnostic_probabilities), final_quantiles)
        ),
        "final_maximum_source_log_ratio": float(np.max(final_log_ratio[finite])),
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "wall_seconds": time.time() - started,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
