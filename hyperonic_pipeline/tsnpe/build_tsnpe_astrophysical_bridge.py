#!/usr/bin/env python3
"""Build a TSNPE-owned full-prior bridge for the hyperonic A1 target.

The bridge is fitted directly from the sampler-neutral prior-generated physics
caches.  Its target is the fixed astrophysical likelihood only; the nuclear
observation is deliberately excluded because it is simulated after rejection
sampling for conditional neural-posterior training.  A uniform-prior component
keeps the proposal density positive over the complete canonical 9D prior.
"""

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


UNIFORM_SCHEMA = "ddb-hyperonic-clean-uniform-training-cache-v1"
SUPPORT_SCHEMA = "ddb-hyperonic-clean-support-training-cache-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_joblib(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_cache(path: Path, schema: str) -> dict[str, np.ndarray]:
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
            raise RuntimeError(f"{path} lacks {sorted(missing)}")
        if str(archive["schema"].item()) != schema:
            raise RuntimeError(f"unexpected cache schema in {path}")
        if bool(archive["reference_present"]):
            raise RuntimeError(f"reference inference artifact declared by {path}")
        if np.asarray(archive["forbidden_artifacts_used"]).size:
            raise RuntimeError(f"forbidden inference ancestry declared by {path}")
        payload = {
            key: np.asarray(archive[key])
            for key in (
                "theta",
                "prediction",
                "log_astrophysical",
                "target_valid",
                "prior_low",
                "prior_high",
            )
        }
        if "total_proposals" in archive.files:
            payload["total_proposals"] = np.asarray(archive["total_proposals"])
        if "cbox" in archive.files:
            payload["cbox"] = np.asarray(archive["cbox"])
    rows = len(payload["theta"])
    if (
        payload["theta"].shape != (rows, 9)
        or payload["prediction"].shape != (rows, 7)
        or payload["log_astrophysical"].shape != (rows,)
        or payload["target_valid"].shape != (rows,)
    ):
        raise RuntimeError(f"invalid cache arrays in {path}")
    return payload


def to_unconstrained(
    theta: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    coordinate = 2.0 * (theta - low) / (high - low) - 1.0
    coordinate = np.clip(coordinate, -1.0 + 1.0e-12, 1.0 - 1.0e-12)
    return np.arctanh(coordinate)


def from_unconstrained(
    value: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    coordinate = np.tanh(value)
    theta = low + 0.5 * (coordinate + 1.0) * (high - low)
    if not np.all((theta > low) & (theta < high)):
        raise RuntimeError("GMM transform reached the numerical prior boundary")
    return theta


def log_transform_jacobian(
    theta: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    coordinate = 2.0 * (theta - low) / (high - low) - 1.0
    if not np.all(np.abs(coordinate) < 1.0):
        raise RuntimeError("proposal density requested outside the open prior")
    return (
        np.log(2.0 / (high - low))[None, :]
        - np.log1p(-(coordinate * coordinate))
    ).sum(axis=1)


def model_log_density(
    specification: dict[str, object],
    theta: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    transformed = to_unconstrained(theta, low, high)
    mean = np.asarray(specification["mean"], dtype=np.float64)
    scale = np.asarray(specification["scale"], dtype=np.float64)
    standardized = (transformed - mean) / scale
    return (
        specification["model"].score_samples(standardized)
        - np.log(scale).sum()
        + log_transform_jacobian(theta, low, high)
    )


def ensemble_log_density(
    models: list[dict[str, object]],
    theta: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    chunk: int = 20_000,
) -> np.ndarray:
    output = np.empty(len(theta), dtype=np.float64)
    for start in range(0, len(theta), chunk):
        stop = min(start + chunk, len(theta))
        terms = np.vstack(
            [
                model_log_density(model, theta[start:stop], low, high)
                for model in models
            ]
        )
        output[start:stop] = logsumexp(terms, axis=0) - math.log(len(models))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform-cache", type=Path, action="append", required=True)
    parser.add_argument("--support-cache", type=Path, action="append", required=True)
    parser.add_argument("--bridge-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-proposals", type=int, default=600_000)
    parser.add_argument("--support-proposals", type=int, default=160_000_000)
    parser.add_argument("--support-radius", type=float, default=3.0)
    parser.add_argument("--pilot", type=int, default=20_000)
    parser.add_argument("--count", type=int, default=80_000)
    parser.add_argument("--training-rows", type=int, default=120_000)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--defensive-fraction", type=float, default=0.10)
    parser.add_argument("--regularization", type=float, default=2.0e-3)
    parser.add_argument("--jitter", type=float, default=0.03)
    parser.add_argument("--maximum-iterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260921)
    arguments = parser.parse_args()
    if arguments.pilot < 1 or arguments.count < 1:
        raise ValueError("pilot and training candidate counts must be positive")
    if not 0.0 < arguments.defensive_fraction < 1.0:
        raise ValueError("defensive fraction must lie strictly between zero and one")
    for path in (arguments.bridge_cache, arguments.checkpoint, arguments.output):
        if path.exists():
            raise FileExistsError(f"refusing to replace {path}")

    started = time.time()
    uniform = [load_cache(path, UNIFORM_SCHEMA) for path in arguments.uniform_cache]
    support = [load_cache(path, SUPPORT_SCHEMA) for path in arguments.support_cache]
    all_inputs = uniform + support
    low = np.asarray(uniform[0]["prior_low"], dtype=np.float64)
    high = np.asarray(uniform[0]["prior_high"], dtype=np.float64)
    if low.shape != (9,) or high.shape != (9,) or np.any(low >= high):
        raise RuntimeError("canonical hyperonic prior is invalid")
    for item in all_inputs:
        if not (
            np.array_equal(item["prior_low"], low)
            and np.array_equal(item["prior_high"], high)
        ):
            raise RuntimeError("input caches do not share the canonical prior")
    for item in support:
        if int(item["total_proposals"]) != arguments.support_proposals:
            raise RuntimeError("support proposal count changed")
        if float(item["cbox"]) != arguments.support_radius:
            raise RuntimeError("support radius changed")

    theta = np.concatenate(
        [np.asarray(item["theta"], dtype=np.float64) for item in all_inputs]
    )
    prediction = np.concatenate(
        [np.asarray(item["prediction"], dtype=np.float64) for item in all_inputs]
    )
    log_astrophysical = np.concatenate(
        [
            np.asarray(item["log_astrophysical"], dtype=np.float64)
            for item in all_inputs
        ]
    )
    valid = np.concatenate(
        [np.asarray(item["target_valid"], dtype=bool) for item in all_inputs]
    )
    if not np.all((theta >= low) & (theta <= high)):
        raise RuntimeError("bridge source has an out-of-prior parameter row")

    observation = np.asarray(
        [
            0.153,
            -16.1,
            230.0,
            32.5,
            0.505714285714279,
            1.24142857142857,
            2.4857142857143,
        ],
        dtype=np.float64,
    )
    sigma = np.asarray(
        [
            0.005,
            0.2,
            40.0,
            1.8,
            0.194285714285714,
            0.608571428571429,
            1.38285714285714,
        ],
        dtype=np.float64,
    )
    support_indicator = np.all(
        np.abs((prediction - observation) / sigma) <= arguments.support_radius,
        axis=1,
    )
    support_start = sum(len(item["theta"]) for item in uniform)
    if not support_indicator[support_start:].all():
        raise RuntimeError("deterministic support row lies outside its declared box")
    correction = math.log(arguments.base_proposals) - np.log(
        arguments.base_proposals
        + arguments.support_proposals * support_indicator.astype(np.float64)
    )
    log_astrophysical = np.where(valid, log_astrophysical, -np.inf)
    metadata = {
        "schema_version": 1,
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_owned_astrophysical_bridge_source",
        "model": "ddb-hyperonic",
        "full_prior_dimension": 9,
        "target": "uniform_prior_times_fixed_astrophysical_likelihood",
        "nuclear_observation_used_in_bridge_target": False,
        "base_total_proposals": arguments.base_proposals,
        "support_total_proposals": arguments.support_proposals,
        "support_radius_sigma": arguments.support_radius,
        "parents": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (*arguments.uniform_cache, *arguments.support_cache)
        ],
        "forbidden_artifacts_used": [],
    }
    atomic_savez(
        arguments.bridge_cache,
        theta=theta,
        log_astrophysical=log_astrophysical,
        log_proposal_correction=correction,
        prior_low=low,
        prior_high=high,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )

    log_weight = correction + log_astrophysical
    finite = np.isfinite(log_weight)
    normalized = np.zeros(len(log_weight), dtype=np.float64)
    normalized[finite] = np.exp(log_weight[finite] - logsumexp(log_weight[finite]))
    source_ess = float(1.0 / np.sum(normalized * normalized))
    if source_ess < 1_000.0:
        raise RuntimeError(f"astrophysical bridge source ESS is too small: {source_ess}")

    models: list[dict[str, object]] = []
    transformed_all = to_unconstrained(theta, low, high)
    for member in range(arguments.ensemble_members):
        rng = np.random.default_rng(arguments.seed + 1000 * member)
        index = rng.choice(
            len(theta), size=arguments.training_rows, replace=True, p=normalized
        )
        transformed = transformed_all[index]
        mean = transformed.mean(axis=0)
        scale = transformed.std(axis=0)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 1.0e-10):
            raise RuntimeError("degenerate bridge standardization")
        standardized = (transformed - mean) / scale
        standardized += rng.normal(0.0, arguments.jitter, standardized.shape)
        model = GaussianMixture(
            n_components=arguments.components,
            covariance_type="full",
            reg_covar=arguments.regularization,
            max_iter=arguments.maximum_iterations,
            n_init=2,
            init_params="k-means++",
            random_state=arguments.seed + 1000 * member,
        )
        model.fit(standardized)
        if not model.converged_:
            raise RuntimeError(f"bridge GMM member {member} did not converge")
        models.append(
            {
                "model": model,
                "mean": mean,
                "scale": scale,
                "iterations": int(model.n_iter_),
            }
        )

    checkpoint = {
        "schema": "ddb-hyperonic-independent-tsnpe-astro-bridge-v1",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_owned_normalized_astrophysical_bridge",
        "models": models,
        "prior_low": low,
        "prior_high": high,
        "defensive_fraction": arguments.defensive_fraction,
        "source_cache_sha256": sha256(arguments.bridge_cache),
        "seed": arguments.seed,
        "forbidden_artifacts_used": [],
    }
    atomic_joblib(arguments.checkpoint, checkpoint)

    total = arguments.pilot + arguments.count
    defensive_rows = int(round(total * arguments.defensive_fraction))
    learned_rows = total - defensive_rows
    counts = np.full(len(models), learned_rows // len(models), dtype=int)
    counts[: learned_rows % len(models)] += 1
    pieces: list[np.ndarray] = []
    component: list[np.ndarray] = []
    for member, (specification, count) in enumerate(zip(models, counts, strict=True)):
        model = specification["model"]
        model.random_state = arguments.seed + 10_000 + member
        standardized, _ = model.sample(int(count))
        transformed = (
            standardized * np.asarray(specification["scale"])[None, :]
            + np.asarray(specification["mean"])[None, :]
        )
        pieces.append(from_unconstrained(transformed, low, high))
        component.append(np.full(int(count), member, dtype=np.uint8))
    rng = np.random.default_rng(arguments.seed + 20_000)
    pieces.append(rng.uniform(low, high, size=(defensive_rows, 9)))
    component.append(np.full(defensive_rows, len(models), dtype=np.uint8))
    proposal = np.vstack(pieces)
    proposal_component = np.concatenate(component)
    permutation = rng.permutation(total)
    proposal = proposal[permutation]
    proposal_component = proposal_component[permutation]

    learned_logq = ensemble_log_density(models, proposal, low, high)
    prior_logq = -float(np.log(high - low).sum())
    logq = np.logaddexp(
        math.log1p(-arguments.defensive_fraction) + learned_logq,
        math.log(arguments.defensive_fraction) + prior_logq,
    )
    if not np.isfinite(logq).all():
        raise RuntimeError("TSNPE bridge produced a non-finite normalized density")

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
        "uniform_prior_weight": arguments.defensive_fraction,
        "bridge_target_includes_nuclear_observation": False,
        "source_cache": str(arguments.bridge_cache.resolve()),
        "source_cache_sha256": sha256(arguments.bridge_cache),
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
        "ensemble_members": arguments.ensemble_members,
        "components_per_member": arguments.components,
        "training_rows_per_member": arguments.training_rows,
        "fit_iterations": [int(item["iterations"]) for item in models],
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
