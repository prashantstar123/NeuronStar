#!/usr/bin/env python3
"""Build the held-out-certified GMM plus frozen-TSNPE 9D proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammaln, logsumexp

from build_dual_stage_flow_proposal_9d import (
    load_proposal,
    proposal_density,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def transform(theta, low, high, mean, scale):
    unit = (theta - low) / (high - low)
    if not np.all((unit > 0.0) & (unit < 1.0)):
        raise RuntimeError("bounded transform received a boundary row")
    value = (np.log(unit) - np.log1p(-unit) - mean) / scale
    jacobian = -np.log(scale).sum() - np.sum(
        np.log(high - low) + np.log(unit) + np.log1p(-unit), axis=1
    )
    return value, jacobian


def inverse_transform(value, low, high, mean, scale):
    logit = value * scale + mean
    unit = np.empty_like(logit)
    positive = logit >= 0.0
    unit[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    unit[~positive] = exponential / (1.0 + exponential)
    theta = low + (high - low) * unit
    if not np.all((theta > low) & (theta < high)):
        raise RuntimeError("proposal generated a numerical boundary row")
    return theta


def t_log_density(value, location, shape, degrees):
    dimension = value.shape[1]
    sign, logdet = np.linalg.slogdet(shape)
    if sign <= 0:
        raise RuntimeError("Student-t shape is not positive definite")
    delta = value - location
    quadratic = np.einsum("ni,ij,nj->n", delta, np.linalg.inv(shape), delta)
    constant = (
        gammaln((degrees + dimension) / 2.0)
        - gammaln(degrees / 2.0)
        - 0.5 * (dimension * np.log(degrees * np.pi) + logdet)
    )
    return constant - 0.5 * (degrees + dimension) * np.log1p(quadratic / degrees)


def gmm_log_density(value, weights, means, covariances):
    pieces = []
    dimension = value.shape[1]
    for weight, mean, covariance in zip(weights, means, covariances, strict=True):
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise RuntimeError("GMM covariance is not positive definite")
        delta = value - mean
        quadratic = np.einsum(
            "ni,ij,nj->n", delta, np.linalg.inv(covariance), delta
        )
        pieces.append(
            np.log(weight)
            - 0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic)
        )
    return logsumexp(np.column_stack(pieces), axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--parent-hybrid-model", type=Path, required=True)
    parser.add_argument("--old-proposal", type=Path, required=True)
    parser.add_argument("--old-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--new-proposal", type=Path, required=True)
    parser.add_argument("--new-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production flow-density replay")

    bridge_report = json.loads(arguments.bridge.with_suffix(".json").read_text())
    bridge_hash = sha256(arguments.bridge)
    if (
        bridge_report.get("status") != "PASS"
        or bridge_report.get("lineage_class") != "independent_tsnpe"
        or bridge_report.get("role") != "tsnpe_crossvalidated_gmm_defensive_bridge"
        or bridge_report.get("full_prior_dimension") != 9
        or bridge_report.get("forbidden_artifacts_used")
        or bridge_report.get("output_sha256") != bridge_hash
        or bridge_report.get("worst_held_out_predicted_ess", 0.0)
        < bridge_report.get("minimum_worst_predicted_ess", np.inf)
    ):
        raise RuntimeError("cross-validated bridge failed production gates")
    with np.load(arguments.bridge, allow_pickle=False) as archive:
        gmm_weights = np.asarray(archive["gmm_weights"], dtype=np.float64)
        gmm_means = np.asarray(archive["gmm_means"], dtype=np.float64)
        gmm_covariances = np.asarray(archive["gmm_covariances"], dtype=np.float64)
        mean = np.asarray(archive["transform_mean"], dtype=np.float64)
        scale = np.asarray(archive["transform_scale"], dtype=np.float64)
        broad_location = np.asarray(archive["broad_location"], dtype=np.float64)
        broad_shape = np.asarray(archive["broad_shape"], dtype=np.float64)
        broad_degrees = float(np.asarray(archive["broad_degrees"]).item())
        repair_fractions = np.asarray(archive["repair_fractions"], dtype=np.float64)
        parent_fraction = float(np.asarray(archive["parent_tsnpe_fraction"]).item())
        low = np.asarray(archive["prior_low"], dtype=np.float64)
        high = np.asarray(archive["prior_high"], dtype=np.float64)
        if np.asarray(archive["forbidden_artifacts_used"]).size:
            raise RuntimeError("cross-validated bridge declares forbidden ancestry")

    old_report, old_hash, old_flows, old_model = load_proposal(
        arguments.old_proposal, arguments.old_checkpoints, device
    )
    new_report, new_hash, new_flows, new_model = load_proposal(
        arguments.new_proposal, arguments.new_checkpoints, device
    )
    if (
        not np.array_equal(low, old_model["prior_low"])
        or not np.array_equal(high, old_model["prior_high"])
        or not np.array_equal(low, new_model["prior_low"])
        or not np.array_equal(high, new_model["prior_high"])
    ):
        raise RuntimeError("bridge and frozen flows use different full priors")

    parent_report = json.loads(
        arguments.parent_hybrid_model.with_suffix(".json").read_text()
    )
    parent_hash = sha256(arguments.parent_hybrid_model)
    if (
        parent_report.get("status") != "PASS"
        or parent_report.get("lineage_class") != "independent_tsnpe"
        or parent_report.get("role") != "tsnpe_optimized_hybrid_model"
        or parent_report.get("old_proposal_sha256") != old_hash
        or parent_report.get("new_proposal_sha256") != new_hash
        or parent_report.get("forbidden_artifacts_used")
        or parent_report.get("output_sha256") != parent_hash
    ):
        raise RuntimeError("parent TSNPE hybrid failed lineage gates")
    with np.load(arguments.parent_hybrid_model, allow_pickle=False) as archive:
        parent_weights = np.asarray(archive["optimized_weights"], dtype=np.float64)
        parent_low = np.asarray(archive["prior_low"], dtype=np.float64)
        parent_high = np.asarray(archive["prior_high"], dtype=np.float64)
        if np.asarray(archive["forbidden_artifacts_used"]).size:
            raise RuntimeError("parent TSNPE hybrid declares forbidden ancestry")
    if (
        parent_weights.shape != (5,)
        or not np.isclose(parent_weights.sum(), 1.0)
        or not np.array_equal(parent_low, low)
        or not np.array_equal(parent_high, high)
    ):
        raise RuntimeError("parent TSNPE hybrid parameters are invalid")

    rows = int(bridge_report["candidate_rows"])
    global_weights = np.concatenate(
        [
            (1.0 - parent_fraction) * repair_fractions,
            parent_fraction * parent_weights,
        ]
    )
    raw_counts = global_weights * rows
    counts = np.floor(raw_counts).astype(np.int64)
    for index in np.argsort(raw_counts - counts)[::-1][: rows - int(counts.sum())]:
        counts[index] += 1
    if int(counts.sum()) != rows or np.any(counts < 1):
        raise RuntimeError("invalid hybrid component counts")

    rng = np.random.default_rng(arguments.seed)

    def draw_gmm(count):
        component = rng.choice(len(gmm_weights), size=count, p=gmm_weights)
        output = np.empty((count, 9), dtype=np.float64)
        for index in range(len(gmm_weights)):
            selected = np.flatnonzero(component == index)
            if len(selected):
                output[selected] = rng.multivariate_normal(
                    gmm_means[index], gmm_covariances[index], size=len(selected)
                )
        return inverse_transform(output, low, high, mean, scale)

    def draw_t(location, shape, degrees, count, model_mean, model_scale):
        gaussian = rng.multivariate_normal(np.zeros(9), shape, size=count)
        chi_square = rng.chisquare(degrees, size=count)
        value = location + gaussian / np.sqrt(chi_square[:, None] / degrees)
        return inverse_transform(value, low, high, model_mean, model_scale)

    def draw_flow(flows, model, count, seed):
        base, remainder = divmod(count, len(flows))
        pieces = []
        for member, flow in enumerate(flows):
            local = base + (member < remainder)
            torch.manual_seed(seed + member)
            torch.cuda.manual_seed_all(seed + member)
            with torch.no_grad():
                value = flow().sample((local,)).detach().cpu().numpy().astype(np.float64)
            pieces.append(
                inverse_transform(
                    value,
                    low,
                    high,
                    model["transform_mean"],
                    model["transform_scale"],
                )
            )
        return np.vstack(pieces)

    pieces = [
        draw_gmm(int(counts[0])),
        draw_t(
            broad_location,
            broad_shape,
            broad_degrees,
            int(counts[1]),
            mean,
            scale,
        ),
        rng.uniform(low, high, size=(int(counts[2]), 9)),
        draw_flow(old_flows, old_model, int(counts[3]), arguments.seed + 10_000),
        draw_t(
            old_model["broad_location"],
            old_model["broad_shape"],
            old_model["broad_degrees"],
            int(counts[4]),
            old_model["transform_mean"],
            old_model["transform_scale"],
        ),
        draw_flow(new_flows, new_model, int(counts[5]), arguments.seed + 20_000),
        draw_t(
            new_model["broad_location"],
            new_model["broad_shape"],
            new_model["broad_degrees"],
            int(counts[6]),
            new_model["transform_mean"],
            new_model["transform_scale"],
        ),
        rng.uniform(low, high, size=(int(counts[7]), 9)),
    ]
    theta = np.vstack(pieces)
    component = np.concatenate(
        [np.full(len(piece), index, dtype=np.uint8) for index, piece in enumerate(pieces)]
    )
    permutation = rng.permutation(rows)
    theta = theta[permutation]
    component = component[permutation]

    value, jacobian = transform(theta, low, high, mean, scale)
    logq_gmm = gmm_log_density(value, gmm_weights, gmm_means, gmm_covariances) + jacobian
    logq_broad = t_log_density(
        value, broad_location, broad_shape, broad_degrees
    ) + jacobian
    logq_uniform = np.full(rows, -float(np.log(high - low).sum()), dtype=np.float64)
    logq_repair = logsumexp(
        np.column_stack(
            [
                np.log(repair_fractions[0]) + logq_gmm,
                np.log(repair_fractions[1]) + logq_broad,
                np.log(repair_fractions[2]) + logq_uniform,
            ]
        ),
        axis=1,
    )
    _, old_flow, old_broad, old_uniform = proposal_density(
        theta, old_flows, old_model, device, arguments.batch_size
    )
    _, new_flow, new_broad, new_uniform = proposal_density(
        theta, new_flows, new_model, device, arguments.batch_size
    )
    if np.max(np.abs(old_uniform - new_uniform)) > 1.0e-12:
        raise RuntimeError("frozen flow generations disagree on uniform density")
    logq_parent = logsumexp(
        np.column_stack(
            [
                np.log(parent_weights[0]) + old_flow,
                np.log(parent_weights[1]) + old_broad,
                np.log(parent_weights[2]) + new_flow,
                np.log(parent_weights[3]) + new_broad,
                np.log(parent_weights[4]) + old_uniform,
            ]
        ),
        axis=1,
    )
    logq_final = logsumexp(
        np.column_stack(
            [
                np.log1p(-parent_fraction) + logq_repair,
                np.log(parent_fraction) + logq_parent,
            ]
        ),
        axis=1,
    )
    if not np.isfinite(logq_final).all():
        raise RuntimeError("final hybrid proposal has non-finite density")

    metadata = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_crossvalidated_hybrid_full_prior_proposal_candidates",
        "model": "ddb-hyperonic",
        "full_prior_dimension": 9,
        "bridge_sha256": bridge_hash,
        "parent_hybrid_model_sha256": parent_hash,
        "old_proposal_sha256": old_hash,
        "new_proposal_sha256": new_hash,
        "component_order": [
            "crossfit_GMM",
            "crossfit_defensive_Student_t",
            "crossfit_uniform_full_prior",
            "parent_old_flow",
            "parent_old_Student_t",
            "parent_new_flow",
            "parent_new_Student_t",
            "parent_uniform_full_prior",
        ],
        "component_counts": counts.tolist(),
        "component_weights": global_weights.tolist(),
        "held_out_validation_diagnostics": bridge_report["validation_stages"],
        "worst_held_out_predicted_ess": bridge_report[
            "worst_held_out_predicted_ess"
        ],
        "seed": arguments.seed,
        "forbidden_artifacts_used": [],
    }
    atomic_save(
        arguments.output,
        theta=theta,
        log_proposal=logq_final,
        logq_repair=logq_repair,
        logq_parent_tsnpe=logq_parent,
        proposal_component=component,
        component_counts=counts,
        component_weights=global_weights,
        prior_low=low,
        prior_high=high,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    report = {
        **metadata,
        "candidate_rows": rows,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
