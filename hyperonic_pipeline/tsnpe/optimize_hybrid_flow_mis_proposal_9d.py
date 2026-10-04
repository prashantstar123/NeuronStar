#!/usr/bin/env python3
"""Optimize and sample a defensive hybrid of two clean TSNPE flow generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

from build_dual_stage_flow_proposal_9d import (
    atomic_save,
    load_proposal,
    proposal_density,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inverse_transform(value, model):
    logit = value * model["transform_scale"] + model["transform_mean"]
    unit = np.empty_like(logit)
    positive = logit >= 0.0
    unit[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    unit[~positive] = exponential / (1.0 + exponential)
    theta = model["prior_low"] + (model["prior_high"] - model["prior_low"]) * unit
    if not np.all((theta > model["prior_low"]) & (theta < model["prior_high"])):
        raise RuntimeError("hybrid proposal generated a numerical boundary row")
    return theta


def load_exact_stage(path: Path):
    report = json.loads(path.with_suffix(".json").read_text())
    path_hash = sha256(path)
    if (
        report.get("lineage_class") != "independent_tsnpe"
        or report.get("status") not in {"PASS", "FAIL_ESS_GATE"}
        or report.get("target_gate", {}).get("status") != "PASS"
        or report.get("forbidden_artifacts_used")
        or report.get("output_sha256") != path_hash
    ):
        raise RuntimeError(f"exact bridge stage failed provenance gates: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {
            key: np.asarray(archive[key], dtype=np.float64)
            for key in (
                "theta",
                "log_target",
                "log_proposal",
                "prior_low",
                "prior_high",
            )
        }
    return report, path_hash, data


def component_log_densities(theta, flows, model, device, batch_size):
    _, qflow, qbroad, quniform = proposal_density(
        theta, flows, model, device, batch_size
    )
    return qflow, qbroad, quniform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-proposal", type=Path, required=True)
    parser.add_argument("--old-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--new-proposal", type=Path, required=True)
    parser.add_argument("--new-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--old-exact-stage", type=Path, required=True)
    parser.add_argument("--validation-stage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-rows", type=int, default=100_000)
    parser.add_argument("--minimum-predicted-ess", type=float, default=1_200.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise FileExistsError(f"refusing to replace {arguments.output_dir}")
    if arguments.candidate_rows < 1:
        raise ValueError("--candidate-rows must be positive")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for hybrid-flow density evaluation")

    old_report, old_proposal_hash, old_flows, old_model = load_proposal(
        arguments.old_proposal, arguments.old_checkpoints, device
    )
    new_report, new_proposal_hash, new_flows, new_model = load_proposal(
        arguments.new_proposal, arguments.new_checkpoints, device
    )
    if (
        not np.array_equal(old_model["prior_low"], new_model["prior_low"])
        or not np.array_equal(old_model["prior_high"], new_model["prior_high"])
    ):
        raise RuntimeError("flow generations use different full priors")

    old_exact_report, old_exact_hash, old_exact = load_exact_stage(
        arguments.old_exact_stage
    )
    validation_report, validation_hash, validation = load_exact_stage(
        arguments.validation_stage
    )
    for stage in (old_exact, validation):
        if (
            not np.array_equal(stage["prior_low"], old_model["prior_low"])
            or not np.array_equal(stage["prior_high"], old_model["prior_high"])
        ):
            raise RuntimeError("bridge stage uses a different full prior")

    bridge_payload = []
    for name, report, stage in (
        ("old_exact", old_exact_report, old_exact),
        ("validation", validation_report, validation),
    ):
        old_mix, old_flow, old_broad, uniform = proposal_density(
            stage["theta"], old_flows, old_model, device, arguments.batch_size
        )
        _, new_flow, new_broad, new_uniform = proposal_density(
            stage["theta"], new_flows, new_model, device, arguments.batch_size
        )
        if np.max(np.abs(uniform - new_uniform)) > 1.0e-12:
            raise RuntimeError("flow generations disagree on uniform full-prior density")
        components = np.column_stack(
            [old_flow, old_broad, new_flow, new_broad, uniform]
        )
        bridge_payload.append(
            {
                "name": name,
                "report": report,
                "stage": stage,
                "old_mix": old_mix,
                "components": components,
            }
        )

    old_rows = len(old_exact["theta"])
    total_rows = old_rows + arguments.candidate_rows
    outer_fractions = np.asarray(
        [old_rows / total_rows, arguments.candidate_rows / total_rows],
        dtype=np.float64,
    )
    lower = np.asarray([0.04, 0.10, 0.04, 0.10, 0.02], dtype=np.float64)

    def stage_metric(weights, payload):
        qhybrid = logsumexp(
            np.log(weights)[None, :] + payload["components"], axis=1
        )
        qtotal = logsumexp(
            np.column_stack(
                [
                    np.log(outer_fractions[0]) + payload["old_mix"],
                    np.log(outer_fractions[1]) + qhybrid,
                ]
            ),
            axis=1,
        )
        stage = payload["stage"]
        valid = (
            np.isfinite(stage["log_target"])
            & np.isfinite(stage["log_proposal"])
            & np.isfinite(qtotal)
        )
        log_second = float(
            logsumexp(
                2.0 * stage["log_target"][valid]
                - qtotal[valid]
                - stage["log_proposal"][valid]
            )
            - np.log(len(stage["theta"]))
        )
        normalized = log_second - 2.0 * float(payload["report"]["log_evidence"])
        return normalized, int(valid.sum()), qhybrid, qtotal

    def objective(weights):
        if np.any(weights <= 0.0):
            return 1.0e6
        metrics = np.asarray([stage_metric(weights, item)[0] for item in bridge_payload])
        return float(logsumexp(20.0 * metrics) / 20.0)

    starts = [
        np.asarray([0.25, 0.20, 0.25, 0.25, 0.05]),
        np.asarray([0.15, 0.30, 0.15, 0.35, 0.05]),
        np.asarray([0.35, 0.15, 0.30, 0.15, 0.05]),
        np.asarray([0.10, 0.35, 0.10, 0.40, 0.05]),
    ]
    solutions = []
    for start in starts:
        start = np.maximum(start, lower)
        start = lower + (1.0 - lower.sum()) * (
            (start - lower) / np.sum(start - lower)
        )
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(float(value), 1.0) for value in lower],
            constraints={"type": "eq", "fun": lambda value: float(value.sum() - 1.0)},
            options={"maxiter": 250, "ftol": 1.0e-10},
        )
        solutions.append(result)
    best = min(solutions, key=lambda item: objective(item.x))
    if not best.success:
        raise RuntimeError(f"hybrid-weight optimizer failed: {best.message}")
    weights = np.asarray(best.x, dtype=np.float64)
    weights /= weights.sum()
    if (
        not np.isfinite(weights).all()
        or np.any(weights < lower - 1.0e-10)
        or abs(float(weights.sum()) - 1.0) > 1.0e-10
    ):
        raise RuntimeError("hybrid-weight optimizer returned invalid weights")
    diagnostics = {}
    for payload in bridge_payload:
        normalized, finite_rows, _, _ = stage_metric(weights, payload)
        fraction = float(np.exp(-normalized))
        diagnostics[payload["name"]] = {
            "finite_rows": finite_rows,
            "normalized_log_second_moment": normalized,
            "predicted_ess_fraction": fraction,
            "predicted_combined_ess": float(fraction * total_rows),
        }
    worst_predicted = min(
        item["predicted_combined_ess"] for item in diagnostics.values()
    )
    if worst_predicted < arguments.minimum_predicted_ess:
        raise RuntimeError(
            f"optimized hybrid forecast {worst_predicted:.3f} is below "
            f"the required {arguments.minimum_predicted_ess:.3f}"
        )

    raw_counts = weights * arguments.candidate_rows
    counts = np.floor(raw_counts).astype(np.int64)
    for index in np.argsort(raw_counts - counts)[::-1][
        : arguments.candidate_rows - int(counts.sum())
    ]:
        counts[index] += 1
    if int(counts.sum()) != arguments.candidate_rows or np.any(counts < 1):
        raise RuntimeError("invalid optimized component counts")

    def draw_flow(flows, model, rows, seed):
        base, remainder = divmod(rows, len(flows))
        parts = []
        for member, flow in enumerate(flows):
            local_rows = base + (member < remainder)
            torch.manual_seed(seed + member)
            torch.cuda.manual_seed_all(seed + member)
            with torch.no_grad():
                value = flow().sample((local_rows,)).detach().cpu().numpy().astype(np.float64)
            parts.append(inverse_transform(value, model))
        return np.vstack(parts)

    rng = np.random.default_rng(arguments.seed)

    def draw_broad(model, rows):
        gaussian = rng.multivariate_normal(
            np.zeros(9), model["broad_shape"], size=rows
        )
        chi_square = rng.chisquare(model["broad_degrees"], size=rows)
        value = model["broad_location"] + gaussian / np.sqrt(
            chi_square[:, None] / model["broad_degrees"]
        )
        return inverse_transform(value, model)

    parts = [
        draw_flow(old_flows, old_model, int(counts[0]), arguments.seed + 10_000),
        draw_broad(old_model, int(counts[1])),
        draw_flow(new_flows, new_model, int(counts[2]), arguments.seed + 20_000),
        draw_broad(new_model, int(counts[3])),
        rng.uniform(
            old_model["prior_low"],
            old_model["prior_high"],
            size=(int(counts[4]), 9),
        ),
    ]
    theta = np.vstack(parts)
    component = np.concatenate(
        [np.full(len(part), index, dtype=np.uint8) for index, part in enumerate(parts)]
    )
    old_mix_new, old_flow_new, old_broad_new, uniform_new = proposal_density(
        theta, old_flows, old_model, device, arguments.batch_size
    )
    _, new_flow_new, new_broad_new, new_uniform_new = proposal_density(
        theta, new_flows, new_model, device, arguments.batch_size
    )
    if np.max(np.abs(uniform_new - new_uniform_new)) > 1.0e-12:
        raise RuntimeError("uniform density mismatch on optimized candidates")
    component_density_new = np.column_stack(
        [old_flow_new, old_broad_new, new_flow_new, new_broad_new, uniform_new]
    )
    hybrid_new = logsumexp(
        np.log(weights)[None, :] + component_density_new, axis=1
    )

    old_stage_components = bridge_payload[0]["components"]
    hybrid_on_old = logsumexp(
        np.log(weights)[None, :] + old_stage_components, axis=1
    )
    old_mix_on_old = bridge_payload[0]["old_mix"]
    old_replay_error = float(
        np.max(np.abs(old_mix_on_old - old_exact["log_proposal"]))
    )
    if old_replay_error > 5.0e-4:
        raise RuntimeError(f"old proposal replay mismatch: {old_replay_error}")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = arguments.output_dir / "optimized_hybrid_model.npz"
    model_metadata = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_optimized_hybrid_model",
        "component_order": [
            "old_flow_ensemble",
            "old_defensive_Student_t",
            "new_flow_ensemble",
            "new_defensive_Student_t",
            "uniform_full_prior",
        ],
        "old_proposal_sha256": old_proposal_hash,
        "new_proposal_sha256": new_proposal_hash,
        "old_exact_stage_sha256": old_exact_hash,
        "validation_stage_sha256": validation_hash,
        "forbidden_artifacts_used": [],
    }
    atomic_save(
        model_path,
        optimized_weights=weights,
        optimized_component_counts=counts,
        outer_stage_fractions=outer_fractions,
        prior_low=old_model["prior_low"],
        prior_high=old_model["prior_high"],
        metadata=np.asarray(json.dumps(model_metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    model_report = {
        **model_metadata,
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "optimized_weights": weights.tolist(),
        "optimized_component_counts": counts.tolist(),
        "outer_stage_fractions": outer_fractions.tolist(),
        "bridge_diagnostics": diagnostics,
        "worst_predicted_combined_ess": worst_predicted,
        "minimum_predicted_ess": arguments.minimum_predicted_ess,
        "output": str(model_path.resolve()),
        "output_sha256": sha256(model_path),
    }
    model_path.with_suffix(".json").write_text(
        json.dumps(model_report, indent=2, sort_keys=True) + "\n"
    )
    model_hash = model_report["output_sha256"]

    cross_path = arguments.output_dir / "old_exact_cross_density.npz"
    cross_metadata = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_dual_stage_old_exact_cross_density",
        "old_exact_stage_sha256": old_exact_hash,
        "old_proposal_sha256": old_proposal_hash,
        "new_proposal_sha256": model_hash,
        "rows": old_rows,
        "forbidden_artifacts_used": [],
    }
    atomic_save(
        cross_path,
        logq_old_stage=old_mix_on_old,
        logq_new_stage=hybrid_on_old,
        metadata=np.asarray(json.dumps(cross_metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    cross_report = {
        **cross_metadata,
        "old_density_replay_max_abs_error": old_replay_error,
        "output": str(cross_path.resolve()),
        "output_sha256": sha256(cross_path),
    }
    cross_path.with_suffix(".json").write_text(
        json.dumps(cross_report, indent=2, sort_keys=True) + "\n"
    )

    candidate_path = arguments.output_dir / "dual_stage_new_candidates.npz"
    group_component = np.where(component <= 1, 0, np.where(component <= 3, 1, 2)).astype(
        np.uint8
    )
    group_counts = np.bincount(group_component, minlength=3).astype(np.int64)
    candidate_metadata = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_dual_stage_new_proposal_candidates",
        "model": "ddb-hyperonic",
        "full_prior_dimension": 9,
        "old_proposal_sha256": old_proposal_hash,
        "new_proposal_sha256": model_hash,
        "old_exact_stage_sha256": old_exact_hash,
        "new_component_counts": group_counts.tolist(),
        "optimized_subcomponent_counts": counts.tolist(),
        "optimized_subcomponent_weights": weights.tolist(),
        "outer_stage_counts": [old_rows, arguments.candidate_rows],
        "outer_stage_fractions": outer_fractions.tolist(),
        "validation_stage_sha256": validation_hash,
        "forbidden_artifacts_used": [],
    }
    atomic_save(
        candidate_path,
        theta=theta,
        logq_old_stage=old_mix_new,
        logq_new_stage=hybrid_new,
        proposal_component=group_component,
        component_counts=group_counts,
        prior_low=old_model["prior_low"],
        prior_high=old_model["prior_high"],
        metadata=np.asarray(json.dumps(candidate_metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    candidate_report = {
        **candidate_metadata,
        "candidate_rows": len(theta),
        "bridge_diagnostics": diagnostics,
        "worst_predicted_combined_ess": worst_predicted,
        "minimum_predicted_ess": arguments.minimum_predicted_ess,
        "output": str(candidate_path.resolve()),
        "output_sha256": sha256(candidate_path),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    candidate_path.with_suffix(".json").write_text(
        json.dumps(candidate_report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "model": model_report,
                "cross_density": cross_report,
                "new_candidates": candidate_report,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
