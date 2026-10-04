#!/usr/bin/env python3
"""Train and freeze a clean bounded adaptive-flow proposal for final IS."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, gammaln, logsumexp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    unit = expit(value * scale + mean)
    theta = low + (high - low) * unit
    if not np.all((theta > low) & (theta < high)):
        raise RuntimeError("flow proposal generated a numerical boundary row")
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
    return constant - 0.5 * (degrees + dimension) * np.log1p(
        quadratic / degrees
    )


def batched_flow_log_prob(flow, value, device, batch_size):
    output = []
    with torch.no_grad():
        distribution = flow()
        for start in range(0, len(value), batch_size):
            batch = torch.as_tensor(
                value[start : start + batch_size], dtype=torch.float32, device=device
            )
            output.append(distribution.log_prob(batch).detach().cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptation-stage", type=Path, required=True)
    parser.add_argument(
        "--bridge-validation-stage",
        type=Path,
        help="Optional independent exact stage used only to predict proposal ESS.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--training-resamples", type=int, default=120_000)
    parser.add_argument("--validation-resamples", type=int, default=30_000)
    parser.add_argument("--jitter", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--transforms", type=int, default=8)
    parser.add_argument("--hidden-width", type=int, default=192)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--flow-draws-per-member", type=int, default=20_000)
    parser.add_argument("--broad-draws", type=int, default=10_000)
    parser.add_argument("--uniform-draws", type=int, default=5_000)
    parser.add_argument("--broad-degrees", type=float, default=10.0)
    parser.add_argument("--broad-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise FileExistsError(f"refusing to replace {arguments.output_dir}")
    if (
        arguments.ensemble < 2
        or arguments.broad_degrees <= 2.0
        or arguments.transforms < 1
        or arguments.hidden_width < 1
        or arguments.hidden_layers < 1
        or arguments.bins < 2
    ):
        raise ValueError("ensemble must exceed one and t degrees must exceed two")

    import zuko

    started = time.time()
    stage_report = json.loads(
        arguments.adaptation_stage.with_suffix(".json").read_text()
    )
    if (
        stage_report.get("lineage_class") != "independent_tsnpe"
        or stage_report.get("role") not in {
            "tsnpe_adaptive_full_prior_mis_posterior",
            "tsnpe_frozen_adaptive_flow_full_prior_mis_posterior",
            "tsnpe_dual_stage_full_prior_mis_posterior",
            "tsnpe_crossvalidation_training_stage",
        }
        or stage_report.get("status") not in {"PASS", "FAIL_ESS_GATE"}
        or stage_report.get("forbidden_artifacts_used")
        or stage_report.get("target_gate", {}).get("status") != "PASS"
        or stage_report.get("output_sha256") != sha256(arguments.adaptation_stage)
    ):
        raise RuntimeError("adaptation stage failed clean exact-target gates")
    with np.load(arguments.adaptation_stage, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64)
        weight = np.asarray(archive["normalized_weight"], dtype=np.float64)
        low = np.asarray(archive["prior_low"], dtype=np.float64)
        high = np.asarray(archive["prior_high"], dtype=np.float64)
        adaptation_log_target = np.asarray(archive["log_target"], dtype=np.float64)
        adaptation_log_proposal = np.asarray(
            archive["log_proposal"], dtype=np.float64
        )
    if theta.shape[1:] != (9,) or weight.shape != (len(theta),):
        raise RuntimeError("adaptation arrays have wrong shapes")
    if not np.isclose(weight.sum(), 1.0, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("adaptation weights do not sum to one")

    bridge_validation = None
    bridge_validation_report = None
    if arguments.bridge_validation_stage is not None:
        bridge_validation_report = json.loads(
            arguments.bridge_validation_stage.with_suffix(".json").read_text()
        )
        if (
            bridge_validation_report.get("lineage_class") != "independent_tsnpe"
            or bridge_validation_report.get("status") not in {"PASS", "FAIL_ESS_GATE"}
            or bridge_validation_report.get("forbidden_artifacts_used")
            or bridge_validation_report.get("target_gate", {}).get("status") != "PASS"
            or bridge_validation_report.get("output_sha256")
            != sha256(arguments.bridge_validation_stage)
        ):
            raise RuntimeError("bridge validation stage failed clean exact-target gates")
        with np.load(arguments.bridge_validation_stage, allow_pickle=False) as archive:
            bridge_validation = {
                "theta": np.asarray(archive["theta"], dtype=np.float64),
                "log_target": np.asarray(archive["log_target"], dtype=np.float64),
                "log_proposal": np.asarray(archive["log_proposal"], dtype=np.float64),
                "prior_low": np.asarray(archive["prior_low"], dtype=np.float64),
                "prior_high": np.asarray(archive["prior_high"], dtype=np.float64),
            }
        if (
            bridge_validation["theta"].shape[1:] != (9,)
            or not np.array_equal(bridge_validation["prior_low"], low)
            or not np.array_equal(bridge_validation["prior_high"], high)
        ):
            raise RuntimeError("bridge validation stage uses a different 9D prior")

    unit = np.clip((theta - low) / (high - low), 1.0e-8, 1.0 - 1.0e-8)
    logit = np.log(unit) - np.log1p(-unit)
    transform_mean = np.sum(weight[:, None] * logit, axis=0)
    transform_variance = np.sum(
        weight[:, None] * (logit - transform_mean) ** 2, axis=0
    )
    transform_scale = np.sqrt(np.maximum(transform_variance, 1.0e-6))
    standardized = (logit - transform_mean) / transform_scale

    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production adaptive-flow fitting")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    flows = []
    flow_reports = []
    for member in range(arguments.ensemble):
        member_seed = arguments.seed + member * 1009
        rng = np.random.default_rng(member_seed)
        train_index = rng.choice(
            len(theta), size=arguments.training_resamples, replace=True, p=weight
        )
        validation_index = rng.choice(
            len(theta), size=arguments.validation_resamples, replace=True, p=weight
        )
        train = standardized[train_index] + rng.normal(
            0.0, arguments.jitter, size=(arguments.training_resamples, 9)
        )
        validation = standardized[validation_index] + rng.normal(
            0.0, arguments.jitter, size=(arguments.validation_resamples, 9)
        )
        torch.manual_seed(member_seed)
        np.random.seed(member_seed)
        torch.cuda.manual_seed_all(member_seed)
        flow = zuko.flows.NSF(
            9,
            transforms=arguments.transforms,
            hidden_features=[arguments.hidden_width] * arguments.hidden_layers,
            bins=arguments.bins,
        ).to(device)
        optimizer = torch.optim.AdamW(
            flow.parameters(), lr=arguments.learning_rate, weight_decay=1.0e-5
        )
        train_tensor = torch.as_tensor(train, dtype=torch.float32, device=device)
        validation_tensor = torch.as_tensor(
            validation, dtype=torch.float32, device=device
        )
        best_state = None
        best_validation = float("inf")
        best_epoch = -1
        stale = 0
        history = []
        for epoch in range(arguments.epochs):
            flow.train()
            permutation = torch.randperm(len(train_tensor), device=device)
            running = 0.0
            rows = 0
            for start in range(0, len(train_tensor), arguments.batch_size):
                index = permutation[start : start + arguments.batch_size]
                loss = -flow().log_prob(train_tensor[index]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
                optimizer.step()
                running += float(loss.detach()) * len(index)
                rows += len(index)
            flow.eval()
            validation_sum = 0.0
            with torch.no_grad():
                distribution = flow()
                for start in range(0, len(validation_tensor), arguments.batch_size):
                    batch = validation_tensor[start : start + arguments.batch_size]
                    validation_sum += float(-distribution.log_prob(batch).sum())
            train_loss = running / rows
            validation_loss = validation_sum / len(validation_tensor)
            history.append(
                {"epoch": epoch + 1, "train_nll": train_loss, "validation_nll": validation_loss}
            )
            if validation_loss < best_validation - 1.0e-4:
                best_validation = validation_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(flow.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= arguments.patience:
                break
        if best_state is None:
            raise RuntimeError("adaptive flow never produced a finite validation state")
        flow.load_state_dict(best_state, strict=True)
        flow.eval()
        checkpoint = arguments.output_dir / f"adaptive_flow_{member:02d}.pt"
        torch.save(flow.state_dict(), checkpoint)
        flows.append(flow)
        flow_reports.append(
            {
                "member": member,
                "seed": member_seed,
                "best_epoch": best_epoch,
                "best_validation_nll": best_validation,
                "epochs_run": len(history),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "history": history,
            }
        )
        del train_tensor, validation_tensor

    flow_theta_parts = []
    for member, flow in enumerate(flows):
        torch.manual_seed(arguments.seed + 50_000 + member)
        torch.cuda.manual_seed_all(arguments.seed + 50_000 + member)
        with torch.no_grad():
            sample = (
                flow()
                .sample((arguments.flow_draws_per_member,))
                .detach().cpu().numpy().astype(np.float64)
            )
        flow_theta_parts.append(
            inverse_transform(
                sample, low, high, transform_mean, transform_scale
            )
        )
    theta_flow = np.vstack(flow_theta_parts)

    rng = np.random.default_rng(arguments.seed + 90_000)
    centered = standardized - np.sum(weight[:, None] * standardized, axis=0)
    covariance = (centered * weight[:, None]).T @ centered
    covariance = 0.9 * covariance + 0.1 * np.diag(np.diag(covariance))
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    covariance = (eigenvector * np.maximum(eigenvalue, 0.01)) @ eigenvector.T
    broad_location = np.sum(weight[:, None] * standardized, axis=0)
    broad_shape = covariance * arguments.broad_scale**2 * (
        (arguments.broad_degrees - 2.0) / arguments.broad_degrees
    )
    gaussian = rng.multivariate_normal(
        np.zeros(9), broad_shape, size=arguments.broad_draws
    )
    chi_square = rng.chisquare(arguments.broad_degrees, size=arguments.broad_draws)
    broad_standardized = broad_location + gaussian / np.sqrt(
        chi_square[:, None] / arguments.broad_degrees
    )
    theta_broad = inverse_transform(
        broad_standardized, low, high, transform_mean, transform_scale
    )
    theta_uniform = rng.uniform(
        low, high, size=(arguments.uniform_draws, 9)
    )
    output_theta = np.vstack([theta_flow, theta_broad, theta_uniform])
    flow_rows = len(theta_flow)
    component = np.concatenate(
        [
            np.zeros(flow_rows, dtype=np.uint8),
            np.ones(arguments.broad_draws, dtype=np.uint8),
            np.full(arguments.uniform_draws, 2, dtype=np.uint8),
        ]
    )
    transformed, jacobian = transform(
        output_theta, low, high, transform_mean, transform_scale
    )
    member_logq = np.vstack(
        [
            batched_flow_log_prob(
                flow, transformed, device, arguments.batch_size
            )
            for flow in flows
        ]
    )
    logq_flow_ensemble = logsumexp(member_logq, axis=0) - np.log(len(flows)) + jacobian
    logq_broad = t_log_density(
        transformed, broad_location, broad_shape, arguments.broad_degrees
    ) + jacobian
    logq_uniform = np.full(
        len(output_theta), -float(np.log(high - low).sum()), dtype=np.float64
    )
    counts = np.asarray(
        [flow_rows, arguments.broad_draws, arguments.uniform_draws], dtype=np.int64
    )
    fractions = counts.astype(np.float64) / counts.sum()

    # Predict the asymptotic ESS fraction of this new proposal using stored
    # exact target values.  These are proposal diagnostics only; fresh exact
    # evaluations remain mandatory.
    def bridge_diagnostic(query_theta, query_target, query_proposal, query_logz):
        query_transformed, query_jacobian = transform(
            query_theta, low, high, transform_mean, transform_scale
        )
        query_member_logq = np.vstack(
            [
                batched_flow_log_prob(
                    flow, query_transformed, device, arguments.batch_size
                )
                for flow in flows
            ]
        )
        query_qflow = (
            logsumexp(query_member_logq, axis=0)
            - np.log(len(flows))
            + query_jacobian
        )
        query_qbroad = t_log_density(
            query_transformed,
            broad_location,
            broad_shape,
            arguments.broad_degrees,
        ) + query_jacobian
        query_quniform = np.full(
            len(query_theta), -float(np.log(high - low).sum()), dtype=np.float64
        )
        query_qnew = logsumexp(
            np.column_stack(
                [
                    np.log(fractions[0]) + query_qflow,
                    np.log(fractions[1]) + query_qbroad,
                    np.log(fractions[2]) + query_quniform,
                ]
            ),
            axis=1,
        )
        valid = (
            np.isfinite(query_target)
            & np.isfinite(query_proposal)
            & np.isfinite(query_qnew)
        )
        if not valid.any():
            raise RuntimeError("no finite rows for proposal bridge diagnostic")
        log_second_moment = float(
            logsumexp(
                2.0 * query_target[valid]
                - query_qnew[valid]
                - query_proposal[valid]
            )
            - np.log(len(query_theta))
        )
        predicted_fraction = float(np.exp(2.0 * query_logz - log_second_moment))
        return {
            "status": "DIAGNOSTIC_ONLY_FRESH_EXACT_EVALUATION_REQUIRED",
            "finite_rows": int(valid.sum()),
            "log_second_moment": log_second_moment,
            "predicted_ess_fraction": predicted_fraction,
            "predicted_candidate_ess": float(predicted_fraction * counts.sum()),
        }

    adaptation_bridge = bridge_diagnostic(
        theta,
        adaptation_log_target,
        adaptation_log_proposal,
        float(stage_report["log_evidence"]),
    )
    validation_bridge = None
    if bridge_validation is not None:
        validation_bridge = bridge_diagnostic(
            bridge_validation["theta"],
            bridge_validation["log_target"],
            bridge_validation["log_proposal"],
            float(bridge_validation_report["log_evidence"]),
        )
    metadata = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_frozen_adaptive_flow_full_prior_proposal",
        "model": "ddb-hyperonic",
        "full_prior_dimension": 9,
        "method": "frozen_adaptive_flow_ensemble_plus_Student_t_plus_uniform_prior",
        "adaptation_stage": {
            "path": str(arguments.adaptation_stage.resolve()),
            "sha256": sha256(arguments.adaptation_stage),
        },
        "bridge_validation_stage": (
            {
                "path": str(arguments.bridge_validation_stage.resolve()),
                "sha256": sha256(arguments.bridge_validation_stage),
            }
            if arguments.bridge_validation_stage is not None
            else None
        ),
        "component_counts": counts.tolist(),
        "component_fractions": fractions.tolist(),
        "flow_members": flow_reports,
        "seed": arguments.seed,
        "forbidden_artifacts_used": [],
    }
    output_path = arguments.output_dir / "final_adaptive_flow_candidates.npz"
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            theta=output_theta,
            logq_flow_ensemble=logq_flow_ensemble,
            logq_broad=logq_broad,
            logq_uniform_full_prior=logq_uniform,
            proposal_component=component,
            component_counts=counts,
            prior_low=low,
            prior_high=high,
            transform_mean=transform_mean,
            transform_scale=transform_scale,
            broad_location=broad_location,
            broad_shape=broad_shape,
            broad_degrees=np.asarray(arguments.broad_degrees),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
    os.replace(temporary, output_path)
    report = {
        **metadata,
        "candidate_rows": len(output_theta),
        "training_configuration": {
            "ensemble": arguments.ensemble,
            "architecture": "unconditional_zuko_NSF",
            "transforms": arguments.transforms,
            "hidden_features": [arguments.hidden_width] * arguments.hidden_layers,
            "bins": arguments.bins,
            "training_resamples_per_member": arguments.training_resamples,
            "validation_resamples_per_member": arguments.validation_resamples,
            "jitter": arguments.jitter,
            "maximum_epochs": arguments.epochs,
            "patience": arguments.patience,
        },
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "adaptation_bridge_diagnostic": adaptation_bridge,
        "independent_validation_bridge_diagnostic": validation_bridge,
        "wall_seconds": time.time() - started,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output_path.with_suffix(".json").write_text(report_text)
    (arguments.output_dir / "run_report.json").write_text(report_text)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
