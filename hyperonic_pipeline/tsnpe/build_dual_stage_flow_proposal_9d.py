#!/usr/bin/env python3
"""Build a clean two-generation TSNPE deterministic-MIS proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammaln, logsumexp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform(theta, low, high, mean, scale):
    unit = (theta - low) / (high - low)
    if not np.all((unit > 0.0) & (unit < 1.0)):
        raise RuntimeError("density transform received a boundary row")
    value = (np.log(unit) - np.log1p(-unit) - mean) / scale
    jacobian = -np.log(scale).sum() - np.sum(
        np.log(high - low) + np.log(unit) + np.log1p(-unit), axis=1
    )
    return value, jacobian


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


def load_proposal(path, checkpoints, device):
    import zuko

    report = json.loads(path.with_suffix(".json").read_text())
    path_hash = sha256(path)
    if (
        report.get("status") != "PASS"
        or report.get("lineage_class") != "independent_tsnpe"
        or report.get("role") != "tsnpe_frozen_adaptive_flow_full_prior_proposal"
        or report.get("full_prior_dimension") != 9
        or report.get("forbidden_artifacts_used")
        or report.get("output_sha256") != path_hash
    ):
        raise RuntimeError(f"proposal failed clean-lineage gates: {path}")
    configuration = report["training_configuration"]
    members = report["flow_members"]
    if len(checkpoints) != len(members):
        raise RuntimeError("checkpoint count does not match proposal ensemble")
    flows = []
    for checkpoint, member in zip(checkpoints, members, strict=True):
        if sha256(checkpoint) != member["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
        flow = zuko.flows.NSF(
            9,
            transforms=int(configuration["transforms"]),
            hidden_features=list(configuration["hidden_features"]),
            bins=int(configuration["bins"]),
        ).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        # Zuko renamed the diagonal-normal base buffers between the version
        # used for the first clean generation and the current runtime.  The
        # tensors themselves are identical; normalize only these two keys and
        # let the full stored-density replay below certify the conversion.
        expected_keys = set(flow.state_dict())
        if "base.loc" in state and "base._0" in expected_keys:
            state["base._0"] = state.pop("base.loc")
        if "base.scale" in state and "base._1" in expected_keys:
            state["base._1"] = state.pop("base.scale")
        if "base._0" in state and "base.loc" in expected_keys:
            state["base.loc"] = state.pop("base._0")
        if "base._1" in state and "base.scale" in expected_keys:
            state["base.scale"] = state.pop("base._1")
        flow.load_state_dict(state, strict=True)
        flow.eval()
        flows.append(flow)
    with np.load(path, allow_pickle=False) as archive:
        model = {
            key: np.asarray(archive[key], dtype=np.float64)
            for key in (
                "prior_low",
                "prior_high",
                "transform_mean",
                "transform_scale",
                "broad_location",
                "broad_shape",
            )
        }
        model["broad_degrees"] = float(np.asarray(archive["broad_degrees"]).item())
        model["theta"] = np.asarray(archive["theta"], dtype=np.float64)
        model["stored_logq_flow"] = np.asarray(
            archive["logq_flow_ensemble"], dtype=np.float64
        )
        model["stored_logq_broad"] = np.asarray(archive["logq_broad"], dtype=np.float64)
        model["stored_logq_uniform"] = np.asarray(
            archive["logq_uniform_full_prior"], dtype=np.float64
        )
        model["proposal_component"] = np.asarray(
            archive["proposal_component"], dtype=np.uint8
        )
        model["component_counts"] = np.asarray(
            archive["component_counts"], dtype=np.int64
        )
    model["fractions"] = model["component_counts"] / model["component_counts"].sum()
    return report, path_hash, flows, model


def proposal_density(theta, flows, model, device, batch_size):
    transformed, jacobian = transform(
        theta,
        model["prior_low"],
        model["prior_high"],
        model["transform_mean"],
        model["transform_scale"],
    )
    member_logq = np.vstack(
        [batched_flow_log_prob(flow, transformed, device, batch_size) for flow in flows]
    )
    qflow = logsumexp(member_logq, axis=0) - np.log(len(flows)) + jacobian
    qbroad = t_log_density(
        transformed,
        model["broad_location"],
        model["broad_shape"],
        model["broad_degrees"],
    ) + jacobian
    quniform = np.full(
        len(theta),
        -float(np.log(model["prior_high"] - model["prior_low"]).sum()),
        dtype=np.float64,
    )
    fractions = model["fractions"]
    mixture = logsumexp(
        np.column_stack(
            [
                np.log(fractions[0]) + qflow,
                np.log(fractions[1]) + qbroad,
                np.log(fractions[2]) + quniform,
            ]
        ),
        axis=1,
    )
    return mixture, qflow, qbroad, quniform


def atomic_save(path, **payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-proposal", type=Path, required=True)
    parser.add_argument("--old-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--old-exact-stage", type=Path, required=True)
    parser.add_argument(
        "--bridge-validation-stage",
        type=Path,
        help="Optional independent exact stage for the outer-mixture ESS forecast.",
    )
    parser.add_argument("--new-proposal", type=Path, required=True)
    parser.add_argument("--new-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise FileExistsError(f"refusing to replace {arguments.output_dir}")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dual-stage density replay")

    old_report, old_hash, old_flows, old_model = load_proposal(
        arguments.old_proposal, arguments.old_checkpoints, device
    )
    new_report, new_hash, new_flows, new_model = load_proposal(
        arguments.new_proposal, arguments.new_checkpoints, device
    )
    if (
        not np.array_equal(old_model["prior_low"], new_model["prior_low"])
        or not np.array_equal(old_model["prior_high"], new_model["prior_high"])
    ):
        raise RuntimeError("old and new flow generations use different priors")

    exact_report = json.loads(arguments.old_exact_stage.with_suffix(".json").read_text())
    exact_hash = sha256(arguments.old_exact_stage)
    if (
        exact_report.get("lineage_class") != "independent_tsnpe"
        or exact_report.get("status") not in {"PASS", "FAIL_ESS_GATE"}
        or exact_report.get("target_gate", {}).get("status") != "PASS"
        or exact_report.get("forbidden_artifacts_used")
        or exact_report.get("output_sha256") != exact_hash
        or exact_report.get("candidate", {}).get("sha256") != old_hash
    ):
        raise RuntimeError("old exact stage failed clean-lineage gates")
    with np.load(arguments.old_exact_stage, allow_pickle=False) as archive:
        old_theta = np.asarray(archive["theta"], dtype=np.float64)
        old_stored_density = np.asarray(archive["log_proposal"], dtype=np.float64)
        old_log_target = np.asarray(archive["log_target"], dtype=np.float64)
        if np.asarray(archive["forbidden_artifacts_used"]).size:
            raise RuntimeError("old exact stage declares forbidden ancestry")

    replay_old, _, _, _ = proposal_density(
        old_theta, old_flows, old_model, device, arguments.batch_size
    )
    old_replay_error = float(np.max(np.abs(replay_old - old_stored_density)))
    if old_replay_error > 5.0e-4:
        raise RuntimeError(f"old proposal density replay mismatch: {old_replay_error}")
    new_on_old, _, _, _ = proposal_density(
        old_theta, new_flows, new_model, device, arguments.batch_size
    )

    new_theta = new_model["theta"]
    new_replay, _, _, _ = proposal_density(
        new_theta, new_flows, new_model, device, arguments.batch_size
    )
    new_stored_density = logsumexp(
        np.column_stack(
            [
                np.log(new_model["fractions"][0]) + new_model["stored_logq_flow"],
                np.log(new_model["fractions"][1]) + new_model["stored_logq_broad"],
                np.log(new_model["fractions"][2]) + new_model["stored_logq_uniform"],
            ]
        ),
        axis=1,
    )
    new_replay_error = float(np.max(np.abs(new_replay - new_stored_density)))
    if new_replay_error > 5.0e-4:
        raise RuntimeError(f"new proposal density replay mismatch: {new_replay_error}")
    old_on_new, _, _, _ = proposal_density(
        new_theta, old_flows, old_model, device, arguments.batch_size
    )

    stage_counts = np.asarray([len(old_theta), len(new_theta)], dtype=np.int64)
    stage_fractions = stage_counts.astype(np.float64) / stage_counts.sum()

    def bridge_diagnostic(
        bridge_theta,
        bridge_log_target,
        bridge_log_sampling_density,
        bridge_log_evidence,
    ):
        bridge_qold, _, _, _ = proposal_density(
            bridge_theta, old_flows, old_model, device, arguments.batch_size
        )
        bridge_qnew, _, _, _ = proposal_density(
            bridge_theta, new_flows, new_model, device, arguments.batch_size
        )
        bridge_qmix = logsumexp(
            np.column_stack(
                [
                    np.log(stage_fractions[0]) + bridge_qold,
                    np.log(stage_fractions[1]) + bridge_qnew,
                ]
            ),
            axis=1,
        )
        valid = (
            np.isfinite(bridge_log_target)
            & np.isfinite(bridge_log_sampling_density)
            & np.isfinite(bridge_qmix)
        )
        log_second = float(
            logsumexp(
                2.0 * bridge_log_target[valid]
                - bridge_qmix[valid]
                - bridge_log_sampling_density[valid]
            )
            - np.log(len(bridge_theta))
        )
        fraction = float(np.exp(2.0 * bridge_log_evidence - log_second))
        return {
            "status": "DIAGNOSTIC_ONLY_FRESH_EXACT_EVALUATION_REQUIRED",
            "finite_rows": int(valid.sum()),
            "log_second_moment": log_second,
            "predicted_ess_fraction": fraction,
            "predicted_combined_ess": float(fraction * stage_counts.sum()),
        }

    adaptation_bridge = bridge_diagnostic(
        old_theta,
        old_log_target,
        old_stored_density,
        float(exact_report["log_evidence"]),
    )
    independent_bridge = None
    validation_hash = None
    if arguments.bridge_validation_stage is not None:
        validation_report = json.loads(
            arguments.bridge_validation_stage.with_suffix(".json").read_text()
        )
        validation_hash = sha256(arguments.bridge_validation_stage)
        if (
            validation_report.get("lineage_class") != "independent_tsnpe"
            or validation_report.get("status") not in {"PASS", "FAIL_ESS_GATE"}
            or validation_report.get("target_gate", {}).get("status") != "PASS"
            or validation_report.get("forbidden_artifacts_used")
            or validation_report.get("output_sha256") != validation_hash
        ):
            raise RuntimeError("independent bridge stage failed clean-lineage gates")
        with np.load(arguments.bridge_validation_stage, allow_pickle=False) as archive:
            validation_theta = np.asarray(archive["theta"], dtype=np.float64)
            validation_target = np.asarray(archive["log_target"], dtype=np.float64)
            validation_proposal = np.asarray(archive["log_proposal"], dtype=np.float64)
            validation_low = np.asarray(archive["prior_low"], dtype=np.float64)
            validation_high = np.asarray(archive["prior_high"], dtype=np.float64)
        if (
            not np.array_equal(validation_low, old_model["prior_low"])
            or not np.array_equal(validation_high, old_model["prior_high"])
        ):
            raise RuntimeError("independent bridge stage uses a different prior")
        independent_bridge = bridge_diagnostic(
            validation_theta,
            validation_target,
            validation_proposal,
            float(validation_report["log_evidence"]),
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    cross_path = arguments.output_dir / "old_exact_cross_density.npz"
    cross_metadata = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_dual_stage_old_exact_cross_density",
        "old_exact_stage_sha256": exact_hash,
        "old_proposal_sha256": old_hash,
        "new_proposal_sha256": new_hash,
        "rows": len(old_theta),
        "forbidden_artifacts_used": [],
    }
    atomic_save(
        cross_path,
        logq_old_stage=old_stored_density,
        logq_new_stage=new_on_old,
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
    candidate_metadata = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_dual_stage_new_proposal_candidates",
        "model": "ddb-hyperonic",
        "full_prior_dimension": 9,
        "old_proposal_sha256": old_hash,
        "new_proposal_sha256": new_hash,
        "old_exact_stage_sha256": exact_hash,
        "new_component_counts": new_model["component_counts"].tolist(),
        "outer_stage_counts": stage_counts.tolist(),
        "outer_stage_fractions": stage_fractions.tolist(),
        "bridge_validation_stage_sha256": validation_hash,
        "forbidden_artifacts_used": [],
    }
    atomic_save(
        candidate_path,
        theta=new_theta,
        logq_old_stage=old_on_new,
        logq_new_stage=new_stored_density,
        proposal_component=new_model["proposal_component"],
        component_counts=new_model["component_counts"],
        prior_low=new_model["prior_low"],
        prior_high=new_model["prior_high"],
        metadata=np.asarray(json.dumps(candidate_metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    candidate_report = {
        **candidate_metadata,
        "candidate_rows": len(new_theta),
        "new_density_replay_max_abs_error": new_replay_error,
        "adaptation_outer_mixture_bridge": adaptation_bridge,
        "independent_outer_mixture_bridge": independent_bridge,
        "output": str(candidate_path.resolve()),
        "output_sha256": sha256(candidate_path),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    candidate_path.with_suffix(".json").write_text(
        json.dumps(candidate_report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {"cross_density": cross_report, "new_candidates": candidate_report},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
