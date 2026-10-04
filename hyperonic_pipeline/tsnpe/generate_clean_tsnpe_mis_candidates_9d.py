#!/usr/bin/env python3
"""Build a clean full-prior 9D defensive-MIS proposal from two TSNPE fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


A1_OBSERVATION = np.asarray(
    [0.153, -16.1, 230.0, 32.5, 0.505714285714279,
     1.24142857142857, 2.4857142857143],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_run(directory: Path) -> tuple[dict, Path, Path, np.ndarray]:
    report_path = directory / "run_report.json"
    estimator_path = directory / "density_estimator.pt"
    posterior_path = directory / "posterior_FINAL.npy"
    require(report_path.is_file(), f"missing report: {report_path}")
    require(estimator_path.is_file(), f"missing estimator: {estimator_path}")
    require(posterior_path.is_file(), f"missing posterior: {posterior_path}")
    report = json.loads(report_path.read_text())
    require(
        report.get("status") == "PROPOSAL_READY_FOR_NEXT_ROUND_OR_MIS",
        f"run is not frozen for MIS: {report_path}",
    )
    require(report.get("lineage_class") == "independent_tsnpe", "wrong lineage")
    require(report.get("role") == "tsnpe_round_density_estimator", "wrong role")
    require(report.get("model") == "ddb-hyperonic", "wrong EOS model")
    require(report.get("full_prior_dimension") == 9, "not a full 9D run")
    require(not report.get("forbidden_artifacts_used"), "forbidden ancestry")
    require(report.get("estimator_sha256") == sha256(estimator_path), "estimator hash mismatch")
    require(report.get("posterior_sha256") == sha256(posterior_path), "posterior hash mismatch")
    require(report.get("rounds_consumed") == [0, 1, 2], "incomplete round prefix")
    for parent in report.get("parents", []):
        require(int(parent.get("rows", 0)) > 0, "empty training parent")
        require(len(str(parent.get("sha256", ""))) == 64, "invalid parent hash")
    posterior = np.asarray(np.load(posterior_path, allow_pickle=False), dtype=np.float64)
    require(posterior.ndim == 2 and posterior.shape[1] == 9, "wrong posterior shape")
    require(np.isfinite(posterior).all(), "non-finite posterior row")
    return report, estimator_path, posterior_path, posterior


def normalized_log_density(posterior, observed, theta, device, batch_size):
    values = []
    with torch.no_grad():
        for start in range(0, len(theta), batch_size):
            batch = torch.as_tensor(
                theta[start : start + batch_size], dtype=torch.float32, device=device
            )
            values.append(
                posterior.log_prob(batch, x=observed, norm_posterior=True)
                .detach().cpu().numpy()
            )
    return np.concatenate(values).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run-dir", type=Path, required=True)
    parser.add_argument("--second-run-dir", type=Path, required=True)
    parser.add_argument("--uniform-draws", type=int, default=5_000)
    parser.add_argument("--uniform-seed", type=int, required=True)
    parser.add_argument("--density-batch", type=int, default=4_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")
    if arguments.uniform_draws < 1 or arguments.density_batch < 1:
        raise ValueError("draw and batch counts must be positive")

    from sbi.inference.posteriors import DirectPosterior
    from sbi.utils import BoxUniform

    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production candidate generation")
    first, estimator1_path, posterior1_path, theta1 = load_run(arguments.first_run_dir)
    second, estimator2_path, posterior2_path, theta2 = load_run(arguments.second_run_dir)
    require(first["seed"] != second["seed"], "flow fits must use independent seeds")
    require(first["parents"] == second["parents"], "flow fits do not share clean training parents")
    require(
        first["training_configuration"] == second["training_configuration"],
        "flow training configurations differ",
    )
    prior_low = np.asarray(first["prior_low"], dtype=np.float64)
    prior_high = np.asarray(first["prior_high"], dtype=np.float64)
    require(
        prior_low.shape == (9,) and prior_high.shape == (9,), "wrong prior shape"
    )
    require(
        np.array_equal(prior_low, np.asarray(second["prior_low"], dtype=np.float64))
        and np.array_equal(prior_high, np.asarray(second["prior_high"], dtype=np.float64)),
        "flow fits use different full priors",
    )
    require(
        np.all((theta1 >= prior_low) & (theta1 <= prior_high))
        and np.all((theta2 >= prior_low) & (theta2 <= prior_high)),
        "flow sample outside full prior",
    )

    rng = np.random.default_rng(arguments.uniform_seed)
    theta_uniform = rng.uniform(
        prior_low, prior_high, size=(arguments.uniform_draws, 9)
    )
    theta = np.vstack([theta1, theta2, theta_uniform])
    component = np.concatenate(
        [
            np.zeros(len(theta1), dtype=np.uint8),
            np.ones(len(theta2), dtype=np.uint8),
            np.full(len(theta_uniform), 2, dtype=np.uint8),
        ]
    )

    low_tensor = torch.as_tensor(prior_low, dtype=torch.float32, device=device)
    high_tensor = torch.as_tensor(prior_high, dtype=torch.float32, device=device)
    prior = BoxUniform(low=low_tensor, high=high_tensor)
    estimator1 = torch.load(estimator1_path, map_location=device, weights_only=False)
    estimator2 = torch.load(estimator2_path, map_location=device, weights_only=False)
    estimator1.eval()
    estimator2.eval()
    posterior1 = DirectPosterior(estimator1, prior=prior, device=str(device))
    posterior2 = DirectPosterior(estimator2, prior=prior, device=str(device))
    observed = torch.as_tensor(A1_OBSERVATION, dtype=torch.float32, device=device)
    logq1 = normalized_log_density(
        posterior1, observed, theta, device, arguments.density_batch
    )
    logq2 = normalized_log_density(
        posterior2, observed, theta, device, arguments.density_batch
    )
    logq_uniform = np.full(
        len(theta), -float(np.log(prior_high - prior_low).sum()), dtype=np.float64
    )
    require(np.isfinite(logq1).all(), "first normalized flow density is non-finite")
    require(np.isfinite(logq2).all(), "second normalized flow density is non-finite")

    parents = [
        {
            "run_directory": str(arguments.first_run_dir.resolve()),
            "seed": int(first["seed"]),
            "estimator_sha256": sha256(estimator1_path),
            "posterior_sha256": sha256(posterior1_path),
        },
        {
            "run_directory": str(arguments.second_run_dir.resolve()),
            "seed": int(second["seed"]),
            "estimator_sha256": sha256(estimator2_path),
            "posterior_sha256": sha256(posterior2_path),
        },
    ]
    counts = [len(theta1), len(theta2), len(theta_uniform)]
    metadata = {
        "schema_version": 1,
        "status": "PASS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_full_prior_mis_candidates",
        "model": "ddb-hyperonic",
        "full_prior_dimension": 9,
        "components": ["flow_seed_a", "flow_seed_b", "uniform_full_prior"],
        "component_counts": counts,
        "uniform_seed": arguments.uniform_seed,
        "parents": parents,
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            theta=theta,
            logq_flow_seed_a=logq1,
            logq_flow_seed_b=logq2,
            logq_uniform_full_prior=logq_uniform,
            proposal_component=component,
            component_counts=np.asarray(counts, dtype=np.int64),
            prior_low=prior_low,
            prior_high=prior_high,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
    os.replace(temporary, arguments.output)
    report = {
        **metadata,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "candidate_rows": len(theta),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
