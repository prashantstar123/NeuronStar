#!/usr/bin/env python3
"""Fit a clean 9D NPE density from cumulative TSNPE-owned rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


A1_OBSERVATION = np.asarray(
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def width90(sample: np.ndarray) -> np.ndarray:
    return np.percentile(sample, 95.0, axis=0) - np.percentile(sample, 5.0, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-data", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hidden-features", type=int, default=256)
    parser.add_argument("--transforms", type=int, default=10)
    parser.add_argument("--max-training-epochs", type=int, default=400)
    parser.add_argument("--posterior-samples", type=int, default=20_000)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise FileExistsError(f"refusing to replace {arguments.output_dir}")
    if arguments.max_training_epochs < 1 or arguments.posterior_samples < 1:
        raise ValueError("training epochs and posterior samples must be positive")

    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform

    started = time.time()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production TSNPE fitting")
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(arguments.seed)

    theta_parts = []
    x_parts = []
    parents = []
    prior_low = None
    prior_high = None
    expected_rounds = list(range(len(arguments.round_data)))
    observed_rounds = []
    for path in arguments.round_data:
        report_path = path.with_suffix(".json")
        report = json.loads(report_path.read_text())
        if report.get("lineage_class") != "independent_tsnpe":
            raise RuntimeError(f"round lacks independent TSNPE lineage: {path}")
        if report.get("role") != "tsnpe_exact_round":
            raise RuntimeError(f"wrong round role: {path}")
        if report.get("forbidden_artifacts_used"):
            raise RuntimeError(f"round declares forbidden ancestry: {path}")
        if report.get("output_sha256") != sha256(path):
            raise RuntimeError(f"round hash does not match report: {path}")
        with np.load(path, allow_pickle=False) as archive:
            theta = np.asarray(archive["theta"], dtype=np.float64)
            x = np.asarray(archive["x"], dtype=np.float64)
            mask = np.asarray(archive["training_mask"], dtype=bool)
            local_low = np.asarray(archive["prior_low"], dtype=np.float64)
            local_high = np.asarray(archive["prior_high"], dtype=np.float64)
            forbidden = np.asarray(archive["forbidden_artifacts_used"])
        if forbidden.size:
            raise RuntimeError(f"round archive declares forbidden ancestry: {path}")
        if theta.shape[1:] != (9,) or x.shape != (len(theta), 7) or mask.shape != (len(theta),):
            raise RuntimeError(f"invalid array shapes in {path}")
        if not np.isfinite(theta[mask]).all() or not np.isfinite(x[mask]).all():
            raise RuntimeError(f"non-finite accepted training row in {path}")
        if prior_low is None:
            prior_low, prior_high = local_low, local_high
        elif not (
            np.array_equal(prior_low.astype(np.float32), local_low.astype(np.float32))
            and np.array_equal(prior_high.astype(np.float32), local_high.astype(np.float32))
        ):
            # Later-round proposals are sampled by the float32 SBI posterior, so
            # their serialized bounds can carry a harmless float32 round trip.
            # Require exact equality in the numerical precision actually used by
            # BoxUniform below; this still rejects any physically different bound.
            raise RuntimeError("rounds use different prior bounds")
        theta_parts.append(theta[mask])
        x_parts.append(x[mask])
        observed_rounds.append(int(report["round_index"]))
        parents.append(
            {"path": str(path.resolve()), "sha256": sha256(path), "rows": int(mask.sum())}
        )
    if observed_rounds != expected_rounds:
        raise RuntimeError(
            f"round data must be an ordered complete prefix {expected_rounds}, got {observed_rounds}"
        )
    theta = np.vstack(theta_parts)
    x = np.vstack(x_parts)
    if len(theta) < 100:
        raise RuntimeError("too few accepted simulations for production NPE")

    low = torch.as_tensor(prior_low, dtype=torch.float32, device=device)
    high = torch.as_tensor(prior_high, dtype=torch.float32, device=device)
    prior = BoxUniform(low=low, high=high)
    builder = posterior_nn(
        model="nsf",
        hidden_features=arguments.hidden_features,
        num_transforms=arguments.transforms,
    )
    inference = NPE(prior=prior, density_estimator=builder, device=str(device))
    estimator = inference.append_simulations(
        torch.as_tensor(theta, dtype=torch.float32, device=device),
        torch.as_tensor(x, dtype=torch.float32, device=device),
        exclude_invalid_x=True,
    ).train(
        force_first_round_loss=True,
        show_train_summary=False,
        max_num_epochs=arguments.max_training_epochs,
    )
    posterior = inference.build_posterior(estimator).set_default_x(
        torch.as_tensor(A1_OBSERVATION, dtype=torch.float32, device=device)
    )
    sample = (
        posterior.sample(
            (arguments.posterior_samples,),
            show_progress_bars=False,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    if sample.shape != (arguments.posterior_samples, 9):
        raise RuntimeError(f"unexpected posterior sample shape {sample.shape}")
    if not np.isfinite(sample).all() or not np.all(
        (sample >= prior_low) & (sample <= prior_high)
    ):
        raise RuntimeError("posterior sample contains invalid or out-of-prior rows")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    estimator_path = arguments.output_dir / "density_estimator.pt"
    posterior_path = arguments.output_dir / "posterior_FINAL.npy"
    torch.save(estimator, estimator_path)
    np.save(posterior_path, sample)
    report = {
        "status": "PROPOSAL_READY_FOR_NEXT_ROUND_OR_MIS",
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_round_density_estimator",
        "model": "ddb-hyperonic",
        "round_index": observed_rounds[-1],
        "rounds_consumed": observed_rounds,
        "full_prior_dimension": 9,
        "prior_low": prior_low.tolist(),
        "prior_high": prior_high.tolist(),
        "training_rows": int(len(theta)),
        "training_rows_by_round": [int(len(part)) for part in theta_parts],
        "training_configuration": {
            "network": "NSF",
            "hidden_features": arguments.hidden_features,
            "transforms": arguments.transforms,
            "force_first_round_loss": True,
            "maximum_epochs": arguments.max_training_epochs,
        },
        "seed": arguments.seed,
        "parents": parents,
        "forbidden_artifacts_used": [],
        "posterior_median": np.median(sample, axis=0).tolist(),
        "posterior_width90": width90(sample).tolist(),
        "estimator": str(estimator_path.resolve()),
        "estimator_sha256": sha256(estimator_path),
        "posterior": str(posterior_path.resolve()),
        "posterior_sha256": sha256(posterior_path),
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "wall_seconds": time.time() - started,
    }
    report_path = arguments.output_dir / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
