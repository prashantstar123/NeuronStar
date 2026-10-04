#!/usr/bin/env python3
"""Train the nuclear-plus-maximum-mass seed flow used by fresh TSNPE.

This estimator defines only the initial restricted-prior support.  It is not a
paper posterior: every final TSNPE result is independently corrected by the
shared full target and deterministic-mixture importance sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eos.ddb import PRIOR_HIGH, PRIOR_LOW
from likelihoods.nuclear import A1_OBSERVATION, A1_SIGMA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_moments(values: np.ndarray, weight: np.ndarray):
    normalized = weight / np.sum(weight)
    mean = np.sum(values * normalized[:, None], axis=0)
    variance = np.sum(normalized[:, None] * (values - mean) ** 2, axis=0)
    scale = np.sqrt(variance)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError("seed-flow standardization has a non-positive scale")
    return mean, scale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=26_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.steps < 1 or arguments.batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    if arguments.smoke:
        arguments.steps = min(arguments.steps, 3)
        arguments.batch_size = min(arguments.batch_size, 64)

    import zuko

    output = arguments.output_dir
    checkpoint = output / "flow_nuclear_mmax.pt"
    standardization = output / "flow_nuclear_mmax_std.npz"
    report_path = output / "run_report.json"
    existing = [path for path in (checkpoint, standardization, report_path) if path.exists()]
    if existing and not arguments.force:
        raise FileExistsError(
            "seed-flow output already exists; let the campaign resume from its "
            "validated receipt or pass --force explicitly"
        )
    output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with np.load(arguments.bank, allow_pickle=False) as archive:
        required = {"theta", "X", "MM"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"training bank is missing {sorted(missing)}")
        theta = np.asarray(archive["theta"], dtype=np.float64)
        prediction = np.asarray(archive["X"], dtype=np.float64)
        maximum_mass = np.asarray(archive["MM"], dtype=np.float64)
        base_rows = int(archive["N0"]) if "N0" in archive.files else len(theta)
    if theta.ndim != 2 or theta.shape[1] != 7:
        raise ValueError("seed-flow theta must have shape (N,7)")
    if prediction.shape != theta.shape or maximum_mass.shape != (len(theta),):
        raise ValueError("seed-flow bank arrays have inconsistent shapes")
    if not 0 < base_rows <= len(theta):
        raise ValueError("seed-flow bank has an invalid uniform-base row count")
    # A common physics bank may append observation-box/radius-corner
    # enrichment after its original uniform-prior population.  Those rows are
    # not a uniform-prior sample.  The TSNPE-owned support flow therefore uses
    # only the leading N0 rows and never consumes another inference method's
    # checkpoint, posterior, or proposal geometry.
    theta = theta[:base_rows]
    prediction = prediction[:base_rows]
    maximum_mass = maximum_mass[:base_rows]
    valid = (
        np.isfinite(theta).all(axis=1)
        & np.isfinite(prediction).all(axis=1)
        & np.isfinite(maximum_mass)
        & np.all((theta >= PRIOR_LOW) & (theta <= PRIOR_HIGH), axis=1)
    )
    theta = theta[valid]
    prediction = prediction[valid]
    maximum_mass = maximum_mass[valid]
    if len(theta) < 100 and not arguments.smoke:
        raise RuntimeError("too few valid bank rows for seed-flow training")

    maximum_mass_weight = 1.0 / (
        1.0 + np.exp(-(maximum_mass - 2.0) / 0.05)
    )
    theta_mean, theta_scale = weighted_moments(theta, maximum_mass_weight)
    x_mean, x_scale = weighted_moments(prediction, maximum_mass_weight)

    device = torch.device(
        arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    theta_tensor = torch.as_tensor(
        (theta - theta_mean) / theta_scale,
        dtype=torch.float32,
        device=device,
    )
    prediction_tensor = torch.as_tensor(
        prediction, dtype=torch.float32, device=device
    )
    weight_tensor = torch.as_tensor(
        maximum_mass_weight, dtype=torch.float32, device=device
    )
    sigma_tensor = torch.as_tensor(A1_SIGMA, dtype=torch.float32, device=device)
    x_mean_tensor = torch.as_tensor(x_mean, dtype=torch.float32, device=device)
    x_scale_tensor = torch.as_tensor(x_scale, dtype=torch.float32, device=device)

    flow = zuko.flows.NSF(
        7,
        7,
        transforms=8,
        hidden_features=[192, 192, 192],
        bins=10,
    ).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), arguments.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, arguments.steps
    )
    losses = []
    for step in range(arguments.steps):
        index = torch.multinomial(
            weight_tensor, arguments.batch_size, replacement=True
        )
        noisy = prediction_tensor[index] + torch.randn(
            (arguments.batch_size, 7), device=device
        ) * sigma_tensor
        context = (noisy - x_mean_tensor) / x_scale_tensor
        loss = -flow(context).log_prob(theta_tensor[index]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 1000 == 0 or step == arguments.steps - 1:
            losses.append({"step": step, "nll": float(loss.detach().cpu())})

    torch.save(flow.state_dict(), checkpoint)
    np.savez(
        standardization,
        tm=theta_mean,
        ts=theta_scale,
        xm=x_mean,
        xs=x_scale,
        observation=A1_OBSERVATION,
        sigma=A1_SIGMA,
        prior_low=PRIOR_LOW,
        prior_high=PRIOR_HIGH,
    )
    with torch.no_grad():
        observed = torch.as_tensor(
            ((A1_OBSERVATION - x_mean) / x_scale)[None],
            dtype=torch.float32,
            device=device,
        )
        probe = flow(observed).sample((512,)).reshape(512, 7)
        probe = probe.cpu().numpy() * theta_scale + theta_mean
    probe_in_bounds = np.all(
        (probe >= PRIOR_LOW) & (probe <= PRIOR_HIGH), axis=1
    )
    if not arguments.smoke and probe_in_bounds.mean() < 0.5:
        raise RuntimeError("fresh seed flow puts less than half its probe mass in prior")

    report = {
        "status": "SMOKE_COMPLETED" if arguments.smoke else "SEED_READY",
        "role": "TSNPE round-zero support only",
        "scientific_posterior_certified": False,
        "bank": str(arguments.bank),
        "bank_sha256": sha256(arguments.bank),
        "uniform_base_rows": base_rows,
        "enriched_rows_excluded": True,
        "valid_training_rows": int(len(theta)),
        "steps": arguments.steps,
        "batch_size": arguments.batch_size,
        "seed": arguments.seed,
        "device": str(device),
        "probe_in_prior_fraction": float(probe_in_bounds.mean()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "standardization": str(standardization),
        "standardization_sha256": sha256(standardization),
        "losses": losses,
        "wall_seconds": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
