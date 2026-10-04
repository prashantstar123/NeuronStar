#!/usr/bin/env python3
"""Train the nucleonic flow-matching A-NET from fresh scenario data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.anet.proposal import DeepSets, VectorField
from workflows.data_gate import sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scenario-batch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heun-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if min(
        arguments.epochs,
        arguments.scenario_batch,
        arguments.hidden,
        arguments.layers,
        arguments.heun_steps,
    ) < 1:
        raise ValueError("all A-NET training settings must be positive")
    if arguments.smoke:
        arguments.epochs = min(arguments.epochs, 2)

    checkpoint = arguments.output_dir / "anet_net.pt"
    standardization = arguments.output_dir / "anet_std.npz"
    report_path = arguments.output_dir / "run_report.json"
    if any(path.exists() for path in (checkpoint, standardization, report_path)) and not arguments.force:
        raise FileExistsError(
            "A-NET output exists; resume from its validated campaign receipt or "
            "pass --force explicitly"
        )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with np.load(arguments.scenarios, allow_pickle=False) as archive:
        required = {"clouds", "dial", "theta_len", "theta", "gtheta", "ntr", "metadata"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"scenario archive is missing {sorted(missing)}")
        clouds = np.asarray(archive["clouds"], dtype=np.float64)
        dial = np.asarray(archive["dial"], dtype=np.float64)
        lengths = np.asarray(archive["theta_len"], dtype=np.int64)
        theta_all = np.asarray(archive["theta"], dtype=np.float64)
        prediction_all = np.asarray(archive["gtheta"], dtype=np.float64)
        training_scenarios = int(archive["ntr"])
        metadata = json.loads(str(archive["metadata"].item()))
    if clouds.ndim != 4 or clouds.shape[1:] != (3, 256, 2):
        raise ValueError("nucleonic A-NET clouds must have shape (S,3,256,2)")
    if dial.shape != (len(clouds), 2) or lengths.shape != (len(clouds),):
        raise ValueError("scenario context arrays have inconsistent shapes")
    if lengths.sum() != len(theta_all) or prediction_all.shape != theta_all.shape:
        raise ValueError("scenario posterior rows are inconsistent")
    if theta_all.shape[1] != 7:
        raise ValueError("nucleonic A-NET theta must be seven-dimensional")
    if not 0 < training_scenarios < len(clouds):
        raise ValueError("scenario archive has no train/validation split")

    with np.load(arguments.bank, allow_pickle=False) as bank:
        bank_prediction = np.asarray(bank["X"], dtype=np.float64)
        bank_maximum_mass = np.asarray(bank["MM"], dtype=np.float64)
        base_rows = int(bank["N0"]) if "N0" in bank.files else len(bank_prediction)
    # Nuclear-context standardization follows the uniform-prior, valid-stellar
    # bank.  The enriched rows are deliberately excluded from these moments.
    valid_bank = (
        np.isfinite(bank_prediction[:base_rows]).all(axis=1)
        & np.isfinite(bank_maximum_mass[:base_rows])
    )
    x_mean = bank_prediction[:base_rows][valid_bank].mean(axis=0)
    x_scale = bank_prediction[:base_rows][valid_bank].std(axis=0)
    theta_mean = theta_all.mean(axis=0)
    theta_scale = theta_all.std(axis=0)
    if np.any(x_scale <= 0) or np.any(theta_scale <= 0):
        raise RuntimeError("A-NET standardization contains a non-positive scale")

    centres = np.asarray(
        [metadata["centres"][name] for name in ("j0030", "j0740", "j0437")],
        dtype=np.float64,
    )
    slot_scale = np.asarray(
        [
            metadata["slot_standard_deviation"][name]
            for name in ("j0030", "j0740", "j0437")
        ],
        dtype=np.float64,
    )
    cloud_mean = centres.mean(axis=0)
    cloud_scale = np.sqrt((slot_scale**2 + centres**2).mean(axis=0) - cloud_mean**2)
    if np.any(cloud_scale <= 0):
        raise RuntimeError("A-NET cloud standardization is singular")

    gamma_mean, gamma_scale = 0.0, 0.346
    rho_mean, rho_scale = 1.2, 0.115
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

    device = torch.device(
        arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(arguments.seed)
    cloud_tensor = torch.as_tensor(
        (clouds - cloud_mean) / cloud_scale,
        dtype=torch.float32,
        device=device,
    )
    dial_tensor = torch.as_tensor(
        (dial - [gamma_mean, rho_mean]) / [gamma_scale, rho_scale],
        dtype=torch.float32,
        device=device,
    )
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    theta_tensors = [
        torch.as_tensor(
            (theta_all[offsets[index] : offsets[index + 1]] - theta_mean)
            / theta_scale,
            dtype=torch.float32,
            device=device,
        )
        for index in range(len(clouds))
    ]
    prediction_tensors = [
        torch.as_tensor(
            prediction_all[offsets[index] : offsets[index + 1]],
            dtype=torch.float32,
            device=device,
        )
        for index in range(len(clouds))
    ]
    sigma_tensor = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    x_mean_tensor = torch.as_tensor(x_mean, dtype=torch.float32, device=device)
    x_scale_tensor = torch.as_tensor(x_scale, dtype=torch.float32, device=device)

    deep_sets = DeepSets(128, 64).to(device)
    vector_field = VectorField(7, 3 * 64 + 7 + 2, arguments.hidden, arguments.layers).to(
        device
    )
    optimizer = torch.optim.Adam(
        list(deep_sets.parameters()) + list(vector_field.parameters()),
        arguments.learning_rate,
        weight_decay=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, arguments.epochs
    )

    def fresh_nuclear(index):
        return (
            prediction_tensors[index]
            + torch.randn_like(prediction_tensors[index]) * sigma_tensor
            - x_mean_tensor
        ) / x_scale_tensor

    validation_noise = {}
    validation_generator = torch.Generator(device=device)
    for index in range(training_scenarios, len(clouds)):
        validation_generator.manual_seed(1000 + index)
        validation_noise[index] = (
            prediction_tensors[index]
            + torch.randn(
                prediction_tensors[index].shape,
                device=device,
                generator=validation_generator,
            )
            * sigma_tensor
            - x_mean_tensor
        ) / x_scale_tensor

    def validation_loss():
        generator = torch.Generator(device=device).manual_seed(555)
        total = 0.0
        with torch.no_grad():
            for index in range(training_scenarios, len(clouds)):
                target = theta_tensors[index]
                rows = len(target)
                context = torch.cat(
                    [
                        deep_sets(cloud_tensor[index : index + 1]).expand(rows, -1),
                        validation_noise[index],
                        dial_tensor[index : index + 1].expand(rows, -1),
                    ],
                    dim=1,
                )
                time_value = torch.rand(rows, 1, device=device, generator=generator)
                noise = torch.randn(target.shape, device=device, generator=generator)
                interpolated = (
                    1.0 - (1.0 - 1e-3) * time_value
                ) * noise + time_value * target
                velocity = target - (1.0 - 1e-3) * noise
                total += float(
                    ((vector_field(interpolated, time_value, context) - velocity) ** 2)
                    .mean()
                    .cpu()
                )
        return total / (len(clouds) - training_scenarios)

    best = float("inf")
    history = []
    for epoch in range(arguments.epochs):
        permutation = torch.randperm(training_scenarios).tolist()
        total = 0.0
        batches = 0
        for start in range(0, training_scenarios, arguments.scenario_batch):
            group = permutation[start : start + arguments.scenario_batch]
            contexts = []
            targets = []
            for index in group:
                target = theta_tensors[index]
                rows = len(target)
                contexts.append(
                    torch.cat(
                        [
                            deep_sets(cloud_tensor[index : index + 1]).expand(rows, -1),
                            fresh_nuclear(index),
                            dial_tensor[index : index + 1].expand(rows, -1),
                        ],
                        dim=1,
                    )
                )
                targets.append(target)
            context = torch.cat(contexts, dim=0)
            target = torch.cat(targets, dim=0)
            time_value = torch.rand(len(target), 1, device=device)
            noise = torch.randn_like(target)
            interpolated = (
                1.0 - (1.0 - 1e-3) * time_value
            ) * noise + time_value * target
            velocity = target - (1.0 - 1e-3) * noise
            loss = ((vector_field(interpolated, time_value, context) - velocity) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        if epoch % 10 == 0 or epoch == arguments.epochs - 1:
            value = validation_loss()
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": total / batches,
                    "validation_loss": value,
                }
            )
            if value < best:
                best = value
                torch.save(
                    {"ds": deep_sets.state_dict(), "vf": vector_field.state_dict()},
                    checkpoint,
                )
            print(
                f"[anet-train] epoch {epoch}: train={total / batches:.4f} "
                f"validation={value:.4f} best={best:.4f}",
                flush=True,
            )

    np.savez(
        standardization,
        tmth=theta_mean,
        tsth=theta_scale,
        xm=x_mean,
        xs=x_scale,
        cmean=cloud_mean,
        cstd=cloud_scale,
        GL_M=np.float64(gamma_mean),
        GL_S=np.float64(gamma_scale),
        RH_M=np.float64(rho_mean),
        RH_S=np.float64(rho_scale),
        ntr=np.int64(training_scenarios),
        nreal=np.int64(metadata.get("identity_scenarios", 15)),
        OBS=observation,
        SIG=sigma,
        CTX=np.int64(201),
        dsd=np.int64(128),
        dse=np.int64(64),
        kpts=np.int64(256),
        heun_steps=np.int64(arguments.heun_steps),
        scenario_sha256=np.asarray(sha256(arguments.scenarios)),
        bank_sha256=np.asarray(sha256(arguments.bank)),
    )
    report = {
        "status": "SMOKE_COMPLETED" if arguments.smoke else "PROPOSAL_READY_FOR_IS",
        "method": "nucleonic FMPE A-NET",
        "scientific_posterior_certified": False,
        "certification_stage": "run exact A-NET+IS for every posterior configuration",
        "scenarios": str(arguments.scenarios),
        "scenarios_sha256": sha256(arguments.scenarios),
        "bank": str(arguments.bank),
        "bank_sha256": sha256(arguments.bank),
        "epochs": arguments.epochs,
        "training_scenarios": training_scenarios,
        "posterior_training_rows": int(len(theta_all)),
        "seed": arguments.seed,
        "device": str(device),
        "best_validation_loss": best,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "standardization": str(standardization),
        "standardization_sha256": sha256(standardization),
        "wall_seconds": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
