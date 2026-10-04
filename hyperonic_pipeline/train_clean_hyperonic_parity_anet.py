#!/usr/bin/env python3
"""Train a DeepSets+FMPE A-NET by the locked nucleonic-parity workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from inference.anet.proposal import DeepSets, VectorField
from likelihoods.nuclear import A1_OBSERVATION, A1_SIGMA


SOURCES = ("j0030", "j0740", "j0437")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_prediction_moments(paths: list[Path]):
    count = 0
    total = np.zeros(7, dtype=np.float64)
    total_square = np.zeros(7, dtype=np.float64)
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "schema",
                "prediction",
                "reference_present",
                "forbidden_artifacts_used",
            }
            missing = required - set(archive.files)
            if missing:
                raise RuntimeError(f"base cache {path} is missing {sorted(missing)}")
            if str(archive["schema"].item()) != "ddb-hyperonic-clean-uniform-training-cache-v1":
                raise RuntimeError(f"unexpected base-cache schema in {path}")
            if bool(archive["reference_present"]):
                raise RuntimeError(f"reference contamination declared by {path}")
            if archive["forbidden_artifacts_used"].size:
                raise RuntimeError(f"forbidden ancestry declared by {path}")
            prediction = np.asarray(archive["prediction"], dtype=np.float64)
        valid = np.isfinite(prediction).all(axis=1)
        local = prediction[valid]
        count += len(local)
        total += local.sum(axis=0)
        total_square += np.square(local).sum(axis=0)
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "rows": int(len(prediction)),
                "finite_prediction_rows": int(len(local)),
            }
        )
    if count < 2:
        raise RuntimeError("base caches contain too few finite predictions")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    if np.any(scale <= 0):
        raise RuntimeError("base-cache nuclear standardization is singular")
    return mean, scale, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--scenario-batch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heun-steps", type=int, default=64)
    parser.add_argument("--learned-dim", type=int, choices=(7, 9), default=9)
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
    if (
        any(path.exists() for path in (checkpoint, standardization, report_path))
        and not arguments.force
    ):
        raise FileExistsError(f"A-NET output already exists in {arguments.output_dir}")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with np.load(arguments.scenarios, allow_pickle=False) as archive:
        required = {
            "clouds",
            "theta_len",
            "theta",
            "gtheta",
            "ntr",
            "metadata",
            "reference_present",
            "forbidden_artifacts_used",
        }
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"scenario archive is missing {sorted(missing)}")
        if bool(archive["reference_present"]):
            raise RuntimeError("scenario archive declares reference contamination")
        if archive["forbidden_artifacts_used"].size:
            raise RuntimeError("scenario archive declares forbidden ancestry")
        clouds = np.asarray(archive["clouds"], dtype=np.float64)
        lengths = np.asarray(archive["theta_len"], dtype=np.int64)
        theta_all = np.asarray(archive["theta"], dtype=np.float64)
        prediction_all = np.asarray(archive["gtheta"], dtype=np.float64)
        training_scenarios = int(archive["ntr"])
        metadata = json.loads(str(archive["metadata"].item()))
    if clouds.ndim != 4 or clouds.shape[1:] != (3, 256, 2):
        raise RuntimeError("A-NET clouds must have shape (S,3,256,2)")
    if lengths.shape != (len(clouds),):
        raise RuntimeError("scenario lengths and clouds are inconsistent")
    if lengths.sum() != len(theta_all):
        raise RuntimeError("scenario posterior lengths do not sum to theta rows")
    if theta_all.shape[1:] != (9,) or prediction_all.shape != (len(theta_all), 7):
        raise RuntimeError("scenario theta/prediction dimensions are invalid")
    if not np.isfinite(theta_all).all() or not np.isfinite(prediction_all).all():
        raise RuntimeError("scenario posterior arrays contain non-finite values")
    if not 0 < training_scenarios < len(clouds):
        raise RuntimeError("scenario archive has no train/validation split")
    if metadata.get("held_out_sources_absent") != ["J0614", "J1231", "J1614"]:
        raise RuntimeError("scenario metadata does not lock all held-out sources")
    if metadata.get("reference_present") is not False:
        raise RuntimeError("scenario metadata lacks a negative reference declaration")
    if metadata.get("forbidden_artifacts_used") != []:
        raise RuntimeError("scenario metadata declares forbidden artifacts")

    x_mean, x_scale, base_cache_records = base_prediction_moments(
        arguments.base_cache
    )
    prior_low = np.asarray(metadata["prior_low"], dtype=np.float64)
    prior_high = np.asarray(metadata["prior_high"], dtype=np.float64)
    if prior_low.shape != (9,) or prior_high.shape != (9,) or np.any(prior_low >= prior_high):
        raise RuntimeError("scenario metadata contains invalid prior bounds")
    if not np.all((theta_all >= prior_low) & (theta_all <= prior_high)):
        raise RuntimeError("scenario theta contains an out-of-prior row")
    theta_box = np.clip(
        2.0 * (theta_all - prior_low) / (prior_high - prior_low) - 1.0,
        -1.0 + 1.0e-9,
        1.0 - 1.0e-9,
    )
    learned_dim = int(arguments.learned_dim)
    transformed_theta = np.arctanh(theta_box[:, :learned_dim])
    theta_mean = transformed_theta.mean(axis=0)
    theta_scale = transformed_theta.std(axis=0)
    if np.any(theta_scale <= 0):
        raise RuntimeError("A-NET theta standardization is singular")
    centres = np.asarray(
        [metadata["centres"][name] for name in SOURCES], dtype=np.float64
    )
    slot_scale = np.asarray(
        [metadata["slot_standard_deviation"][name] for name in SOURCES],
        dtype=np.float64,
    )
    cloud_mean = centres.mean(axis=0)
    cloud_scale = np.sqrt(
        (slot_scale**2 + centres**2).mean(axis=0) - cloud_mean**2
    )
    if np.any(cloud_scale <= 0):
        raise RuntimeError("A-NET cloud standardization is singular")

    device = torch.device(
        arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(arguments.seed)
    cloud_tensor = torch.as_tensor(
        (clouds - cloud_mean) / cloud_scale,
        dtype=torch.float32,
        device=device,
    )
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    theta_tensors = [
        torch.as_tensor(
            (transformed_theta[offsets[index] : offsets[index + 1]] - theta_mean)
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
    sigma_tensor = torch.as_tensor(A1_SIGMA, dtype=torch.float32, device=device)
    x_mean_tensor = torch.as_tensor(x_mean, dtype=torch.float32, device=device)
    x_scale_tensor = torch.as_tensor(x_scale, dtype=torch.float32, device=device)

    deep_sets = DeepSets(128, 64).to(device)
    vector_field = VectorField(
        learned_dim, 3 * 64 + 7, arguments.hidden, arguments.layers
    ).to(device)
    optimizer = torch.optim.Adam(
        list(deep_sets.parameters()) + list(vector_field.parameters()),
        arguments.learning_rate,
        weight_decay=1.0e-6,
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
                    ],
                    dim=1,
                )
                time_value = torch.rand(rows, 1, device=device, generator=generator)
                noise = torch.randn(target.shape, device=device, generator=generator)
                interpolated = (
                    1.0 - (1.0 - 1.0e-3) * time_value
                ) * noise + time_value * target
                velocity = target - (1.0 - 1.0e-3) * noise
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
                1.0 - (1.0 - 1.0e-3) * time_value
            ) * noise + time_value * target
            velocity = target - (1.0 - 1.0e-3) * noise
            loss = (
                (vector_field(interpolated, time_value, context) - velocity) ** 2
            ).mean()
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
                f"[hyperonic-parity-train] epoch {epoch}: "
                f"train={total / batches:.4f} validation={value:.4f} "
                f"best={best:.4f}",
                flush=True,
            )

    np.savez(
        standardization,
        tmth=theta_mean,
        tsth=theta_scale,
        prior_low=prior_low,
        prior_high=prior_high,
        theta_transform=np.asarray(
            "canonical_box_atanh_all_9d"
            if learned_dim == 9
            else "canonical_box_atanh_first_7d_plus_uniform_2d"
        ),
        learned_dim=np.int64(learned_dim),
        xm=x_mean,
        xs=x_scale,
        cmean=cloud_mean,
        cstd=cloud_scale,
        ntr=np.int64(training_scenarios),
        nreal=np.int64(metadata.get("identity_scenarios", 15)),
        OBS=np.asarray(A1_OBSERVATION, dtype=np.float64),
        SIG=np.asarray(A1_SIGMA, dtype=np.float64),
        CTX=np.int64(199),
        dsd=np.int64(128),
        dse=np.int64(64),
        kpts=np.int64(256),
        heun_steps=np.int64(arguments.heun_steps),
        scenario_sha256=np.asarray(sha256(arguments.scenarios)),
        base_cache_sha256=np.asarray(
            [record["sha256"] for record in base_cache_records]
        ),
        reference_present=np.bool_(False),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    report = {
        "status": "SMOKE_COMPLETED" if arguments.smoke else "PROPOSAL_READY_FOR_IS",
        "method": (
            "nine-dimensional hyperonic DeepSets + FMPE A-NET"
            if learned_dim == 9
            else "seven-dimensional hyperonic DeepSets + FMPE A-NET with "
            "two canonical-prior hyperon coordinates"
        ),
        "learned_dim": learned_dim,
        "nucleonic_parity": True,
        "scientific_posterior_certified": False,
        "certification_stage": "fresh exact nine-dimensional importance sampling",
        "scenarios": str(arguments.scenarios.resolve()),
        "scenarios_sha256": sha256(arguments.scenarios),
        "base_caches": base_cache_records,
        "epochs": arguments.epochs,
        "training_scenarios": training_scenarios,
        "posterior_training_rows": int(len(theta_all)),
        "seed": arguments.seed,
        "device": str(device),
        "best_validation_loss": best,
        "history": history,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "standardization": str(standardization.resolve()),
        "standardization_sha256": sha256(standardization),
        "reference_present": False,
        "forbidden_artifacts_used": [],
        "held_out_sources_absent": ["J0614", "J1231", "J1614"],
        "wall_seconds": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
