#!/usr/bin/env python3
"""Sample a frozen parity A-NET plus a clean defensive Student mixture."""

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
import torch
from scipy.special import logsumexp

from clean_student_t_ensemble import (
    STUDENT_ENSEMBLE_SCHEMA,
    checkpoint_density as student_density,
    sample_checkpoint as sample_student,
)
from inference.anet.proposal import DeepSets, VectorField, mass_radius_clouds
from workflows.nuclear_scenarios import NUCLEAR_SCENARIO_NAMES, nuclear_observation
from workflows.source_scenarios import SOURCE_SCENARIO_NAMES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ParityFMPE:
    def __init__(
        self,
        checkpoint: Path,
        standardization: Path,
        *,
        device: str,
        hidden: int = 512,
        layers: int = 4,
        steps: int = 64,
    ):
        self.device = torch.device(device)
        self.steps = int(steps)
        if self.steps < 1:
            raise ValueError("Heun steps must be positive")
        with np.load(standardization, allow_pickle=False) as saved:
            if bool(saved["reference_present"]) or saved["forbidden_artifacts_used"].size:
                raise RuntimeError("A-NET standardization declares forbidden ancestry")
            transform = str(saved["theta_transform"].item())
            self.learned_dim = int(
                saved["learned_dim"]
                if "learned_dim" in saved.files
                else len(saved["tmth"])
            )
            expected_transform = {
                7: "canonical_box_atanh_first_7d_plus_uniform_2d",
                9: "canonical_box_atanh_all_9d",
            }.get(self.learned_dim)
            if transform != expected_transform:
                raise RuntimeError("unexpected A-NET theta transform")
            self.theta_mean = np.asarray(saved["tmth"], dtype=np.float64)
            self.theta_scale = np.asarray(saved["tsth"], dtype=np.float64)
            self.cloud_mean = np.asarray(saved["cmean"], dtype=np.float64)
            self.cloud_scale = np.asarray(saved["cstd"], dtype=np.float64)
            self.x_mean = np.asarray(saved["xm"], dtype=np.float64)
            self.x_scale = np.asarray(saved["xs"], dtype=np.float64)
            self.prior_low = np.asarray(saved["prior_low"], dtype=np.float64)
            self.prior_high = np.asarray(saved["prior_high"], dtype=np.float64)
            self.cloud_points = int(saved["kpts"])
            dsd = int(saved["dsd"])
            dse = int(saved["dse"])
            context_dimension = int(saved["CTX"])
        if (
            self.theta_mean.shape != (self.learned_dim,)
            or self.theta_scale.shape != (self.learned_dim,)
            or self.x_mean.shape != (7,)
            or self.x_scale.shape != (7,)
            or self.prior_low.shape != (9,)
            or self.prior_high.shape != (9,)
            or context_dimension != 199
        ):
            raise RuntimeError("A-NET standardization has an invalid schema")
        if np.any(self.theta_scale <= 0) or np.any(self.x_scale <= 0):
            raise RuntimeError("A-NET standardization contains a non-positive scale")
        self.encoder = DeepSets(dsd, dse).to(self.device)
        self.flow = VectorField(
            self.learned_dim, context_dimension, hidden, layers
        ).to(self.device)
        try:
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=self.device)
        self.encoder.load_state_dict(state["ds"])
        self.flow.load_state_dict(state["vf"])
        self.encoder.eval()
        self.flow.eval()

    def context(self, clouds: np.ndarray, observation: np.ndarray, rows: int):
        if np.asarray(clouds).shape != (3, self.cloud_points, 2):
            raise ValueError("conditioning cloud has the wrong shape")
        cloud = torch.as_tensor(
            ((np.asarray(clouds, dtype=np.float64) - self.cloud_mean) / self.cloud_scale)[None],
            dtype=torch.float32,
            device=self.device,
        )
        nuclear = torch.as_tensor(
            ((np.asarray(observation, dtype=np.float64) - self.x_mean) / self.x_scale)[None],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            encoded = torch.cat([self.encoder(cloud), nuclear], dim=1)
        return encoded.expand(rows, -1)

    def _velocity_and_divergence(self, value, time_value, context):
        """Return the field and its exact trace Jacobian without backprop loops.

        The vector field is an MLP containing only Linear and exact-GELU
        layers.  Propagating its small input Jacobian alongside the ordinary
        forward pass gives the same divergence as autograd while fusing the
        nine directional derivatives into batched matrix multiplications.
        """

        with torch.no_grad():
            feature = torch.cat(
                [value, self.flow.time_embedding(time_value), context], dim=1
            )
            jacobian = None
            for layer in self.flow.net:
                if isinstance(layer, torch.nn.Linear):
                    feature = layer(feature)
                    if jacobian is None:
                        jacobian = layer.weight[:, : self.learned_dim].unsqueeze(0)
                        jacobian = jacobian.expand(len(value), -1, -1)
                    else:
                        jacobian = torch.einsum(
                            "oi,bid->bod", layer.weight, jacobian
                        )
                elif isinstance(layer, torch.nn.GELU):
                    derivative = 0.5 * (
                        1.0 + torch.erf(feature / math.sqrt(2.0))
                    ) + feature * torch.exp(-0.5 * feature.square()) / math.sqrt(
                        2.0 * math.pi
                    )
                    feature = layer(feature)
                    jacobian = jacobian * derivative.unsqueeze(-1)
                else:
                    raise RuntimeError(
                        f"unsupported vector-field layer {type(layer).__name__}"
                    )
            if jacobian is None or jacobian.shape[1:] != (
                self.learned_dim,
                self.learned_dim,
            ):
                raise RuntimeError("invalid analytic vector-field Jacobian")
            divergence = torch.diagonal(jacobian, dim1=1, dim2=2).sum(dim=1)
        return feature, divergence

    def _velocity_and_divergence_autograd(self, value, time_value, context):
        """Reference implementation retained for numerical audit tests."""

        differentiable = value.detach().requires_grad_(True)
        velocity = self.flow(differentiable, time_value, context)
        divergence = torch.zeros(len(value), device=self.device)
        for dimension in range(self.learned_dim):
            gradient = torch.autograd.grad(
                velocity[:, dimension].sum(),
                differentiable,
                retain_graph=dimension < self.learned_dim - 1,
            )[0]
            divergence = divergence + gradient[:, dimension]
        return velocity.detach(), divergence.detach()

    def sample(self, clouds, observation, rows, generator):
        context = self.context(clouds, observation, rows)
        state = torch.randn(
            (rows, self.learned_dim), generator=generator, device=self.device
        )
        step = 1.0 / self.steps
        with torch.no_grad():
            for index in range(self.steps):
                time_0 = torch.full((rows, 1), index * step, device=self.device)
                velocity_0 = self.flow(state, time_0, context)
                predictor = state + step * velocity_0
                velocity_1 = self.flow(predictor, time_0 + step, context)
                state = state + 0.5 * step * (velocity_0 + velocity_1)
        standardized = state.cpu().numpy().astype(np.float64)
        transformed = standardized * self.theta_scale + self.theta_mean
        box = np.tanh(transformed)
        theta = self.prior_low[: self.learned_dim] + 0.5 * (box + 1.0) * (
            self.prior_high[: self.learned_dim]
            - self.prior_low[: self.learned_dim]
        )
        if self.learned_dim < len(self.prior_low):
            unit = torch.rand(
                (rows, len(self.prior_low) - self.learned_dim),
                generator=generator,
                device=self.device,
            ).cpu().numpy().astype(np.float64)
            tail = self.prior_low[self.learned_dim :] + unit * (
                self.prior_high[self.learned_dim :]
                - self.prior_low[self.learned_dim :]
            )
            theta = np.column_stack([theta, tail])
        return np.clip(
            theta,
            np.nextafter(self.prior_low, self.prior_high),
            np.nextafter(self.prior_high, self.prior_low),
        )

    def log_density(self, theta, clouds, observation, batch_size):
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        inside = np.all(
            (theta > self.prior_low) & (theta < self.prior_high), axis=1
        )
        output = np.full(len(theta), -np.inf, dtype=np.float64)
        width = self.prior_high - self.prior_low
        for start in range(0, len(theta), batch_size):
            stop = min(start + batch_size, len(theta))
            local_inside = inside[start:stop]
            if not local_inside.any():
                continue
            physical = theta[start:stop][local_inside]
            box = np.clip(
                2.0
                * (physical[:, : self.learned_dim] - self.prior_low[: self.learned_dim])
                / width[: self.learned_dim]
                - 1.0,
                -1.0 + 1.0e-12,
                1.0 - 1.0e-12,
            )
            transformed = np.arctanh(box)
            state = torch.as_tensor(
                ((transformed - self.theta_mean) / self.theta_scale).astype(np.float32),
                device=self.device,
            )
            rows = len(state)
            context = self.context(clouds, observation, rows)
            integral_divergence = torch.zeros(rows, device=self.device)
            step = 1.0 / self.steps
            for index in range(self.steps, 0, -1):
                time_1 = torch.full((rows, 1), index * step, device=self.device)
                time_0 = torch.full((rows, 1), (index - 1) * step, device=self.device)
                velocity_1, divergence_1 = self._velocity_and_divergence(
                    state, time_1, context
                )
                predictor = state - step * velocity_1
                velocity_0, divergence_0 = self._velocity_and_divergence(
                    predictor, time_0, context
                )
                state = state - 0.5 * step * (velocity_1 + velocity_0)
                integral_divergence += 0.5 * step * (
                    divergence_1 + divergence_0
                )
            base = (
                -0.5 * (state * state).sum(dim=1)
                - 0.5 * self.learned_dim * math.log(2.0 * math.pi)
                - integral_divergence
            ).detach().cpu().numpy().astype(np.float64)
            physical_logq = (
                base
                - np.log(self.theta_scale).sum()
                + (
                    np.log(2.0 / width[: self.learned_dim])[None, :]
                    - np.log1p(-np.square(box))
                ).sum(axis=1)
            )
            if self.learned_dim < len(self.prior_low):
                physical_logq -= np.log(width[self.learned_dim :]).sum()
            local = np.full(stop - start, -np.inf, dtype=np.float64)
            local[local_inside] = physical_logq
            output[start:stop] = local
            print(f"[parity-proposal-density] {stop}/{len(theta)}", flush=True)
        return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anet-checkpoint", type=Path, required=True)
    parser.add_argument("--standardization", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--source-scenario", choices=SOURCE_SCENARIO_NAMES, required=True)
    parser.add_argument("--nuclear-scenario", choices=NUCLEAR_SCENARIO_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=25_000)
    parser.add_argument("--flow-weight", type=float, default=0.70)
    parser.add_argument("--cloud-seed", type=int, default=777_001)
    parser.add_argument("--proposal-seed", type=int, default=20_260_944)
    parser.add_argument("--flow-steps", type=int, default=64)
    parser.add_argument("--draw-batch", type=int, default=20_000)
    parser.add_argument("--density-batch", type=int, default=3_000)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.source_scenario != "A1" and arguments.nuclear_scenario != "A1":
        raise ValueError("change either NICER source or nuclear observation, not both")
    if not 0.0 < arguments.flow_weight < 1.0:
        raise ValueError("flow weight must lie strictly between zero and one")
    if min(arguments.draws, arguments.draw_batch, arguments.density_batch) < 1:
        raise ValueError("proposal counts must be positive")
    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    training = json.loads(arguments.training_report.read_text())
    if training.get("reference_present") is not False or training.get("forbidden_artifacts_used") != []:
        raise RuntimeError("training receipt declares forbidden ancestry")
    if training.get("held_out_sources_absent") != ["J0614", "J1231", "J1614"]:
        raise RuntimeError("training receipt does not exclude all held-out sources")
    if training.get("checkpoint_sha256") != sha256(arguments.anet_checkpoint):
        raise RuntimeError("training receipt/checkpoint hash mismatch")
    if training.get("standardization_sha256") != sha256(arguments.standardization):
        raise RuntimeError("training receipt/standardization hash mismatch")

    proposal = ParityFMPE(
        arguments.anet_checkpoint,
        arguments.standardization,
        device=arguments.device,
        steps=arguments.flow_steps,
    )
    student = joblib.load(arguments.student_checkpoint)
    if student.get("schema") != STUDENT_ENSEMBLE_SCHEMA:
        raise RuntimeError("unrecognized defensive Student checkpoint")
    if not (
        np.array_equal(np.asarray(student["prior_low"]), proposal.prior_low)
        and np.array_equal(np.asarray(student["prior_high"]), proposal.prior_high)
    ):
        raise RuntimeError("A-NET and defensive Student priors differ")

    clouds = mass_radius_clouds(
        arguments.data_root,
        points=proposal.cloud_points,
        seed=arguments.cloud_seed,
        source_scenario=arguments.source_scenario,
        source_data_root=arguments.source_data_root,
        verify_data=True,
    )
    observation = nuclear_observation(arguments.nuclear_scenario)
    started = time.time()
    assignment_rng = np.random.default_rng(arguments.proposal_seed)
    component = (
        assignment_rng.random(arguments.draws) >= arguments.flow_weight
    ).astype(np.int8)
    flow_index = np.flatnonzero(component == 0)
    student_index = np.flatnonzero(component == 1)
    theta = np.empty((arguments.draws, 9), dtype=np.float64)
    generator = torch.Generator(device=proposal.device).manual_seed(
        arguments.proposal_seed + 1
    )
    for start in range(0, len(flow_index), arguments.draw_batch):
        local_index = flow_index[start : start + arguments.draw_batch]
        theta[local_index] = proposal.sample(
            clouds, observation, len(local_index), generator
        )
    theta[student_index] = sample_student(
        student,
        len(student_index),
        np.random.default_rng(arguments.proposal_seed + 2),
    )
    draw_seconds = time.time() - started

    density_started = time.time()
    flow_logq = proposal.log_density(
        theta, clouds, observation, arguments.density_batch
    )
    prior_logq = -float(np.log(proposal.prior_high - proposal.prior_low).sum())
    student_logq = student_density(student, theta, prior_logq)
    mixture_logq = logsumexp(
        np.vstack(
            [
                math.log(arguments.flow_weight) + flow_logq,
                math.log1p(-arguments.flow_weight) + student_logq,
            ]
        ),
        axis=0,
    )
    density_seconds = time.time() - density_started
    if not np.isfinite(theta).all() or not np.isfinite(mixture_logq).all():
        raise RuntimeError("non-finite hybrid proposal output")
    if not np.all(
        (theta >= proposal.prior_low) & (theta <= proposal.prior_high)
    ):
        raise RuntimeError("hybrid proposal draw lies outside the canonical prior")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp-{os.getpid()}"
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray("ddb-hyperonic-clean-parity-anet-mixture-proposal-v1"),
            theta=theta,
            logq=mixture_logq,
            flow_logq=flow_logq,
            student_logq=student_logq,
            component=component,
            cloud=clouds,
            prior_low=proposal.prior_low,
            prior_high=proposal.prior_high,
            flow_weight=np.float64(arguments.flow_weight),
            checkpoint_sha256=np.asarray(sha256(arguments.anet_checkpoint)),
            standardization_sha256=np.asarray(sha256(arguments.standardization)),
            training_report_sha256=np.asarray(sha256(arguments.training_report)),
            student_checkpoint_sha256=np.asarray(sha256(arguments.student_checkpoint)),
            nuclear_scenario=np.asarray(arguments.nuclear_scenario),
            source_scenario=np.asarray(arguments.source_scenario),
            nuclear_observation=observation,
            evaluated_source_artifact_absent_from_training=np.bool_(
                arguments.source_scenario != "A1"
            ),
            held_out_sources_absent=np.asarray(["J0614", "J1231", "J1614"]),
            flow_steps=np.int64(arguments.flow_steps),
            cloud_seed=np.int64(arguments.cloud_seed),
            proposal_seed=np.int64(arguments.proposal_seed),
            reference_present=np.bool_(False),
            forbidden_artifacts_used=np.asarray([], dtype="U1"),
        )
    os.replace(temporary, arguments.output)
    report = {
        "status": "PASS",
        "source_scenario": arguments.source_scenario,
        "nuclear_scenario": arguments.nuclear_scenario,
        "draws": arguments.draws,
        "flow_rows": int(len(flow_index)),
        "student_rows": int(len(student_index)),
        "flow_weight": arguments.flow_weight,
        "flow_steps": arguments.flow_steps,
        "draw_seconds": draw_seconds,
        "density_seconds": density_seconds,
        "wall_seconds": time.time() - started,
        "checkpoint_sha256": sha256(arguments.anet_checkpoint),
        "standardization_sha256": sha256(arguments.standardization),
        "training_report_sha256": sha256(arguments.training_report),
        "student_checkpoint_sha256": sha256(arguments.student_checkpoint),
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "held_out_sources_absent": ["J0614", "J1231", "J1614"],
        "reference_present": False,
        "forbidden_artifacts_used": [],
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
