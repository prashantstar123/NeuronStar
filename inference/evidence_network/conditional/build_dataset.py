#!/usr/bin/env python3
"""Build jointly amortized NICER-conditioned EN training scenarios.

The target for each synthetic NICER configuration is the restricted
astrophysical tilt used by the normalized-flow evidence identity.  No
conventional or neural posterior enters the construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from inference.evidence_network.cache import (  # noqa: E402
    DEFAULT_NUCLEAR_OBSERVATION,
    DEFAULT_NUCLEAR_SIGMA,
)
from inference.evidence_network.conditional.batched_nicer import (  # noqa: E402
    BatchedShiftedNICER,
)
from likelihoods.nicer import build_nicer_grid  # noqa: E402
from workflows.source_scenarios import BASE_NICER_SOURCES  # noqa: E402


SOURCE_NAMES = ("j0030", "j0740", "j0437")
IDENTITY = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
HELD_OUT_FILENAMES = (
    "J0614_mrsamples.dat",
    "J1231_wmrsamples.txt",
    "J1614_STU_mrsamples_post_equal_weights.dat",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def weighted_source_table(path: Path, source):
    table = np.loadtxt(path)
    weight = (
        np.ones(len(table), dtype=np.float64)
        if source.weight_column is None
        else np.asarray(table[:, source.weight_column], dtype=np.float64)
    )
    weight /= weight.sum()
    mass = np.asarray(table[:, source.mass_column], dtype=np.float64)
    radius = np.asarray(table[:, source.radius_column], dtype=np.float64)
    centre = np.asarray(
        [np.sum(weight * mass), np.sum(weight * radius)], dtype=np.float64
    )
    return mass, radius, weight, centre


class ScenarioGenerator:
    def __init__(
        self,
        data_root: Path,
        points: int,
        seed: int,
        identity_fraction: float,
        one_slot_fraction: float,
    ) -> None:
        self.data_root = data_root
        self.points = int(points)
        self.rng = np.random.default_rng(seed)
        self.identity_fraction = float(identity_fraction)
        self.one_slot_fraction = float(one_slot_fraction)
        if (
            self.identity_fraction < 0
            or self.one_slot_fraction < 0
            or self.identity_fraction + self.one_slot_fraction > 1
        ):
            raise ValueError("invalid conditional-NICER mode fractions")
        self.centres: dict[str, tuple[float, float]] = {}
        self.pools: dict[str, np.ndarray] = {}
        self.source_records: list[dict[str, object]] = []
        for name, source in zip(SOURCE_NAMES, BASE_NICER_SOURCES, strict=True):
            path = data_root / source.filename
            mass, radius, weight, centre = weighted_source_table(path, source)
            self.centres[name] = (float(centre[0]), float(centre[1]))
            # Mirror build_nicer_grid exactly: for these large source files,
            # seed-0 probability-proportional sampling without replacement is
            # followed by an unweighted KDE.  The conditioning cloud must
            # describe that same empirical distribution.
            if len(mass) > 20_000:
                selected = np.random.default_rng(0).choice(
                    len(mass), size=20_000, replace=False, p=weight
                )
            else:
                selected = self.rng.choice(
                    len(mass), size=len(mass), replace=True, p=weight
                )
            self.pools[name] = np.column_stack(
                [mass[selected], radius[selected]]
            )
            self.source_records.append(
                {
                    "slot": name,
                    "filename": source.filename,
                    "sha256": sha256(path),
                    "rows": int(len(mass)),
                }
            )

    def draw_scale(self) -> float:
        return float(np.exp(self.rng.uniform(np.log(0.28), np.log(2.2))))

    def draw_transform(self) -> np.ndarray:
        return np.asarray(
            [
                self.rng.uniform(-0.5, 0.25),
                self.rng.uniform(-1.6, 1.6),
                self.draw_scale(),
                self.draw_scale(),
            ],
            dtype=np.float32,
        )

    def draw(self, count: int):
        transforms = np.repeat(IDENTITY[None, None, :], count * 3, axis=0).reshape(
            count, 3, 4
        )
        modes = np.empty(count, dtype=np.int8)
        clouds = np.empty((count, 3, self.points, 2), dtype=np.float32)
        for scenario in range(count):
            selector = self.rng.random()
            if selector < self.identity_fraction:
                active: tuple[int, ...] = ()
                modes[scenario] = 0
            elif selector < self.identity_fraction + self.one_slot_fraction:
                active = (int(self.rng.integers(3)),)
                modes[scenario] = 1
            else:
                active = (0, 1, 2)
                modes[scenario] = 3
            for slot in active:
                transforms[scenario, slot] = self.draw_transform()
            for slot, name in enumerate(SOURCE_NAMES):
                centre = np.asarray(self.centres[name], dtype=np.float64)
                pool = self.pools[name]
                selected = pool[
                    self.rng.integers(0, len(pool), size=self.points)
                ]
                mass_shift, radius_shift, mass_scale, radius_scale = transforms[
                    scenario, slot
                ]
                clouds[scenario, slot, :, 0] = (
                    centre[0]
                    + mass_scale * (selected[:, 0] - centre[0])
                    + mass_shift
                )
                clouds[scenario, slot, :, 1] = (
                    centre[1]
                    + radius_scale * (selected[:, 1] - centre[1])
                    + radius_shift
                )
        return transforms, clouds, modes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=2000)
    parser.add_argument("--validation-scenarios", type=int, default=200)
    parser.add_argument("--samples-per-scenario", type=int, default=2048)
    parser.add_argument("--cloud-points", type=int, default=256)
    parser.add_argument("--scenario-batch", type=int, default=16)
    parser.add_argument("--row-chunk", type=int, default=50_000)
    parser.add_argument("--quadrature-points", type=int, default=40)
    parser.add_argument("--restriction-radius", type=float, default=6.0)
    parser.add_argument("--minimum-tilt-ess", type=float, default=10.0)
    parser.add_argument("--identity-fraction", type=float, default=0.25)
    parser.add_argument("--one-slot-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists() and not arguments.force:
        raise FileExistsError(arguments.output)
    if arguments.smoke:
        arguments.scenarios = min(arguments.scenarios, 24)
        arguments.validation_scenarios = min(arguments.validation_scenarios, 4)
        arguments.samples_per_scenario = min(arguments.samples_per_scenario, 128)
        # Exercise the same vectorized scenario batching used in production.
        # Keep large row chunks on the GPU during the smoke test.  Shrinking
        # this to a few thousand rows made the diagnostic dominated by Python
        # launch overhead and did not exercise the production batching path.
        arguments.quadrature_points = min(arguments.quadrature_points, 12)
        arguments.minimum_tilt_ess = min(arguments.minimum_tilt_ess, 1.0)
    if not 0 < arguments.validation_scenarios < arguments.scenarios:
        raise ValueError("invalid training/validation scenario split")
    for filename in HELD_OUT_FILENAMES:
        if (arguments.data_root / filename).exists():
            raise RuntimeError(
                f"held-out NICER source is present in training root: {filename}"
            )

    started = time.perf_counter()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    with np.load(arguments.bank, allow_pickle=False) as bank:
        required = {"schema", "model", "X", "R", "MG", "MM", "LA", "LPC", "lpc_all_lse", "metadata"}
        missing = required - set(bank.files)
        if missing:
            raise RuntimeError(f"conditional physics bank lacks {sorted(missing)}")
        if str(bank["schema"].item()) != "conditional-en-independent-physics-v1":
            raise RuntimeError("unexpected conditional physics-bank schema")
        model_name = str(bank["model"].item())
        prediction = np.asarray(bank["X"], dtype=np.float32)
        radius = np.asarray(bank["R"], dtype=np.float32)
        mass_grid = np.asarray(bank["MG"], dtype=np.float32)
        maximum_mass = np.asarray(bank["MM"], dtype=np.float32)
        log_astro = np.asarray(bank["LA"], dtype=np.float32)
        correction = np.asarray(bank["LPC"], dtype=np.float32)
        denominator = float(bank["lpc_all_lse"])
        parent_metadata = json.loads(str(bank["metadata"].item()))
    if parent_metadata.get("forbidden_artifacts_used"):
        raise RuntimeError("conditional physics bank has forbidden ancestry")
    if parent_metadata.get("held_out_sources_used"):
        raise RuntimeError("conditional physics bank consumed a held-out source")
    rows = len(prediction)
    if radius.shape != (rows, len(mass_grid)):
        raise RuntimeError("conditional physics-bank curve shape is invalid")

    scenario_generator = ScenarioGenerator(
        arguments.data_root,
        arguments.cloud_points,
        arguments.seed,
        arguments.identity_fraction,
        arguments.one_slot_fraction,
    )
    # This stage consumes only the already-frozen curve bank and observational
    # NICER grids.  Keeping EOS/TOV construction out of this process both
    # avoids redundant work and makes the no-new-physics-evaluation timing
    # boundary explicit.
    common_grid = dict(nM=150, nR=150, max_rows=20_000, bw=0.08, seed=0)
    interpolators = {
        name: build_nicer_grid(
            arguments.data_root / source.filename,
            mcol=source.mass_column,
            rcol=source.radius_column,
            wcol_or_None=source.weight_column,
            **common_grid,
        )
        for name, source in zip(SOURCE_NAMES, BASE_NICER_SOURCES, strict=True)
    }
    covered = mass_grid >= 1.0
    evaluator = BatchedShiftedNICER(
        interpolators,
        radius[:, covered],
        mass_grid[covered],
        maximum_mass,
        scenario_generator.centres,
        device=str(device),
        row_chunk=arguments.row_chunk,
        quadrature_points=arguments.quadrature_points,
    )
    base_log_weight = torch.as_tensor(
        correction + log_astro, dtype=torch.float32, device=device
    )
    prediction_tensor = torch.as_tensor(prediction, device=device)
    sigma_tensor = torch.as_tensor(
        DEFAULT_NUCLEAR_SIGMA.astype(np.float32), device=device
    )
    support = torch.max(
        torch.abs(
            (
                prediction_tensor
                - torch.as_tensor(
                    DEFAULT_NUCLEAR_OBSERVATION.astype(np.float32), device=device
                )
            )
            / sigma_tensor
        ),
        dim=1,
    ).values <= arguments.restriction_radius
    base_log_weight = torch.where(
        support, base_log_weight, torch.full_like(base_log_weight, -torch.inf)
    )

    identity = np.empty((3, rows), dtype=np.float32)
    identity_transform = torch.as_tensor(IDENTITY[None, :], device=device)
    identity_started = time.perf_counter()
    for start, stop in evaluator.chunks():
        quadrature = evaluator.curve_quadrature(start, stop)
        for slot, name in enumerate(SOURCE_NAMES):
            identity[slot, start:stop] = (
                evaluator.evaluate_chunk(name, identity_transform, quadrature)[0]
                .detach()
                .cpu()
                .numpy()
            )
    identity_seconds = time.perf_counter() - identity_started

    accepted_clouds: list[np.ndarray] = []
    accepted_transforms: list[np.ndarray] = []
    accepted_modes: list[int] = []
    accepted_logc: list[float] = []
    accepted_ess: list[float] = []
    accepted_samples: list[np.ndarray] = []
    attempts = 0
    label_started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(arguments.seed + 101)
    while len(accepted_clouds) < arguments.scenarios:
        requested = min(
            arguments.scenario_batch,
            max(arguments.scenario_batch, arguments.scenarios - len(accepted_clouds)),
        )
        transforms, clouds, modes = scenario_generator.draw(requested)
        attempts += requested
        transform_tensor = torch.as_tensor(transforms, device=device)
        log_weight = base_log_weight[None, :].expand(requested, -1).clone()
        for start, stop in evaluator.chunks():
            quadrature = evaluator.curve_quadrature(start, stop)
            for slot, name in enumerate(SOURCE_NAMES):
                active = np.flatnonzero(
                    np.any(transforms[:, slot] != IDENTITY[None, :], axis=1)
                )
                if len(active) == 0:
                    continue
                active_tensor = torch.as_tensor(active, device=device)
                changed = evaluator.evaluate_chunk(
                    name, transform_tensor[active_tensor, slot], quadrature
                )
                baseline = torch.as_tensor(
                    identity[slot, start:stop], device=device
                )[None, :]
                log_weight[active_tensor, start:stop] += changed - baseline
        log_sum = torch.logsumexp(log_weight, dim=1)
        log_sum_two = torch.logsumexp(2.0 * log_weight, dim=1)
        ess = torch.exp(2.0 * log_sum - log_sum_two)
        keep = torch.isfinite(log_sum) & (ess >= arguments.minimum_tilt_ess)
        if keep.any():
            kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
            normalized = torch.exp(
                log_weight[kept_indices] - log_sum[kept_indices, None]
            )
            sampled = torch.multinomial(
                normalized,
                arguments.samples_per_scenario,
                replacement=True,
                generator=generator,
            )
            noisy = prediction_tensor[sampled] + torch.randn(
                (*sampled.shape, 7), device=device, generator=generator
            ) * sigma_tensor[None, None, :]
            kept_cpu = kept_indices.cpu().numpy()
            noisy_cpu = noisy.cpu().numpy().astype(np.float32)
            logc_cpu = (log_sum[kept_indices] - denominator).cpu().numpy()
            ess_cpu = ess[kept_indices].cpu().numpy()
            for local, original in enumerate(kept_cpu):
                if len(accepted_clouds) >= arguments.scenarios:
                    break
                accepted_clouds.append(clouds[original])
                accepted_transforms.append(transforms[original])
                accepted_modes.append(int(modes[original]))
                accepted_logc.append(float(logc_cpu[local]))
                accepted_ess.append(float(ess_cpu[local]))
                accepted_samples.append(noisy_cpu[local])
        print(
            f"[conditional-en-data] accepted={len(accepted_clouds)}/"
            f"{arguments.scenarios} attempted={attempts} "
            f"last-ESS={ess.detach().cpu().numpy().round(1).tolist()}",
            flush=True,
        )
    label_seconds = time.perf_counter() - label_started

    clouds_array = np.asarray(accepted_clouds, dtype=np.float32)
    transforms_array = np.asarray(accepted_transforms, dtype=np.float32)
    samples_array = np.asarray(accepted_samples, dtype=np.float32)
    logc_array = np.asarray(accepted_logc, dtype=np.float64)
    ess_array = np.asarray(accepted_ess, dtype=np.float64)
    modes_array = np.asarray(accepted_modes, dtype=np.int8)
    metadata = {
        "schema_version": 2,
        "lineage_class": "independent_conditional_evidence_network",
        "model": model_name,
        "physics_bank": str(arguments.bank.resolve()),
        "physics_bank_sha256": sha256(arguments.bank),
        "baseline_sources": scenario_generator.source_records,
        "held_out_sources_absent": ["J0614", "J1231", "J1614"],
        "held_out_source_files_used": [],
        "forbidden_artifacts_used": [],
        "amortized_inputs": ["three_NICER_clouds", "nuclear_observation"],
        "transform_domain": {
            "mass_shift": [-0.5, 0.25],
            "radius_shift": [-1.6, 1.6],
            "scale": [0.28, 2.2],
            "identity_fraction": arguments.identity_fraction,
            "one_slot_fraction": arguments.one_slot_fraction,
            "all_slot_fraction": 1.0
            - arguments.identity_fraction
            - arguments.one_slot_fraction,
        },
        "affine_density_jacobian_applied": True,
        "conditioning_cloud_matches_kde_subsample": {
            "maximum_rows": 20_000,
            "selection_seed": 0,
            "weighted_selection_without_replacement": True,
        },
        "restriction_radius_sigma": arguments.restriction_radius,
        "query_contract": {
            "source_specific_cache": False,
            "source_specific_training": False,
            "new_likelihood_evaluations": 0,
        },
    }
    atomic_savez(
        arguments.output,
        schema=np.asarray("conditional-en-scenarios-v2"),
        clouds=clouds_array,
        transforms=transforms_array,
        mode=modes_array,
        y=samples_array,
        logC=logc_array,
        tilt_ess=ess_array,
        ntr=np.int64(arguments.scenarios - arguments.validation_scenarios),
        OBS=DEFAULT_NUCLEAR_OBSERVATION,
        SIG=DEFAULT_NUCLEAR_SIGMA,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = {
        "status": "SMOKE_COMPLETED" if arguments.smoke else "PASS",
        "stage": "joint NICER+nuclear conditional EN scenario construction",
        "model": model_name,
        "scenarios": arguments.scenarios,
        "training_scenarios": arguments.scenarios - arguments.validation_scenarios,
        "attempted_scenarios": attempts,
        "samples_per_scenario": arguments.samples_per_scenario,
        "minimum_tilt_ess": float(ess_array.min()),
        "median_tilt_ess": float(np.median(ess_array)),
        "maximum_tilt_ess": float(ess_array.max()),
        "identity_likelihood_seconds": identity_seconds,
        "scenario_label_seconds": label_seconds,
        "construction_wall_seconds": time.perf_counter() - started,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "metadata": metadata,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
