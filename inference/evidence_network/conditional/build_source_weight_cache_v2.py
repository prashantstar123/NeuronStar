#!/usr/bin/env python3
"""Build exact Green-EN source weights from the expanded hyperonic bank.

Two details of this builder (version 2):

* the unchanged-source term is subtracted from the exact per-source NICER
  components stored by the EOS/TOV likelihood replay;
* a square-root continuation is used for newly generated source densities at
  the maximum-mass endpoint.

No held-out NICER source or output from a comparison inference method is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from inference.evidence_network.cache import (  # noqa: E402
    DEFAULT_NUCLEAR_OBSERVATION as OBS,
    DEFAULT_NUCLEAR_SIGMA as SIG,
)
from inference.evidence_network.conditional.batched_mixture_v3 import (  # noqa: E402
    BatchedMixtureNICER,
)
from inference.evidence_network.conditional.batched_nicer import (  # noqa: E402
    BatchedShiftedNICER,
)
from inference.evidence_network.conditional.build_dataset import (  # noqa: E402
    SOURCE_NAMES,
    weighted_source_table,
)
from inference.evidence_network.conditional.build_dataset_v3 import (  # noqa: E402
    BroadDensityGenerator,
    HELD_OUT_FILENAMES,
    nuclear_design,
    stabilized_log_matrix_product,
)
from likelihoods.nicer import build_nicer_grid  # noqa: E402
from workflows.source_scenarios import BASE_NICER_SOURCES  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def square_root_endpoint(
    mass: torch.Tensor,
    radius: torch.Tensor,
    radius_grid: np.ndarray,
    mass_grid: np.ndarray,
    maximum_mass: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continue the final stable branch as R=R_end+c*sqrt(Mmax-M)."""

    radius_table = torch.as_tensor(radius_grid, device=device)
    mass_table = torch.as_tensor(mass_grid, device=device)
    maximum = torch.as_tensor(maximum_mass, device=device)
    finite = torch.isfinite(radius_table) & (mass_table[None] <= maximum[:, None])
    finite_count = finite.sum(dim=1)
    has_curve = finite_count > 0
    has_two = finite_count > 1
    last = (len(mass_grid) - 1) - torch.argmax(finite.flip(1).float(), dim=1)
    column = torch.arange(len(mass_grid), device=device)[None, :]
    before_last = finite & (column < last[:, None])
    previous_candidate = (len(mass_grid) - 1) - torch.argmax(
        before_last.flip(1).float(), dim=1
    )
    previous = torch.where(has_two, previous_candidate, last)
    radius_last = radius_table.gather(1, last[:, None])[:, 0]
    radius_previous = radius_table.gather(1, previous[:, None])[:, 0]
    root_last = torch.sqrt((maximum - mass_table[last]).clamp_min(0))
    root_previous = torch.sqrt((maximum - mass_table[previous]).clamp_min(0))
    coefficient = torch.where(
        has_two & (root_previous > root_last + 1.0e-6),
        (radius_previous - radius_last) / (root_previous - root_last),
        torch.zeros_like(radius_last),
    )
    endpoint_radius = radius_last - coefficient * root_last
    beyond = mass > mass_table[previous][:, None]
    corrected = torch.where(
        beyond & has_two[:, None],
        endpoint_radius[:, None]
        + coefficient[:, None]
        * torch.sqrt((maximum[:, None] - mass).clamp_min(0)),
        radius,
    )
    return corrected, beyond & has_curve[:, None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=8192)
    parser.add_argument("--validation-scenarios", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--identity-fraction", type=float, default=0.02)
    parser.add_argument("--one-slot-fraction", type=float, default=0.83)
    parser.add_argument("--scenario-batch", type=int, default=32)
    parser.add_argument("--minimum-tilt-ess", type=float, default=20.0)
    parser.add_argument(
        "--maximum-restriction-delta",
        type=float,
        default=1.0e-3,
        help=(
            "Reject a synthetic scenario when any nuclear-design label changes "
            "by more than this amount after restricting to cached bank rows."
        ),
    )
    parser.add_argument(
        "--nuclear-support-sigma",
        type=float,
        default=6.0,
        help=(
            "Retain finite bank rows within this standardized nuclear-data "
            "box when writing the source-weight cache. The full-bank labels "
            "remain the validation authority."
        ),
    )
    parser.add_argument("--nuclear-designs", type=int, default=64)
    parser.add_argument("--mass-centre-min", type=float, default=0.85)
    parser.add_argument("--mass-centre-max", type=float, default=2.25)
    parser.add_argument("--radius-centre-min", type=float, default=8.5)
    parser.add_argument("--radius-centre-max", type=float, default=16.5)
    parser.add_argument("--mass-sigma-min", type=float, default=0.015)
    parser.add_argument("--mass-sigma-max", type=float, default=0.48)
    parser.add_argument("--radius-sigma-min", type=float, default=0.12)
    parser.add_argument("--radius-sigma-max", type=float, default=2.30)
    parser.add_argument("--correlation-limit", type=float, default=0.95)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.with_suffix(".sourcelogw.npy").exists():
        raise FileExistsError(arguments.output)
    for filename in HELD_OUT_FILENAMES:
        if (arguments.baseline_root / filename).exists():
            raise RuntimeError(f"held-out NICER source present: {filename}")

    started = time.perf_counter()
    device = torch.device("cuda")
    with np.load(arguments.bank, allow_pickle=False) as source:
        required = {"schema", "X", "R", "MG", "MM", "LA", "LPC", "NIC",
                    "lpc_all_lse", "metadata"}
        missing = required - set(source.files)
        if missing:
            raise RuntimeError(f"expanded bank lacks {sorted(missing)}")
        if str(source["schema"].item()) != "conditional-en-independent-physics-v2":
            raise RuntimeError("unexpected expanded-bank schema")
        metadata = json.loads(str(source["metadata"].item()))
        if metadata.get("forbidden_artifacts_used") or metadata.get("held_out_sources_used"):
            raise RuntimeError("expanded-bank ancestry gate failed")
        prediction = np.asarray(source["X"], dtype=np.float32)
        radius = np.asarray(source["R"], dtype=np.float32)
        mass_grid = np.asarray(source["MG"], dtype=np.float32)
        maximum_mass = np.asarray(source["MM"], dtype=np.float32)
        log_astro = np.asarray(source["LA"], dtype=np.float32)
        correction = np.asarray(source["LPC"], dtype=np.float32)
        exact_nicer = np.asarray(source["NIC"], dtype=np.float32)
        denominator = float(source["lpc_all_lse"])
    rows = len(prediction)
    if exact_nicer.shape != (rows, 3):
        raise RuntimeError("exact per-source NICER table has the wrong shape")

    generator = BroadDensityGenerator(
        arguments.baseline_root,
        1024,
        arguments.seed,
        single_components_only=True,
        mass_centre_range=(arguments.mass_centre_min, arguments.mass_centre_max),
        radius_centre_range=(arguments.radius_centre_min, arguments.radius_centre_max),
        mass_sigma_range=(arguments.mass_sigma_min, arguments.mass_sigma_max),
        radius_sigma_range=(arguments.radius_sigma_min, arguments.radius_sigma_max),
        correlation_limit=arguments.correlation_limit,
    )
    grid_options = dict(nM=150, nR=150, max_rows=20_000, bw=0.08, seed=0)
    interpolators = {
        name: build_nicer_grid(
            arguments.baseline_root / specification.filename,
            mcol=specification.mass_column,
            rcol=specification.radius_column,
            wcol_or_None=specification.weight_column,
            **grid_options,
        )
        for name, specification in zip(SOURCE_NAMES, BASE_NICER_SOURCES, strict=True)
    }
    centres = {}
    for name, specification in zip(SOURCE_NAMES, BASE_NICER_SOURCES, strict=True):
        _, _, _, centre = weighted_source_table(
            arguments.baseline_root / specification.filename, specification
        )
        centres[name] = (float(centre[0]), float(centre[1]))
    covered = mass_grid >= 1.0
    evaluator = BatchedShiftedNICER(
        interpolators,
        radius[:, covered],
        mass_grid[covered],
        maximum_mass,
        centres,
        device="cuda",
        row_chunk=25_000,
        quadrature_points=40,
    )
    mixture = BatchedMixtureNICER(relative_floor=1.0e-6)
    quadrature = []
    endpoint_nodes = 0
    for start, stop in evaluator.chunks():
        mass, curve_radius, valid, lower, upper = evaluator.curve_quadrature(start, stop)
        fixed_radius, affected = square_root_endpoint(
            mass,
            curve_radius,
            radius[start:stop, covered],
            mass_grid[covered],
            maximum_mass[start:stop],
            device,
        )
        endpoint_nodes += int(affected.sum().item())
        quadrature.append((start, stop, (mass, fixed_radius, valid, lower, upper)))

    base = torch.as_tensor(correction + log_astro, dtype=torch.float32, device=device)
    exact_nicer_gpu = torch.as_tensor(exact_nicer, dtype=torch.float32, device=device)
    prediction_gpu = torch.as_tensor(prediction, device=device)
    observation_gpu = torch.as_tensor(OBS.astype(np.float32), device=device)
    sigma_gpu = torch.as_tensor(SIG.astype(np.float32), device=device)
    if arguments.nuclear_support_sigma <= 0:
        raise ValueError("--nuclear-support-sigma must be positive")
    support = torch.max(
        torch.abs((prediction_gpu - observation_gpu) / sigma_gpu), dim=1
    ).values <= arguments.nuclear_support_sigma
    keep = support & torch.isfinite(base)
    kept_index = torch.nonzero(keep).flatten()
    designs, _ = nuclear_design(arguments.nuclear_designs, arguments.seed + 503)
    design_gpu = torch.as_tensor(designs.astype(np.float32), device=device)
    nuclear_log = -0.5 * torch.sum(
        ((prediction_gpu[None] - design_gpu[:, None]) / sigma_gpu) ** 2, dim=2
    )
    nuclear_max = nuclear_log.max(dim=1).values
    nuclear_factor = torch.exp(nuclear_log - nuclear_max[:, None])
    del nuclear_log
    nuclear_factor_box = torch.where(
        support[None], nuclear_factor, torch.zeros_like(nuclear_factor)
    )

    cache_path = arguments.output.with_suffix(".sourcelogw.npy")
    cache = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=(arguments.scenarios, int(kept_index.numel())),
    )
    accepted = {key: [] for key in (
        "weight", "mean", "covariance", "active", "family", "mode",
        "logC", "ess", "logZ_full", "logZ_box",
    )}
    count = attempts = restriction_rejected = 0
    label_started = time.perf_counter()
    while count < arguments.scenarios:
        weight, mean, covariance, active, family, mode, _ = generator.draw(
            arguments.scenario_batch,
            identity_fraction=arguments.identity_fraction,
            one_slot_fraction=arguments.one_slot_fraction,
        )
        attempts += arguments.scenario_batch
        weight_gpu, mean_gpu, covariance_gpu = (
            torch.as_tensor(value, device=device)
            for value in (weight, mean, covariance)
        )
        source_log_weight = base[None].expand(arguments.scenario_batch, -1).clone()
        for start, stop, local_quadrature in quadrature:
            for slot in range(3):
                changed = np.flatnonzero(active[:, slot])
                if not len(changed):
                    continue
                selected = torch.as_tensor(changed, device=device)
                replacement = mixture.evaluate_chunk(
                    weight_gpu[selected, slot, :1],
                    mean_gpu[selected, slot, :1],
                    covariance_gpu[selected, slot, :1],
                    local_quadrature,
                )
                source_log_weight[selected, start:stop] += (
                    replacement
                    - exact_nicer_gpu[start:stop, slot][None]
                )
        source_log_weight = torch.where(
            torch.isfinite(base)[None],
            source_log_weight,
            torch.full_like(source_log_weight, -torch.inf),
        )
        # Invalid EOS rows carry LA=-inf and may also have NaN per-source
        # components.  Mask them before logsumexp so ``-inf - NaN`` cannot
        # poison an otherwise valid scenario.
        restricted = torch.where(
            (support & torch.isfinite(base))[None],
            source_log_weight,
            torch.full_like(source_log_weight, -torch.inf),
        )
        log_sum = torch.logsumexp(restricted, dim=1)
        ess = torch.exp(2 * log_sum - torch.logsumexp(2 * restricted, dim=1))
        retain = torch.isfinite(log_sum) & (
            (ess >= arguments.minimum_tilt_ess)
            | torch.as_tensor(mode == 0, device=device)
        )
        if retain.any():
            selected = torch.nonzero(retain).flatten()
            full_evidence = (
                stabilized_log_matrix_product(
                    source_log_weight[selected], nuclear_factor, nuclear_max
                )
                - denominator
            ).cpu().numpy()
            box_evidence = (
                stabilized_log_matrix_product(
                    source_log_weight[selected], nuclear_factor_box, nuclear_max
                )
                - denominator
            ).cpu().numpy()
            selected_weight = source_log_weight[selected][:, kept_index].cpu().numpy()
            selected_logc = (log_sum[selected] - denominator).cpu().numpy()
            selected_ess = ess[selected].cpu().numpy()
            restriction_ok = np.max(
                np.abs(box_evidence - full_evidence), axis=1
            ) <= arguments.maximum_restriction_delta
            restriction_rejected += int((~restriction_ok).sum())
            selected_cpu = selected.cpu().numpy()
            for local in np.flatnonzero(restriction_ok):
                original = selected_cpu[local]
                if count >= arguments.scenarios:
                    break
                cache[count] = selected_weight[local]
                values = {
                    "weight": weight[original], "mean": mean[original],
                    "covariance": covariance[original], "active": active[original],
                    "family": family[original], "mode": int(mode[original]),
                    "logC": float(selected_logc[local]),
                    "ess": float(selected_ess[local]),
                    "logZ_full": full_evidence[local],
                    "logZ_box": box_evidence[local],
                }
                for key, value in values.items():
                    accepted[key].append(value)
                count += 1
        if attempts % 1600 == 0:
            print(
                f"[cache-v2] accepted={count}/{arguments.scenarios} "
                f"attempted={attempts} {time.perf_counter()-label_started:.0f}s",
                flush=True,
            )
    cache.flush()
    label_seconds = time.perf_counter() - label_started
    output = {key: np.asarray(value) for key, value in accepted.items()}
    np.savez(
        arguments.output,
        schema=np.asarray("green-source-weight-cache-v2"),
        mixture_weight=output["weight"].astype(np.float32),
        mixture_mean=output["mean"].astype(np.float32),
        mixture_covariance=output["covariance"].astype(np.float32),
        active_mask=output["active"].astype(bool),
        family=output["family"].astype(np.int8),
        mode=output["mode"].astype(np.int8),
        logC=output["logC"].astype(np.float64),
        tilt_ess=output["ess"].astype(np.float64),
        logZ_full=output["logZ_full"].astype(np.float64),
        logZ_box=output["logZ_box"].astype(np.float64),
        nuclear_observations=designs,
        kept_bank_rows=kept_index.cpu().numpy(),
        ntr=np.int64(arguments.scenarios - arguments.validation_scenarios),
        OBS=OBS,
        SIG=SIG,
        denominator=np.float64(denominator),
    )
    restriction_delta = output["logZ_box"] - output["logZ_full"]
    identity_rows = np.flatnonzero(output["mode"] == 0)
    identity_error = 0.0
    if len(identity_rows):
        expected = base[kept_index].cpu().numpy()
        identity_error = float(
            np.max(np.abs(np.asarray(cache[identity_rows[0]]) - expected))
        )
    report = {
        "status": "PASS",
        "schema": "green-source-weight-cache-v2",
        "scenarios": arguments.scenarios,
        "attempted": attempts,
        "in_box_finite_rows": int(kept_index.numel()),
        "label_seconds": label_seconds,
        "restriction_rejected_scenarios": restriction_rejected,
        "wall_seconds": time.perf_counter() - started,
        "bank": str(arguments.bank.resolve()),
        "bank_sha256": sha256(arguments.bank),
        "cache_sha256": sha256(cache_path),
        "index_sha256": sha256(arguments.output),
        "exact_identity_subtraction": True,
        "square_root_endpoint_nodes": endpoint_nodes,
        "identity_source_weight_max_abs_error": identity_error,
        "execution": {
            "hostname": socket.gethostname(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
        },
        "scenario_configuration": {
            "seed": arguments.seed,
            "validation_scenarios": arguments.validation_scenarios,
            "identity_fraction": arguments.identity_fraction,
            "one_slot_fraction": arguments.one_slot_fraction,
            "scenario_batch": arguments.scenario_batch,
            "minimum_tilt_ess": arguments.minimum_tilt_ess,
            "maximum_restriction_delta": arguments.maximum_restriction_delta,
            "nuclear_designs": arguments.nuclear_designs,
            "mode_counts": {
                str(value): int(np.sum(output["mode"] == value))
                for value in np.unique(output["mode"])
            },
            "single_slot_counts": {
                SOURCE_NAMES[slot]: int(
                    np.sum(
                        (output["mode"] == 1)
                        & output["active"][:, slot]
                    )
                )
                for slot in range(3)
            },
        },
        "restriction_delta": {
            "maximum_absolute": float(np.max(np.abs(restriction_delta))),
            "p999_absolute": float(np.quantile(np.abs(restriction_delta), 0.999)),
        },
        "source_support": {
            "nuclear_box_sigma": arguments.nuclear_support_sigma,
            "mass_centre": list(generator.mass_centre_range),
            "radius_centre_km": list(generator.radius_centre_range),
            "mass_sigma": list(generator.mass_sigma_range),
            "radius_sigma_km": list(generator.radius_sigma_range),
            "correlation": [-generator.correlation_limit, generator.correlation_limit],
        },
        "held_out_source_files_used": [],
        "forbidden_artifacts_used": [],
    }
    finite_delta = bool(np.isfinite(restriction_delta).all())
    report["restriction_delta"]["all_finite"] = finite_delta
    if (
        identity_error > 1.0e-6
        or not finite_delta
        or report["restriction_delta"]["maximum_absolute"]
        > arguments.maximum_restriction_delta
    ):
        report["status"] = "FAIL_INTERNAL_GATE"
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
