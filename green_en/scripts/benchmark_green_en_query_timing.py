#!/usr/bin/env python3
"""Component query-timing benchmark for the frozen Green-EN networks (query times of Table VIII).

Measurement definitions:

* checkpoint load: deserialize one checkpoint, construct the network, move it to
  the GPU and synchronize (per member, sequential);
* density table: the deterministic CPU NICER table (``build_nicer_grid`` with the
  frozen options), built once per new source and reused by all members;
* first and warm neural calls: repetition 0 and repetition 1 of the identical
  evaluation, each synchronized, for the five-setting identity batch and for
  each new NICER source.

Reported aggregates (as in Table VIII):
five-setting entry = three sequential checkpoint loads + three warm identity
batches; new-source entry = one density table + three warm member evaluations.
The network code is imported unchanged from ``query_green_frozen_v5.py``; the
log-evidence of every member is recorded so it can be compared with the frozen
query outputs.  No likelihood, TOV solve or training is performed.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.evidence_network.conditional.query_green_frozen_v5 import (  # noqa: E402
    GreenKernel,
    log_overlap,
    unit,
)
from likelihoods.nicer import build_nicer_grid  # noqa: E402
from workflows.nuclear_scenarios import NUCLEAR_SCENARIO_NAMES, nuclear_observation  # noqa: E402
from workflows.source_scenarios import SOURCE_SUBSTITUTIONS  # noqa: E402

OBSERVATION = np.asarray(
    [0.153, -16.1, 230.0, 32.5, 0.505714285714279, 1.24142857142857, 2.4857142857143]
)
SIGMA = np.asarray(
    [0.005, 0.2, 40.0, 1.8, 0.19428571428571, 0.608571428571428, 1.382857142857144]
)
GRID_OPTIONS = dict(nM=150, nR=150, max_rows=20_000, bw=0.08, seed=0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_member(path: Path, device: torch.device):
    sync(device)
    started = time.perf_counter()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_type") != "green_function_en_v1":
        raise RuntimeError(f"unexpected checkpoint type in {path}")
    model = GreenKernel().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    sync(device)
    seconds = time.perf_counter() - started
    return model, payload, seconds


def identity_call(model, payload, paper, device):
    sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        values = (model.identity(paper) + float(payload["mean"])).double().cpu().numpy()
    sync(device)
    return values, time.perf_counter() - started


def source_call(model, payload, interpolator, slot, paper, device):
    """The Green-kernel held-out evaluation of v5, timed like v5's network_seconds."""
    mean = float(payload["mean"])
    mass_points, radius_points = map(int, payload["lattice"])
    domain = tuple(map(float, payload["domain"]))
    mass_low, mass_high, radius_low, radius_high = domain
    sync(device)
    started = time.perf_counter()
    mass = torch.linspace(mass_low, mass_high, mass_points, device=device)
    radius = torch.linspace(radius_low, radius_high, radius_points, device=device)
    mass_mesh, radius_mesh = torch.meshgrid(mass, radius, indexing="ij")
    lattice = unit(mass_mesh.flatten(), radius_mesh.flatten(), domain)
    mass_weight = torch.full_like(mass, float(mass[1] - mass[0]))
    radius_weight = torch.full_like(radius, float(radius[1] - radius[0]))
    mass_weight[[0, -1]] *= 0.5
    radius_weight[[0, -1]] *= 0.5
    log_weight = torch.log((mass_weight[:, None] * radius_weight[None, :]).flatten())
    grid_mass, grid_radius = (
        torch.as_tensor(value, dtype=torch.float32, device=device)
        for value in interpolator.grid
    )
    log_density = torch.as_tensor(
        np.asarray(interpolator.values), dtype=torch.float32, device=device
    )
    grid_mass_mesh, grid_radius_mesh = torch.meshgrid(grid_mass, grid_radius, indexing="ij")
    keep = (
        (log_density > float(interpolator.fill_value) + 1.0e-6)
        & (grid_mass_mesh >= mass_low)
        & (grid_mass_mesh <= mass_high)
        & (grid_radius_mesh >= radius_low)
        & (grid_radius_mesh <= radius_high)
    )
    grid_mass_weight = torch.full_like(grid_mass, float(grid_mass[1] - grid_mass[0]))
    grid_radius_weight = torch.full_like(grid_radius, float(grid_radius[1] - grid_radius[0]))
    grid_mass_weight[[0, -1]] *= 0.5
    grid_radius_weight[[0, -1]] *= 0.5
    source_log_weight = (
        log_density
        + torch.log(grid_mass_weight[:, None] * grid_radius_weight[None, :])
    )[keep][None, :]
    a1 = paper[0:1]
    with torch.inference_mode():
        full_kernel = model(lattice, a1, slot) + mean
        floor_overlap = torch.logsumexp(full_kernel + log_weight[:, None], dim=0)[0]
        source_kernel = model(
            unit(grid_mass_mesh[keep], grid_radius_mesh[keep], domain), a1, slot
        ) + mean
        main_overlap = log_overlap(source_log_weight, source_kernel)[0, 0]
        value = torch.logaddexp(
            main_overlap,
            torch.as_tensor(float(interpolator.fill_value), device=device) + floor_overlap,
        )
        sync(device)
        value = float(value.cpu())
    return value, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nucleonic", type=Path, nargs=3, required=True)
    parser.add_argument("--hyperonic", type=Path, nargs=3, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    device = torch.device(arguments.device)

    paper = torch.as_tensor(
        np.stack(
            [
                (np.asarray(nuclear_observation(name)) - OBSERVATION) / SIGMA
                for name in NUCLEAR_SCENARIO_NAMES
            ]
        ),
        dtype=torch.float32,
        device=device,
    )

    tables = {}
    table_seconds = {}
    source_files = {}
    for case, specification in SOURCE_SUBSTITUTIONS.items():
        path = arguments.heldout_root / specification.filename
        started = time.perf_counter()
        tables[case] = build_nicer_grid(
            path,
            mcol=specification.mass_column,
            rcol=specification.radius_column,
            wcol_or_None=specification.weight_column,
            **GRID_OPTIONS,
        )
        table_seconds[case] = time.perf_counter() - started
        source_files[case] = {"path": str(path.resolve()), "sha256": sha256(path)}

    sectors = {}
    for sector, paths in (("nucleonic", arguments.nucleonic), ("hyperonic", arguments.hyperonic)):
        members = {}
        for path in paths:
            model, payload, load_seconds = load_member(path, device)
            identity0, identity_first = identity_call(model, payload, paper, device)
            identity1, identity_warm = identity_call(model, payload, paper, device)
            sources = {}
            for case, specification in SOURCE_SUBSTITUTIONS.items():
                value0, first = source_call(
                    model, payload, tables[case], specification.slot, paper, device
                )
                value1, warm = source_call(
                    model, payload, tables[case], specification.slot, paper, device
                )
                sources[case] = {
                    "log_evidence": value1,
                    "repeat_identical": value0 == value1,
                    "first_seconds": first,
                    "warm_seconds": warm,
                }
            members[path.name] = {
                "checkpoint_sha256": sha256(path),
                "load_seconds": load_seconds,
                "identity_first_seconds": identity_first,
                "identity_warm_seconds": identity_warm,
                "identity_log_evidence": dict(
                    zip(NUCLEAR_SCENARIO_NAMES, map(float, identity1))
                ),
                "identity_repeat_identical": bool(np.array_equal(identity0, identity1)),
                "sources": sources,
            }
            del model
        load_sum = sum(item["load_seconds"] for item in members.values())
        identity_warm_sum = sum(item["identity_warm_seconds"] for item in members.values())
        new_sources = {}
        for case in SOURCE_SUBSTITUTIONS:
            warm_sum = sum(item["sources"][case]["warm_seconds"] for item in members.values())
            new_sources[case] = {
                "density_table_cpu_seconds_once": table_seconds[case],
                "neural_warm_sequential_ensemble_seconds": warm_sum,
                "loaded_warm_end_to_end_seconds": table_seconds[case] + warm_sum,
            }
        sectors[sector] = {
            "members": members,
            "checkpoint_load_sequential_ensemble_seconds": load_sum,
            "five_setting_warm_sequential_ensemble_seconds": identity_warm_sum,
            "five_setting_entry_seconds_load_plus_warm": load_sum + identity_warm_sum,
            "new_source_queries": new_sources,
        }

    report = {
        "schema": "green-en-query-timing-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "definitions": {
            "five_setting_entry": "three sequential checkpoint loads plus three warm five-setting identity batches (one per member)",
            "new_source_entry": "one CPU density table plus three warm member evaluations, sequential on one GPU",
            "density_table_reuse": "built once per source and reused by every member of both sectors",
        },
        "density_table_cpu_seconds": table_seconds,
        "source_files": source_files,
        "sectors": sectors,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        sector: {
            "five_setting_entry_s": round(data["five_setting_entry_seconds_load_plus_warm"], 5),
            "load_s": round(data["checkpoint_load_sequential_ensemble_seconds"], 5),
            "warm_batch_s": round(data["five_setting_warm_sequential_ensemble_seconds"], 6),
            "new_sources_s": {
                case: round(item["loaded_warm_end_to_end_seconds"], 4)
                for case, item in data["new_source_queries"].items()
            },
        }
        for sector, data in sectors.items()
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
