#!/usr/bin/env python3
"""Apply a frozen Green-function EN to untouched empirical NICER densities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from likelihoods.nicer import build_nicer_grid  # noqa: E402
from workflows.nuclear_scenarios import (  # noqa: E402
    NUCLEAR_SCENARIO_NAMES,
    nuclear_observation,
)
from workflows.source_scenarios import SOURCE_SUBSTITUTIONS  # noqa: E402


MASS_LOW, MASS_HIGH, RADIUS_LOW, RADIUS_HIGH = 1.0, 2.6, 8.2, 18.4
HYPERONIC_ULTRANEST = {
    "A1": (-28.759, 0.126),
    "K0_200": (-29.394, 0.358),
    "K0_260": (-28.681, 0.225),
    "Jsym_29": (-28.552, 0.120),
    "Jsym_36": (-29.016, 0.226),
    "J0614": (-31.533, 0.259),
    "J1231": (-29.240, 0.195),
    "J1614": (-28.789, 0.109),
}
NUCLEONIC_ULTRANEST = {
    "A1": (-25.467, 0.242),
    "K0_200": (-25.590, 0.110),
    "K0_260": (-25.760, 0.210),
    "Jsym_29": (-25.580, 0.090),
    "Jsym_36": (-25.700, 0.060),
    "J0614": (-26.972, 0.160),
    "J1231": (-26.448, 0.213),
    "J1614": (-25.565, 0.286),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class GreenKernel(nn.Module):
    """Frozen architecture used by the independently trained Green EN."""

    def __init__(self, width: int = 256, bands: int = 6) -> None:
        super().__init__()
        self.register_buffer(
            "freq", (2.0 ** torch.arange(bands)) * np.pi, persistent=True
        )
        self.trunk = nn.Sequential(
            nn.Linear(2 + 4 * bands, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width),
        )
        self.slot = nn.Parameter(torch.zeros(3, width))
        self.tnet = nn.Sequential(
            nn.Linear(7, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width),
        )
        self.head = nn.Sequential(
            nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        self.ux = nn.Linear(width, 1)
        self.vt = nn.Linear(width, 1)
        self.identity_head = nn.Sequential(
            nn.Linear(7, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1),
        )

    def forward(
        self, coordinate: torch.Tensor, observation: torch.Tensor, slot: int
    ) -> torch.Tensor:
        phase = coordinate[:, :, None] * self.freq
        spatial = self.trunk(
            torch.cat(
                [
                    coordinate,
                    torch.sin(phase).flatten(1),
                    torch.cos(phase).flatten(1),
                ],
                dim=1,
            )
        ) + self.slot[slot]
        nuclear = self.tnet(observation)
        return (
            self.head(spatial[:, None, :] * (1.0 + nuclear[None, :, :]))
            .squeeze(-1)
            + self.ux(spatial)
            + self.vt(nuclear).T
        )

    def identity(self, observation: torch.Tensor) -> torch.Tensor:
        return self.identity_head(observation).squeeze(-1)


def unit(
    mass: torch.Tensor,
    radius: torch.Tensor,
    domain=(MASS_LOW, MASS_HIGH, RADIUS_LOW, RADIUS_HIGH),
) -> torch.Tensor:
    mass_low, mass_high, radius_low, radius_high = domain
    return torch.stack(
        [
            (mass - mass_low) / (mass_high - mass_low),
            (radius - radius_low) / (radius_high - radius_low),
        ],
        dim=-1,
    )


def log_overlap(log_density_weight: torch.Tensor, log_kernel: torch.Tensor) -> torch.Tensor:
    source_maximum = log_density_weight.max(dim=1, keepdim=True).values
    kernel_maximum = log_kernel.max(dim=0, keepdim=True).values
    return (
        torch.log(
            torch.exp(log_density_weight - source_maximum)
            @ torch.exp(log_kernel - kernel_maximum)
            + 1.0e-37
        )
        + source_maximum
        + kernel_maximum
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sector", choices=("hyperonic", "nucleonic"), default="hyperonic")
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    checkpoint_hash_before = sha256(arguments.checkpoint)
    payload = torch.load(
        arguments.checkpoint, map_location="cpu", weights_only=False
    )
    if payload.get("model_type") != "green_function_en_v1":
        raise RuntimeError("unexpected frozen Green-EN checkpoint")
    device = torch.device(arguments.device)
    ultranest = (
        HYPERONIC_ULTRANEST
        if arguments.sector == "hyperonic"
        else NUCLEONIC_ULTRANEST
    )
    model = GreenKernel().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    mean = float(payload["mean"])
    mass_points, radius_points = map(int, payload["lattice"])
    domain = tuple(map(float, payload["domain"]))
    if len(domain) != 4 or not (
        domain[0] < domain[1] and domain[2] < domain[3]
    ):
        raise RuntimeError("invalid frozen Green domain")
    mass_low, mass_high, radius_low, radius_high = domain

    observation = np.asarray(
        [0.153, -16.1, 230.0, 32.5, 0.505714285714279,
         1.24142857142857, 2.4857142857143],
        dtype=np.float64,
    )
    sigma = np.asarray(
        [0.005, 0.2, 40.0, 1.8, 0.19428571428571,
         0.608571428571428, 1.382857142857144],
        dtype=np.float64,
    )
    paper = torch.as_tensor(
        np.stack(
            [
                (np.asarray(nuclear_observation(name)) - observation) / sigma
                for name in NUCLEAR_SCENARIO_NAMES
            ]
        ),
        dtype=torch.float32,
        device=device,
    )
    mass = torch.linspace(mass_low, mass_high, mass_points, device=device)
    radius = torch.linspace(radius_low, radius_high, radius_points, device=device)
    mass_mesh, radius_mesh = torch.meshgrid(mass, radius, indexing="ij")
    lattice = unit(mass_mesh.flatten(), radius_mesh.flatten(), domain)
    mass_weight = torch.full_like(mass, float(mass[1] - mass[0]))
    radius_weight = torch.full_like(radius, float(radius[1] - radius[0]))
    mass_weight[[0, -1]] *= 0.5
    radius_weight[[0, -1]] *= 0.5
    log_weight = torch.log(
        (mass_weight[:, None] * radius_weight[None, :]).flatten()
    )

    results = {}
    with torch.inference_mode():
        identity_started = time.perf_counter()
        identity = (model.identity(paper) + mean).double().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        identity_seconds = time.perf_counter() - identity_started
    for name, value in zip(NUCLEAR_SCENARIO_NAMES, identity, strict=True):
        reference, reference_sigma = ultranest[name]
        results[name] = {
            "log_evidence": float(value),
            "ultranest_log_evidence": reference,
            "ultranest_sigma": reference_sigma,
            "delta": float(value - reference),
            "separation_ultranest_sigma": float(abs(value - reference) / reference_sigma),
            "neural_seconds_shared_batch": identity_seconds,
        }

    source_receipts = []
    grid_options = dict(nM=150, nR=150, max_rows=20_000, bw=0.08, seed=0)
    a1 = paper[0:1]
    for case, specification in SOURCE_SUBSTITUTIONS.items():
        path = arguments.heldout_root / specification.filename
        table_started = time.perf_counter()
        interpolator = build_nicer_grid(
            path,
            mcol=specification.mass_column,
            rcol=specification.radius_column,
            wcol_or_None=specification.weight_column,
            **grid_options,
        )
        table_seconds = time.perf_counter() - table_started
        query_started = time.perf_counter()
        grid_mass, grid_radius = (
            torch.as_tensor(value, dtype=torch.float32, device=device)
            for value in interpolator.grid
        )
        log_density = torch.as_tensor(
            np.asarray(interpolator.values), dtype=torch.float32, device=device
        )
        grid_mass_mesh, grid_radius_mesh = torch.meshgrid(
            grid_mass, grid_radius, indexing="ij"
        )
        keep = (
            (log_density > float(interpolator.fill_value) + 1.0e-6)
            & (grid_mass_mesh >= mass_low)
            & (grid_mass_mesh <= mass_high)
            & (grid_radius_mesh >= radius_low)
            & (grid_radius_mesh <= radius_high)
        )
        grid_mass_weight = torch.full_like(
            grid_mass, float(grid_mass[1] - grid_mass[0])
        )
        grid_radius_weight = torch.full_like(
            grid_radius, float(grid_radius[1] - grid_radius[0])
        )
        grid_mass_weight[[0, -1]] *= 0.5
        grid_radius_weight[[0, -1]] *= 0.5
        source_log_weight = (
            log_density
            + torch.log(
                grid_mass_weight[:, None] * grid_radius_weight[None, :]
            )
        )[keep][None, :]
        with torch.inference_mode():
            full_kernel = model(lattice, a1, specification.slot) + mean
            floor_overlap = torch.logsumexp(
                full_kernel + log_weight[:, None], dim=0
            )[0]
            source_kernel = model(
                unit(grid_mass_mesh[keep], grid_radius_mesh[keep], domain),
                a1,
                specification.slot,
            ) + mean
            main_overlap = log_overlap(source_log_weight, source_kernel)[0, 0]
            value = torch.logaddexp(
                main_overlap,
                torch.as_tensor(
                    float(interpolator.fill_value), device=device
                ) + floor_overlap,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            value = float(value.cpu())
        network_seconds = time.perf_counter() - query_started
        reference, reference_sigma = ultranest[case]
        results[case] = {
            "log_evidence": value,
            "ultranest_log_evidence": reference,
            "ultranest_sigma": reference_sigma,
            "delta": value - reference,
            "separation_ultranest_sigma": abs(value - reference) / reference_sigma,
            "network_seconds": network_seconds,
            "certified_table_build_seconds": table_seconds,
        }
        source_receipts.append(
            {
                "case": case,
                "slot": int(specification.slot),
                "source": str(path.resolve()),
                "source_sha256": sha256(path),
                "table_nodes": int(keep.sum()),
                "grid_fill": float(interpolator.fill_value),
            }
        )

    checkpoint_hash_after = sha256(arguments.checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("frozen checkpoint changed during query")
    report = {
        "status": "HELDOUT_QUERY_COMPLETE",
        "method": "certified_density_green_function_evidence_network",
        "sector": arguments.sector,
        "domain": list(domain),
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "results": results,
        "source_receipts": source_receipts,
        "independence": {
            "query_retraining": False,
            "query_likelihood_evaluations": False,
            "query_tov_evaluations": False,
            "ultranest_proposals_used": False,
            "anet_used": False,
            "tsnpe_used": False,
            "heldout_sources_used_during_training": False,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
