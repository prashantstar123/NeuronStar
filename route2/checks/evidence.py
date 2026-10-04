#!/usr/bin/env python3
"""Route 2 check: Evidence Network (Green EN) -> Table VI evidence values, from the six frozen networks.

1. Each frozen member (3 nucleonic, 3 hyperonic; route2/fixtures/evidence/checkpoints) is queried at the
   five nuclear configurations and the three held-out NICER sources (48 values). The network code is the
   production query code (query_green_frozen_v5, the engine behind the v7/v8 query gates) and the NICER
   density tables use the frozen options.
2. The 16 Table VI values are rebuilt with the paper's aggregation: centre = log of the mean evidence of
   the three members; uncertainty = sqrt(seed scatter^2 + exact-bank Monte Carlo^2 + calibration^2)
   (nucleonic: the larger of this value and the stored per-row value in
   route2/fixtures/evidence/nucleonic_stored_sigma.json; held-out sources: plus the network-minus-bank residual in
   quadrature).
3. Both are compared with the published results (results/evidence/green_en_vs_ultranest_16_rows.json).
   On the production GPU stack the member values are bit-identical; on another CPU/GPU the tolerance is
   1e-5 in log Z (typical difference 1e-6). The three-decimal values printed in Table VI must be identical,
   except where the published value lies within 1e-5 of a rounding boundary (the nucleonic J0614 evidence,
   -27.1695003, is 3e-7 from one); such a last-digit flip is reported, not failed.
4. For networks trained again from scratch (docs/route2/STEPS_EVIDENCE_NETWORK.md), --checkpoints, --training-reports
   and --exact-bank point to their files (same file names as in route2/fixtures/evidence/); the report then holds
   their 16 Table VI values, and the comparison measures the difference from the published networks.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from inference.evidence_network.conditional.query_green_frozen_v5 import GreenKernel, log_overlap, unit
from likelihoods.nicer import build_nicer_grid
from workflows.nuclear_scenarios import NUCLEAR_SCENARIO_NAMES, nuclear_observation
from workflows.source_scenarios import SOURCE_SUBSTITUTIONS

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "route2/fixtures/evidence"
OBSERVATION = np.asarray([0.153, -16.1, 230.0, 32.5, 0.505714285714279, 1.24142857142857, 2.4857142857143])
SIGMA = np.asarray([0.005, 0.2, 40.0, 1.8, 0.19428571428571, 0.608571428571428, 1.382857142857144])
GRID_OPTIONS = dict(nM=150, nR=150, max_rows=20_000, bw=0.08, seed=0)
SEEDS = {"nucleonic": (9211, 9212, 9213), "hyperonic": (9311, 9312, 9313)}
CONFIGURATIONS = ("A1", "K0_200", "K0_260", "Jsym_29", "Jsym_36", "J0614", "J1231", "J1614")
NUCLEAR = set(CONFIGURATIONS[:5])
SOURCE_TO_BASELINE = {"J0614": "j0437", "J1231": "j0030", "J1614": "j0740"}
TOLERANCE = 1.0e-5


def load_member(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_type") != "green_function_en_v1":
        raise RuntimeError(f"unexpected checkpoint type in {path}")
    model = GreenKernel().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def source_value(model, payload, interpolator, slot, paper, device) -> float:
    """Held-out-source evaluation, identical to the production query (v5)."""
    mean = float(payload["mean"])
    mass_points, radius_points = map(int, payload["lattice"])
    domain = tuple(map(float, payload["domain"]))
    mass_low, mass_high, radius_low, radius_high = domain
    mass = torch.linspace(mass_low, mass_high, mass_points, device=device)
    radius = torch.linspace(radius_low, radius_high, radius_points, device=device)
    mass_mesh, radius_mesh = torch.meshgrid(mass, radius, indexing="ij")
    lattice = unit(mass_mesh.flatten(), radius_mesh.flatten(), domain)
    mass_weight = torch.full_like(mass, float(mass[1] - mass[0]))
    radius_weight = torch.full_like(radius, float(radius[1] - radius[0]))
    mass_weight[[0, -1]] *= 0.5
    radius_weight[[0, -1]] *= 0.5
    log_weight = torch.log((mass_weight[:, None] * radius_weight[None, :]).flatten())
    grid_mass, grid_radius = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in interpolator.grid)
    log_density = torch.as_tensor(np.asarray(interpolator.values), dtype=torch.float32, device=device)
    grid_mass_mesh, grid_radius_mesh = torch.meshgrid(grid_mass, grid_radius, indexing="ij")
    keep = ((log_density > float(interpolator.fill_value) + 1.0e-6) & (grid_mass_mesh >= mass_low)
            & (grid_mass_mesh <= mass_high) & (grid_radius_mesh >= radius_low) & (grid_radius_mesh <= radius_high))
    grid_mass_weight = torch.full_like(grid_mass, float(grid_mass[1] - grid_mass[0]))
    grid_radius_weight = torch.full_like(grid_radius, float(grid_radius[1] - grid_radius[0]))
    grid_mass_weight[[0, -1]] *= 0.5
    grid_radius_weight[[0, -1]] *= 0.5
    source_log_weight = (log_density + torch.log(grid_mass_weight[:, None] * grid_radius_weight[None, :]))[keep][None, :]
    a1 = paper[0:1]
    with torch.inference_mode():
        full_kernel = model(lattice, a1, slot) + mean
        floor_overlap = torch.logsumexp(full_kernel + log_weight[:, None], dim=0)[0]
        source_kernel = model(unit(grid_mass_mesh[keep], grid_radius_mesh[keep], domain), a1, slot) + mean
        main_overlap = log_overlap(source_log_weight, source_kernel)[0, 0]
        value = torch.logaddexp(main_overlap, torch.as_tensor(float(interpolator.fill_value), device=device) + floor_overlap)
    return float(value.cpu())


def replay_members(device: torch.device, heldout_root: Path, checkpoints: Path) -> dict:
    paper = torch.as_tensor(np.stack([(np.asarray(nuclear_observation(n)) - OBSERVATION) / SIGMA
                                      for n in NUCLEAR_SCENARIO_NAMES]), dtype=torch.float32, device=device)
    tables = {case: build_nicer_grid(heldout_root / spec.filename, mcol=spec.mass_column, rcol=spec.radius_column,
                                     wcol_or_None=spec.weight_column, **GRID_OPTIONS)
              for case, spec in SOURCE_SUBSTITUTIONS.items()}
    values = {}
    for sector, seeds in SEEDS.items():
        for seed in seeds:
            model, payload = load_member(checkpoints / f"{sector}_green_seed{seed}.pt", device)
            with torch.inference_mode():
                identity = (model.identity(paper) + float(payload["mean"])).double().cpu().numpy()
            member = dict(zip(NUCLEAR_SCENARIO_NAMES, map(float, identity)))
            for case, spec in SOURCE_SUBSTITUTIONS.items():
                member[case] = source_value(model, payload, tables[case], spec.slot, paper, device)
            values[(sector, seed)] = member
    return values


def rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def aggregate(members: dict, training_reports: Path, exact_bank: Path) -> dict:
    """The paper's Table VI aggregation."""
    stored = json.loads((FIX / "nucleonic_stored_sigma.json").read_text())
    stored_sigma = {(row["sector"], row["configuration"]): float(row["sigma"]) for row in stored["rows"]}
    rows = {}
    for sector, seeds in SEEDS.items():
        reports = [json.loads((training_reports / f"{sector}_green_seed{s}.json").read_text()) for s in seeds]
        exact = json.loads((exact_bank / f"{sector}_exact_bank.json").read_text())
        identity_calibration = rms(np.concatenate([np.asarray(r["identity_paper_residual"]) for r in reports]))
        source_calibration = {}
        for configuration, baseline in SOURCE_TO_BASELINE.items():
            synthetic = rms([r["blind_synthetic_one_source"]["paper"][baseline]["rmse"] for r in reports])
            transfer = rms(np.concatenate([np.asarray(r["real_baseline_clouds_no_encoding"][baseline]["TOTAL_error"])
                                           for r in reports]))
            source_calibration[configuration] = math.hypot(synthetic, transfer)
        for configuration in CONFIGURATIONS:
            member_values = np.asarray([members[(sector, s)][configuration] for s in seeds], dtype=np.float64)
            centre = float(logsumexp(member_values) - math.log(len(member_values)))
            seed_scatter = float(np.std(member_values, ddof=1))
            bank_row = exact["nuclear_results" if configuration in NUCLEAR else "heldout_results"][configuration]
            calibration = identity_calibration if configuration in NUCLEAR else source_calibration[configuration]
            three_term = math.sqrt(seed_scatter ** 2 + float(bank_row["naive_monte_carlo_error"]) ** 2 + calibration ** 2)
            base = max(stored_sigma[(sector, configuration)], three_term) if sector == "nucleonic" else three_term
            sigma = base if configuration in NUCLEAR else math.hypot(base, abs(centre - float(bank_row["log_evidence"])))
            rows[(sector, configuration)] = {"members": member_values.tolist(), "centre": centre, "sigma": sigma}
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    parser.add_argument("--heldout-root", type=Path, default=ROOT / "data/observations")
    parser.add_argument("--report", type=Path, default=ROOT / "build/route2/evidence_check.json")
    parser.add_argument("--checkpoints", type=Path, default=FIX / "checkpoints")
    parser.add_argument("--training-reports", type=Path, default=FIX / "training_reports")
    parser.add_argument("--exact-bank", type=Path, default=FIX / "exact_bank")
    arguments = parser.parse_args()
    started = time.time()
    device = torch.device(arguments.device)
    members = replay_members(device, arguments.heldout_root, arguments.checkpoints)
    rows = aggregate(members, arguments.training_reports, arguments.exact_bank)
    published = {(r["sector"], r["configuration"]): r
              for r in json.loads((ROOT / "results/evidence/green_en_vs_ultranest_16_rows.json").read_text())["rows"]}
    worst_member = max(abs(a - b) for key, row in rows.items() for a, b in zip(row["members"], published[key]["member_log_evidence"]))
    worst_centre = max(abs(row["centre"] - published[key]["en_log_evidence"]) for key, row in rows.items())
    worst_sigma = max(abs(row["sigma"] - published[key]["en_sigma"]) for key, row in rows.items())
    # The paper prints three decimals. A printed digit may only differ where the published value itself lies within
    # the tolerance of a rounding boundary (two values sit 3e-7 and 2e-6 from one); such cases are listed.
    printed_mismatch, boundary_cases = [], []
    for key, row in rows.items():
        for replayed, published_value in ((row["centre"], published[key]["en_log_evidence"]),
                                          (row["sigma"], published[key]["en_sigma"])):
            if f"{replayed:.3f}" == f"{published_value:.3f}":
                continue
            distance_to_boundary = abs((abs(published_value) * 1000) % 1 - 0.5) / 1000
            at_boundary = abs(replayed - published_value) <= TOLERANCE and distance_to_boundary <= TOLERANCE
            (boundary_cases if at_boundary else printed_mismatch).append("/".join(key))
    printed_ok = not printed_mismatch
    ok = max(worst_member, worst_centre, worst_sigma) <= TOLERANCE and printed_ok
    report = {"status": "PASS" if ok else "FAIL", "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "torch": torch.__version__,
              "member_values": 48, "largest_member_difference": worst_member,
              "largest_centre_difference": worst_centre, "largest_sigma_difference": worst_sigma,
              "tolerance_log_z": TOLERANCE, "printed_table_vi_values_identical": printed_ok,
              "printed_digit_differences_at_rounding_boundaries": boundary_cases,
              "seconds": round(time.time() - started, 1),
              "rows": {f"{s}/{c}": v for (s, c), v in rows.items()}}
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=1) + "\n")
    print(f"[evidence] {report['status']}: 48 member values and 16 Table VI evidences rebuilt from the frozen networks "
          f"on {device} in {report['seconds']} s; largest differences: member {worst_member:.1e}, "
          f"centre {worst_centre:.1e}, sigma {worst_sigma:.1e} (tolerance {TOLERANCE:.0e}); printed Table VI values "
          f"{'identical' if printed_ok else 'DIFFERENT: ' + ', '.join(printed_mismatch)}"
          + (f" (last digit at a rounding boundary: {', '.join(boundary_cases)})" if boundary_cases else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
