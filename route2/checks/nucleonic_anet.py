#!/usr/bin/env python3
"""Route 2 check: nucleonic A-NET posteriors (A1, J0614, J1231, the four nuclear shifts, J1614), replayed.

1. Weights. The stored importance weights and ESS of every A-NET run are recomputed from the saved exact
   log-likelihoods (log L) and proposal densities (log q). For A1, J0614 and J1231 the NICER single-weight
   correction terms are applied as in the paper (ESS 525.0, 445.8, 1548.8), and the A1 posterior shipped in
   results/nucleonic/a1_parameter_posteriors.npz is rebuilt exactly. The nuclear-shift seeds are pooled as in
   the paper (ESS 395, 993, 1279, 756) and the J1614 seeds give log Z = -26.383 +/- 0.016 and ESS 3296. All
   comparisons are exact or within 1e-12.
2. Resamples and curve selections. The stored resample indices and the draws behind the plotted
   nuclear-shift bands (index hashes in results/nucleonic/nuclear_amortization_verification.json) are redrawn
   from the recomputed weights with the stored seeds and must be identical. The J1614 weights must equal
   those in results/nucleonic/j1614_anet_mass_radius_curves.npz.
3. Likelihood. The exact single-weight target is rebuilt from data/observations and log L is recomputed for 14
   saved rows per configuration (the 6 largest weights and 8 random accepted rows; tolerance 1e-4).
4. Proposal density. A-NET samples by integrating its flow (64 Heun steps) from a random normal draw, and
   log q is accumulated along that path. The random draw came from the GPU generator, so it is recovered here
   instead: Newton's method on a float64 copy of the flow solves "path(start) = saved sample" for each of the
   14 rows. The production float32 path is then rerun from that start. It must land on the saved sample
   (within 1e-4 in standardized units), and its log q must equal the saved value (tolerance 2e-3, well below
   the log Z uncertainties of 0.016-0.05). The conditioning clouds are redrawn from the NICER data (the J1614
   ones must equal the stored per-seed clouds exactly).

Run from the repository root:  PYTHONPATH=$PWD python -m route2.checks.nucleonic_anet [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from eos import get_eos
from inference.anet.proposal import FMPEProposal, mass_radius_clouds
from workflows.a1_problem import A1Problem
from workflows.nuclear_scenarios import nuclear_observation

from route2.common import OBSERVATIONS, ROOT, prepare_observations

FIX = ROOT / "route2/fixtures/nucleonic_anet"
CHECKPOINT = ROOT / "inference/anet/checkpoints/fmpe9c_net.pt"
STANDARDIZATION = ROOT / "inference/anet/checkpoints/fmpe9c_std.npz"
CONFIGURATIONS = ("A1", "J0614", "J1231", "K0_200", "K0_260", "Jsym_29", "Jsym_36", "J1614")
SHIFT_ORDER = ["K0_200", "K0_260", "Jsym_29", "Jsym_36"]  # order of verify_and_render_nuclear_amortization.py
CURVE_DRAWS = 8000
TOLERANCE = {"arithmetic": 1.0e-12, "log_likelihood": 1.0e-4, "log_q": 2.0e-3, "path_endpoint": 1.0e-4}


def normalized(logweight: np.ndarray) -> np.ndarray:
    finite = np.isfinite(logweight)
    weight = np.exp(np.where(finite, logweight - np.max(logweight[finite]), -np.inf))
    return weight / weight.sum()


def ess(weight: np.ndarray) -> float:
    return float(weight.sum() ** 2 / np.sum(weight * weight))


def close(a, b, tolerance=TOLERANCE["arithmetic"]) -> bool:
    return bool(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))) <= tolerance)


def check_weights(archive: dict, expected: dict) -> dict:
    """Steps 1 and 2: weights, corrections, pooling, resamples and curve selections."""
    results, corrected_weights = {}, {}
    for run in ("A1", "J0614", "J1231"):
        weight = normalized(archive[f"{run}__logl"] - archive[f"{run}__logq"])
        positive = weight > 0.0
        delta = archive[f"{run}__delta"]
        corrected = np.zeros_like(weight)  # compute_tableIII_corrected_ess.py, main path
        corrected[positive] = weight[positive] * np.exp(delta[positive] - np.max(delta[positive]))
        corrected /= corrected.sum()
        index = np.random.default_rng(1).choice(len(weight), size=len(weight), replace=True, p=weight)
        record = expected["configurations"][run]
        results[run] = {
            "stored_weights": close(weight, archive[f"{run}__w"]),
            "stored_ess": close(ess(weight), record["stored_ess"], 1e-9),
            "corrected_weights": close(corrected, archive[f"{run}__corrected_weight"], 5e-15),
            "corrected_ess": close(ess(corrected), record["corrected_ess"], 1e-9),
            "resample": bool(np.array_equal(index, archive[f"{run}__resample_index"])),
            "ess": ess(corrected)}
        corrected_weights[run] = corrected
    with np.load(ROOT / "results/nucleonic/a1_parameter_posteriors.npz", allow_pickle=False) as compact:
        corrected = corrected_weights["A1"]
        positive = corrected > 0.0
        results["A1"]["shipped_posterior"] = bool(
            np.array_equal(compact["theta_anet_is"], archive["A1__raw"][positive])
            and np.array_equal(compact["weight_anet_is"], corrected[positive] / corrected[positive].sum()))

    shipped = json.loads((ROOT / "results/nucleonic/nuclear_amortization_verification.json").read_text())["cases"]
    for case in SHIFT_ORDER:
        record = expected["configurations"][case]
        checks = {}
        logweight = []
        for run in record["runs"]:
            weight = normalized(archive[f"{run}__logl"] - archive[f"{run}__logq"])
            index = np.random.default_rng(271828).choice(len(weight), len(weight), p=weight)
            checks[f"{run}_weights"] = close(weight, archive[f"{run}__w"])
            checks[f"{run}_ess"] = close(ess(weight), expected["runs"][run]["stored_ess"], 1e-9)
            checks[f"{run}_resample"] = bool(np.array_equal(index, archive[f"{run}__resample_index"]))
            logweight.append(archive[f"{run}__logl"] - archive[f"{run}__logq"])
        pooled = normalized(np.concatenate(logweight))
        seed = 2026082700 + 100 * SHIFT_ORDER.index(case)
        if case == "K0_260":  # the paper's deterministic (systematic) selection for this case
            curve_index = np.searchsorted(np.cumsum(pooled), (np.arange(CURVE_DRAWS) + 0.5) / CURVE_DRAWS, side="left")
        else:
            curve_index = np.random.default_rng(seed).choice(len(pooled), CURVE_DRAWS, replace=True, p=pooled)
        checks["pooled_ess"] = close(1.0 / np.sum(pooled ** 2), record["pooled_ess"], 1e-9)
        checks["pooled_ess_matches_shipped"] = close(1.0 / np.sum(pooled ** 2), shipped[case]["anet"]["ess"], 1e-9)
        checks["curve_selection"] = (hashlib.sha256(np.ascontiguousarray(curve_index).tobytes()).hexdigest()
                                     == shipped[case]["curve_selection"]["anet_index_sha256"]
                                     and seed == shipped[case]["curve_selection"]["anet_seed"])
        checks["ess"] = float(1.0 / np.sum(pooled ** 2))
        results[case] = checks

    record = expected["configurations"]["J1614"]
    logl, logq = archive["J1614__logl"], archive["J1614__logq"]
    plugin = get_eos("ddb")
    log_prior = -float(np.sum(np.log(np.asarray(plugin.prior_high) - np.asarray(plugin.prior_low))))
    logweight = logl + log_prior - logq
    weight = normalized(logweight)
    finite = np.isfinite(logweight)
    count = len(logweight)
    log_evidence = float(logsumexp(logweight[finite]) - math.log(count))
    pooled_ess = 1.0 / float(np.sum(weight ** 2))
    # The pooled evidence is the mean of the four equal-size seed evidences; its error combines the four
    # delta-method errors (compute_importance_weights) weighted by the seed evidences.
    seed_evidence, seed_error = [], []
    for seed in np.unique(archive["J1614__seed_label"]):
        local = logweight[archive["J1614__seed_label"] == seed]
        valid = np.isfinite(local)
        scaled = np.zeros(len(local))
        scaled[valid] = np.exp(local[valid] - np.max(local[valid]))
        seed_evidence.append(np.max(local[valid]) + math.log(scaled.mean()))
        seed_error.append(math.sqrt(np.var(scaled, ddof=1) / len(scaled)) / scaled.mean())
    relative = np.exp(np.asarray(seed_evidence) - max(seed_evidence))
    pooled_error = float(np.sqrt(np.sum((relative * np.asarray(seed_error)) ** 2)) / relative.sum())
    with np.load(ROOT / "results/nucleonic/j1614_anet_mass_radius_curves.npz", allow_pickle=False) as curves:
        shipped_weights = np.asarray(curves["weight"], dtype=np.float64)
    results["J1614"] = {
        "log_weights": close(logweight[finite], archive["J1614__logw"][finite], 1e-10),
        "weights": close(weight, archive["J1614__w"]),
        "pooled_ess": close(pooled_ess, record["pooled_ess"], 1e-9),
        "log_evidence": close(log_evidence, record["log_evidence"], 1e-10),
        "log_evidence_standard_error": close(pooled_error, record["log_evidence_standard_error"], 1e-12),
        "shipped_curve_weights": close(shipped_weights / shipped_weights.sum(), weight[finite] / weight[finite].sum(), 1e-15),
        "ess": pooled_ess, "log_evidence_value": log_evidence}
    for checks in results.values():
        checks["pass"] = all(value for value in checks.values() if isinstance(value, bool))
    return results


class PathReplay:
    """Recompute A-NET's path log q for saved samples (step 4)."""

    def __init__(self, device: str):
        self.proposal = FMPEProposal(CHECKPOINT, STANDARDIZATION, device=device, steps=64)
        self.field64 = copy.deepcopy(self.proposal.vector_field).double()
        self.steps = self.proposal.steps

    def _path64(self, start, context):
        step = 1.0 / self.steps
        value = start
        for index in range(self.steps):
            time_ = torch.full((len(value), 1), index * step, dtype=torch.float64, device=value.device)
            first = self.field64(value, time_, context)
            second = self.field64(value + step * first, time_ + step, context)
            value = value + step * 0.5 * (first + second)
        return value

    def _reverse64(self, end, context):
        step = 1.0 / self.steps
        value = end
        with torch.no_grad():
            for index in range(self.steps, 0, -1):
                time_ = torch.full((len(value), 1), index * step, dtype=torch.float64, device=value.device)
                first = self.field64(value, time_, context)
                second = self.field64(value - step * first, time_ - step, context)
                value = value - step * 0.5 * (first + second)
        return value

    def recover_start(self, end, context):
        """Newton's method for path(start) = end, started from the reverse integration."""
        start = self._reverse64(end, context)
        for _ in range(12):
            variable = start.detach().requires_grad_(True)
            reached = self._path64(variable, context)
            residual = (reached - end).detach()
            if float(residual.abs().max()) < 1.0e-12:
                break
            jacobian = torch.stack([torch.autograd.grad(reached[:, k].sum(), variable, retain_graph=k < 6)[0]
                                    for k in range(7)], dim=1)
            start = (start - torch.linalg.solve(jacobian, residual.unsqueeze(-1)).squeeze(-1)).detach()
        return start.detach()

    def log_q(self, theta, clouds, observation, physical: bool):
        """Returns (log q along the production float32 path, largest endpoint error)."""
        proposal = self.proposal
        rows = len(theta)
        context = proposal._context(clouds, observation, 0.0, 1.2, rows)
        end32 = torch.tensor((theta - proposal.theta_mean) / proposal.theta_scale, dtype=torch.float32,
                             device=proposal.device)
        start = self.recover_start(end32.double(), context.double()).float()
        # The production loop of FMPEProposal.sample_with_log_density, started from the recovered draw.
        value = start
        log_density = -0.5 * 7 * np.log(2 * np.pi) - 0.5 * (value ** 2).sum(1)
        step = 1.0 / self.steps
        for index in range(self.steps):
            time_ = torch.full((rows, 1), index * step, device=proposal.device)
            with torch.no_grad():
                velocity1 = proposal.vector_field(value, time_, context)
            divergence1 = proposal._divergence(value, time_, context)
            predictor = value + step * velocity1
            with torch.no_grad():
                velocity2 = proposal.vector_field(predictor, time_ + step, context)
            divergence2 = proposal._divergence(predictor, time_ + step, context)
            value = (value + step * 0.5 * (velocity1 + velocity2)).detach()
            log_density = log_density - step * 0.5 * (divergence1 + divergence2)
        result = log_density.cpu().numpy().astype(np.float64)
        if physical:
            result = result - np.sum(np.log(proposal.theta_scale))
        return result, float((value - end32).abs().max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    parser.add_argument("--report", type=Path, default=ROOT / "build/route2/nucleonic_anet_check.json")
    arguments = parser.parse_args()
    started = time.time()
    prepare_observations()
    expected = json.loads((FIX / "expected.json").read_text())
    with np.load(FIX / "archives.npz", allow_pickle=False) as data:
        archive = {key: data[key] for key in data.files}
    with np.load(FIX / "replay_rows.npz", allow_pickle=False) as data:
        rows_fixture = {key: data[key] for key in data.files}
    with np.load(STANDARDIZATION, allow_pickle=False) as saved:
        standardized_observation = np.asarray(saved["OBS"], dtype=np.float64)
    if not np.array_equal(standardized_observation, nuclear_observation("A1")):
        raise SystemExit("FAIL: the A-NET standardization file does not carry the A1 observation")

    arithmetic = check_weights(archive, expected)
    print("[nucleonic-anet] weights, corrections, pooling, resamples and curve selections: "
          + ", ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}" for k, v in arithmetic.items()), flush=True)

    print(f"[nucleonic-anet] rebuilding the exact targets from {OBSERVATIONS.relative_to(ROOT)} ...", flush=True)
    problems = {source: A1Problem.from_data_root(OBSERVATIONS, source_data_root=OBSERVATIONS, workers=1,
                                                 verify_data=True, source_scenario=source, model="ddb")
                for source in ("A1", "J0614", "J1231", "J1614")}
    replay = PathReplay(arguments.device)
    j1614_clouds = {}
    for seed in range(4):
        clouds = mass_radius_clouds(OBSERVATIONS, points=replay.proposal.cloud_points, seed=seed,
                                    source_scenario="J1614", source_data_root=OBSERVATIONS)
        if not np.array_equal(clouds, rows_fixture[f"J1614__clouds_seed{seed}"]):
            raise SystemExit(f"FAIL: J1614 conditioning clouds of seed {seed} differ from the stored ones")
        j1614_clouds[seed] = clouds

    reports = {}
    for configuration in CONFIGURATIONS:
        case_started = time.time()
        record = expected["configurations"][configuration]
        rows, theta = rows_fixture[f"{configuration}__rows"], rows_fixture[f"{configuration}__theta"]
        runs = record["runs"]
        if record["kind"] == "source":
            logl = (archive[f"{configuration}__logl"] + archive[f"{configuration}__delta"])[rows]
            logq_saved = archive[f"{configuration}__logq"][rows]
            source, nuclear = configuration, "A1"
        elif record["kind"] == "nuclear_shift":
            logl = np.concatenate([archive[f"{run}__logl"] for run in runs])[rows]
            logq_saved = np.concatenate([archive[f"{run}__logq"] for run in runs])[rows]
            source, nuclear = "A1", configuration
        else:
            logl, logq_saved = archive["J1614__logl"][rows], archive["J1614__logq"][rows]
            source, nuclear = "J1614", "A1"
        problem = problems[source]
        if nuclear != "A1":
            problem = dataclasses.replace(problem, nuclear_scenario=nuclear, target=dataclasses.replace(
                problem.target, nuclear_observation=nuclear_observation(nuclear)))
        replayed_logl = problem.evaluate_batch(theta)
        logl_difference = float(np.max(np.abs(replayed_logl - logl)))

        observation = nuclear_observation(nuclear)
        if record["kind"] == "nuclear_shift":
            for run in runs:
                if not np.array_equal(archive[f"{run}__nuclear_observation"], observation):
                    raise SystemExit(f"FAIL: {run} was not conditioned on the {nuclear} observation")
        replayed_logq = np.empty(len(rows))
        endpoint = 0.0
        if configuration == "J1614":
            seed_of_row = archive["J1614__seed_label"][rows]
            groups = {int(seed): np.flatnonzero(seed_of_row == seed) for seed in np.unique(seed_of_row)}
            for seed, members in groups.items():
                values, error = replay.log_q(theta[members], j1614_clouds[seed], observation, physical=True)
                replayed_logq[members] = values
                endpoint = max(endpoint, error)
        else:
            clouds = mass_radius_clouds(OBSERVATIONS, points=replay.proposal.cloud_points, seed=0,
                                        source_scenario=source, source_data_root=OBSERVATIONS)
            replayed_logq, endpoint = replay.log_q(theta, clouds, observation, physical=False)
        logq_difference = float(np.max(np.abs(replayed_logq - logq_saved)))
        report = {
            "weights": arithmetic[configuration],
            "likelihood": {"pass": logl_difference <= TOLERANCE["log_likelihood"], "replayed_rows": int(len(rows)),
                           "largest_difference": logl_difference},
            "proposal": {"pass": logq_difference <= TOLERANCE["log_q"] and endpoint <= TOLERANCE["path_endpoint"],
                         "replayed_rows": int(len(rows)), "largest_log_q_difference": logq_difference,
                         "largest_path_endpoint_error": endpoint},
        }
        report["pass"] = all(report[step]["pass"] for step in ("weights", "likelihood", "proposal"))
        reports[configuration] = report
        print(f"[nucleonic-anet] {configuration:8s} {'PASS' if report['pass'] else 'FAIL'}  ESS "
              f"{arithmetic[configuration]['ess']:.1f}, log L diff {logl_difference:.0e}, log q diff "
              f"{logq_difference:.0e} (path end {endpoint:.0e})  ({time.time() - case_started:.0f} s)", flush=True)

    ok = all(report["pass"] for report in reports.values())
    summary = {"status": "PASS" if ok else "FAIL", "device": arguments.device, "torch": torch.__version__,
               "tolerances": TOLERANCE, "seconds": round(time.time() - started, 1), "configurations": reports}
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[nucleonic-anet] {summary['status']}: 8 configurations; weights, corrections, resamples, likelihoods and "
          f"proposal densities replayed in {summary['seconds']} s on {arguments.device}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
