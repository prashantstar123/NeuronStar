#!/usr/bin/env python3
"""Route 2 check: nucleonic TSNPE+MIS posterior and evidence for A1, replayed from the saved final stage.

The final importance sampling (MIS) of the paper is the output of inference/tsnpe/run_mis.py with its default settings
(seed 0) and the frozen TSNPE estimator: 105,000 parameter rows from a three-part mixture, 60,000 from the frozen flow,
15,000 from a broad and 30,000 from a mild box-truncated Gaussian. Each row has the weight
log w = log L + log prior - log q, where q is the whole mixture density. The release file
tsnpe_nucleonic_mis_a1_20261003.npz is that output: theta, log L, the three component densities at every row, the
mixture density, log w, the normalized weights, ESS, log Z and its error. No other method's result enters it.

1. MIS arithmetic (1e-12). The mixture density is rebuilt from the three stored component densities and the counts
   60,000/15,000/30,000 (balance heuristic), then log w, the normalized weights, ESS = 1 / sum w_i^2 = 30,762.70,
   log Z = log(mean of exp(log w)) over all 105,000 rows = -25.54883, and its error, with the production code
   (inference.common.compute_importance_weights; inference/tsnpe/run_mis.py stage_report and log_mixture_density).
   They must equal the stored values and the run report. The 3 rows with no finite weight must be exactly the rows
   whose log L is not finite.
2. Proposal replay (exact). The broad Gaussian is rebuilt from the 60,000 stored flow rows and the mild Gaussian from
   the first 75,000 rows, as run_mis.py builds them (importance-weighted mean and covariance, scales 2.5 and 1.5,
   seeds 101/102 and 201/202, 2,000,000 draws for the share inside the prior box). Their densities at all 105,000 rows
   and the 45,000 Gaussian draws must equal the stored ones (1e-9; observed 0).
3. Frozen estimator, loaded exactly as run_mis.py loads it (release file tsnpe_nucleonic_density_estimator.pt; sbi
   DirectPosterior, BoxUniform prior on the 7-D box, A1 observation):
   - its log density at the 14 rows of step 4 equals the stored flow density within 0.01. sbi normalizes the
     box-truncated flow with a Monte Carlo estimate of its mass inside the box, made anew for every batch of 4,000 rows
     (standard error about 0.0014 in log), so the stored rows of different batches and this evaluation carry slightly
     different constants;
   - 20,000 fresh draws (torch seed 0) match the stored training output posterior_FINAL.npy. Test: two-sample
     Kolmogorov-Smirnov per parameter; fail if any D exceeds the critical value for a false-alarm rate of 1e-3 over
     all 7 parameters (D = 0.022; typical D is 0.01);
   - the stored training history (history.json, astro_meta.json: 6 rounds, 30,000 + 5 x 20,000 simulations,
     132,000 with the pilot), the flow's size (10 transforms, 256 hidden units) and the mixture counts
     (60,000/15,000/30,000) equal the defaults of inference/tsnpe/train.py and run_mis.py.
4. Exact log L (1e-4). The A1 target is rebuilt from data/observations and log L is recomputed for 14 rows (the 6
   largest weights and 8 random rows, seed 20261003). It must equal the stored log L within 1e-4, the target
   certificate's tolerance (observed about 1e-6). The rebuilt target must also reject the 3 rows without a finite
   weight (log L <= -1e29).
5. Route 1 identity (exact). theta_tsnpe_mis and weight_tsnpe_mis in results/nucleonic/a1_parameter_posteriors.npz are
   the 104,846 rows with a positive weight and their normalized weights, bit for bit.
6. Code identity: the 7 files of inference/tsnpe must be byte-identical to the stored reference (SHA-256 in
   expected.json).

Run from the repository root:  PYTHONPATH=$PWD python -m route2.checks.tsnpe_nucleonic [--device cpu|cuda]
The two large inputs come from the GitHub release via route2.fetch (group tsnpe_nucleonic);
ROUTE2_LOCAL_MIRROR=/path/to/dir copies them from a local directory instead.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import time
from pathlib import Path

from route2.common import OBSERVATIONS, ROOT, prepare_observations, sha256  # sets JAX to CPU; import first
from route2.fetch import fetch

import numpy as np
import sbi
import torch
from scipy.special import kolmogi
from scipy.stats import ks_2samp

from eos.ddb import PRIOR_HIGH, PRIOR_LOW
from inference.common import compute_importance_weights
from inference.tsnpe.mixture import TruncatedGaussian, weighted_mean_covariance
from inference.tsnpe.run_mis import (evaluate_flow_log_density, log_mixture_density, stage_report,
                                     target_log_density)
from likelihoods.nuclear import A1_OBSERVATION
from workflows.a1_problem import A1Problem

FIX = ROOT / "route2/fixtures/tsnpe_nucleonic"
ARCHIVE = "tsnpe_nucleonic_mis_a1_20261003.npz"
ESTIMATOR = "tsnpe_nucleonic_density_estimator.pt"
INVALID = -1.0e29  # the target marks a row invalid with -1e30 per failed likelihood term
TOLERANCE = {"arithmetic": 1.0e-12, "proposal_replay": 1.0e-9, "log_likelihood": 1.0e-4,
             "flow_normalization": 1.0e-2, "flow_row_to_row": 1.0e-2, "ks_family_alpha": 1.0e-3}


def argument_defaults(script: Path, names: set[str]) -> dict[str, object]:
    """Read the argparse defaults of a production script without running it."""
    defaults = {}
    for node in ast.walk(ast.parse(script.read_text())):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value in names):
            for keyword in node.keywords:
                if keyword.arg == "default":
                    defaults[node.args[0].value] = ast.literal_eval(keyword.value)
    return defaults


def relative_difference(value: float, reference: float) -> float:
    return abs(value / reference - 1.0)


def check_code(expected: dict) -> dict:
    """Step 6: the TSNPE package is byte-identical to the stored reference (SHA-256 in expected.json)."""
    differing = [name for name, digest in expected["code_identity"].items() if sha256(ROOT / name) != digest]
    return {"pass": not differing, "files": len(expected["code_identity"]), "differing": differing}


def component_counts(archive: dict, expected: dict) -> list[int]:
    labels = [str(label) for label in archive["component_labels"]]
    counts = [int(np.sum(archive["proposal_component"] == index)) for index in range(len(labels))]
    if labels != ["flow", "broad_truncated_gaussian", "mild_truncated_gaussian"] or counts != [
            expected["mis"]["component_counts"][label] for label in labels]:
        raise SystemExit(f"FAIL: unexpected mixture components {labels} {counts}")
    if np.any(np.diff(archive["proposal_component"].astype(int)) < 0):
        raise SystemExit("FAIL: the mixture components are not stored in the order flow, broad, mild")
    return counts


def check_mis(archive: dict, expected: dict) -> dict:
    """Step 1: mixture density, weights, ESS, log Z and its error from the stored component densities."""
    counts = component_counts(archive, expected)
    log_proposal = log_mixture_density([archive["logq_flow"], archive["logq_broad"], archive["logq_mild"]], counts)
    importance = compute_importance_weights(target_log_density(archive["logL"]), log_proposal)
    replay = stage_report(importance, len(log_proposal))
    report = expected["mis"]["run_report"]
    differences = {
        "mixture_density": float(np.max(np.abs(log_proposal - archive["logq"]))),
        "log_weight": float(np.max(np.abs(np.where(np.isfinite(importance.log_weight), importance.log_weight, 0.0)
                                          - np.where(np.isfinite(archive["logw"]), archive["logw"], 0.0)))),
        "normalized_weight": float(np.max(np.abs(importance.normalized_weight - archive["w"]))),
        "ess_relative": relative_difference(importance.ess, float(archive["ESS"])),
        "log_evidence": abs(importance.log_evidence - float(archive["logZ"])),
        "log_evidence_error": abs(importance.log_evidence_standard_error - float(archive["logZ_standard_error"])),
        "report_ess_relative": relative_difference(replay["ess"], report["ess"]),
        "report_log_evidence": abs(replay["log_evidence"] - report["log_evidence"]),
        "report_log_evidence_error": abs(replay["log_evidence_standard_error"] - report["log_evidence_standard_error"]),
    }
    no_weight = ~np.isfinite(archive["logw"])
    record = {
        "ess": importance.ess, "log_evidence": importance.log_evidence,
        "log_evidence_error": importance.log_evidence_standard_error, "differences": differences,
        "stored_scalars_equal_expected": (float(archive["ESS"]), float(archive["logZ"]), float(archive["logZ_standard_error"]))
        == (expected["mis"]["stored"]["ESS"], expected["mis"]["stored"]["logZ"], expected["mis"]["stored"]["logZ_standard_error"]),
        "same_infinite_rows": bool(np.array_equal(no_weight, ~np.isfinite(importance.log_weight))),
        "rows_without_weight_are_invalid_rows": bool(np.array_equal(no_weight, ~(np.isfinite(archive["logL"])
                                                                                 & (archive["logL"] > -1e50)))
                                                     and int(no_weight.sum()) == expected["mis"]["rows_without_weight"]),
        "rows": int(len(log_proposal)) == expected["mis"]["rows"],
    }
    record["pass"] = (all(value <= TOLERANCE["arithmetic"] for value in differences.values())
                      and all(value for value in record.values() if isinstance(value, bool)))
    return record


def check_proposal(archive: dict, expected: dict) -> dict:
    """Step 2: rebuild the broad and mild Gaussians from the stored rows, as run_mis.py builds them."""
    counts = component_counts(archive, expected)
    settings = expected["proposal"]
    theta, log_likelihood, log_flow = archive["theta"], archive["logL"], archive["logq_flow"]
    flow_rows, broad_rows = counts[0], counts[0] + counts[1]
    flow_importance = compute_importance_weights(target_log_density(log_likelihood[:flow_rows]), log_flow[:flow_rows])
    broad = TruncatedGaussian.from_covariance(
        *weighted_mean_covariance(theta[:flow_rows], flow_importance.log_weight), PRIOR_LOW, PRIOR_HIGH,
        scale=settings["broad_scale"], rng=np.random.default_rng(settings["seed"] + 101),
        normalization_draws=settings["normalization_draws"])
    broad_draws = broad.sample(counts[1], rng=np.random.default_rng(settings["seed"] + 102))
    intermediate = log_mixture_density([log_flow[:broad_rows], broad.log_density(theta[:broad_rows])], counts[:2])
    intermediate_importance = compute_importance_weights(target_log_density(log_likelihood[:broad_rows]), intermediate)
    mild = TruncatedGaussian.from_covariance(
        *weighted_mean_covariance(theta[:broad_rows], intermediate_importance.log_weight), PRIOR_LOW, PRIOR_HIGH,
        scale=settings["mild_scale"], rng=np.random.default_rng(settings["seed"] + 201),
        normalization_draws=settings["normalization_draws"])
    mild_draws = mild.sample(counts[2], rng=np.random.default_rng(settings["seed"] + 202))
    differences = {
        "broad_draws": float(np.max(np.abs(broad_draws - theta[flow_rows:broad_rows]))),
        "mild_draws": float(np.max(np.abs(mild_draws - theta[broad_rows:]))),
        "broad_density": float(np.max(np.abs(broad.log_density(theta) - archive["logq_broad"]))),
        "mild_density": float(np.max(np.abs(mild.log_density(theta) - archive["logq_mild"]))),
    }
    record = {"normalizations": {"broad": broad.normalization, "mild": mild.normalization},
              "normalizations_equal_report": [broad.normalization, mild.normalization]
              == [settings["truncation_normalizations"]["broad"], settings["truncation_normalizations"]["mild"]],
              "differences": differences}
    record["pass"] = (record["normalizations_equal_report"]
                      and all(value <= TOLERANCE["proposal_replay"] for value in differences.values()))
    return record


def replay_rows(archive: dict, expected: dict) -> tuple[np.ndarray, np.ndarray]:
    """The 6 largest weights and 8 random positive-weight rows; the rows without a finite weight."""
    selection = expected["row_selection"]
    weight = archive["w"]
    largest = np.argsort(-weight, kind="mergesort")[: selection["largest"]]
    others = np.setdiff1d(np.flatnonzero(weight > 0.0), largest)
    chosen = np.random.default_rng(selection["seed"]).choice(others, selection["random"], replace=False)
    return np.concatenate([largest, np.sort(chosen)]), np.flatnonzero(~np.isfinite(archive["logw"]))


def check_estimator(path: Path, archive: dict, rows: np.ndarray, expected: dict, device: str) -> dict:
    """Step 3: the frozen flow, its draws, and the stored training settings."""
    from sbi.inference.posteriors import DirectPosterior
    from sbi.utils import BoxUniform

    estimator = torch.load(path, map_location=device, weights_only=False)  # as in run_mis.py
    prior = BoxUniform(low=torch.as_tensor(PRIOR_LOW, dtype=torch.float32, device=device),
                       high=torch.as_tensor(PRIOR_HIGH, dtype=torch.float32, device=device))
    posterior = DirectPosterior(posterior_estimator=estimator, prior=prior, device=device)
    observed = torch.as_tensor(A1_OBSERVATION, dtype=torch.float32, device=device)

    torch.manual_seed(0)
    log_flow = evaluate_flow_log_density(posterior, observed, archive["theta"][rows], device)
    offset = log_flow - archive["logq_flow"][rows]
    constant = float(np.median(offset))

    torch.manual_seed(expected["estimator"]["ks_torch_seed"])
    draws = posterior.sample((expected["estimator"]["ks_draws"],), x=observed, show_progress_bars=False)
    draws = draws.detach().cpu().numpy().astype(np.float64)
    reference = np.load(FIX / "training/posterior_FINAL.npy", allow_pickle=False).astype(np.float64)
    tests = [ks_2samp(draws[:, k], reference[:, k]) for k in range(reference.shape[1])]
    statistic = [float(test.statistic) for test in tests]
    n, m = len(draws), len(reference)
    critical = float(kolmogi(TOLERANCE["ks_family_alpha"] / reference.shape[1]) * math.sqrt((n + m) / (n * m)))

    history = json.loads((FIX / "training/history.json").read_text())
    meta = json.loads((FIX / "training/astro_meta.json").read_text())
    train = argument_defaults(ROOT / "inference/tsnpe/train.py",
                              {"--pilot", "--n0", "--n", "--max-rounds", "--hidden-features", "--transforms"})
    mis = argument_defaults(ROOT / "inference/tsnpe/run_mis.py", {"--flow-draws", "--broad-draws", "--mild-draws"})
    simulations = [int(entry["nsim"]) for entry in history]
    coupling = [module for module in estimator.modules()
                if type(module).__name__ == "PiecewiseRationalQuadraticCouplingTransform"]
    hidden = next(module for module in estimator.modules() if type(module).__name__ == "ResidualNet").initial_layer
    counts = expected["mis"]["component_counts"]
    settings = {
        "rounds_and_simulations": simulations == [train["--n0"]] + [train["--n"]] * (train["--max-rounds"] - 1),
        "pilot_and_total": meta["nsim_total"] - sum(simulations) == train["--pilot"],
        "flow_size": len(coupling) == train["--transforms"] and hidden.out_features == train["--hidden-features"],
        "mixture_counts": (counts["flow"], counts["broad_truncated_gaussian"], counts["mild_truncated_gaussian"])
        == (mis["--flow-draws"], mis["--broad-draws"], mis["--mild-draws"]),
    }
    record = {
        "finite_log_density": bool(np.isfinite(log_flow).all()),
        "normalization_constant": constant,
        "row_to_row_difference": float(np.max(np.abs(offset - constant))),
        "ks_statistic": statistic, "ks_pvalue": [float(test.pvalue) for test in tests], "ks_critical": critical,
        "training_record_matches_defaults": settings,
        "flow_size": {"transforms": len(coupling), "hidden_features": hidden.out_features,
                      "bins": int(coupling[0].num_bins)},
    }
    record["pass"] = (record["finite_log_density"] and max(statistic) <= critical
                      and abs(constant) <= TOLERANCE["flow_normalization"]
                      and record["row_to_row_difference"] <= TOLERANCE["flow_row_to_row"]
                      and all(settings.values()))
    return record


def check_likelihood(archive: dict, rows: np.ndarray, invalid: np.ndarray) -> dict:
    """Step 4: the rebuilt A1 target reproduces log L on the replay rows and rejects the invalid rows."""
    problem = A1Problem.from_data_root(OBSERVATIONS, source_data_root=OBSERVATIONS, workers=1, verify_data=True,
                                       source_scenario="A1", model="ddb")
    values = problem.evaluate_batch(np.vstack([archive["theta"][rows], archive["theta"][invalid]]))
    count = len(rows)
    difference = np.abs(values[:count] - archive["logL"][rows])
    record = {"replayed_rows": count, "rows": rows.tolist(), "largest_difference": float(difference.max()),
              "differences": difference.tolist(), "invalid_rows": int(len(invalid)),
              "invalid_values": values[count:].tolist(),
              "invalid_rows_rejected": bool(np.all(values[count:] <= INVALID))}
    record["pass"] = record["largest_difference"] <= TOLERANCE["log_likelihood"] and record["invalid_rows_rejected"]
    return record


def check_route1(archive: dict, expected: dict) -> dict:
    """Step 5: the Route 1 compact posterior is the positive-weight part of the archive."""
    keep = archive["w"] > 0.0
    weight = archive["w"][keep] / archive["w"][keep].sum()
    with np.load(ROOT / expected["route1"]["file"], allow_pickle=False) as compact:
        shipped_theta, shipped_weight = compact["theta_tsnpe_mis"], compact["weight_tsnpe_mis"]
    record = {"positive_rows": int(keep.sum()), "positive_rows_expected": int(keep.sum()) == expected["route1"]["positive_rows"],
              "theta_identical": bool(np.array_equal(shipped_theta, archive["theta"][keep])),
              "weights_identical": bool(np.array_equal(shipped_weight, weight))}
    record["pass"] = all(value for value in record.values() if isinstance(value, bool))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    parser.add_argument("--report", type=Path, default=ROOT / "build/route2/tsnpe_nucleonic_check.json")
    arguments = parser.parse_args()
    started = time.time()
    prepare_observations()
    expected = json.loads((FIX / "expected.json").read_text())
    paths = {name: fetch(name) for name in (ARCHIVE, ESTIMATOR)}  # each verified against its SHA-256
    with np.load(paths[ARCHIVE], allow_pickle=False) as data:
        archive = {key: data[key] for key in data.files}
    rows, invalid = replay_rows(archive, expected)
    steps = {}

    steps["code_identity"] = check_code(expected)
    print(f"[tsnpe-nucleonic] code identity      {'PASS' if steps['code_identity']['pass'] else 'FAIL'}  "
          f"{steps['code_identity']['files']} files of inference/tsnpe byte-identical to the stored reference", flush=True)

    steps["mis_arithmetic"] = mis = check_mis(archive, expected)
    print(f"[tsnpe-nucleonic] MIS arithmetic     {'PASS' if mis['pass'] else 'FAIL'}  ESS {mis['ess']:.6f}, "
          f"log Z {mis['log_evidence']:.8f} +/- {mis['log_evidence_error']:.8f} (largest difference "
          f"{max(mis['differences'].values()):.0e})", flush=True)

    steps["proposal_replay"] = proposal = check_proposal(archive, expected)
    print(f"[tsnpe-nucleonic] proposal replay    {'PASS' if proposal['pass'] else 'FAIL'}  broad and mild Gaussians "
          f"rebuilt from the stored rows; densities and 45,000 draws largest difference "
          f"{max(proposal['differences'].values()):.0e}; box shares {proposal['normalizations']['broad']}, "
          f"{proposal['normalizations']['mild']}", flush=True)

    steps["frozen_estimator"] = flow = check_estimator(paths[ESTIMATOR], archive, rows, expected, arguments.device)
    print(f"[tsnpe-nucleonic] frozen estimator   {'PASS' if flow['pass'] else 'FAIL'}  loads on {arguments.device}; "
          f"flow density at {len(rows)} rows = stored + {flow['normalization_constant']:.4f} (row to row "
          f"{flow['row_to_row_difference']:.1e}); KS vs posterior_FINAL max D {max(flow['ks_statistic']):.4f} "
          f"(limit {flow['ks_critical']:.4f}); training settings = code defaults: "
          f"{all(flow['training_record_matches_defaults'].values())}", flush=True)

    print(f"[tsnpe-nucleonic] rebuilding the exact A1 target from {OBSERVATIONS.relative_to(ROOT)} ...", flush=True)
    steps["exact_likelihood"] = likelihood = check_likelihood(archive, rows, invalid)
    print(f"[tsnpe-nucleonic] exact log L        {'PASS' if likelihood['pass'] else 'FAIL'}  "
          f"{likelihood['replayed_rows']} rows, largest difference {likelihood['largest_difference']:.1e} "
          f"(tolerance {TOLERANCE['log_likelihood']:.0e}); {likelihood['invalid_rows']} rows without a weight "
          f"{'rejected' if likelihood['invalid_rows_rejected'] else 'NOT rejected'} by the target", flush=True)

    steps["route1_identity"] = route1 = check_route1(archive, expected)
    print(f"[tsnpe-nucleonic] Route 1 identity   {'PASS' if route1['pass'] else 'FAIL'}  theta and weights of "
          f"{route1['positive_rows']} positive rows {'bit-identical' if route1['pass'] else 'DIFFERENT'} to "
          f"{expected['route1']['file']}", flush=True)

    ok = all(step["pass"] for step in steps.values())
    summary = {"status": "PASS" if ok else "FAIL", "device": arguments.device, "torch": torch.__version__,
               "sbi": sbi.__version__, "tolerances": TOLERANCE, "seconds": round(time.time() - started, 1),
               "steps": steps}
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[tsnpe-nucleonic] {summary['status']}: MIS weights, ESS and log Z, the proposal, the frozen flow, exact "
          f"log L and the Route 1 posterior replayed in {summary['seconds']} s on {arguments.device}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
