#!/usr/bin/env python3
"""Route 2 check: hyperonic A-NET posteriors and evidences (A1 and the seven queries), replayed.

For each configuration (A1; K0 = 200, 260 MeV; Jsym = 29, 36 MeV; J0614, J1231, J1614 substituted):

1. Certificate. The importance weights, ESS, largest weight and log Z are recomputed from the saved proposal
   densities (log q) and exact likelihoods (log L) with the certification formula of the pipeline and
   compared with the stored certificate (tolerance 1e-9).
2. Resample. The 6,000-draw posterior sample is redrawn from the recomputed weights with its stored seed;
   it must be identical to the stored sample.
3. Proposal. The conditioning clouds are redrawn from the NICER data (seed 777001), and the component labels
   and the first Student draws are redrawn from the stored proposal seeds; all must be identical (Student
   draws within 1e-10). For 14 saved rows per configuration (the 6 largest weights, 6 random valid rows and
   2 random rejected rows) the proposal density is recomputed from the frozen networks: both A-NET heads
   (1,024 Heun steps) and the Student mixture. Tolerance 2e-3 in the mixture log q, which is what enters the
   weights: below the change of the density between 1,024 and 2,048 steps (95th percentile 3.6e-3 and
   4.3e-3) and well below the log Z uncertainty (about 0.012). Each head's own difference is gated after
   weighting by its share of the mixture density at that row (its first-order effect on log q); the raw
   differences are reported too. They grow only far in a head's tail (log q near -2000, share exp(-1000)),
   where float32 accumulation over 1,024 steps leaves relative errors of about 1e-5.
4. Likelihood. The exact target is rebuilt from data/observations, and log L of the same 14 rows is
   recomputed with the production EOS/TOV evaluator (tolerance 1e-4, the certification gate of the
   pipeline; accepted and rejected rows must agree).

The A-NET head draws use the GPU random-number generator, so they are not replayed bit for bit. The
importance weights depend on them only through log q and log L at the saved points, which are replayed.

Run from the repository root:  PYTHONPATH=$PWD python -m route2.checks.hyperonic_anet [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

# Repository packages are imported before the pipeline modules (which add their own paths).
from inference.anet.proposal import mass_radius_clouds
from workflows.a1_problem import A1Problem
from workflows.nuclear_scenarios import nuclear_observation

from route2.common import OBSERVATIONS, ROOT, prepare_observations, sha256, use_hyperonic_pipeline

use_hyperonic_pipeline()

import joblib  # noqa: E402

import fast_portable_hyperonic_likelihood as evaluator  # noqa: E402
from clean_student_t_ensemble import checkpoint_density, sample_checkpoint  # noqa: E402
from sample_clean_hyperonic_parity_mixture import ParityFMPE  # noqa: E402

FIX = ROOT / "route2/fixtures/hyperonic_anet"
CASES = ("A1", "K0_200", "K0_260", "Jsym_29", "Jsym_36", "J0614", "J1231", "J1614")
SOURCE_CASES = ("A1", "J0614", "J1231", "J1614")
FLOW_STEPS = 1024
CLOUD_SEED = 777_001
STUDENT = "student_beta1_48c_df5_s075_seed20260990.joblib"
TOLERANCE = {"certificate": 1.0e-9, "student_draw": 1.0e-10, "student_logq": 1.0e-10, "head_logq": 2.0e-3,
             "log_likelihood": 1.0e-4}


def certify(logq: np.ndarray, logl: np.ndarray, lower: np.ndarray, upper: np.ndarray, combined: bool):
    """The certification formula of certify_clean_anet_scenario(_batches).py."""
    rows = len(logq)
    valid = logl > -1.0e50
    logprior = -float(np.log(upper - lower).sum())
    logweight = np.full(rows, -np.inf)
    logweight[valid] = logprior + logl[valid] - logq[valid]
    if combined:  # the batch certifier uses scipy; the single-batch certifier its own two-line version
        normalizer = float(logsumexp(logweight[valid]))
    else:
        largest = float(np.max(logweight[valid]))
        normalizer = largest + float(np.log(np.exp(logweight[valid] - largest).sum()))
    weight = np.zeros(rows)
    weight[valid] = np.exp(logweight[valid] - normalizer)
    ess = float(1.0 / np.sum(weight * weight))
    return weight, {"valid_rows": int(valid.sum()), "log_evidence": normalizer - math.log(rows),
                    "log_evidence_standard_error": math.sqrt(max(rows / ess - 1.0, 0.0) / rows),
                    "posterior_ess": ess, "maximum_normalized_weight": float(weight.max())}


def load_case_arrays(case: str, expected: dict, shifts) -> dict:
    if case in SOURCE_CASES:
        path = ROOT / expected["certificate_in"]
        if sha256(path) != expected["certificate_sha256"]:
            raise SystemExit(f"FAIL: {path.relative_to(ROOT)} is not the stored certificate (SHA-256 mismatch)")
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in ("theta", "logq", "logl", "normalized_weight", "prior_low", "prior_high")}
        resample_path = ROOT / expected["resample"]["in"]
        if sha256(resample_path) != expected["resample"]["sha256"]:
            raise SystemExit(f"FAIL: {resample_path.relative_to(ROOT)} is not the stored resample (SHA-256 mismatch)")
        with np.load(resample_path, allow_pickle=False) as data:
            arrays["resample_source_index"] = data["source_index"]
            arrays["resample_counts"] = data["counts"]
            arrays["resample_theta"] = data["theta"]
        return arrays
    return {key: shifts[f"{case}__{key}"] for key in ("logq", "logl", "prior_low", "prior_high",
                                                      "resample_source_index", "resample_counts")}


def check_statistics(case, expected, arrays, report) -> np.ndarray:
    weight, stats = certify(arrays["logq"], arrays["logl"], arrays["prior_low"], arrays["prior_high"],
                            combined=len(expected["batches"]) > 1)
    diffs = {key: abs(stats[key] - expected[key]) / (1.0 if key in ("log_evidence", "log_evidence_standard_error")
                                                     else max(abs(expected[key]), 1.0))
             for key in ("log_evidence", "log_evidence_standard_error", "posterior_ess", "maximum_normalized_weight")}
    ok = stats["valid_rows"] == expected["valid_rows"] and max(diffs.values()) <= TOLERANCE["certificate"]
    if "normalized_weight" in arrays:
        diffs["normalized_weight_max_abs"] = float(np.max(np.abs(weight - arrays["normalized_weight"])))
        ok = ok and diffs["normalized_weight_max_abs"] <= 1.0e-15
    report["certificate"] = {"pass": ok, "recomputed": stats, "largest_difference": max(diffs.values()), "differences": diffs}

    rng = np.random.default_rng(expected["resample"]["seed"])
    sampled = rng.choice(len(weight), size=expected["resample"]["draws"], replace=True, p=weight)
    source_index, counts = np.unique(sampled, return_counts=True)
    same = (np.array_equal(source_index, arrays["resample_source_index"])
            and np.array_equal(counts, arrays["resample_counts"]))
    if "resample_theta" in arrays:
        same = same and np.array_equal(arrays["theta"][source_index], arrays["resample_theta"])
    report["resample"] = {"pass": bool(same), "unique_rows": int(len(source_index)), "seed": expected["resample"]["seed"]}
    return weight


def check_proposal(case, expected, arrays, fixture, heads, student, device, report) -> None:
    prefix = f"{case}__"
    source = expected["source_scenario"]
    cloud = mass_radius_clouds(OBSERVATIONS, points=heads[9].cloud_points, seed=CLOUD_SEED, source_scenario=source,
                               source_data_root=OBSERVATIONS, verify_data=True)
    observation = nuclear_observation(expected["nuclear_scenario"])
    cloud_ok = np.array_equal(cloud, fixture[prefix + "cloud"])
    observation_ok = np.array_equal(observation, fixture[prefix + "nuclear_observation"])

    weights = np.asarray(expected["component_weights"], dtype=np.float64)
    labels_ok, draw_diff = True, 0.0
    for number, batch in enumerate(expected["batches"]):
        saved = fixture[f"{prefix}b{number}__component_all"]
        labels = np.random.default_rng(batch["proposal_seed"]).choice(3, size=batch["rows"], p=weights).astype(np.int8)
        labels_ok = labels_ok and np.array_equal(labels, saved)
        draws = sample_checkpoint(student, int(np.sum(saved == 2)), np.random.default_rng(batch["proposal_seed"] + 12))
        first = fixture[f"{prefix}b{number}__student_first_theta"]
        draw_diff = max(draw_diff, float(np.max(np.abs(draws[: len(first)] - first))))

    rows = fixture[prefix + "rows"]
    theta = fixture[prefix + "theta"]
    certificate_theta = arrays.get("theta")
    rows_ok = certificate_theta is None or np.array_equal(certificate_theta[rows], theta)
    with contextlib.redirect_stdout(io.StringIO()):
        head9 = heads[9].log_density(theta, cloud, observation, len(theta))
        head7 = heads[7].log_density(theta, cloud, observation, len(theta))
    prior_logq = -float(np.log(heads[9].prior_high - heads[9].prior_low).sum())
    student_logq = checkpoint_density(student, theta, prior_logq)
    log_weights = np.log(weights)[:, None]
    mixture = logsumexp(log_weights + np.vstack([head9, head7, student_logq]), axis=0)
    # A head's error changes log q (and hence the weight) in proportion to the head's share of the mixture
    # density at that row. Far in a head's tail (share ~ exp(-1000)) its raw float32 error is irrelevant.
    saved_terms = log_weights + np.vstack([fixture[prefix + "head9_logq"], fixture[prefix + "head7_logq"],
                                           fixture[prefix + "student_logq"]])
    share = np.exp(saved_terms - logsumexp(saved_terms, axis=0))
    raw9 = np.abs(head9 - fixture[prefix + "head9_logq"])
    raw7 = np.abs(head7 - fixture[prefix + "head7_logq"])
    differences = {
        "mixture_logq_vs_certificate": float(np.max(np.abs(mixture - arrays["logq"][rows]))),
        "head9_logq_share_weighted": float(np.max(share[0] * raw9)),
        "head7_logq_share_weighted": float(np.max(share[1] * raw7)),
        "head9_logq_raw": float(np.max(raw9)),
        "head7_logq_raw": float(np.max(raw7)),
        "student_logq": float(np.max(np.abs(student_logq - fixture[prefix + "student_logq"]))),
        "student_draws": draw_diff,
    }
    ok = (cloud_ok and observation_ok and labels_ok and rows_ok
          and differences["student_draws"] <= TOLERANCE["student_draw"]
          and differences["student_logq"] <= TOLERANCE["student_logq"]
          and max(differences["mixture_logq_vs_certificate"], differences["head9_logq_share_weighted"],
                  differences["head7_logq_share_weighted"]) <= TOLERANCE["head_logq"])
    report["proposal"] = {"pass": bool(ok), "cloud_identical": bool(cloud_ok), "nuclear_observation_identical": bool(observation_ok),
                          "component_labels_identical": bool(labels_ok), "rows_match_certificate": bool(rows_ok),
                          "replayed_rows": int(len(rows)), "largest_differences": differences}


def check_likelihood(case, expected, arrays, fixture, targets, report) -> None:
    prefix = f"{case}__"
    rows = fixture[prefix + "rows"]
    theta = fixture[prefix + "theta"]
    target = targets[expected["source_scenario"]]
    if expected["nuclear_scenario"] != "A1":  # as build_portable_hyperonic_target_cache.py --base-cache
        target = dataclasses.replace(target, nuclear_observation=nuclear_observation(expected["nuclear_scenario"]))
    properties = evaluator.nuclear_observables(theta)
    replayed = np.full(len(theta), -1.0e100)
    for index, parameters in enumerate(theta):
        forward = evaluator._fast_forward(parameters)
        if forward is None:
            continue
        terms = target.evaluate(parameters, properties[index], forward.density, forward.energy, forward.pressure, forward)
        if np.isfinite(terms.total):
            replayed[index] = terms.total
    saved = arrays["logl"][rows]
    valid = saved > -1.0e50
    same_pattern = bool(np.array_equal(valid, replayed > -1.0e50))
    largest = float(np.max(np.abs(replayed[valid] - saved[valid]))) if same_pattern else float("inf")
    report["likelihood"] = {"pass": same_pattern and largest <= TOLERANCE["log_likelihood"],
                            "replayed_rows": int(len(rows)), "accepted_rows": int(valid.sum()),
                            "accepted_rejected_pattern_identical": same_pattern, "largest_difference": largest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    parser.add_argument("--report", type=Path, default=ROOT / "build/route2/hyperonic_anet_check.json")
    arguments = parser.parse_args()
    started = time.time()
    prepare_observations()
    expected = json.loads((FIX / "expected.json").read_text())["cases"]
    with np.load(FIX / "replay_rows.npz", allow_pickle=False) as data:
        fixture = {key: data[key] for key in data.files}
    with np.load(FIX / "nuclear_shift_certificates.npz", allow_pickle=False) as data:
        shifts = {key: data[key] for key in data.files}

    heads = {}
    for dimension in (9, 7):
        folder = FIX / f"networks/head{dimension}"
        training = json.loads((folder / "run_report.json").read_text())
        if (training["checkpoint_sha256"] != sha256(folder / "anet_net.pt")
                or training["standardization_sha256"] != sha256(folder / "anet_std.npz")):
            raise SystemExit(f"FAIL: the frozen head{dimension} files differ from their training report")
        heads[dimension] = ParityFMPE(folder / "anet_net.pt", folder / "anet_std.npz", device=arguments.device,
                                      steps=FLOW_STEPS)
    student = joblib.load(FIX / "networks/student" / STUDENT)

    print(f"[hyperonic-anet] rebuilding the exact targets from {OBSERVATIONS.relative_to(ROOT)} ...", flush=True)
    targets = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for source in SOURCE_CASES:
            targets[source] = A1Problem.from_data_root(OBSERVATIONS, source_data_root=OBSERVATIONS, workers=1,
                                                       verify_data=True, nuclear_scenario="A1",
                                                       source_scenario=source, model="ddb-hyperonic").target

    reports = {}
    for case in CASES:
        case_started = time.time()
        report: dict = {}
        arrays = load_case_arrays(case, expected[case], shifts)
        check_statistics(case, expected[case], arrays, report)
        check_proposal(case, expected[case], arrays, fixture, heads, student, arguments.device, report)
        check_likelihood(case, expected[case], arrays, fixture, targets, report)
        report["pass"] = all(report[step]["pass"] for step in ("certificate", "resample", "proposal", "likelihood"))
        reports[case] = report
        print(f"[hyperonic-anet] {case:8s} {'PASS' if report['pass'] else 'FAIL'}  "
              f"log Z {report['certificate']['recomputed']['log_evidence']:.4f} "
              f"(diff {report['certificate']['largest_difference']:.0e}), resample "
              f"{'identical' if report['resample']['pass'] else 'DIFFERENT'}, log q diff "
              f"{report['proposal']['largest_differences']['mixture_logq_vs_certificate']:.0e}, "
              f"log L diff {report['likelihood']['largest_difference']:.0e}  ({time.time() - case_started:.0f} s)", flush=True)

    ok = all(report["pass"] for report in reports.values())
    summary = {"status": "PASS" if ok else "FAIL", "device": arguments.device,
               "gpu": torch.cuda.get_device_name(arguments.device) if arguments.device.startswith("cuda") else None,
               "torch": torch.__version__, "tolerances": TOLERANCE, "seconds": round(time.time() - started, 1),
               "cases": reports}
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[hyperonic-anet] {summary['status']}: 8 configurations; certificates, resamples, proposal densities and "
          f"exact likelihoods replayed in {summary['seconds']} s on {arguments.device} "
          f"(report: {arguments.report.relative_to(ROOT) if arguments.report.is_relative_to(ROOT) else arguments.report})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
