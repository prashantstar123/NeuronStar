#!/usr/bin/env python3
"""Route 2 check: UltraNest, both sectors. Evidences of Table VI, corrected posterior weights, exact likelihoods.

Inputs: the stored UltraNest records in route2/fixtures/ultranest/records (evidence_aggregation.json with the
combination rule, run_values.json with the values of all accepted runs, and run_reports/ with the correction
reports of the four hyperonic desktop runs), the posterior and single-weight-term files (release files listed in
route2/downloads.d/ultranest.json) and the Route 1 results in results/. Nothing is sampled again.

A. Table VI, UltraNest column (arithmetic from the stored records).
   Each accepted run is stored with its raw UltraNest log Z +/- error and its NICER single-weight correction
   (a log Z shift and its Monte Carlo error). The corrected per-run value is raw + shift, with the two errors
   added in quadrature. The runs of one configuration are combined with the rule of evidence_aggregation.json:
   the inverse-variance mean, with the formal error multiplied by max(1, sqrt(Q/(n-1))). This rebuilds all 27
   run values and 16 combinations of run_values.json (the UltraNest columns of
   results/evidence/green_en_vs_ultranest_16_rows.json). For the hyperonic K0 and Jsym rows, Table VI uses
   independent seeds only: the four desktop runs share part of their initial random ancestry with a cluster run
   and are left out (results/evidence/table_VI_independent_ultranest_rows.json). The resulting column and the
   separations are compared with Table VI of paper/main.tex (the paper does not mark the seed counts; they are
   checked against the stored records). Where a stored posterior file or report carries the raw log Z of a run
   (J1614 posteriors, hyperonic equal-weight posteriors, the correction reports), it must equal the stored
   record exactly. Tolerance 5e-13 in log Z: float64 rounding only; printed Table VI strings must be identical.

B. Corrected posterior weights and correction terms.
   * A1: log w = log(UltraNest weight) + fixed_0740 - old_0740, normalized; the weights and parameters must
     equal results/nucleonic/a1_parameter_posteriors.npz exactly.
   * The same correction (for J1231 also the J1231 term) for J0614, J1231 and the Jsym=29/36 seeds: the log Z
     shift log(sum w exp(delta)) and its Monte Carlo error sqrt(CV^2/ESS) must equal the stored per-run values.
     Tolerances: 2e-12 for the shift (float64 summation over about 15,000 rows, n*eps = 1.7e-12); 1e-13 for the
     error (the stored value was computed as sqrt(sigma_c^2 - sigma_raw^2), which carries about 2e-14 rounding).
   * Jsym=29/36: per-seed and pooled corrected ESS equal results/nucleonic/nuclear_amortization_verification.json
     (relative 1e-11, float64 summation), and the 8,000-draw curve selection redrawn with the stored seed has
     the stored SHA-256 (exact).
   * J1614 (both sectors): the sampled target is the paper's target (no correction), so the weights are equal;
     the shipped curve files must hold exactly the stored samples with weight 1/n.
   * Hyperonic desktop runs: shift, CV^2, error and ESS values of the stored correction reports.
   * Hyperonic Table VI runs: the equal-weight posterior files carry the raw log Z of the stored records, the
     observation and the prior exactly; their rows share exactly 71/80/107/78 distinct 9D rows with the
     desktop runs, the evidence of shared random ancestry behind the independent-seed rule.

C. Exact likelihood of 4 posterior rows per run with the portable targets (as in nucleonic_anet/hyperonic_anet).
   Every row must have a finite exact log L. The UltraNest runs sampled the double-weight NICER convention; the
   stored per-row NICER terms (J0740 and J1231, single-weight "fixed" and double-weight "old") are recomputed
   from the row's mass-radius curve with the portable grid and the legacy grid and must agree within 1e-4 (the
   certification gate of the pipeline). Maximum mass, R(1.4), R(M_max) and the radius curve must agree with the
   stored terms or curve files within 1e-6 (Msun or km; the maximum-mass factor has slope <= 20 per Msun, so
   1e-6 Msun moves log L by less than 2e-5). The stored files hold no per-row UltraNest log L.

D. Driver smoke (skip with --skip-driver-smoke; about 2.2 minutes on 3 cores; A-C take under a minute).
   * inference/ultranest/run.py --smoke: the portable nucleonic driver checks its target on the 1,870
     certificate rows (its own gate, 1e-4) and writes a prior-probe posterior. No nested sampling is run: the
     driver has no iteration cap and a converged 50-live-point run needs 1e4-1e5 likelihood calls.
   * hyperonic_pipeline/ddb_ultranest_hyp.py --cert: the hyperonic UltraNest driver evaluates its own
     (double-weight convention) target on the C rows of each nuclear shift; it must equal the portable log L
     minus (fixed_0740 - old_0740) within its own gate (1e-4). ddb_ultranest_hyp_j1614.py --cert must equal the
     portable J1614 log L (no correction: J0740 is replaced and the remaining NICER files carry no weights).

Run from the repository root:  PYTHONPATH=$PWD python -m route2.checks.ultranest [--skip-driver-smoke]
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_variable, "3")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from route2.common import OBSERVATIONS, ROOT, prepare_observations, sha256, use_hyperonic_pipeline  # noqa: E402
from route2.fetch import catalogue, fetch  # noqa: E402

# Repository packages are imported before the pipeline modules (which add their own paths).
from eos import get_eos  # noqa: E402
from likelihoods.nicer.single_weight import log_likelihood_one  # noqa: E402
from likelihoods.nuclear import A1_SIGMA  # noqa: E402
from tov import solve_stable_branch  # noqa: E402
from workflows.a1_problem import A1Problem  # noqa: E402
from workflows.nuclear_scenarios import nuclear_observation  # noqa: E402
from workflows.source_scenarios import BASE_NICER_SOURCES, SOURCE_SUBSTITUTIONS  # noqa: E402

use_hyperonic_pipeline()

import fast_portable_hyperonic_likelihood as evaluator  # noqa: E402  (imports the hyperonic UltraNest driver)
import nicer_like as legacy_nicer  # noqa: E402  (NICER module of the UltraNest runs; double-weight convention)

FIX = ROOT / "route2/fixtures/ultranest"
TOLERANCE = {"log_z": 5.0e-13, "shift": 2.0e-12, "mc_error": 1.0e-13, "cv2": 1.0e-12, "ess_relative": 1.0e-11,
             "log_likelihood": 1.0e-4, "mass_msun": 1.0e-6, "radius_km": 1.0e-6}
TEX_LABEL = {"A1": "A1 base", "K0_200": "K_0=200", "K0_260": "K_0=260", "Jsym_29": "J_{\\mathrm{sym}}=29",
             "Jsym_36": "J_{\\mathrm{sym}}=36", "J0614": "J0614", "J1231": "J1231", "J1614": "J1614"}
GRID = dict(nM=150, nR=150, max_rows=20_000, bw=0.08)  # NICER grid options of the target and of the drivers
SHIFT_ORDER = ["K0_200", "K0_260", "Jsym_29", "Jsym_36"]  # seeds of the curve selections: 2026082701 + 100 * index
CURVE_DRAWS = 8000
LINES: list[str] = []


def say(text: str) -> None:
    print(f"[ultranest] {text}", flush=True)
    LINES.append(text)


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def row_view(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=np.float64)
    return value.view(np.dtype((np.void, value.dtype.itemsize * value.shape[1]))).ravel()


# ------------------------------------------------------------------------------------------------ A. Table VI
def combine(pairs: list[tuple[float, float]]) -> dict:
    """The combination rule of evidence_aggregation.json (same operation order as the code that wrote it):
    inverse-variance mean; the formal error is multiplied by max(1, sqrt(Q/(n-1))) for n > 1; a single run is
    unchanged."""
    raw = [1.0 / sigma ** 2 for _, sigma in pairs]
    total = sum(raw)
    normalized = [weight / total for weight in raw]
    centre = sum(weight * logz for weight, (logz, _) in zip(normalized, pairs))
    formal = 1.0 / math.sqrt(total)
    q_value = sum(weight * (logz - centre) ** 2 for weight, (logz, _) in zip(raw, pairs))
    dof = len(pairs) - 1
    ratio = max(1.0, math.sqrt(q_value / dof)) if dof else 1.0
    return {"n": len(pairs), "logz": centre, "formal_sigma": formal, "Q": q_value, "dof": dof,
            "birge_ratio": ratio, "sigma": formal * ratio}


def tex_table_vi() -> list[dict]:
    """The 16 data rows of Table VI in paper/main.tex: UltraNest log Z, error, seed-count marker, separation."""
    tex = (ROOT / "paper/main.tex").read_text()
    start = tex.index("\\label{tab:en-ultranest}")
    body = tex[start:tex.index("\\end{tabular*}", start)]
    rows = []
    for line in body.splitlines():
        cells = line.split("&")
        if len(cells) != 4 or "\\pm" not in line:
            continue
        match = re.search(r"\$(-?\d+\.\d+) \\pm (\d+\.\d+)\$(?:\\textsuperscript\{([ab])\})?", cells[2])
        separation = re.search(r"\$(\d+\.\d+)\$", cells[3])
        rows.append({"label": cells[0].strip(), "logz": match.group(1), "sigma": match.group(2),
                     "marker": match.group(3) or "", "separation": separation.group(1)})
    return rows


def stored_raw_values(expected: dict, files: dict) -> list[tuple[str, str, float, str]]:
    """Raw UltraNest log Z +/- error of the runs whose stored posterior file or report carries them."""
    values = []
    j1614 = expected["nucleonic"]["J1614"]
    saved = load(files[j1614["posterior"]])
    values += [(j1614["run_id"], "raw_results_logz", float(saved["logz"]), j1614["posterior"]),
               (j1614["run_id"], "raw_results_sigma", float(saved["logzerr"]), j1614["posterior"])]
    for record in expected["hyperonic"]["cluster"].values():
        equal = load(files[record["equal_weight"]])
        values += [(record["id"], "raw_results_logz", float(equal["logZ"]), record["equal_weight"]),
                   (record["id"], "raw_results_sigma", float(equal["logZerr"]), record["equal_weight"])]
    for record in expected["hyperonic"]["desktop_b"].values():
        report = json.loads((ROOT / record["report"]).read_text())
        values += [(record["id"], "raw_results_logz", report["raw_logZ"], record["report"]),
                   (record["id"], "raw_results_sigma", report["raw_logZ_error"], record["report"])]
    hyperonic_j1614 = expected["hyperonic"]["J1614"]
    for name, run_id in (("replica_1", "hyp_J1614_replica_1"), ("replica_2", "hyp_J1614_replica_2")):
        replica = hyperonic_j1614["replicas"][name]
        values += [(run_id, "raw_results_logz", replica["logz"], hyperonic_j1614["label"]),
                   (run_id, "raw_results_sigma", replica["logzerr"], hyperonic_j1614["label"])]
    return values


def check_table_vi(expected: dict, raw_values: list) -> dict:
    records = expected["records"]
    hashes = {name: sha256(ROOT / records[name]["file"]) for name in ("aggregation", "runs")}
    records_ok = all(hashes[name] == records[name]["sha256"] for name in hashes)
    aggregation = json.loads((ROOT / records["aggregation"]["file"]).read_text())
    run_values = json.loads((ROOT / records["runs"]["file"]).read_text())
    sixteen = {(r["sector"], r["configuration"]): r
               for r in json.loads((ROOT / expected["sixteen_row_comparison"]["file"]).read_text())["rows"]}
    rows_path = ROOT / expected["table_vi_rows"]["file"]
    records_ok = records_ok and sha256(rows_path) == expected["table_vi_rows"]["sha256"]
    independent = json.loads(rows_path.read_text())["rows"]
    excluded = set(expected["table_vi_rows"]["excluded_constituents"])
    say(f"A stored UltraNest records and the independent-seed Table VI rows carry their expected SHA-256: {verdict(records_ok)}")

    # A1. Per-run corrected log Z (27 runs) and raw values carried by the stored posterior files and reports.
    runs, worst_run, aggregation_same = {}, 0.0, True
    aggregation_runs = {c["id"]: c for row in aggregation["rows"] for c in row["constituents"]}
    for row in run_values["rows"]:
        for c in row["constituents"]:
            logz = c["raw_results_logz"] + c["nicer_weight_correction"]
            sigma = math.sqrt(c["raw_results_sigma"] ** 2 + c["nicer_weight_correction_mc_sigma"] ** 2)
            worst_run = max(worst_run, abs(logz - c["corrected_logz"]), abs(sigma - c["corrected_sigma"]))
            runs[c["id"]] = (logz, sigma)
            if c["id"] in aggregation_runs:
                aggregation_same = aggregation_same and all(aggregation_runs[c["id"]][k] == c[k] for k in (
                    "raw_results_logz", "raw_results_sigma", "nicer_weight_correction", "corrected_logz", "corrected_sigma"))
    stored_runs = {c["id"]: c for row in run_values["rows"] for c in row["constituents"]}
    raw_ok = all(stored_runs[run_id][field] == value for run_id, field, value, _ in raw_values)
    run_ok = worst_run <= TOLERANCE["log_z"] and len(runs) == 27 and aggregation_same and raw_ok
    say(f"A 27 corrected per-run log Z = raw + NICER shift (errors in quadrature): largest difference {worst_run:.1e}; "
        f"{len(raw_values) // 2} raw log Z +/- error equal those in the stored posterior files and reports: "
        f"{verdict(raw_ok)}; runs of evidence_aggregation.json identical in run_values.json: {verdict(aggregation_same)}  "
        f"{verdict(run_ok)}")

    # A2. Combinations: all accepted runs (run_values.json, 16-row file) and independent seeds only (Table VI).
    tex_rows = tex_table_vi()
    rows_report, worst, printed_ok, structure_ok = {}, 0.0, True, len(tex_rows) == 16
    for index, row in enumerate(run_values["rows"]):
        key = (row["sector"], row["configuration"])
        ids = [c["id"] for c in row["constituents"]]
        full = combine([runs[i] for i in ids])
        stored = row["aggregate"]
        sixteen_row = sixteen[key]
        diffs = [abs(full[k] - stored[k]) for k in ("logz", "formal_sigma", "Q", "birge_ratio", "sigma")]
        combined_full = math.hypot(sixteen_row["en_sigma"], full["sigma"])
        separation_full = abs(sixteen_row["en_log_evidence"] - full["logz"]) / combined_full
        counts_ok = full["n"] == stored["n"] and full["dof"] == stored["dof"]
        final, used = full, ids
        if row["sector"] == "hyperonic" and row["configuration"] in independent:
            used = [i for i in ids if i not in excluded]
            final = combine([runs[i] for i in used])
            u = independent[row["configuration"]]
            aggregation_row = next(r for r in aggregation["rows"] if (r["sector"], r["configuration"]) == key)["aggregate"]
            diffs += [abs(final["logz"] - u["ultranest_logz"]), abs(final["sigma"] - u["ultranest_sigma"]),
                      abs(final["logz"] - aggregation_row["logz"]), abs(final["sigma"] - aggregation_row["sigma"]),
                      abs(final["logz"] - row["aggregate_without_desktop_runs"]["logz"]),
                      abs(final["sigma"] - row["aggregate_without_desktop_runs"]["sigma"]),
                      abs(u["all_runs_logz"] - full["logz"]), abs(u["all_runs_sigma"] - full["sigma"]),
                      abs(u["all_runs_separation"] - separation_full)]
            counts_ok = counts_ok and sorted(used) == sorted(u["runs_combined"]) and len(used) < len(ids)
            counts_ok = counts_ok and u["en_logz"] == sixteen_row["en_log_evidence"] and u["en_sigma"] == sixteen_row["en_sigma"]
        else:
            counts_ok = counts_ok and not (set(ids) & excluded)
        # The 16-row file holds the printed Table VI values: independent runs for the four rows above, all runs otherwise.
        combined = math.hypot(sixteen_row["en_sigma"], final["sigma"])
        separation = abs(sixteen_row["en_log_evidence"] - final["logz"]) / combined
        diffs += [abs(final["logz"] - sixteen_row["ultranest_log_evidence"]), abs(final["sigma"] - sixteen_row["ultranest_sigma"]),
                  abs(combined - sixteen_row["combined_en_ultranest_sigma"]), abs(separation - sixteen_row["separation_sigma"])]
        if row["sector"] == "hyperonic" and row["configuration"] in independent:
            diffs.append(abs(separation - independent[row["configuration"]]["separation_full_precision"]))
        counts_ok = counts_ok and final["n"] == sixteen_row["ultranest_accepted_seed_count"]
        tex = tex_rows[index] if index < len(tex_rows) else {}
        printed = (TEX_LABEL[row["configuration"]] in tex.get("label", "") and tex.get("logz") == f"{final['logz']:.3f}"
                   and tex.get("sigma") == f"{final['sigma']:.3f}" and tex.get("separation") == f"{separation:.2f}")
        worst = max(worst, *diffs)
        ok = max(diffs) <= TOLERANCE["log_z"] and counts_ok and printed
        printed_ok, structure_ok = printed_ok and printed, structure_ok and counts_ok
        rows_report[f"{row['sector']}/{row['configuration']}"] = {
            "all_accepted_runs": {"runs": ids, **full}, "table_vi_runs": used, "table_vi_logz": final["logz"],
            "table_vi_sigma": final["sigma"], "table_vi_separation": separation, "largest_difference": max(diffs),
            "printed_row_identical": printed, "pass": ok}
        note = f" (Table VI: {len(used)} independent of {len(ids)})" if used != ids else ""
        say(f"A {row['sector']:9s} {row['configuration']:7s} log Z {final['logz']:.6f} +/- {final['sigma']:.6f} "
            f"(n={final['n']}){note}, separation {separation:.2f}; largest difference {max(diffs):.1e}; "
            f"printed row {'identical' if printed else 'DIFFERENT'}  {verdict(ok)}")
    ok = records_ok and run_ok and worst <= TOLERANCE["log_z"] and printed_ok and structure_ok
    return {"pass": ok, "largest_run_difference": worst_run, "largest_combination_difference": worst,
            "printed_table_vi_identical": printed_ok, "rows": rows_report}


# ------------------------------------------------------------------------------------------------ B. weights
def correction_statistics(weight: np.ndarray, delta: np.ndarray) -> dict:
    """NICER single-weight correction of one run: log Z shift and its Monte Carlo error (correction-report formulas)."""
    normalized = weight / weight.sum()
    ratio = np.exp(delta)
    mean = float(np.sum(normalized * ratio))
    cv2 = float(np.sum(normalized * (ratio / mean - 1.0) ** 2))
    source_ess = float(1.0 / np.sum(normalized ** 2))
    corrected = normalized * ratio
    corrected /= corrected.sum()
    return {"shift": float(logsumexp(np.log(normalized) + delta)), "cv2": cv2, "mc_error": math.sqrt(cv2 / source_ess),
            "source_ess": source_ess, "adjusted_ess": source_ess / (1.0 + cv2),
            "kish_ess": float(1.0 / np.sum(corrected ** 2))}


def relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0)


def check_weights(expected: dict, files: dict, stored_runs: dict) -> dict:
    reports = {}
    nucleonic = expected["nucleonic"]
    shipped = json.loads((ROOT / "results/nucleonic/nuclear_amortization_verification.json").read_text())["cases"]
    corrected_by_run = {}
    for run_id, record in nucleonic["runs"].items():
        posterior, terms = load(files[record["posterior"]]), load(files[record["terms"]])
        same_map = (np.array_equal(posterior["record_index"], terms["record_index"])
                    and str(terms["input_sha256"]) == sha256(files[record["posterior"]]))
        delta = sum(terms[f"fixed_{t}"] - terms[f"old_{t}"] for t in record["correction_terms"])
        stats = correction_statistics(np.asarray(posterior["posterior_weight"], dtype=np.float64), delta)
        stored = stored_runs[run_id]
        diffs = {"shift": abs(stats["shift"] - stored["nicer_weight_correction"]),
                 "mc_error": abs(stats["mc_error"] - stored["nicer_weight_correction_mc_sigma"]),
                 "source_ess": relative(stats["source_ess"], float(posterior["source_weight_ESS"]))}
        ok = (same_map and diffs["shift"] <= TOLERANCE["shift"] and diffs["mc_error"] <= TOLERANCE["mc_error"]
              and diffs["source_ess"] <= TOLERANCE["ess_relative"] and bool(terms["valid_curve"].all()))
        log_weight = np.log(posterior["posterior_weight"]) + delta
        corrected_by_run[run_id] = np.exp(log_weight - logsumexp(log_weight))
        extra = ""
        if run_id == "nuc_a1":  # the A1 posterior weights: log UltraNest weight + fixed_0740 - old_0740, normalized
            verifier = np.log(posterior["posterior_weight"]) + terms["fixed_0740"] - terms["old_0740"]
            weight = np.exp(verifier - logsumexp(verifier))
            compact_path = ROOT / "results/nucleonic/a1_parameter_posteriors.npz"
            compact = load(compact_path)
            summary = json.loads((ROOT / "results/nucleonic/a1_parameter_posteriors.json").read_text())
            normalized = weight / weight.sum()
            ess = float(1.0 / np.sum(normalized * normalized))
            identical = (np.array_equal(compact["weight_ultranest"], weight)
                         and np.array_equal(compact["theta_ultranest"], posterior["theta"])
                         and sha256(compact_path) == nucleonic["a1_expected"]["compact_sha256"])
            ess_ok = (abs(ess - nucleonic["a1_expected"]["ess"]) <= 1.0e-9 and len(weight) == nucleonic["a1_expected"]["rows"]
                      and abs(ess - summary["methods"]["UltraNest"]["ess"]) <= 1.0e-9)
            ok = ok and identical and ess_ok
            diffs["a1_weights_identical"] = identical
            extra = f"; weights and parameters {'identical to' if identical else 'DIFFER from'} results/nucleonic/" \
                    f"a1_parameter_posteriors.npz, ESS {ess:.3f}"
        reports[run_id] = {"pass": ok, **stats, "differences": diffs}
        say(f"B nucleonic {run_id:10s} correction {'+'.join(record['correction_terms'])}: shift {stats['shift']:.6f} "
            f"(diff {diffs['shift']:.0e}), MC error {stats['mc_error']:.2e} (diff {diffs['mc_error']:.0e}){extra}  {verdict(ok)}")

    # Pooled nuclear-shift posteriors: ESS values and the curve selections of the shipped verification file.
    for case, run_ids in (("Jsym_29", ("nuc_j29_s0", "nuc_j29_s1")), ("Jsym_36", ("nuc_j36_s0", "nuc_j36_s1", "nuc_j36_s2"))):
        record = shipped[case]["ultranest"]
        checks = []
        for seed_record, run_id in zip(record["seeds"], run_ids):
            weight = corrected_by_run[run_id]
            run = nucleonic["runs"][run_id]
            checks += [relative(1.0 / float(np.sum(weight ** 2)), seed_record["corrected_seed_ess"]) <= TOLERANCE["ess_relative"],
                       seed_record["rows"] == len(weight),
                       seed_record["input_sha256"] == sha256(files[run["posterior"]]),
                       seed_record["terms_sha256"] == sha256(files[run["terms"]])]
        pooled = np.concatenate([corrected_by_run[r] for r in run_ids])
        pooled = pooled / pooled.sum()
        selection = shipped[case]["curve_selection"]
        seed = 2026082701 + 100 * SHIFT_ORDER.index(case)
        index = np.random.default_rng(seed).choice(len(pooled), CURVE_DRAWS, replace=True, p=pooled)
        digest = hashlib.sha256(np.ascontiguousarray(index).tobytes()).hexdigest()
        ess = float(1.0 / np.sum(pooled ** 2))
        checks += [relative(ess, record["pooled_corrected_ess"]) <= TOLERANCE["ess_relative"],
                   record["pooled_rows"] == len(pooled), seed == selection["ultranest_seed"],
                   digest == selection["ultranest_index_sha256"]]
        ok = all(checks)
        reports[f"{case}_pooled"] = {"pass": ok, "pooled_ess": ess, "curve_selection_identical": digest == selection["ultranest_index_sha256"]}
        say(f"B nucleonic {case} pooled seeds: ESS {ess:.2f}, per-seed ESS, files and the 8,000-draw curve selection "
            f"(seed {seed}) as in results/nucleonic/nuclear_amortization_verification.json  {verdict(ok)}")

    # J1614, nucleonic: the paper's target (no correction), equal weights; the shipped curves hold the stored samples.
    j1614 = nucleonic["J1614"]
    saved = load(files[j1614["posterior"]])
    curves_path = ROOT / "results/nucleonic/j1614_ultranest_mass_radius_curves.npz"
    curves = load(curves_path)
    curve_summary = json.loads((ROOT / "results/nucleonic/j1614_ultranest_mass_radius_curves.json").read_text())
    count = len(saved["samples"])
    ok = (np.array_equal(curves["theta"], saved["samples"]) and np.array_equal(curves["weight"], np.full(count, 1.0 / count))
          and str(curves["input_sha256"]) == sha256(files[j1614["posterior"]])
          and relative(float(1.0 / np.sum((curves["weight"] / curves["weight"].sum()) ** 2)), curve_summary["posterior_ess"]) <= TOLERANCE["ess_relative"]
          and all(float(saved[k]) == float(v) for k, v in j1614["run_report"].items() if k in saved))
    reports["nuc_J1614"] = {"pass": ok}
    say(f"B nucleonic J1614: the paper's target (no correction); results/nucleonic/j1614_ultranest_mass_radius_curves.npz holds "
        f"exactly the {count} stored samples with weight 1/n; log Z {float(saved['logz']):.6f} +/- {float(saved['logzerr']):.6f}  {verdict(ok)}")

    # Hyperonic desktop runs: the stored correction reports.
    for configuration, record in expected["hyperonic"]["desktop_b"].items():
        prepared, terms = load(files[record["input"]]), load(files[record["terms"]])
        report = json.loads((ROOT / record["report"]).read_text())
        report_ok = sha256(ROOT / record["report"]) == expected["records"]["run_reports"][record["id"]]["sha256"]
        stats = correction_statistics(np.asarray(prepared["posterior_weight"], dtype=np.float64), terms["fixed_0740"] - terms["old_0740"])
        primary = report["weighted_primary"]
        diffs = {"shift": abs(stats["shift"] - primary["shift"]), "cv2": abs(stats["cv2"] - primary["coefficient_variation_squared"]),
                 "mc_error": abs(stats["mc_error"] - primary["MC_error_approx"]),
                 "adjusted_ess": relative(stats["adjusted_ess"], primary["adjusted_ratio_ESS"]),
                 "kish_ess": relative(stats["kish_ess"], primary["reweighted_Kish_ESS"]),
                 "source_ess": relative(stats["source_ess"], report["source_posterior_ESS"]),
                 "corrected_logz": abs(report["raw_logZ"] + stats["shift"] - report["corrected_logZ"]),
                 "corrected_sigma": abs(math.hypot(report["raw_logZ_error"], stats["mc_error"]) - report["corrected_logZ_error"])}
        stored = stored_runs[record["id"]]
        same = (report["corrected_logZ"] == stored["corrected_logz"] and report["raw_logZ"] == stored["raw_results_logz"]
                and primary["shift"] == stored["nicer_weight_correction"] and report["correction"] == "0740"
                and str(terms["input_sha256"]) == sha256(files[record["input"]])
                and np.array_equal(prepared["record_index"], terms["record_index"]))
        ok = (report_ok and same and diffs["shift"] <= TOLERANCE["shift"] and diffs["cv2"] <= TOLERANCE["cv2"]
              and diffs["mc_error"] <= TOLERANCE["mc_error"] and max(diffs["adjusted_ess"], diffs["kish_ess"], diffs["source_ess"]) <= TOLERANCE["ess_relative"]
              and max(diffs["corrected_logz"], diffs["corrected_sigma"]) <= TOLERANCE["log_z"])
        reports[record["id"]] = {"pass": ok, **stats, "differences": diffs}
        say(f"B hyperonic {record['id']:18s} correction 0740: shift {stats['shift']:.6f} (diff {diffs['shift']:.0e}), "
            f"MC error {stats['mc_error']:.2e}, ESS {stats['kish_ess']:.1f}; stored report replayed  {verdict(ok)}")

    # Hyperonic Table VI runs: stored equal-weight posteriors; shared rows with the desktop runs.
    plugin = get_eos("ddb-hyperonic")
    for configuration, record in expected["hyperonic"]["cluster"].items():
        equal = load(files[record["equal_weight"]])
        ok = (len(equal["theta"]) == record["rows"] and np.array_equal(equal["nuclear_observation"], nuclear_observation(configuration))
              and np.array_equal(equal["nuclear_sigma"], A1_SIGMA) and np.array_equal(equal["prior_low"], plugin.prior_low)
              and np.array_equal(equal["prior_high"], plugin.prior_high)
              and str(equal["source_sha256"]) == record["primary_summary"]["sha256"])
        desktop = load(files[expected["hyperonic"]["desktop_b"][configuration]["input"]])["theta"]
        shared = np.intersect1d(row_view(desktop), row_view(equal["theta"]))
        copies = int(np.isin(row_view(equal["theta"]), shared).sum())
        target = expected["hyperonic"]["shared_ancestry"]["counts"][configuration]
        ancestry = shared.size == target["shared_distinct_rows"] and copies == target["cluster_copies"]
        ok = ok and ancestry
        reports[f"hyp_cluster_{configuration}"] = {"pass": ok, "shared_distinct_rows": int(shared.size), "cluster_copies": copies}
        say(f"B hyperonic {configuration:7s} Table VI run {record['id']}: {record['rows']} equal-weight rows, observation "
            f"and prior as stored; shares {shared.size} distinct 9D rows ({copies} copies) with the desktop "
            f"run (expected {target['shared_distinct_rows']}/{target['cluster_copies']})  {verdict(ok)}")

    # Hyperonic J1614: the Route 1 curve file is replica 1 with equal weights; replica values as stored.
    record = expected["hyperonic"]["J1614"]
    curves = load(ROOT / "results/hyperonic/J1614/ultranest_mass_radius.npz")
    count = len(curves["theta"])
    replica = record["replicas"]["replica_1"]
    ok = (np.array_equal(curves["weight"], np.full(count, 1.0 / count)) and count == replica["niter"]
          and str(curves["input_sha256"]) == replica["summary_sha256"])
    reports["hyp_J1614"] = {"pass": ok}
    say(f"B hyperonic J1614: the paper's target (no correction); results/hyperonic/J1614/ultranest_mass_radius.npz is replica 1 "
        f"({count} samples, weight 1/n)  {verdict(ok)}")
    return reports


# ------------------------------------------------------------------------------------------------ C. likelihood
class Targets:
    """Portable targets (nucleonic and hyperonic) and the NICER grids of both conventions."""

    def __init__(self):
        spec = {"0740": BASE_NICER_SOURCES[1], "1231": SOURCE_SUBSTITUTIONS["J1231"]}
        with contextlib.redirect_stdout(io.StringIO()):
            self.nucleonic = {source: A1Problem.from_data_root(OBSERVATIONS, source_data_root=OBSERVATIONS, workers=1,
                                                               verify_data=True, source_scenario=source, model="ddb")
                              for source in ("A1", "J0614", "J1231", "J1614")}
            self.hyperonic = {source: A1Problem.from_data_root(OBSERVATIONS, source_data_root=OBSERVATIONS, workers=1,
                                                               verify_data=True, source_scenario=source,
                                                               model="ddb-hyperonic").target
                              for source in ("A1", "J1614")}
        # The target's own J0740 and J1231 grids (single-weight convention).
        self.fixed = {"0740": self.nucleonic["A1"].target.nicer_interpolators[BASE_NICER_SOURCES[1].slot],
                      "1231": self.nucleonic["J1231"].target.nicer_interpolators[SOURCE_SUBSTITUTIONS["J1231"].slot]}
        # The double-weight convention, exactly as the UltraNest drivers build it (legacy module, same options).
        self.old = {key: legacy_nicer.build_nicer_grid(str(OBSERVATIONS / s.filename), mcol=s.mass_column,
                                                       rcol=s.radius_column, wcol_or_None=s.weight_column, **GRID)
                    for key, s in spec.items()}
        self.plugin = get_eos("ddb")

    def nucleonic_problem(self, nuclear: str, source: str):
        problem = self.nucleonic[source]
        if nuclear == "A1":
            return problem
        return dataclasses.replace(problem, nuclear_scenario=nuclear, target=dataclasses.replace(
            problem.target, nuclear_observation=nuclear_observation(nuclear)))

    def nucleonic_branches(self, theta: np.ndarray) -> list:
        padded = np.vstack([theta, np.repeat(theta[-1:], 16 - len(theta), axis=0)])
        _, energy, pressure = self.plugin.core_eos_batch(padded)
        return [solve_stable_branch(energy[k], pressure[k]) for k in range(len(theta))]

    def terms(self, mass, radius, maximum_mass) -> dict:
        values = {}
        for key in ("0740", "1231"):
            values[f"fixed_{key}"] = log_likelihood_one(mass, radius, maximum_mass, self.fixed[key])
            values[f"old_{key}"] = legacy_nicer.logL_nicer_one(mass, radius, maximum_mass, self.old[key])
        return values


def curve_differences(mass, radius, maximum_mass, grid, reference: dict) -> dict:
    """Maximum mass, R(1.4), R(M_max) and the radius curve on the saved mass grid."""
    on_grid = np.interp(grid, mass, radius, left=np.nan, right=np.nan)
    saved = np.asarray(reference["radius"], dtype=np.float64)
    same_support = bool(np.array_equal(np.isfinite(on_grid), np.isfinite(saved)))
    return {"maximum_mass": abs(maximum_mass - reference["maximum_mass"]),
            "radius_1p4": abs(float(np.interp(1.4, mass, radius)) - reference["radius_1p4"]),
            "radius_at_maximum_mass": abs(float(radius[np.argmax(mass)]) - reference["radius_at_maximum_mass"]),
            "radius_curve": float(np.nanmax(np.abs(on_grid - saved))) if same_support else float("inf")}


def check_rows(expected: dict, files: dict, rows: dict, targets: Targets) -> tuple[dict, dict]:
    reports, driver_rows = {}, {}

    def finish(name: str, log_l: list, term_diff: float, mass_diff: float, radius_diff: float, same_theta: bool) -> None:
        finite = bool(np.all(np.asarray(log_l) > -1.0e50))
        ok = (finite and same_theta and term_diff <= TOLERANCE["log_likelihood"] and mass_diff <= TOLERANCE["mass_msun"]
              and radius_diff <= TOLERANCE["radius_km"])
        reports[name] = {"pass": ok, "rows": len(log_l), "log_l": [float(v) for v in log_l], "all_finite": finite,
                         "rows_match_stored_file": same_theta, "largest_term_difference": term_diff,
                         "largest_mass_difference": mass_diff, "largest_radius_difference": radius_diff}
        say(f"C {name:24s} {len(log_l)} rows: exact log L {min(log_l):.3f} .. {max(log_l):.3f}; NICER terms diff "
            f"{term_diff:.0e}, M_max diff {mass_diff:.0e}, radius diff {radius_diff:.0e}  {verdict(ok)}")

    # Nucleonic runs with stored single-weight terms.
    for run_id, record in expected["nucleonic"]["runs"].items():
        index, theta = rows[f"{run_id}__rows"], rows[f"{run_id}__theta"]
        posterior, terms = load(files[record["posterior"]]), load(files[record["terms"]])
        problem = targets.nucleonic_problem(record["nuclear_scenario"], record["source_scenario"])
        log_l = problem.evaluate_batch(theta)
        term_diff = mass_diff = 0.0
        for k, branch in enumerate(targets.nucleonic_branches(theta)):
            replayed = targets.terms(branch.mass, branch.radius, branch.maximum_mass)
            term_diff = max(term_diff, *(abs(v - terms[key][index[k]]) for key, v in replayed.items()))
            mass_diff = max(mass_diff, abs(branch.maximum_mass - terms["maximum_mass"][index[k]]))
        finish(f"nucleonic {run_id}", list(log_l), term_diff, mass_diff, 0.0, np.array_equal(posterior["theta"][index], theta))

    # Nucleonic J1614 (the paper's target): TOV quantities against the shipped curve file.
    curves = load(ROOT / "results/nucleonic/j1614_ultranest_mass_radius_curves.npz")
    index = rows["nuc_J1614__rows"]
    theta = curves["theta"][index]
    log_l = targets.nucleonic["J1614"].evaluate_batch(theta)
    mass_diff = radius_diff = 0.0
    for k, branch in enumerate(targets.nucleonic_branches(theta)):
        d = curve_differences(branch.mass, branch.radius, branch.maximum_mass, curves["mass_grid"],
                              {"maximum_mass": curves["maximum_mass"][index[k]], "radius_1p4": curves["radius_1p4"][index[k]],
                               "radius_at_maximum_mass": curves["radius_at_maximum_mass"][index[k]], "radius": curves["radius"][index[k]]})
        mass_diff = max(mass_diff, d["maximum_mass"])
        radius_diff = max(radius_diff, d["radius_1p4"], d["radius_at_maximum_mass"], d["radius_curve"])
    finish("nucleonic J1614", list(log_l), 0.0, mass_diff, radius_diff, True)

    # Hyperonic: forward model and target of the pipeline (as in hyperonic_anet).
    def hyperonic(theta: np.ndarray, target) -> tuple[list, list]:
        properties = evaluator.nuclear_observables(theta)
        forwards, log_l = [], []
        for k, parameters in enumerate(theta):
            forward = evaluator._fast_forward(parameters)
            forwards.append(forward)
            value = -1.0e100 if forward is None else target.evaluate(parameters, properties[k], forward.density, forward.energy,
                                                                     forward.pressure, forward).total
            log_l.append(float(value) if np.isfinite(value) else -1.0e100)
        return forwards, log_l

    for configuration, record in expected["hyperonic"]["desktop_b"].items():
        run_id = record["id"]
        index, theta = rows[f"{run_id}__rows"], rows[f"{run_id}__theta"]
        prepared, terms = load(files[record["input"]]), load(files[record["terms"]])
        target = dataclasses.replace(targets.hyperonic["A1"], nuclear_observation=nuclear_observation(configuration))
        forwards, log_l = hyperonic(theta, target)
        term_diff = mass_diff = 0.0
        for k, forward in enumerate(forwards):
            if forward is None:
                term_diff = float("inf")
                continue
            replayed = targets.terms(forward.mass, forward.radius, forward.maximum_mass)
            term_diff = max(term_diff, *(abs(v - terms[key][index[k]]) for key, v in replayed.items()))
            mass_diff = max(mass_diff, abs(forward.maximum_mass - terms["maximum_mass"][index[k]]))
        driver_rows[configuration] = [(theta, np.asarray(log_l) - (terms["fixed_0740"][index] - terms["old_0740"][index]))]
        finish(f"hyperonic {run_id}", log_l, term_diff, mass_diff, 0.0, np.array_equal(prepared["theta"][index], theta))

    grid = rows["hyp_cluster__MG"]
    for configuration, record in expected["hyperonic"]["cluster"].items():
        prefix = f"hyp_cluster_{configuration}"
        index, theta = rows[f"{prefix}__rows"], rows[f"{prefix}__theta"]
        equal = load(files[record["equal_weight"]])
        target = dataclasses.replace(targets.hyperonic["A1"], nuclear_observation=nuclear_observation(configuration))
        forwards, log_l = hyperonic(theta, target)
        mass_diff = radius_diff = 0.0
        delta = []
        for k, forward in enumerate(forwards):
            if forward is None:
                mass_diff = float("inf")
                continue
            d = curve_differences(forward.mass, forward.radius, forward.maximum_mass, grid,
                                  {"maximum_mass": rows[f"{prefix}__MM"][k], "radius_1p4": rows[f"{prefix}__R14"][k],
                                   "radius_at_maximum_mass": rows[f"{prefix}__RMAX"][k], "radius": rows[f"{prefix}__Rg"][k]})
            mass_diff = max(mass_diff, d["maximum_mass"])
            radius_diff = max(radius_diff, d["radius_1p4"], d["radius_at_maximum_mass"], d["radius_curve"])
            replayed = targets.terms(forward.mass, forward.radius, forward.maximum_mass)
            delta.append(replayed["fixed_0740"] - replayed["old_0740"])
        if len(delta) == len(theta):
            driver_rows[configuration].append((theta, np.asarray(log_l) - np.asarray(delta)))
        finish(f"hyperonic {record['id']} (Table VI)", log_l, 0.0, mass_diff, radius_diff,
               np.array_equal(equal["theta"][index], theta))

    curves = load(ROOT / "results/hyperonic/J1614/ultranest_mass_radius.npz")
    index = rows["hyp_J1614__rows"]
    theta = curves["theta"][index]
    forwards, log_l = hyperonic(theta, targets.hyperonic["J1614"])
    mass_diff = radius_diff = 0.0
    for k, forward in enumerate(forwards):
        if forward is None:
            mass_diff = float("inf")
            continue
        d = curve_differences(forward.mass, forward.radius, forward.maximum_mass, curves["mass_grid"],
                              {"maximum_mass": curves["maximum_mass"][index[k]], "radius_1p4": curves["radius_1p4"][index[k]],
                               "radius_at_maximum_mass": curves["radius_at_maximum_mass"][index[k]], "radius": curves["radius"][index[k]]})
        mass_diff = max(mass_diff, d["maximum_mass"])
        radius_diff = max(radius_diff, d["radius_1p4"], d["radius_at_maximum_mass"], d["radius_curve"])
    driver_rows["J1614"] = [(theta, np.asarray(log_l))]
    finish("hyperonic J1614", log_l, 0.0, mass_diff, radius_diff, True)
    return reports, driver_rows


# ------------------------------------------------------------------------------------------------ D. drivers
def driver_environment(work: Path) -> dict:
    """CPU only, one thread per worker, and HOME inside the work folder so that the drivers (which add the
    home folder and their own project folder to the import path) can only import the repository modules."""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", HOME=str(work),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
                       PYTHONPATH=str(ROOT))
    environment.update(
        CERTIFIED_DDB_DIR=str(ROOT / "legacy_stack/validated_code_DDB"), CERTIFIED_ASTRO_DIR=str(ROOT / "legacy_stack/ddb_astro_mod"),
        HYPERON_PROJECT_DIR=str(ROOT / "hyperonic_pipeline"), DDB_OBS_DATA_DIR=str(OBSERVATIONS),
        DDB_J0437_FILE=str(OBSERVATIONS / "J0437_post_equal_weights.dat"), DDB_GW_FILE=str(OBSERVATIONS / "GW170817_GWTC-1.hdf5"),
        DDB_J0614_FILE=str(OBSERVATIONS / SOURCE_SUBSTITUTIONS["J0614"].filename),
        DDB_J1231_FILE=str(OBSERVATIONS / SOURCE_SUBSTITUTIONS["J1231"].filename),
        DDB_J1614_FILE=str(OBSERVATIONS / SOURCE_SUBSTITUTIONS["J1614"].filename))
    return environment


def check_drivers(driver_rows: dict) -> dict:
    reports = {}
    work = ROOT / "build/route2/ultranest_driver_smoke"  # replaced on every run
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    environment = driver_environment(work)

    started = time.time()
    output = work / "nucleonic"
    run = subprocess.run([sys.executable, str(ROOT / "inference/ultranest/run.py"), "--data-root", str(OBSERVATIONS),
                          "--output-dir", str(output), "--workers", "3", "--smoke"],
                         env=environment, cwd=work, capture_output=True, text=True)
    ok = run.returncode == 0 and (output / "run_report.json").is_file()
    gate = {}
    if ok:
        report = json.loads((output / "run_report.json").read_text())
        gate = report["target_certification"]
        with np.load(report["summary"], allow_pickle=False) as summary:
            schema = {"samples", "logz", "logzerr", "ncall", "target_certificate_sha256", "smoke_prior_probe"} <= set(summary.files)
        ok = (report["status"] == "SMOKE_COMPLETED" and gate["status"] == "PASS" and gate["rows"] == 1870
              and gate["max_abs_log_likelihood_error"] <= gate["tolerance"] == 1.0e-4 and schema)
    reports["nucleonic_run_py_smoke"] = {"pass": ok, "returncode": run.returncode, "seconds": round(time.time() - started, 1),
                                         "gate": gate, "stderr_tail": run.stderr[-2000:] if not ok else ""}
    say(f"D inference/ultranest/run.py --smoke: target gate on {gate.get('rows', 0)} certificate rows, largest log L "
        f"error {gate.get('max_abs_log_likelihood_error', float('nan')):.1e} (driver tolerance 1e-4), prior-probe "
        f"posterior written ({time.time() - started:.0f} s)  {verdict(ok)}")

    for configuration, groups in driver_rows.items():
        started = time.time()
        theta = np.vstack([g[0] for g in groups])
        reference = np.concatenate([g[1] for g in groups])
        folder = work / configuration
        folder.mkdir()
        np.savez(folder / "reference.npz", theta=theta, logl=reference)
        observation = nuclear_observation("A1" if configuration == "J1614" else configuration)
        np.savez(folder / "nuclear.npz", OBS=observation, SIG=A1_SIGMA)
        driver = "ddb_ultranest_hyp_j1614.py" if configuration == "J1614" else "ddb_ultranest_hyp.py"
        env = dict(environment, DDB_NUCLEAR_BANK=str(folder / "nuclear.npz"))
        if configuration == "J1614":
            env.pop("DDB_SWAP", None)  # this driver replaces J0740 by J1614 by default
        else:
            env["DDB_SWAP"] = ""
        run = subprocess.run([sys.executable, str(ROOT / "hyperonic_pipeline" / driver), "--cert", str(folder / "reference.npz"),
                              "--nw", "3"], env=env, cwd=folder, capture_output=True, text=True)
        match = re.search(r"max\|diff\| ([0-9.eE+-]+)", run.stdout)
        largest = float(match.group(1)) if match else float("inf")
        ok = run.returncode == 0 and "[CERT] PASS" in run.stdout and "valid-pattern match: True" in run.stdout
        reports[f"hyperonic_driver_{configuration}"] = {
            "pass": ok, "driver": driver, "rows": int(len(theta)), "largest_difference": largest,
            "seconds": round(time.time() - started, 1), "stdout_tail": run.stdout[-1500:], "stderr_tail": run.stderr[-1500:] if not ok else ""}
        what = ("portable J1614 log L (no correction)" if configuration == "J1614"
                else "portable log L - (fixed_0740 - old_0740)")
        say(f"D {driver} --cert {configuration:7s} {len(theta)} rows: driver target = {what}, largest difference "
            f"{largest:.1e} (driver gate 1e-4)  {verdict(ok)}")
    return reports


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="accepted for route2/run.py; this check runs on CPU")
    parser.add_argument("--skip-driver-smoke", action="store_true", help="skip stage D (the driver smoke runs)")
    parser.add_argument("--report", type=Path, default=ROOT / "build/route2/ultranest_check.json")
    arguments = parser.parse_args()
    started = time.time()
    for module in (evaluator, evaluator.legacy, evaluator.hyperonic_solver, legacy_nicer, evaluator.legacy.rn,
                   sys.modules["generate_ddbhy_bank"], sys.modules["fast_hyperonic_likelihood_shard"]):
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise SystemExit(f"FAIL: {module.__name__} was imported from outside the repository: {module.__file__}")
    prepare_observations()
    expected = json.loads((FIX / "expected.json").read_text())
    with np.load(FIX / "replay_rows.npz", allow_pickle=False) as data:
        rows = {key: data[key] for key in data.files}
    entries = catalogue()
    files = {name: fetch(name, entries) for name, entry in entries.items() if entry["group"] == "ultranest"}
    run_values = json.loads((ROOT / expected["records"]["runs"]["file"]).read_text())
    stored_runs = {c["id"]: c for row in run_values["rows"] for c in row["constituents"]}

    table_vi = check_table_vi(expected, stored_raw_values(expected, files))
    weights = check_weights(expected, files, stored_runs)
    say(f"C rebuilding the portable targets from {OBSERVATIONS.relative_to(ROOT)} ...")
    targets = Targets()
    likelihood, driver_rows = check_rows(expected, files, rows, targets)
    drivers = {} if arguments.skip_driver_smoke else check_drivers(driver_rows)

    stages = {"A_table_vi": table_vi["pass"], "B_weights": all(r["pass"] for r in weights.values()),
              "C_likelihood": all(r["pass"] for r in likelihood.values()),
              "D_driver_smoke": all(r["pass"] for r in drivers.values()) if drivers else "skipped"}
    ok = all(value is True or value == "skipped" for value in stages.values())
    summary = {"status": verdict(ok), "stages": stages, "tolerances": TOLERANCE, "seconds": round(time.time() - started, 1),
               "table_vi": table_vi, "weights": weights, "likelihood": likelihood, "drivers": drivers, "lines": LINES}
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"[ultranest] {summary['status']}: Table VI UltraNest column (27 runs, 16 rows), corrected weights, "
          f"{sum(r['rows'] for r in likelihood.values())} exact-likelihood rows"
          f"{'' if not drivers else ' and ' + str(len(drivers)) + ' driver smoke runs'} in {summary['seconds']} s "
          f"(report: {arguments.report.relative_to(ROOT) if arguments.report.is_relative_to(ROOT) else arguments.report})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
