#!/usr/bin/env python3
"""Route 2 check: the hyperonic TSNPE+MIS posterior and evidence (A1), replayed from the stored products.

The published result is an exact importance-sampling posterior on 150,000 rows drawn from a frozen proposal:
90% a cross-validated mixture (a bounded-logit Gaussian mixture, a broad Student-t and the uniform prior) and
10% the frozen parent TSNPE mixture (two neural-spline-flow ensembles, two Student-t components and the
uniform prior). Every row has an exact A1 log-likelihood.

1. Weights and summaries. The weights, ESS, log Z, its standard error and the largest weight are recomputed
   from the saved log L and log q with the formula of combine_crossvalidated_hybrid_is_9d.py and compared
   with the stored arrays and summary values. This is deterministic float64 arithmetic, so the tolerance is
   1e-12 (absolute for log Z, its error and the weights; relative for the ESS).
2. Resamples. The 20,000-row posterior index (seed 2026095501) and the 6,000-draw curve resample (seed
   20260921, prepare_frozen_tsnpe_curve_resample.py) are redrawn. Both must be identical to the stored ones;
   Route 1's results/hyperonic/A1/tsnpe_resample6000.npz must be the stored curve resample, byte for byte.
3. Proposal density at 14 rows (the 4 largest weights, one accepted row of each of the 8 proposal components
   spread over the three exact-likelihood shards, and 2 rejected rows), where
   log q = logsumexp(log 0.9 + logq_repair, log 0.1 + logq_parent).
   - logq_repair is the density of the frozen cross-validated mixture, recomputed with the density code of
     its builder (build_crossvalidated_hybrid_proposal_9d.py). Float64 analytic arithmetic: tolerance 1e-12.
   - logq_parent is recomputed from the frozen parent mixture, loading both flow ensembles with the production
     load_proposal and proposal_density; every file hash is checked against the hash stored in the file that
     references it. The flows are float32 networks that production evaluated on an RTX 4090, so another
     device gives slightly different float32 arithmetic. Tolerance 2e-3 in log q. This is about twice the
     largest CPU-versus-GPU flow difference seen when the frozen ensembles re-evaluate 40,000 of their own
     stored rows (after weighting by the flow's share of the density; the median raw difference is 3e-5).
     A change of 2e-3 in log q changes a weight by 0.2%, far below the log Z error (0.026).
   - Flow self-replay: each ensemble re-evaluates 10 of its own stored rows (6 flow, 2 Student-t, 2 uniform
     draws). The flow term is gated after weighting by its share of the proposal density (2e-3, as above;
     far in a flow's tail its raw float32 error is irrelevant); the Student-t and uniform terms are float64
     (1e-12). This also verifies the network architecture read from the proposal reports.
4. Likelihood. The exact A1 target is rebuilt from data/observations, and log L of the same 14 rows is
   recomputed with the production EOS/TOV evaluator (fast_portable_hyperonic_likelihood), exactly as
   route2/checks/hyperonic_anet.py does. Tolerance 1e-4, the certification gate of the pipeline; accepted
   and rejected rows must agree. The saved values were computed with the portable target code
   (workflows.A1Problem, the SciPy port of the EOS), so the same rows are also evaluated with it, and the
   frozen 64-row target certificate (the gate of every exact shard) is re-run. The nuclear term needs no
   EOS/TOV solve, so it is also recomputed on all 128,707 accepted rows (same 1e-4 tolerance). The report
   gives the difference between the two EOS implementations and the difference to the saved values. The
   latter comes from the batched nuclear-matter computation, whose result depends slightly on the compiled
   array shape and on the machine (the saved values came from three machines).

Run from the repository root:  PYTHONPATH=$PWD python -m route2.checks.tsnpe_hyperonic [--device cpu|cuda]
The large inputs are fetched (and verified) with route2.fetch; ROUTE2_LOCAL_MIRROR=<dir> copies them from a
local folder instead of the GitHub release.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

# Repository packages are imported before the pipeline modules (which add their own paths).
from likelihoods.nuclear.gaussian import gaussian_log_likelihood, observable_vector
from workflows.a1_problem import A1Problem
from workflows.resources import DEFAULT_HYPERONIC_TARGET_CERTIFICATE

from route2.common import OBSERVATIONS, ROOT, prepare_observations, sha256, use_hyperonic_pipeline
from route2.fetch import catalogue, fetch

use_hyperonic_pipeline()
TSNPE_CODE = ROOT / "hyperonic_pipeline/tsnpe"
sys.path.insert(0, str(TSNPE_CODE))

import fast_portable_hyperonic_likelihood as evaluator  # noqa: E402
import build_crossvalidated_hybrid_proposal_9d as final_builder  # noqa: E402
from build_dual_stage_flow_proposal_9d import load_proposal, proposal_density  # noqa: E402

FIX = ROOT / "route2/fixtures/tsnpe_hyperonic"
TAG = "[tsnpe-hyperonic]"
TOLERANCE = {"weights": 1.0e-12, "analytic_logq": 1.0e-12, "flow_logq": 2.0e-3, "log_likelihood": 1.0e-4}
TERMS = ("nuclear", "maximum_mass", "nicer", "gw170817", "pqcd")  # column order of the saved likelihood_components
REJECTED = -1.0e50


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def metadata(archive: dict) -> dict:
    return json.loads(str(archive["metadata"]))


def largest(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if np.size(a) else 0.0


def check_inputs(expected: dict, report: dict) -> dict:
    """Fetch the release files and close the hash chain posterior -> candidate -> parents -> checkpoints."""
    entries = catalogue()
    names = [expected["posterior"]["release_name"], expected["candidate"]["release_name"]]
    for flow in expected["flows"].values():
        proposal = flow["proposal_release_name"]
        names += [proposal, proposal.replace(".npz", ".json"), *flow["checkpoint_release_names"]]
    paths = {name: fetch(name, entries) for name in names}
    posterior = load(paths[expected["posterior"]["release_name"]])
    candidate = load(paths[expected["candidate"]["release_name"]])
    bridge_path, parent_path = FIX / expected["fixtures"]["bridge"], FIX / expected["fixtures"]["parent_mixture"]
    candidate_meta = metadata(candidate)
    links = {
        "posterior is the published posterior": sha256(paths[expected["posterior"]["release_name"]]) == expected["posterior"]["sha256"],
        "posterior references the candidate": metadata(posterior)["candidate"]["sha256"]
        == sha256(paths[expected["candidate"]["release_name"]]) == expected["candidate"]["sha256"],
        "candidate references the repair mixture": candidate_meta["bridge_sha256"] == sha256(bridge_path),
        "candidate references the parent mixture": candidate_meta["parent_hybrid_model_sha256"] == sha256(parent_path),
    }
    flows = {}
    for name, flow in expected["flows"].items():
        proposal_path = paths[flow["proposal_release_name"]]
        checkpoints = [paths[n] for n in flow["checkpoint_release_names"]]
        links[f"candidate references the {name} flow proposal"] = candidate_meta[f"{name}_proposal_sha256"] == sha256(proposal_path)
        stored_hashes = [member["checkpoint_sha256"] for member in metadata(load(proposal_path))["flow_members"]]
        links[f"{name} flow checkpoints match the proposal"] = stored_hashes == [sha256(p) for p in checkpoints]
        flows[name] = {"proposal": proposal_path, "checkpoints": checkpoints}
    code = json.loads((TSNPE_CODE / "CODE_MANIFEST.json").read_text())["files"]
    listed_code = {Path(entry["file"]).name: entry["sha256"] for entry in code}
    for module in ("build_crossvalidated_hybrid_proposal_9d.py", "build_dual_stage_flow_proposal_9d.py"):
        links[f"code {module} matches CODE_MANIFEST.json"] = sha256(TSNPE_CODE / module) == listed_code[module]
    ok = all(links.values())
    report["inputs"] = {"pass": ok, "release_files": len(paths), "links": links}
    print(f"{TAG} inputs      {'PASS' if ok else 'FAIL'}  {len(paths)} release files verified; hash chain posterior -> "
          f"candidate -> cross-validated/parent mixtures -> 2 flow proposals -> {sum(len(f['checkpoints']) for f in flows.values())} "
          f"checkpoints {'closes' if ok else 'BROKEN'}", flush=True)
    return {"posterior": posterior, "candidate": candidate, "bridge": load(bridge_path), "parent": load(parent_path),
            "flows": flows}


def check_weights(posterior: dict, expected: dict, report: dict) -> np.ndarray:
    """The certification formula of combine_crossvalidated_hybrid_is_9d.py (its lines 130-150)."""
    rows = len(posterior["theta"])
    low, high = posterior["prior_low"], posterior["prior_high"]
    log_prior = -float(np.log(high - low).sum())
    valid = posterior["valid"].astype(bool)
    log_target = np.full(rows, -np.inf, dtype=np.float64)
    target_valid = valid & np.isfinite(posterior["log_likelihood"]) & (posterior["log_likelihood"] > REJECTED)
    log_target[target_valid] = posterior["log_likelihood"][target_valid] + log_prior
    log_weight = log_target - posterior["log_proposal"]
    finite = np.isfinite(log_weight)
    maximum = float(np.max(log_weight[finite]))
    scaled = np.zeros(rows, dtype=np.float64)
    scaled[finite] = np.exp(log_weight[finite] - maximum)
    weight = scaled / scaled.sum()
    ess = float(1.0 / np.sum(weight**2))
    evidence_scaled = float(np.mean(scaled))
    log_evidence = float(maximum + np.log(evidence_scaled))
    log_evidence_error = float(np.std(scaled, ddof=1) / np.sqrt(rows) / evidence_scaled)

    published = expected["posterior"]
    same_masks = (np.array_equal(np.isfinite(log_target), np.isfinite(posterior["log_target"]))
                  and np.array_equal(finite, np.isfinite(posterior["log_weight"])))
    differences = {
        "ess_relative": abs(ess - published["ess"]) / published["ess"],
        "log_evidence": abs(log_evidence - published["log_evidence"]),
        "log_evidence_standard_error": abs(log_evidence_error - published["log_evidence_standard_error"]),
        "maximum_normalized_weight": abs(float(weight.max()) - published["maximum_normalized_weight"]),
        "stored_ESS_relative": abs(ess - float(posterior["ESS"])) / published["ess"],
        "stored_logZ": abs(log_evidence - float(posterior["logZ"])),
        "stored_logZ_standard_error": abs(log_evidence_error - float(posterior["logZ_standard_error"])),
        "log_target_max_abs": largest(log_target[target_valid], posterior["log_target"][target_valid]),
        "log_weight_max_abs": largest(log_weight[finite], posterior["log_weight"][finite]),
        "normalized_weight_max_abs": largest(weight, posterior["normalized_weight"]),
    }
    counts = {"rows": rows, "valid_rows": int(target_valid.sum()), "finite_weight_rows": int(finite.sum())}
    ok = (same_masks and rows == published["candidate_rows"] and counts["valid_rows"] == published["valid_rows"]
          and counts["finite_weight_rows"] == published["finite_weight_rows"]
          and max(differences.values()) <= TOLERANCE["weights"])
    recomputed = {"ess": ess, "log_evidence": log_evidence, "log_evidence_standard_error": log_evidence_error,
                  "maximum_normalized_weight": float(weight.max()), **counts}
    report["weights"] = {"pass": bool(ok), "recomputed": recomputed, "largest_difference": max(differences.values()),
                         "differences": differences, "masks_identical": bool(same_masks)}
    print(f"{TAG} weights     {'PASS' if ok else 'FAIL'}  ESS {ess:.6f}, log Z {log_evidence:.6f} +- "
          f"{log_evidence_error:.6f}, largest weight {weight.max():.6f}, {counts['valid_rows']} valid rows "
          f"(largest difference {max(differences.values()):.0e}, tolerance {TOLERANCE['weights']:.0e})", flush=True)
    return weight


def check_resamples(posterior: dict, weight: np.ndarray, expected: dict, report: dict) -> None:
    published = expected["posterior"]
    index = np.random.default_rng(published["resample_seed"]).choice(
        len(weight), size=published["posterior_rows"], replace=True, p=weight)
    index_ok = (np.array_equal(index, posterior["posterior_index"])
                and np.array_equal(posterior["theta"][index], posterior["posterior"]))

    curve = expected["curve_resample"]
    route1 = ROOT / curve["in"]
    route1_is_published = sha256(route1) == curve["sha256"]
    shipped = load(route1)
    # prepare_frozen_tsnpe_curve_resample.py draws from the weights stored in the posterior file.
    drawn = np.random.default_rng(curve["seed"]).choice(
        len(weight), size=curve["draws"], replace=True, p=posterior["normalized_weight"])
    source_index, counts = np.unique(drawn, return_counts=True)
    curve_ok = (route1_is_published and int(shipped["seed"]) == curve["seed"]
                and int(shipped["posterior_draws"]) == curve["draws"]
                and np.array_equal(source_index, shipped["source_index"]) and np.array_equal(counts, shipped["counts"])
                and np.array_equal(posterior["theta"][source_index], shipped["theta"]))
    ok = index_ok and curve_ok
    report["resamples"] = {"pass": bool(ok), "posterior_index_identical": bool(index_ok),
                           "posterior_index": {"seed": published["resample_seed"], "rows": published["posterior_rows"]},
                           "curve_resample_identical": bool(curve_ok), "route1_file_is_the_published_file": bool(route1_is_published),
                           "curve_resample": {"seed": curve["seed"], "draws": curve["draws"],
                                              "unique_rows": int(len(source_index))}}
    print(f"{TAG} resamples   {'PASS' if ok else 'FAIL'}  posterior index ({published['posterior_rows']} rows, seed "
          f"{published['resample_seed']}) {'identical' if index_ok else 'DIFFERENT'}; curve resample ({curve['draws']} "
          f"draws, seed {curve['seed']}, {len(source_index)} unique) {'identical' if curve_ok else 'DIFFERENT'} to the "
          f"stored file and Route 1's {curve['in']}", flush=True)


def load_flows(inputs: dict, device: torch.device) -> dict:
    flows = {}
    for name, files in inputs["flows"].items():
        _, _, networks, model = load_proposal(files["proposal"], files["checkpoints"], device)
        flows[name] = {"networks": networks, "model": model}
    return flows


def check_proposal(inputs: dict, fixture: dict, flows: dict, device: torch.device, report: dict) -> None:
    candidate, posterior, bridge = inputs["candidate"], inputs["posterior"], inputs["bridge"]
    rows = fixture["rows"]
    theta = candidate["theta"][rows]
    fraction = float(bridge["parent_tsnpe_fraction"])
    # Relations on all rows (exact bookkeeping) and identity of the saved replay rows.
    stored_mixture = logsumexp(np.column_stack([np.log1p(-fraction) + candidate["logq_repair"],
                                                np.log(fraction) + candidate["logq_parent_tsnpe"]]), axis=1)
    bookkeeping = {
        "candidate_rows_are_posterior_rows": bool(np.array_equal(candidate["theta"], posterior["theta"])
                                                  and np.array_equal(candidate["log_proposal"], posterior["log_proposal"])),
        "stored_log_q_is_the_0.9/0.1_mixture_of_its_parts_all_rows": largest(stored_mixture, candidate["log_proposal"]),
        "fixture_rows_match": bool(all(np.array_equal(fixture[key], candidate[key][rows])
                                       for key in ("theta", "logq_repair", "logq_parent_tsnpe", "log_proposal"))),
    }

    low, high = bridge["prior_low"], bridge["prior_high"]
    value, jacobian = final_builder.transform(theta, low, high, bridge["transform_mean"], bridge["transform_scale"])
    logq_gmm = final_builder.gmm_log_density(value, bridge["gmm_weights"], bridge["gmm_means"],
                                             bridge["gmm_covariances"]) + jacobian
    logq_broad = final_builder.t_log_density(value, bridge["broad_location"], bridge["broad_shape"],
                                             float(bridge["broad_degrees"])) + jacobian
    logq_uniform = np.full(len(theta), -float(np.log(high - low).sum()), dtype=np.float64)
    fractions = bridge["repair_fractions"]
    logq_repair = logsumexp(np.column_stack([np.log(fractions[0]) + logq_gmm, np.log(fractions[1]) + logq_broad,
                                             np.log(fractions[2]) + logq_uniform]), axis=1)

    parent_weights = inputs["parent"]["optimized_weights"]
    _, old_flow, old_broad, old_uniform = proposal_density(theta, flows["old"]["networks"], flows["old"]["model"], device, 8192)
    _, new_flow, new_broad, new_uniform = proposal_density(theta, flows["new"]["networks"], flows["new"]["model"], device, 8192)
    logq_parent = logsumexp(np.column_stack([
        np.log(parent_weights[0]) + old_flow, np.log(parent_weights[1]) + old_broad,
        np.log(parent_weights[2]) + new_flow, np.log(parent_weights[3]) + new_broad,
        np.log(parent_weights[4]) + old_uniform]), axis=1)
    logq = logsumexp(np.column_stack([np.log1p(-fraction) + logq_repair, np.log(fraction) + logq_parent]), axis=1)
    differences = {
        "logq_repair": largest(logq_repair, candidate["logq_repair"][rows]),
        "logq_parent": largest(logq_parent, candidate["logq_parent_tsnpe"][rows]),
        "logq_vs_candidate": largest(logq, candidate["log_proposal"][rows]),
        "logq_vs_posterior": largest(logq, posterior["log_proposal"][rows]),
        "uniform_old_vs_new": largest(old_uniform, new_uniform),
    }
    parent_share = np.exp(np.log(fraction) + candidate["logq_parent_tsnpe"][rows] - candidate["log_proposal"][rows])

    self_replay = {}
    for name, flow in flows.items():
        model = flow["model"]
        selected = fixture[f"{name}_flow_rows"]
        same = all(np.array_equal(fixture[f"{name}_flow_{key}"], model[source][selected]) for key, source in (
            ("theta", "theta"), ("logq_flow_ensemble", "stored_logq_flow"), ("logq_broad", "stored_logq_broad"),
            ("logq_uniform_full_prior", "stored_logq_uniform"), ("proposal_component", "proposal_component")))
        mixture, qflow, qbroad, quniform = proposal_density(model["theta"][selected], flow["networks"], model, device, 8192)
        stored_terms = np.log(model["fractions"])[:, None] + np.vstack([
            model["stored_logq_flow"][selected], model["stored_logq_broad"][selected], model["stored_logq_uniform"][selected]])
        stored = logsumexp(stored_terms, axis=0)
        share = np.exp(stored_terms[0] - stored)
        raw = np.abs(qflow - model["stored_logq_flow"][selected])
        self_replay[name] = {"rows": int(len(selected)), "fixture_rows_match": bool(same),
                             "flow_share_weighted": float(np.max(share * raw)), "flow_raw": float(np.max(raw)),
                             "student_t": largest(qbroad, model["stored_logq_broad"][selected]),
                             "uniform": largest(quniform, model["stored_logq_uniform"][selected]),
                             "mixture": largest(mixture, stored)}

    ok = (bookkeeping["candidate_rows_are_posterior_rows"] and bookkeeping["fixture_rows_match"]
          and bookkeeping["stored_log_q_is_the_0.9/0.1_mixture_of_its_parts_all_rows"] <= TOLERANCE["analytic_logq"]
          and differences["logq_repair"] <= TOLERANCE["analytic_logq"]
          and differences["uniform_old_vs_new"] <= TOLERANCE["analytic_logq"]
          and max(differences["logq_parent"], differences["logq_vs_candidate"], differences["logq_vs_posterior"])
          <= TOLERANCE["flow_logq"]
          and all(r["fixture_rows_match"] and r["student_t"] <= TOLERANCE["analytic_logq"]
                  and r["uniform"] <= TOLERANCE["analytic_logq"]
                  and max(r["flow_share_weighted"], r["mixture"]) <= TOLERANCE["flow_logq"] for r in self_replay.values()))
    report["proposal"] = {"pass": bool(ok), "replayed_rows": int(len(rows)), "rows": rows.tolist(),
                          "components": fixture["proposal_component"].tolist(),
                          "exact_shard": fixture["exact_shard"].tolist(), "bookkeeping": bookkeeping,
                          "largest_differences": differences, "parent_share_of_log_q": parent_share.round(4).tolist(),
                          "flow_self_replay": self_replay}
    worst_self = max(max(r["flow_share_weighted"], r["mixture"]) for r in self_replay.values())
    print(f"{TAG} proposal    {'PASS' if ok else 'FAIL'}  {len(rows)} rows: logq_repair diff {differences['logq_repair']:.0e} "
          f"(tol {TOLERANCE['analytic_logq']:.0e}), logq_parent diff {differences['logq_parent']:.0e}, log q diff "
          f"{differences['logq_vs_candidate']:.0e} (tol {TOLERANCE['flow_logq']:.0e}); flow self-replay "
          f"{sum(r['rows'] for r in self_replay.values())} rows {worst_self:.0e}", flush=True)


def check_likelihood(inputs: dict, fixture: dict, expected: dict, report: dict) -> None:
    posterior = inputs["posterior"]
    rows = fixture["rows"]
    theta = posterior["theta"][rows]
    saved = posterior["log_likelihood"][rows]
    saved_terms = posterior["likelihood_components"][rows]
    accepted = saved > REJECTED
    with contextlib.redirect_stdout(io.StringIO()):
        problem = A1Problem.from_data_root(OBSERVATIONS, source_data_root=OBSERVATIONS, workers=1, verify_data=True,
                                           nuclear_scenario="A1", source_scenario="A1", model="ddb-hyperonic")
    target = problem.target

    # Production EOS/TOV evaluator, as route2/checks/hyperonic_anet.py.
    properties = evaluator.nuclear_observables(theta)
    production = np.full(len(theta), -1.0e100)
    production_terms = np.full((len(theta), len(TERMS)), np.nan)
    for index, parameters in enumerate(theta):
        forward = evaluator._fast_forward(parameters)
        if forward is None:
            continue
        terms = target.evaluate(parameters, properties[index], forward.density, forward.energy, forward.pressure, forward)
        if np.isfinite(terms.total):
            production[index] = terms.total
            production_terms[index] = [getattr(terms, name) for name in TERMS]
    # The portable target code that computed the saved values, and its frozen 64-row certificate.
    with contextlib.redirect_stdout(io.StringIO()):
        portable, portable_terms = problem.evaluate_batch(theta, return_components=True)
        gate = problem.certify(DEFAULT_HYPERONIC_TARGET_CERTIFICATE)

    patterns = {"production": bool(np.array_equal(accepted, production > REJECTED)),
                "portable": bool(np.array_equal(accepted, portable > REJECTED))}
    same = all(patterns.values())
    per_term = {}
    if same:
        differences = {"production_vs_saved": largest(production[accepted], saved[accepted]),
                       "portable_vs_saved": largest(portable[accepted], saved[accepted]),
                       "production_vs_portable": largest(production[accepted], portable[accepted])}
        for label, values in (("production_vs_saved", production_terms), ("portable_vs_saved", portable_terms)):
            per_term[label] = {name: largest(values[accepted, k], saved_terms[accepted, k]) for k, name in enumerate(TERMS)}
        per_term["production_vs_portable"] = {name: largest(production_terms[accepted, k], portable_terms[accepted, k])
                                              for k, name in enumerate(TERMS)}
        # Where each saved value was computed (the three exact-likelihood shards).
        per_term["production_vs_saved_by_exact_shard"] = {
            shard["where"]: largest(production[accepted & (fixture["exact_shard"] == number)],
                                    saved[accepted & (fixture["exact_shard"] == number)])
            for number, shard in enumerate(expected["exact_shards"])}
    else:
        differences = {key: float("inf") for key in ("production_vs_saved", "portable_vs_saved", "production_vs_portable")}
    published_gate = expected["posterior"]["target_gate"]
    gate_ok = (gate.get("status") == "PASS" and gate.get("valid_mask_equal") is True
               and gate.get("rows") == published_gate["rows"] and gate.get("actual_valid_rows") == published_gate["actual_valid_rows"]
               and gate.get("max_abs_log_likelihood_error", np.inf) <= TOLERANCE["log_likelihood"])

    # The nuclear term needs no EOS/TOV solve, so it is replayed on every accepted row, in the production batches
    # of 2,500 rows. Its batched nuclear-matter computation depends slightly on the compiled array shape and on
    # the machine; this shows how large that effect is over the whole posterior.
    everywhere = np.flatnonzero(posterior["log_likelihood"] > REJECTED)
    nuclear = gaussian_log_likelihood(
        observable_vector(posterior["theta"][everywhere], evaluator.nuclear_observables(posterior["theta"][everywhere])),
        target.nuclear_observation, target.nuclear_sigma)
    nuclear_difference = np.abs(nuclear - posterior["likelihood_components"][everywhere, 0])
    nuclear_all = {"rows": int(len(everywhere)), "largest": float(nuclear_difference.max()),
                   "weighted_mean": float(np.sum(posterior["normalized_weight"][everywhere] * nuclear_difference)),
                   "by_exact_shard": {}}
    for low_high, shard in zip((s["rows"] for s in expected["exact_shards"]), expected["exact_shards"]):
        inside = (everywhere >= low_high[0]) & (everywhere < low_high[1])
        values = nuclear_difference[inside]
        nuclear_all["by_exact_shard"][shard["where"]] = {
            "rows": int(inside.sum()), "largest": float(values.max()), "median": float(np.median(values)),
            "percentile_99.9": float(np.percentile(values, 99.9)), "fraction_above_1e-9": float(np.mean(values > 1.0e-9))}
    ok = (same and gate_ok and max(differences.values()) <= TOLERANCE["log_likelihood"]
          and nuclear_all["largest"] <= TOLERANCE["log_likelihood"])
    report["likelihood"] = {"pass": bool(ok), "replayed_rows": int(len(rows)), "accepted_rows": int(accepted.sum()),
                            "accepted_rejected_pattern_identical": patterns, "largest_difference": differences,
                            "largest_difference_per_term": per_term, "nuclear_term_all_accepted_rows": nuclear_all,
                            "target_certificate": {"status": gate.get("status"), "rows": gate.get("rows"),
                                                   "valid_rows": gate.get("actual_valid_rows"),
                                                   "max_abs_log_likelihood_error": gate.get("max_abs_log_likelihood_error"),
                                                   "published_max_abs_log_likelihood_error": published_gate["max_abs_log_likelihood_error"]}}
    print(f"{TAG} likelihood  {'PASS' if ok else 'FAIL'}  {len(rows)} rows ({int(accepted.sum())} accepted, rejected rows "
          f"{'agree' if same else 'DISAGREE'}): production evaluator vs saved {differences['production_vs_saved']:.1e} "
          f"(tol {TOLERANCE['log_likelihood']:.0e}); portable evaluator vs saved {differences['portable_vs_saved']:.1e}; "
          f"production vs portable {differences['production_vs_portable']:.1e}; target certificate "
          f"{gate.get('status')} ({gate.get('max_abs_log_likelihood_error', float('nan')):.1e}); nuclear term on all "
          f"{nuclear_all['rows']} accepted rows {nuclear_all['largest']:.1e}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda (only the flow densities use it)")
    parser.add_argument("--report", type=Path, default=ROOT / "build/route2/tsnpe_hyperonic_check.json")
    arguments = parser.parse_args()
    started = time.time()
    torch.set_num_threads(min(3, torch.get_num_threads()))  # the replay is tiny; stay polite on shared machines
    device = torch.device(arguments.device)
    prepare_observations()
    expected = json.loads((FIX / "expected.json").read_text())
    fixture = load(FIX / "replay_rows.npz")

    report: dict = {}
    inputs = check_inputs(expected, report)
    weight = check_weights(inputs["posterior"], expected, report)
    check_resamples(inputs["posterior"], weight, expected, report)
    flows = load_flows(inputs, device)
    check_proposal(inputs, fixture, flows, device, report)
    print(f"{TAG} rebuilding the exact A1 target from {OBSERVATIONS.relative_to(ROOT)} ...", flush=True)
    check_likelihood(inputs, fixture, expected, report)

    steps = ("inputs", "weights", "resamples", "proposal", "likelihood")
    ok = all(report[step]["pass"] for step in steps)
    summary = {"status": "PASS" if ok else "FAIL", "device": arguments.device,
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "tolerances": TOLERANCE,
               "seconds": round(time.time() - started, 1), **report}
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(summary, indent=1) + "\n")
    shown = arguments.report.relative_to(ROOT) if arguments.report.is_relative_to(ROOT) else arguments.report
    print(f"{TAG} {summary['status']}: A1 weights, resamples, proposal densities and exact likelihoods replayed in "
          f"{summary['seconds']} s on {arguments.device} (report: {shown})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
