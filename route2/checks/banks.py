#!/usr/bin/env python3
"""Route 2 check: rows of every bank that feeds the paper, recomputed with the shipped code.

Banks (their hashes, and the relations verified on ALL rows when the fixtures were extracted, are in
route2/fixtures/banks/expected.json):

  nucleonic           shared prior physics bank, exact astrophysical table, uniform-prior support bank,
                      independent common-physics cache
  hyperonic           four uniform-prior banks, nine A-NET training caches (uniform, support, proposal),
                      Stage-A MIS evaluation
  Green EN            nucleonic and hyperonic training banks (with their S_ext extensions)

For 30-40 deterministically chosen rows per bank (8 per uniform bank, 3-6 per cache part) the check replays:

1. Parameter draws. Every row with a stored seed is redrawn from that seed (numpy PCG64 streams: the
   uniform streams seed 1234/2026 and 91-94, and SeedSequence([seed, stream, block]) for the enrichment and
   support screens; the Student proposal draws from seed 20260992). Exact: random streams are deterministic.
2. Forward physics, with the production functions of the bank that is checked (inference/anet/build_bank.py,
   hyperonic_pipeline/generate_ddbhy_bank.py, build_clean_uniform_hyperonic_training_cache.py,
   fast_hyperonic_likelihood_shard.py, workflows/a1_problem.py): nuclear-matter properties, R(M), Lambda(M),
   maximum mass, R1.4, the pQCD and GW grids and every stored log-likelihood column.
3. Bookkeeping. Support flags and counting corrections (logw_prior, LPC) are recomputed from the replayed
   physics with the stored proposal counts; forward failures and rejected rows must be the same rows.

Tolerances:
- draws, flags, failure codes, rejection patterns and counting corrections: exact (they are deterministic;
  logw_prior is compared to 1e-15, the float64 rounding of one log).
- the Student proposal draws (made on another machine) are compared to 1e-10, as in the hyperonic A-NET check:
  the Student mixture's linear algebra rounds differently on other BLAS builds (observed 1.2e-12).
- nuclear-matter properties: |difference| <= 1e-4 of the nuclear sigma, the prediction gate of the
  pipeline (prediction_tolerance_sigma). K0 is a finite-difference second derivative (step 1e-4 fm^-3), so its
  last digits depend on how JAX vectorizes the batch; observed about 2e-6 sigma. Where the bank stores X in
  float32 (Green EN), half a float32 spacing of the stored value is added (E0 far from saturation, e.g. -100 MeV,
  rounds by up to 4e-5 sigma).
- stellar curves R(M), Lambda(M), Mmax, R1.4, R(Mmax): relative 1e-6 (fixed-step RK4 TOV; inputs differ only by
  float rounding, observed <= 1e-10). The Green EN banks store float32; float32 rounding (<= 6e-8 relative) is
  inside this tolerance and inside the sigma gate.
- log-likelihood columns: 1e-4 absolute, the certification gate of the pipeline (CERT_TOLERANCE,
  A1Problem.certify). Values at or below -1e29 are floor markers (a source outside its support) and must be
  floor markers in both; they are compared as markers, not as numbers.

The nucleonic exact table was computed with the double-weight NICER grids, which the shipped
legacy_stack/ddb_astro_mod reproduces; its J0740 column is therefore replayed with those grids (the single-weight
value used downstream is reported for information).

Known defect of an intermediate bank: in six rounds (r9-r14) of the hyperonic Stage-A bridge, the shard
evaluated on one machine contains no ordinary likelihood value: its forward
(EOS/TOV) computations ran, but every row whose likelihood needs the NICER/GW densities is stored as rejected
(-1e100) because that machine's likelihood-data step failed. The shipped code evaluates those rows. This bank
only shaped the defensive Student proposal component of the hyperonic A-NET; the A-NET importance
weights use that component's exact density and the exact likelihood, so no posterior, evidence, figure or
table of the paper depends on the defect (the hyperonic_anet check replays all eight results). The check
prints the defect and passes only if the mismatches are exactly this defect; any other mismatch fails.
--strict makes the defect itself fail too.

Run from the repository root:  PYTHONPATH=$PWD python -m route2.checks.banks [--strict]
(CPU only, about 75 s wall and 2 CPU-minutes on a desktop; writes build/route2/banks_check.json)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np

# Repository packages first (the pipeline modules add their own paths, including the home directory).
from eos import get_eos
from inference.anet import build_bank
from likelihoods.gw170817 import load_kde
from likelihoods.nicer import log_likelihood_one
from likelihoods.nuclear import A1_OBSERVATION, A1_SIGMA
from tov import solve_stable_branch
from workflows.a1_problem import A1Problem
from workflows.nuclear_scenarios import nuclear_observation

from route2.common import OBSERVATIONS, ROOT, prepare_observations, sha256, use_hyperonic_pipeline

FIX = ROOT / "route2/fixtures/banks"
STUDENT = FIX / "student/student_beta1_48c_df5_s075_seed20260990.joblib"
NUCLEAR_FILE = ROOT / "build/route2/banks_nuclear_observation.npz"
TOLERANCE = {"nuclear_sigma": 1.0e-4, "curve_relative": 1.0e-6, "log_likelihood": 1.0e-4, "floor": -1.0e29,
             "logw_prior": 1.0e-15, "student_draw": 1.0e-10, "student_logq": 1.0e-10}
SEXT_PROPOSALS = 160_000_000
CASE_SHIFTS = ("A1", "K0_200", "K0_260", "Jsym_29", "Jsym_36")

# The legacy (Stage-A) target reads its data paths when it is imported.
os.environ.pop("DDB_SWAP", None)
os.environ["DDB_OBS_DATA_DIR"] = str(OBSERVATIONS)
os.environ["DDB_J0437_FILE"] = str(OBSERVATIONS / "J0437_post_equal_weights.dat")
os.environ["DDB_GW_FILE"] = str(OBSERVATIONS / "GW170817_GWTC-1.hdf5")
os.environ["DDB_NUCLEAR_BANK"] = str(NUCLEAR_FILE)
use_hyperonic_pipeline()

import joblib  # noqa: E402

import build_clean_uniform_hyperonic_training_cache as training_cache  # noqa: E402
import ddb_ultranest_hyp as legacy_target  # noqa: E402
import fast_hyperonic_likelihood_shard as legacy_shard  # noqa: E402
import generate_ddbhy_bank as uniform_bank  # noqa: E402
import nicer_like as legacy_nicer  # noqa: E402
from clean_student_t_ensemble import checkpoint_density, sample_checkpoint  # noqa: E402


# ------------------------------------------------------------------ comparisons

def worst(values) -> float:
    values = [float(v) for v in values if v is not None and not np.isnan(v)]
    return max(values) if values else 0.0


def same(replayed, stored) -> bool:
    return bool(np.array_equal(np.asarray(replayed), np.asarray(stored), equal_nan=True))


def relative(replayed, stored) -> tuple[bool, float]:
    """NaN pattern identical, then relative difference of the finite values."""
    a, b = np.asarray(replayed, dtype=np.float64), np.asarray(stored, dtype=np.float64)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return False, float("inf")
    finite = np.isfinite(b)
    if not finite.any():
        return True, 0.0
    value = float(np.max(np.abs(a[finite] - b[finite]) / np.maximum(np.abs(b[finite]), 1e-300)))
    return value <= TOLERANCE["curve_relative"], value


def loglike(replayed, stored) -> tuple[bool, float]:
    """NaN and floor-marker patterns identical, then absolute difference of the ordinary values."""
    a, b = np.asarray(replayed, dtype=np.float64), np.asarray(stored, dtype=np.float64)
    floor = TOLERANCE["floor"]
    if not (np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a <= floor, b <= floor)):
        return False, float("inf")
    ordinary = ~np.isnan(b) & (b > floor)
    if not ordinary.any():
        return True, 0.0
    value = float(np.max(np.abs(a[ordinary] - b[ordinary])))
    return value <= TOLERANCE["log_likelihood"], value


def sigma_gate(replayed, stored) -> tuple[bool, float]:
    """|difference| <= 1e-4 nuclear sigma; a float32-stored value may also carry half a float32 spacing."""
    a, b = np.asarray(replayed, dtype=np.float64), np.asarray(stored, dtype=np.float64)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return False, float("inf")
    finite = np.isfinite(b).all(axis=1)
    if not finite.any():
        return True, 0.0
    difference = np.abs(a[finite] - b[finite])
    allowed = TOLERANCE["nuclear_sigma"] * A1_SIGMA
    if np.asarray(stored).dtype == np.float32:
        allowed = allowed + 0.5 * np.spacing(np.abs(np.asarray(stored)[finite]))
    return bool(np.all(difference <= allowed)), float(np.max(difference / A1_SIGMA))


class Item:
    """Collects the comparisons of one bank."""

    def __init__(self, name: str):
        self.name, self.started, self.results, self.notes = name, time.time(), {}, []

    def exact(self, label: str, ok: bool) -> None:
        self.results[label] = {"pass": bool(ok), "kind": "exact"}

    def value(self, label: str, outcome: tuple[bool, float], kind: str) -> None:
        self.results[label] = {"pass": bool(outcome[0]), "kind": kind, "largest_difference": outcome[1]}

    @property
    def ok(self) -> bool:
        return all(entry["pass"] for entry in self.results.values())

    def report(self, rows: int) -> dict:
        kinds = {"sigma": "X {:.0e} sigma", "relative": "curves {:.0e} rel", "loglike": "log L {:.0e}"}
        largest = {kind: worst(entry.get("largest_difference") for entry in self.results.values() if entry["kind"] == kind)
                   for kind in kinds}
        present = [kinds[kind].format(largest[kind]) for kind in kinds
                   if any(entry["kind"] == kind for entry in self.results.values())]
        failed = [label for label, entry in self.results.items() if not entry["pass"]]
        accepted = [label for label, entry in self.results.items() if entry.get("accepted_by_flag")]
        status = (f"; FAILED: {', '.join(failed)}" if failed
                  else f"; exact items identical except {', '.join(accepted)} (documented defect of an intermediate bank, reported and accepted)"
                  if accepted else "; exact items identical")
        print(f"[banks] {self.name:34s} {'PASS' if self.ok else 'FAIL'}  {rows:3d} rows; " + ", ".join(present)
              + status + f"  ({time.time() - self.started:.0f} s)", flush=True)
        for note in self.notes:
            print(f"[banks]    {note}", flush=True)
        return {"pass": self.ok, "rows": rows, "largest": largest, "results": self.results, "notes": self.notes,
                "seconds": round(time.time() - self.started, 1)}


# ------------------------------------------------------------------ draws

def uniform_rows(seed: int, positions, low, high, transform=None) -> np.ndarray:
    """Rows of default_rng(seed).random((n, d)) mapped to the prior box (one 64-bit draw per coordinate)."""
    output = []
    for position in positions:
        generator = np.random.default_rng(seed)
        generator.bit_generator.advance(int(position) * len(low))
        unit = generator.random(len(low))
        output.append(transform(unit) if transform else low + unit * (high - low))
    return np.asarray(output)


def block_rows(seed: int, stream: int, blocks, positions, low, high) -> np.ndarray:
    """Rows of SeedSequence([seed, stream, block]) streams (enrichment and support screens)."""
    output = []
    for block, position in zip(blocks, positions):
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream, int(block)])))
        generator.bit_generator.advance(int(position) * len(low))
        output.append(low + generator.random(len(low)) * (high - low))
    return np.asarray(output)


def nucleonic_draws(stream, block, position) -> np.ndarray:
    plugin = get_eos("ddb")
    low, high = np.asarray(plugin.prior_low), np.asarray(plugin.prior_high)
    output = np.zeros((len(stream), 7))
    for index, (label, number, place) in enumerate(zip(stream, block, position)):
        if label in (0, 1):
            output[index] = uniform_rows((1234, 2026)[label], [place], low, high, plugin.prior_transform)[0]
        elif label in (2, 3, 4):
            output[index] = block_rows(20260729, label - 1, [number], [place], low, high)[0]
        else:  # Green EN S_ext stream
            output[index] = block_rows(20260924, 1, [number], [place], low, high)[0]
    return output


# ------------------------------------------------------------------ nucleonic forward

def corner(values, radius, mass, beta, interval, window):
    predicted = np.column_stack([np.ones(len(values)), values]) @ np.asarray(beta)
    return ((predicted >= interval[0]) & (predicted <= interval[1]) & np.isfinite(radius) & np.isfinite(mass)
            & (radius >= window[0]) & (radius <= window[1]) & (mass >= window[2]))


def memberships(prediction):
    centres = np.asarray([nuclear_observation(name) for name in CASE_SHIFTS])
    boxes = np.all(np.abs((prediction[:, None, :] - centres[None]) / A1_SIGMA) <= 3.0, axis=2)
    region = np.any(boxes, axis=1)
    return boxes[:, 0], region & ~boxes[:, 0]


def nucleonic_forward(theta, targets) -> dict:
    """One EOS/TOV solution per row, then every stored quantity with the production functions."""
    plugin = get_eos("ddb")
    prediction = build_bank.nuclear_predictions(theta, len(theta))
    density, energy, pressure = plugin.core_eos_batch(theta)
    rows = len(theta)
    out = {"X": prediction, "Rg": np.full((rows, 200), np.nan), "Lg": np.full((rows, 200), np.nan),
           "MM": np.full(rows, np.nan), "PQG": np.zeros((rows, 5)), "R14": np.full(rows, np.nan),
           "exact": np.full((rows, 6), np.nan), "j0740_single_weight": np.full(rows, np.nan),
           "components": np.full((rows, 4), np.nan), "NIC": np.full((rows, 3), np.nan), "Rmm": np.full(rows, np.nan)}
    legacy = targets["legacy"]
    target = targets["ddb"].target
    for row in range(rows):
        result = build_bank.stellar_row(energy[row], pressure[row], density[row], build_bank.MASS_GRID,
                                        build_bank.PQCD_DENSITY_GRID)
        if result is not None:
            out["Rg"][row], out["Lg"][row], out["MM"][row], out["PQG"][row], out["R14"][row] = result
        out["exact"][row] = build_bank.exact_row(density[row], energy[row], pressure[row], legacy.nicer_grids,
                                                 legacy.gw_kernel, legacy.gw_chirp_mass)
        try:
            branch = solve_stable_branch(energy[row], pressure[row])
        except (ValueError, FloatingPointError, RuntimeError):
            continue
        terms = target.evaluate(theta[row], prediction[row, 1:], density[row], energy[row], pressure[row], branch)
        out["components"][row] = (terms.maximum_mass, terms.nicer, terms.gw170817, terms.pqcd)
        out["NIC"][row] = [log_likelihood_one(branch.mass, branch.radius, branch.maximum_mass, grid)
                           for grid in target.nicer_interpolators]
        out["j0740_single_weight"][row] = out["NIC"][row, 1]
        out["Rmm"][row] = branch.radius[-1]
    out["GWG"] = build_bank.shifted_gw_grid(out["Lg"], out["MM"], targets["gw3500"], targets["gw3500_chirp"])
    out["LA"] = np.where(np.isfinite(out["components"]).all(1), out["components"].sum(1), np.nan)
    return out


def counting_flags(prediction, r14, mass, constants):
    in_b = np.all(np.abs(prediction - np.asarray(constants["OBS"])) <= constants["cbox"] * np.asarray(constants["SIG"]),
                  axis=1)
    in_c = corner(prediction, r14, mass, constants["beta_r14"], constants["predC_int"], constants["winC"])
    in_d = corner(prediction, r14, mass, constants["beta_r14"], constants["predD_int"], constants["winD"])
    return in_b, in_c, in_d


def check_nucleonic(fix, expected, targets) -> tuple[dict, dict]:
    reports = {}
    constants = expected["nucleonic_prior_bank"]["constants"]
    n0, mb, mc, md = constants["N0"], constants["MtotB"], constants["MtotC"], constants["MtotD"]

    # Shared prior physics bank and the row-aligned exact table.
    item = Item("nucleonic_prior_bank")
    p = "nuc_prior__"
    theta = fix[p + "theta"]
    item.exact("draws", same(nucleonic_draws(fix[p + "stream"], fix[p + "block"], fix[p + "position"]), theta))
    item.exact("GW_chirp_mass", float(targets["gw3500_chirp"]) == constants["Mc_obs"])
    forward = nucleonic_forward(theta, targets)
    item.value("X", sigma_gate(forward["X"], fix[p + "X"]), "sigma")
    item.exact("rho0_column", same(forward["X"][:, 0], fix[p + "X"][:, 0]))
    for key in ("Rg", "Lg", "MM", "R14"):
        item.value(key, relative(forward[key], fix[p + key]), "relative")
    item.value("PQG", loglike(forward["PQG"], fix[p + "PQG"]), "loglike")
    item.value("GWG", loglike(forward["GWG"], fix[p + "GWG"]), "loglike")
    in_b, in_c, in_d = counting_flags(forward["X"], forward["R14"], forward["MM"], constants)
    item.exact("support_flags", same(in_b, fix[p + "in_SB"]) and same(in_c, fix[p + "in_SC"]) and same(in_d, fix[p + "in_SD"]))
    logw = np.log(n0) - np.log(n0 + mb * in_b.astype(np.float64) + mc * in_c + md * in_d)
    difference = float(np.max(np.abs(logw - fix[p + "logw_prior"])))
    item.value("logw_prior", (difference <= TOLERANCE["logw_prior"], difference), "counting")
    reports["nucleonic_prior_bank"] = item.report(len(theta))

    item = Item("nucleonic_exact_table")
    e = "nuc_exact__"
    for column, key in enumerate(("exact_j0030", "exact_j0740", "exact_j0437", "exact_gw")):
        item.value(key, loglike(forward["exact"][:, column], fix[e + key]), "loglike")
    item.value("Rmm", relative(forward["exact"][:, 4], fix[e + "Rmm"]), "relative")
    item.value("Mmax_fresh", relative(forward["exact"][:, 5], fix[e + "Mmax_fresh"]), "relative")
    ordinary = np.isfinite(fix[e + "exact_j0740"]) & (fix[e + "exact_j0740"] > TOLERANCE["floor"])
    shift = float(np.max(np.abs(forward["j0740_single_weight"][ordinary] - fix[e + "exact_j0740"][ordinary])))
    item.notes.append(f"stored J0740 column = double-weight term (replayed with the double-weight grid); the "
                      f"single-weight term used downstream differs by up to {shift:.3f} in log L on these rows")
    reports["nucleonic_exact_table"] = item.report(len(theta))

    # Uniform-prior support bank (rows 0..659,999 of the shared bank; theta, X, MM).
    item = Item("nucleonic_support_bank")
    p = "nuc_support__"
    rows = fix[p + "row"]
    stream = (rows >= 200_000).astype(np.int64)
    item.exact("draws", same(nucleonic_draws(stream, -np.ones(len(rows)), rows - 200_000 * stream), fix[p + "theta"]))
    support = nucleonic_forward(fix[p + "theta"], targets)
    item.value("X", sigma_gate(support["X"], fix[p + "X"]), "sigma")
    item.value("MM", relative(support["MM"], fix[p + "MM"]), "relative")
    reports["nucleonic_support_bank"] = item.report(len(rows))

    # Independent common-physics cache (solved rows of the shared bank; X, LA, LPC).
    item = Item("nucleonic_independent_cache")
    p = "nuc_cache__"
    theta = fix[p + "theta"]
    item.exact("draws", same(nucleonic_draws(fix[p + "stream"], fix[p + "block"], fix[p + "position"]), theta))
    prediction, log_astro, _ = targets["ddb"].evaluate_evidence_cache_batch(theta)
    item.value("X", sigma_gate(prediction, fix[p + "X"]), "sigma")
    item.value("LA", loglike(log_astro, fix[p + "LA"]), "loglike")
    cache_forward = nucleonic_forward(theta, targets)
    in_b, in_c, in_d = counting_flags(cache_forward["X"], cache_forward["R14"], cache_forward["MM"], constants)
    item.exact("support_flags", same(in_b, fix[p + "in_SB"]) and same(in_c, fix[p + "in_SC"]) and same(in_d, fix[p + "in_SD"]))
    lpc = np.log(n0) - np.log(n0 + mb * in_b.astype(np.float64) + mc * in_c + md * in_d)
    difference = float(np.max(np.abs(lpc - fix[p + "LPC"])))
    item.value("LPC", (difference <= TOLERANCE["logw_prior"], difference), "counting")
    reports["nucleonic_independent_cache"] = item.report(len(theta))
    return reports, forward


def check_green_nucleonic(fix, expected, targets) -> dict:
    item = Item("green_en_nucleonic_bank")
    constants = expected["nucleonic_prior_bank"]["constants"]
    p = "en_nuc__"
    theta = fix[p + "theta64"]
    shared = fix[p + "kind"] == 0
    item.exact("draws", same(nucleonic_draws(fix[p + "stream"], fix[p + "block"], fix[p + "position"]), theta))
    item.exact("theta_float32", same(theta.astype(np.float32), fix[p + "theta"]))
    forward = nucleonic_forward(theta, targets)
    item.value("X", sigma_gate(forward["X"], fix[p + "X"]), "sigma")
    item.value("R", relative(forward["Rg"], fix[p + "R"]), "relative")
    item.value("MM", relative(forward["MM"], fix[p + "MM"]), "relative")
    item.value("LA", loglike(forward["LA"], fix[p + "LA"]), "loglike")
    item.value("NIC", loglike(forward["NIC"], fix[p + "NIC"]), "loglike")
    # AC: [max-mass term, NICER sum, GW, pQCD]; for shared-bank rows the stored first column is R(Mmax) and the
    # fourth the remainder LA - R(Mmax) - NICER - GW (as the assembler builds it).
    components = forward["components"].copy()
    components[shared, 0] = forward["Rmm"][shared]
    components[shared, 3] = (forward["LA"][shared] - forward["Rmm"][shared] - forward["NIC"][shared].sum(1)
                             - forward["components"][shared, 2])
    item.value("AC_radius_column", relative(components[shared, 0], fix[p + "AC"][shared, 0]), "relative")
    item.value("AC", loglike(np.column_stack([components[:, 1:], np.where(shared, 0.0, components[:, 0])]),
                             np.column_stack([fix[p + "AC"][:, 1:], np.where(shared, 0.0, fix[p + "AC"][:, 0])])), "loglike")
    # Counting correction: shared-bank flags plus the S_ext box (all rows), with the stored counts.
    in_b, in_c, in_d = counting_flags(forward["X"], forward["R14"], forward["MM"], constants)
    item.exact("support_flags", same(in_b, fix[p + "in_SB"]) and same(in_c, fix[p + "in_SC"]) and same(in_d, fix[p + "in_SD"]))
    b3, sext = memberships(forward["X"])
    n0 = constants["N0"]
    lpc = math.log(n0) - np.log(n0 + constants["MtotB"] * in_b.astype(np.float64) + constants["MtotC"] * in_c
                                + constants["MtotD"] * in_d + SEXT_PROPOSALS * sext.astype(np.float64))
    item.exact("LPC", same(lpc, fix[p + "LPC"]))
    item.exact("sext_rows_outside_B3", bool(not b3[~shared].any() and sext[~shared].all()))
    return item.report(len(theta))


# ------------------------------------------------------------------ hyperonic

def training_forward(theta, target, mass_grid) -> dict:
    plugin = get_eos("ddb-hyperonic")
    prediction = np.empty((len(theta), 7))
    prediction[:, 0] = theta[:, plugin.parameter_names.index("rho0")]
    prediction[:, 1:] = plugin.nuclear_observables_batch(theta)
    rows = [training_cache.evaluate_training_row("ddb-hyperonic", theta[i], prediction[i, 1:], target, mass_grid)
            for i in range(len(theta))]
    out = {"prediction": prediction, "log_astrophysical": np.array([r[0] for r in rows]),
           "astrophysical_components": np.array([r[1] for r in rows]),
           "nicer_source_components": np.array([r[2] for r in rows]), "radius": np.array([r[3] for r in rows]),
           "tidal_lambda": np.array([r[4] for r in rows]), "maximum_mass": np.array([r[5] for r in rows])}
    out["target_valid"] = (np.isfinite(out["log_astrophysical"]) & (out["log_astrophysical"] > -1.0e29)
                           & np.isfinite(out["nicer_source_components"]).all(1) & np.isfinite(out["maximum_mass"])
                           & np.isfinite(out["radius"]).any(1) & np.isfinite(out["tidal_lambda"]).any(1))
    return out


def compare_training(item: Item, forward: dict, fix: dict, prefix: str) -> None:
    item.value("prediction", sigma_gate(forward["prediction"], fix[prefix + "prediction"]), "sigma")
    item.exact("target_valid", same(forward["target_valid"], fix[prefix + "target_valid"]))
    for key in ("log_astrophysical", "astrophysical_components", "nicer_source_components"):
        item.value(key, loglike(forward[key], fix[prefix + key]), "loglike")
    for key in ("radius", "tidal_lambda", "maximum_mass"):
        item.value(key, relative(forward[key], fix[prefix + key]), "relative")


def check_hyperonic(fix, expected, targets) -> dict:
    reports = {}
    certified = uniform_bank.load_certified_forward()
    low, high = uniform_bank.extended_prior_bounds(certified)
    plugin = get_eos("ddb-hyperonic")
    portable_low, portable_high = np.asarray(plugin.prior_low), np.asarray(plugin.prior_high)
    same_box = same(low, portable_low) and same(high, portable_high)
    seeds = {part: expected[f"hyperonic_uniform_bank_part{part}"]["seed"] for part in (1, 2, 3, 4)}
    for part in (1, 2, 3, 4):
        item = Item(f"hyperonic_uniform_bank_part{part}")
        p = f"hyp_bank{part}__"
        theta = fix[p + "theta"]
        item.exact("draws", same(uniform_rows(seeds[part], fix[p + "row"], low, high), theta))
        item.exact("prior_box_equal_portable_plugin", same_box)
        results = [uniform_bank.solve_bank_row(int(row), theta[i], prior_low=low, prior_high=high)
                   for i, row in enumerate(fix[p + "row"])]
        prediction, _ = uniform_bank.compute_certified_nmp(theta, certified, chunk_size=len(theta))
        item.value("X", sigma_gate(prediction, fix[p + "X"]), "sigma")
        item.exact("failure_code", same([r.failure_code for r in results], fix[p + "failure_code"]))
        item.exact("valid", same([r.failure_code == 0 for r in results], fix[p + "valid"]))
        item.value("Rg", relative(np.array([r.radius_grid for r in results]), fix[p + "Rg"]), "relative")
        item.value("Lg", relative(np.array([r.lambda_grid for r in results]), fix[p + "Lg"]), "relative")
        item.value("MM", relative([r.maximum_mass for r in results], fix[p + "MM"]), "relative")
        item.value("R14", relative([r.radius_14 for r in results], fix[p + "R14"]), "relative")
        reports[f"hyperonic_uniform_bank_part{part}"] = item.report(len(theta))

    target = targets["ddb-hyperonic"].target
    mass_grid = uniform_bank.MASS_GRID
    student = joblib.load(STUDENT)
    student_draws = None
    for kind, label, parts in (("u", "uniform", (1, 2, 3, 4)), ("s", "support", (1, 2)), ("p", "proposal", (1, 2, 3))):
        for part in parts:
            item = Item(f"hyperonic_{label}_cache_part{part}")
            p = f"hyp_cache_{kind}{part}__"
            theta = fix[p + "theta"]
            if kind == "u":
                draws = uniform_rows(seeds[part], fix[p + "source_index"], low, high)
            elif kind == "s":
                draws = block_rows(20260729, 1, fix[p + "block"], fix[p + "position"], portable_low, portable_high)
            else:
                if student_draws is None:
                    student_draws = sample_checkpoint(student, 300_000, np.random.default_rng(20260992))
                draws = student_draws[fix[p + "source_index"]]
                prior_logq = -float(np.log(portable_high - portable_low).sum())
                difference = float(np.max(np.abs(checkpoint_density(student, theta, prior_logq) - fix[p + "proposal_logq"])))
                item.value("proposal_logq", (difference <= TOLERANCE["student_logq"], difference), "density")
            if kind == "p":  # the Student draws were made on another machine: BLAS rounding in its affine maps
                difference = float(np.max(np.abs(draws - theta)))
                item.value("draws", (difference <= TOLERANCE["student_draw"], difference), "draw")
            else:
                item.exact("draws", same(draws, theta))
            compare_training(item, training_forward(theta, target, mass_grid), fix, p)
            reports[f"hyperonic_{label}_cache_part{part}"] = item.report(len(theta))
    return reports


def check_green_hyperonic(fix, expected, targets) -> dict:
    item = Item("green_en_hyperonic_bank")
    p = "en_hyp__"
    theta = fix[p + "theta64"]
    plugin = get_eos("ddb-hyperonic")
    low, high = np.asarray(plugin.prior_low), np.asarray(plugin.prior_high)
    draws = np.zeros_like(theta)
    for index, (kind, seed, block, position) in enumerate(fix[p + "draw"]):
        draws[index] = (uniform_rows(seed, [position], low, high)[0] if kind == 0
                        else block_rows(seed, 1, [block], [position], low, high)[0])
    item.exact("draws", same(draws, theta))
    item.exact("theta_float32", same(theta.astype(np.float32), fix[p + "theta"]))
    forward = training_forward(theta, targets["ddb-hyperonic"].target, uniform_bank.MASS_GRID)
    valid = forward["target_valid"]
    item.value("X", sigma_gate(forward["prediction"], fix[p + "X"]), "sigma")
    item.value("R", relative(forward["radius"], fix[p + "R"]), "relative")
    item.value("MM", relative(forward["maximum_mass"], fix[p + "MM"]), "relative")
    item.exact("rejected_rows", same(~valid, ~np.isfinite(fix[p + "LA"])))
    item.value("LA", loglike(np.where(valid, forward["log_astrophysical"], np.nan),
                             np.where(np.isfinite(fix[p + "LA"]), fix[p + "LA"], np.nan)), "loglike")
    item.value("AC", loglike(forward["astrophysical_components"], fix[p + "AC"]), "loglike")
    item.value("NIC", loglike(forward["nicer_source_components"], fix[p + "NIC"]), "loglike")
    parent = fix[p + "kind"] < 3  # the assembler read the parent rows' X back from float32
    x = np.where(parent[:, None], forward["prediction"].astype(np.float32).astype(np.float64), forward["prediction"])
    b3, sext = memberships(x)
    lpc = math.log(600_000) - np.log(600_000 + 320_000_000 * b3.astype(np.float64) + SEXT_PROPOSALS * sext.astype(np.float64))
    item.exact("LPC", same(lpc, fix[p + "LPC"]))
    return item.report(len(theta))


def check_stage_a(fix, expected, targets, accept: bool) -> dict:
    item = Item("hyperonic_stage_a_mis")
    p = "stage_a__"
    theta, stored, in_prefix = fix[p + "theta"], fix[p + "logl"], fix[p + "in_prefix"]
    data = targets["legacy"]
    with contextlib.redirect_stdout(io.StringIO()):
        forwards = [legacy_shard._fast_forward(row) for row in theta]
    nuclear = legacy_shard._nuclear_loglike(theta, data)
    astro = np.asarray([legacy_target._astro_loglike_from_forward(forward, data) for forward in forwards])
    replayed = astro + nuclear
    replayed[~np.isfinite(replayed)] = legacy_target.LOG_LIKELIHOOD_FLOOR
    rejected = stored <= -1.0e50
    ordinary = stored > TOLERANCE["floor"]
    floor_rows = ~rejected & ~ordinary
    difference = float(np.max(np.abs(replayed[ordinary] - stored[ordinary])))
    item.value("ordinary_log_likelihood", (bool(np.all(replayed[ordinary] > TOLERANCE["floor"]))
                                           and difference <= TOLERANCE["log_likelihood"], difference), "loglike")
    floor_difference = float(np.max(np.abs(replayed[floor_rows] - stored[floor_rows]) / np.abs(stored[floor_rows])))
    item.exact("floor_rows", bool(np.all(replayed[floor_rows] <= TOLERANCE["floor"])) and floor_difference <= 1e-12)
    item.exact("rejected_rows_outside_the_six_prefixes", bool(np.all(replayed[rejected & ~in_prefix] <= -1.0e50)))
    inside = rejected & in_prefix
    solved = int(np.sum(replayed[inside] > -1.0e50))
    item.notes.append(f"KNOWN DEFECT of the stored bank: {solved} of {int(inside.sum())} replayed rows that the bank "
                      f"stores as rejected inside the six zero-ordinary-value prefixes are evaluated by the shipped "
                      f"code (largest replayed log L {np.max(replayed[inside]):.1f}); see docs/route2/STEPS_BANKS.md")
    prefixes = expected["hyperonic_stage_a_mis"]["zero_ordinary_value_prefixes"]
    defect_rows = sum(count for prefix in prefixes for key, count in prefix["stored_values"].items() if float(key) <= -1e50)
    item.notes.append(f"the six prefixes hold {sum(p_['rows'][1] - p_['rows'][0] for p_ in prefixes)} rows, "
                      f"{defect_rows} of them stored as rejected")
    item.results["rejected_rows_inside_the_six_prefixes"] = {"pass": solved == 0 or accept, "kind": "documented_defect",
                                                             "replayed_as_evaluated": solved,
                                                             "accepted_by_flag": bool(accept and solved)}
    return item.report(len(theta))


# ------------------------------------------------------------------ main

def build_targets() -> dict:
    NUCLEAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NUCLEAR_FILE, OBS=A1_OBSERVATION, SIG=A1_SIGMA)
    started = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        targets = {model: A1Problem.from_data_root(OBSERVATIONS, workers=1, verify_data=True, model=model)
                   for model in ("ddb", "ddb-hyperonic")}
        targets["legacy"] = legacy_target.build_a1_likelihood_data(verbose=False)
        kernel, chirp = load_kde(OBSERVATIONS / "GW170817_GWTC-1.hdf5", subsample=3500, seed=0)
    targets["gw3500"], targets["gw3500_chirp"] = kernel, chirp
    print(f"[banks] targets rebuilt from {OBSERVATIONS.relative_to(ROOT)} (single-weight and double-weight NICER grids) "
          f"in {time.time() - started:.0f} s", flush=True)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true",
                        help="also fail on the documented Stage-A shard defect (by default it is reported and accepted)")
    parser.add_argument("--device", default="cpu", help="accepted for route2/run.py; the bank physics always runs on CPU")
    parser.add_argument("--report", type=Path, default=ROOT / "build/route2/banks_check.json")
    arguments = parser.parse_args()
    started = time.time()
    prepare_observations()
    manifest = json.loads((FIX / "MANIFEST.json").read_text())
    for entry in manifest["files"]:
        if sha256(ROOT / entry["file"]) != entry["sha256"]:
            raise SystemExit(f"FAIL: {entry['file']} differs from route2/fixtures/banks/MANIFEST.json")
    expected = json.loads((FIX / "expected.json").read_text())
    if sha256(STUDENT) != expected["hyperonic_student_checkpoint"]["sha256"]:
        raise SystemExit("FAIL: the Student checkpoint fixture is not the stored one (SHA-256 mismatch)")
    if not (np.array_equal(expected["nucleonic_prior_bank"]["constants"]["OBS"], A1_OBSERVATION)
            and np.array_equal(expected["nucleonic_prior_bank"]["constants"]["SIG"], A1_SIGMA)):
        raise SystemExit("FAIL: the stored nuclear observation differs from likelihoods/nuclear")
    with np.load(FIX / "rows.npz", allow_pickle=False) as data:
        fix = {key: data[key] for key in data.files}
    targets = build_targets()

    reports, _ = check_nucleonic(fix, expected, targets)
    reports["green_en_nucleonic_bank"] = check_green_nucleonic(fix, expected, targets)
    reports.update(check_hyperonic(fix, expected, targets))
    reports["hyperonic_stage_a_mis"] = check_stage_a(fix, expected, targets, not arguments.strict)
    reports["green_en_hyperonic_bank"] = check_green_hyperonic(fix, expected, targets)

    ok = all(report["pass"] for report in reports.values())
    defect = reports["hyperonic_stage_a_mis"]["results"]["rejected_rows_inside_the_six_prefixes"]
    rows = sum(report["rows"] for report in reports.values())
    summary = {"status": "PASS" if ok else "FAIL", "tolerances": TOLERANCE, "rows_replayed": rows,
               "seconds": round(time.time() - started, 1), "documented_defect": defect,
               "bank_hashes": {name: {"file": record["file"], "sha256": record["sha256"]}
                               for name, record in expected.items() if isinstance(record, dict) and "sha256" in record},
               "banks": reports}
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(summary, indent=1) + "\n")
    failed = [name for name, report in reports.items() if not report["pass"]]
    detail = (f"; documented defect of an intermediate bank: the Stage-A bridge stores as rejected rows that the shipped "
              f"code evaluates ({defect['replayed_as_evaluated']} replayed rows; no paper result depends on it, see "
              f"docs/route2/STEPS_BANKS.md{'; accepted' if defect['accepted_by_flag'] else '; --strict: counted as a failure'})"
              if defect["replayed_as_evaluated"] else "")
    print(f"[banks] {summary['status']}: {len(reports)} bank items, {rows} rows replayed in {summary['seconds']} s"
          + (f"; failed: {', '.join(failed)}" if failed else "") + detail
          + f" (report: {arguments.report.relative_to(ROOT) if arguments.report.is_relative_to(ROOT) else arguments.report})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
