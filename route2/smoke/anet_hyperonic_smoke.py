#!/usr/bin/env python3
"""Route 2 smoke run: hyperonic A-NET (dual parity pipeline) end to end at toy size, on CPU (about 3-4 minutes
on 3 cores).

What runs. Each stage is the unchanged production program in hyperonic_pipeline/ (made importable with the
legacy modules by route2.common.use_hyperonic_pipeline), started as its own process with the production
settings except the sizes:

  1. bank          generate_ddbhy_bank.py: 48 uniform-prior rows, seed 91 (production: 4 x 150,000 rows,
                   seeds 91-94).
  2. uniform cache build_clean_uniform_hyperonic_training_cache.py: exact target factors and fresh curves for
                   every forward-valid bank row (production: the forward-valid rows of the four banks, 62 workers).
  3. support       screen_clean_hyperonic_support_enrichment.py --proposals 40000 (production 160,000,000,
                   seed 20260729, cbox 3), then evaluate_clean_hyperonic_support_cache.py on every screened row.
  4. Student cache clean_student_t_ensemble.py sample: 24 draws (production 300,000, seed 20260992) from the
                   frozen defensive Student checkpoint, then evaluate_clean_hyperonic_proposal_cache.py.
  5. correction    build_clean_hyperonic_augmented_correction.py: the exact counting correction over the three
                   caches, with --base-proposals 48 and --conditional-ess-gate 1 (production 600,000 and 2,000;
                   an ESS of 2,000 cannot exist in a 75-row bank, so this gate is not applied here).
  6. scenarios     build_clean_hyperonic_parity_scenarios.py --smoke --seed 9: 12 scenarios (3 validation,
                   2 identity), at most 200 posterior rows each (production 2,000 scenarios, 6,000 rows).
  7. training      train_clean_hyperonic_parity_anet.py --smoke --seed 2: the nine-dimensional head, 2 epochs
                   (production 600 epochs on a GPU). The new checkpoint is then loaded in the production sampler
                   class and draws 8 rows (32 Heun steps) with finite log densities.
  8. dual sampler  sample_clean_hyperonic_dual_parity_mixture.py: 64 draws with the production weights
                   0.35/0.55/0.10, 1,024 Heun steps, cloud seed 777001 and proposal seed 20260944, from the FROZEN
                   production heads and Student (route2/fixtures/hyperonic_anet/networks), because the full-size
                   training is impossible here and the sampler requires both heads from one scenario bank.
  9. exact IS      build_portable_hyperonic_target_cache.py (A1 target), split_clean_proposal.py (one shard),
                   fast_portable_hyperonic_likelihood.py (1 worker), assemble_likelihood_shards.py.
 10. certificate   certify_clean_anet_scenario.py with the production gates (ESS >= 1000, largest normalized
                   weight <= 0.01). With 64 draws the gates are not expected to pass; the smoke checks that they
                   are applied consistently, not that they pass.

No program needs an edit; every setting above is a command-line option. The command lines are in the report.

What is asserted (no scientific claim is made): every output exists with the documented schema, keys and
shapes; parameters are finite and inside the nine-dimensional prior box; the exact target terms are finite on
every target-valid row; the correction and scenario archives are tied to their caches by hash; the proposal
log density equals the deterministic mixture of its three component densities (1e-12); the certificate's
weights, ESS, log Z and gate decision are recomputed from the proposal and exact log-likelihood (1e-12).

The frozen Student checkpoint was pickled with scikit-learn 1.9.1; with 1.9.0 it loads with a version warning
(the hyperonic A-NET check replays its density to 1e-14).

CPU use: the process and every stage are pinned to 3 CPU cores (the least busy ones), because the JAX CPU
backend ignores OMP_NUM_THREADS; OMP/MKL/OpenBLAS/numba and torch use 3 threads, all worker counts are 1, and
CUDA_VISIBLE_DEVICES is empty. Outputs: build/route2/smoke/anet_hyperonic/ (stage outputs and logs, replaced on
every run) and the report build/route2/smoke/anet_hyperonic_smoke.json. Exit status 0 (PASS) or 1 (FAIL).

Run from the repository root:  PYTHONPATH=$PWD python -m route2.smoke.anet_hyperonic_smoke
"""
from __future__ import annotations

import os

# Before numpy/torch/JAX are imported: no GPU, JAX on CPU, and 3 threads in this process. The stage processes get
# --threads threads (default 3).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_variable] = "3"
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from route2.common import OBSERVATIONS, ROOT, prepare_observations, sha256, use_hyperonic_pipeline  # noqa: E402

NAME = "anet_hyperonic"
TAG = "[smoke-anet-hyperonic]"
WORK = ROOT / "build/route2/smoke" / NAME
REPORT = ROOT / "build/route2/smoke" / f"{NAME}_smoke.json"
PIPELINE = ROOT / "hyperonic_pipeline"
NETWORKS = ROOT / "route2/fixtures/hyperonic_anet/networks"
STUDENT = NETWORKS / "student/student_beta1_48c_df5_s075_seed20260990.joblib"
HELD_OUT = ["J0614", "J1231", "J1614"]
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
# Runs one production program exactly as `python <script> <arguments>` would, after capping torch threads.
BOOTSTRAP = """
import os, runpy, sys
script, use_torch = sys.argv[1], sys.argv[2] == "1"
if use_torch:
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
sys.argv = [script, *sys.argv[3:]]
sys.path[0] = os.path.dirname(os.path.abspath(script))
runpy.run_path(script, run_name="__main__")
"""


# ----------------------------------------------------------------------------------------------------------
# Helpers shared by the three smoke runs (each script is self-contained).

def rel(path: Path) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_absolute() and path.is_relative_to(ROOT) else str(path)


def pin_cpus(count: int) -> list[int]:
    """Restrict this process (and so every stage process) to `count` lightly loaded CPU cores (Linux)."""
    if not hasattr(os, "sched_setaffinity"):
        return []
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) <= count:
        return allowed

    def sample() -> dict[int, tuple[int, int]]:
        times = {}
        with open("/proc/stat") as stream:
            for line in stream:
                name, *fields = line.split()
                if name.startswith("cpu") and name[3:].isdigit():
                    values = [int(value) for value in fields]
                    times[int(name[3:])] = (sum(values), values[3] + values[4])
        return times

    def read(cpu: int, name: str) -> str:
        return Path(f"/sys/devices/system/cpu/cpu{cpu}/{name}").read_text().strip()

    def siblings(cpu: int) -> set[int]:
        """Hardware threads sharing a physical core with `cpu`."""
        members: set[int] = set()
        try:
            for part in read(cpu, "topology/thread_siblings_list").split(","):
                first, _, last = part.partition("-")
                members.update(range(int(first), int(last or first) + 1))
        except (OSError, ValueError):
            pass
        return members - {cpu}

    def top_frequency(cpu: int) -> int:
        try:
            return int(read(cpu, "cpufreq/cpuinfo_max_freq"))
        except (OSError, ValueError):
            return 0

    try:
        before = sample()
        time.sleep(0.5)
        after = sample()
        busy = {cpu: 1.0 - (after[cpu][1] - before[cpu][1]) / max(after[cpu][0] - before[cpu][0], 1)
                for cpu in allowed if cpu in before and cpu in after}
        pairs = {cpu: siblings(cpu) for cpu in busy}
        frequency = {cpu: top_frequency(cpu) for cpu in busy}
        fastest = max(frequency.values())
        chosen: list[int] = []
        # Idle performance cores first (highest top frequency; on hybrid CPUs the efficiency cores are much slower
        # for this code), then idle other cores, then busy ones; within a class, the fewest busy sibling threads,
        # and never both threads of one physical core while another core is available.
        for _ in range(min(count, len(busy))):
            chosen.append(min((cpu for cpu in busy if cpu not in chosen),
                              key=lambda cpu: (busy[cpu] > 0.5, frequency[cpu] < 0.9 * fastest,
                                               any(other in chosen for other in pairs[cpu]),
                                               busy[cpu] + 0.5 * sum(busy.get(other, 0.0) for other in pairs[cpu]),
                                               cpu)))
        chosen.sort()
    except (OSError, ValueError, KeyError):
        chosen = allowed[:count]
    os.sched_setaffinity(0, chosen)
    return chosen


def stage_environment(threads: int, extra_paths: tuple[Path, ...] = ()) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["JAX_PLATFORMS"] = "cpu"
    for variable in THREAD_VARIABLES:
        environment[variable] = str(threads)
    paths = [str(ROOT), *(str(path) for path in extra_paths)]
    paths += [entry for entry in environment.get("PYTHONPATH", "").split(os.pathsep) if entry and entry not in paths]
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


class StageFailure(RuntimeError):
    pass


class Stage:
    """One smoke stage: runs its production programs, then collects assertion failures and key numbers."""

    def __init__(self, name: str, steps: list[tuple[Path, list]], *, uses_torch: bool = False, note: str = ""):
        self.name, self.uses_torch, self.note = name, uses_torch, note
        self.steps = [(script, [rel(value) if isinstance(value, Path) else str(value) for value in arguments])
                      for script, arguments in steps]
        self.failures: list[str] = []
        self.numbers: dict = {}
        self.seconds = 0.0

    @property
    def commands(self) -> list[str]:
        return [" ".join(["python", rel(script), *arguments]) for script, arguments in self.steps]

    def run(self, environment: dict[str, str], logs: Path, first: int = 0) -> None:
        started = time.time()
        for number, (script, arguments) in enumerate(self.steps, start=first):
            log = logs / f"{self.name}_{number}_{script.stem}.log"
            command = [sys.executable, "-c", BOOTSTRAP, str(script), "1" if self.uses_torch else "0", *arguments]
            with log.open("w") as stream:
                code = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                      stderr=subprocess.STDOUT).returncode
            if code != 0:
                self.seconds = time.time() - started
                tail = "\n    ".join(log.read_text(errors="replace").splitlines()[-12:])
                raise StageFailure(f"{rel(script)} exited with status {code}; end of {rel(log)}:\n    {tail}")
        self.seconds = time.time() - started

    def require(self, condition, message: str) -> None:
        if not bool(condition):
            self.failures.append(message)

    def record(self) -> dict:
        return {"pass": not self.failures, "seconds": round(self.seconds, 1), "commands": self.commands,
                "note": self.note, "failures": self.failures, "numbers": self.numbers}


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def shaped(stage: Stage, arrays: dict, shapes: dict, label: str) -> None:
    for key, shape in shapes.items():
        if key not in arrays:
            stage.failures.append(f"{label}: missing {key}")
        elif arrays[key].shape != shape:
            stage.failures.append(f"{label}: {key} has shape {arrays[key].shape}, expected {shape}")


def inside(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    return bool(np.isfinite(values).all() and np.all((values >= low) & (values <= high)))


def text(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def clean_lineage(arrays: dict) -> bool:
    return (not bool(arrays.get("reference_present", False))
            and np.asarray(arrays.get("forbidden_artifacts_used", np.asarray([]))).size == 0)


# ----------------------------------------------------------------------------------------------------------

def check_cache(stage: Stage, path: Path, schema: str, rows: int, low, high, label: str) -> dict:
    cache = load(path)
    grid = len(cache["mass_grid"]) if "mass_grid" in cache else 0
    shaped(stage, cache, {"theta": (rows, 9), "prediction": (rows, 7), "log_astrophysical": (rows,),
                          "astrophysical_components": (rows, 4), "nicer_source_components": (rows, 3),
                          "target_valid": (rows,), "radius": (rows, grid), "maximum_mass": (rows,)}, label)
    if stage.failures:
        return cache
    valid = cache["target_valid"]
    stage.require(text(cache["schema"]) == schema, f"{label}: schema is not {schema}")
    stage.require(clean_lineage(cache), f"{label}: declares a reference or forbidden ancestry")
    stage.require(inside(cache["theta"], low, high), f"{label}: parameters are not finite inside the prior box")
    stage.require(np.isfinite(cache["prediction"]).all(), f"{label}: nuclear predictions are not finite")
    stage.require(valid.any(), f"{label}: no row is target-valid")
    stage.require(np.isfinite(cache["log_astrophysical"][valid]).all()
                  and np.isfinite(cache["astrophysical_components"][valid]).all()
                  and np.isfinite(cache["nicer_source_components"][valid]).all()
                  and np.isfinite(cache["maximum_mass"][valid]).all(),
                  f"{label}: target terms are not finite on every target-valid row")
    stage.require(np.max(np.abs(cache["nicer_source_components"][valid].sum(axis=1)
                                - cache["astrophysical_components"][valid, 1]), initial=0.0) <= 1e-10,
                  f"{label}: the per-source NICER terms do not add up to the NICER factor")
    return cache


def check_bank(stage: Stage, low, high, rows: int) -> None:
    bank = load(WORK / "uniform_bank.npz")
    shaped(stage, bank, {"theta": (rows, 9), "X": (rows, 7), "MG": (200,), "Rg": (rows, 200), "Lg": (rows, 200),
                         "MM": (rows,), "valid": (rows,), "THETA_LOW": (9,), "THETA_HIGH": (9,)}, "uniform_bank.npz")
    if stage.failures:
        return
    valid = bank["valid"]
    stage.require(text(bank["schema_version"]) == "ddbhy-bank-v1", "bank schema is not ddbhy-bank-v1")
    stage.require(np.array_equal(bank["THETA_LOW"], low) and np.array_equal(bank["THETA_HIGH"], high),
                  "bank prior box differs from the hyperonic EOS prior")
    stage.require(inside(bank["theta"], low, high) and int(bank["seed"]) == 91, "bank parameters or seed are wrong")
    stage.require(valid.any(), "no bank row is forward-valid")
    stage.require(np.isfinite(bank["X"][valid]).all() and np.isfinite(bank["MM"][valid]).all()
                  and np.isfinite(bank["Rg"][valid]).any(axis=1).all(),
                  "nuclear predictions, maximum masses or radius curves are not finite on valid rows")
    stage.numbers = {"rows": rows, "forward_valid_rows": int(valid.sum())}


def check_uniform_cache(stage: Stage, low, high) -> None:
    bank = load(WORK / "uniform_bank.npz")
    selected = int(bank["valid"].sum())
    cache = check_cache(stage, WORK / "uniform_cache.npz", "ddb-hyperonic-clean-uniform-training-cache-v1", selected,
                        low, high, "uniform_cache.npz")
    if stage.failures:
        return
    stage.require(np.array_equal(cache["theta"], bank["theta"][bank["valid"]]),
                  "the uniform cache does not hold exactly the forward-valid bank rows")
    stage.require(text(cache["source_bank_sha256"]) == sha256(WORK / "uniform_bank.npz"),
                  "the uniform cache does not record the bank hash")
    stage.numbers = {"rows": selected, "target_valid_rows": int(cache["target_valid"].sum())}


def check_support(stage: Stage, low, high, proposals: int) -> None:
    screen = load(WORK / "support_screen.npz")
    rows = len(screen["theta"])
    shaped(stage, screen, {"theta": (rows, 9), "prediction": (rows, 7)}, "support_screen.npz")
    if stage.failures:
        return
    stage.require(text(screen["schema"]) == "ddb-hyperonic-clean-support-screen-v1" and clean_lineage(screen),
                  "support screen schema or lineage is wrong")
    stage.require(int(screen["total_proposals"]) == proposals and rows >= 1,
                  "support screen proposal count is wrong or it accepted no row")
    standardized = (screen["prediction"] - screen["observation"]) / screen["sigma"]
    stage.require(inside(screen["theta"], low, high) and np.all(np.abs(standardized) <= float(screen["cbox"])),
                  "a screened row lies outside the prior box or the nuclear support box")
    cache = check_cache(stage, WORK / "support_cache.npz", "ddb-hyperonic-clean-support-training-cache-v1", rows,
                        low, high, "support_cache.npz")
    if stage.failures:
        return
    stage.require(np.array_equal(cache["theta"], screen["theta"])
                  and text(cache["screen_sha256"]) == sha256(WORK / "support_screen.npz"),
                  "the support cache is not the full screened set")
    stage.numbers = {"screened_proposals": proposals, "support_rows": rows,
                     "target_valid_rows": int(cache["target_valid"].sum())}


def check_student(stage: Stage, low, high, draws: int) -> None:
    proposal = load(WORK / "student_draws.npz")
    shaped(stage, proposal, {"theta": (draws, 9), "logq": (draws,)}, "student_draws.npz")
    if stage.failures:
        return
    stage.require(text(proposal["schema"]) == "ddb-hyperonic-clean-proposal-v1", "Student draw schema is wrong")
    stage.require(text(proposal["checkpoint_sha256"]) == sha256(STUDENT), "Student draws do not record the frozen checkpoint")
    stage.require(inside(proposal["theta"], low, high) and np.isfinite(proposal["logq"]).all(),
                  "Student draws or their log densities are not finite inside the prior box")
    cache = check_cache(stage, WORK / "proposal_cache.npz", "ddb-hyperonic-clean-proposal-training-cache-v1", draws,
                        low, high, "proposal_cache.npz")
    if stage.failures:
        return
    stage.require(np.array_equal(cache["theta"], proposal["theta"])
                  and float(cache["maximum_logq_replay_error"]) <= float(cache["density_tolerance"]),
                  "the proposal cache does not hold the Student draws or its density replay failed")
    stage.numbers = {"draws": draws, "target_valid_rows": int(cache["target_valid"].sum()),
                     "density_replay_error": float(cache["maximum_logq_replay_error"])}


def check_correction(stage: Stage, caches: list[Path]) -> None:
    correction = load(WORK / "correction.npz")
    rows = [len(load(path)["theta"]) for path in caches]
    shaped(stage, correction, {"log_prior_correction": (sum(rows),), "cache_rows": (3,)}, "correction.npz")
    if stage.failures:
        return
    stage.require(text(correction["schema"]) == "ddb-hyperonic-clean-augmented-counting-correction-v1"
                  and clean_lineage(correction), "correction schema or lineage is wrong")
    stage.require(list(correction["cache_sha256"]) == [sha256(path) for path in caches]
                  and list(correction["cache_rows"]) == rows, "correction is not tied to the three caches")
    stage.require(np.isfinite(correction["log_prior_correction"]).all()
                  and np.all(correction["log_prior_correction"] <= 1e-12),
                  "log counting correction is not finite and non-positive on every row")
    stage.require(np.isfinite(correction["conditional_ess"]) and float(correction["conditional_ess"]) >= 1.0,
                  "conditional ESS is not finite")
    stage.numbers = {"rows": sum(rows), "conditional_ess_at_A1": float(correction["conditional_ess"]),
                     "production_conditional_ess_gate_applied": False}


def check_scenarios(stage: Stage, low, high) -> None:
    path = WORK / "scenarios.npz"
    archive = load(path)
    rows = len(archive["theta"])
    shaped(stage, archive, {"clouds": (12, 3, 256, 2), "transforms": (12, 3, 4), "ess": (12,), "mode": (12,),
                            "theta_len": (12,), "theta": (rows, 9), "gtheta": (rows, 7)}, "scenarios.npz")
    if stage.failures:
        return
    metadata = json.loads(text(archive["metadata"]))
    lengths = archive["theta_len"]
    stage.require(clean_lineage(archive) and metadata.get("held_out_sources_absent") == HELD_OUT,
                  "scenario archive lineage or held-out declaration is wrong")
    stage.require(metadata["counting_correction"]["sha256"] == sha256(WORK / "correction.npz"),
                  "scenario archive does not record the correction hash")
    stage.require(int(archive["ntr"]) == 9 and int(archive["nreal"]) == 2 and metadata["seed"] == 9,
                  "train/identity split or seed is wrong")
    stage.require(int(lengths.sum()) == rows and np.all((lengths >= 1) & (lengths <= 200)),
                  "posterior-row counts are inconsistent")
    stage.require(np.isfinite(archive["clouds"]).all() and np.all(archive["mode"][:2] == 2),
                  "clouds are not finite or the first two scenarios are not identity scenarios")
    stage.require(inside(archive["theta"], low, high) and np.isfinite(archive["gtheta"]).all(),
                  "scenario parameters or predictions are not finite inside the prior box")
    stage.numbers = {"scenarios": len(lengths), "training_scenarios": int(archive["ntr"]), "posterior_rows": rows,
                     "cache_rows": metadata["cache_rows"], "target_valid_rows": metadata["target_valid_rows"]}


def check_training(stage: Stage, low, high, threads: int) -> None:
    folder = WORK / "head9"
    report = json.loads((folder / "run_report.json").read_text())
    standardization = load(folder / "anet_std.npz")
    shaped(stage, standardization, {"tmth": (9,), "tsth": (9,), "xm": (7,), "xs": (7,), "prior_low": (9,),
                                    "prior_high": (9,)}, "anet_std.npz")
    if stage.failures:
        return
    losses = [value for entry in report["history"] for value in (entry["training_loss"], entry["validation_loss"])]
    stage.require(report.get("status") == "SMOKE_COMPLETED" and report.get("epochs") == 2
                  and report.get("learned_dim") == 9 and report.get("seed") == 2,
                  "training report does not record the 2-epoch nine-dimensional smoke run with seed 2")
    stage.require(len(report["history"]) == 2 and np.isfinite(losses).all(), "losses are not finite")
    stage.require(report["checkpoint_sha256"] == sha256(folder / "anet_net.pt")
                  and report["standardization_sha256"] == sha256(folder / "anet_std.npz")
                  and report["held_out_sources_absent"] == HELD_OUT and report["forbidden_artifacts_used"] == [],
                  "checkpoint, standardization or lineage differs from the training report")
    stage.require(text(standardization["theta_transform"]) == "canonical_box_atanh_all_9d"
                  and int(standardization["CTX"]) == 199 and np.all(standardization["tsth"] > 0),
                  "standardization does not declare the nine-dimensional box transform")

    # The new head in the production density class: 8 draws, 32 Heun steps (the heads' production value is 1024).
    import torch

    torch.set_num_threads(threads)
    use_hyperonic_pipeline()
    from inference.anet.proposal import mass_radius_clouds
    from sample_clean_hyperonic_parity_mixture import ParityFMPE
    from workflows.nuclear_scenarios import nuclear_observation

    head = ParityFMPE(folder / "anet_net.pt", folder / "anet_std.npz", device="cpu", steps=32)
    clouds = mass_radius_clouds(OBSERVATIONS, points=head.cloud_points, seed=777_001)
    observation = nuclear_observation("A1")
    theta = head.sample(clouds, observation, 8, torch.Generator().manual_seed(20260944))
    with contextlib.redirect_stdout(io.StringIO()):
        log_q = head.log_density(theta, clouds, observation, 8)
    stage.require(inside(theta, low, high) and np.isfinite(log_q).all(),
                  "the new head does not give finite draws and log densities in the production sampler class")
    stage.numbers = {"epochs": report["epochs"], "best_validation_loss": report["best_validation_loss"],
                     "posterior_training_rows": report["posterior_training_rows"],
                     "sampler_class_draws": len(theta), "sampler_class_log_q_finite": bool(np.isfinite(log_q).all())}


def check_dual(stage: Stage, low, high, draws: int) -> np.ndarray:
    proposal = load(WORK / "proposal/A1_dual_smoke.npz")
    shaped(stage, proposal, {key: (draws,) for key in ("logq", "head9_logq", "head7_logq", "student_logq", "component")}
           | {"theta": (draws, 9), "cloud": (3, 256, 2)}, "A1_dual_smoke.npz")
    if stage.failures:
        return proposal
    weights = proposal["component_weights"]
    mixture = logsumexp(np.vstack([np.log(weights[0]) + proposal["head9_logq"], np.log(weights[1]) + proposal["head7_logq"],
                                   np.log(weights[2]) + proposal["student_logq"]]), axis=0)
    stage.require(text(proposal["schema"]) == "ddb-hyperonic-clean-dual-parity-anet-mixture-v1" and clean_lineage(proposal),
                  "dual proposal schema or lineage is wrong")
    stage.require(np.allclose(weights, [0.35, 0.55, 0.10], rtol=0, atol=1e-15) and int(proposal["flow_steps"]) == 1024
                  and int(proposal["cloud_seed"]) == 777_001 and int(proposal["proposal_seed"]) == 20_260_944,
                  "dual proposal settings are not the production ones")
    stage.require(text(proposal["head9_checkpoint_sha256"]) == sha256(NETWORKS / "head9/anet_net.pt")
                  and text(proposal["head7_checkpoint_sha256"]) == sha256(NETWORKS / "head7/anet_net.pt")
                  and text(proposal["student_checkpoint_sha256"]) == sha256(STUDENT),
                  "dual proposal was not drawn from the frozen networks")
    stage.require(inside(proposal["theta"], low, high) and np.isfinite(proposal["logq"]).all()
                  and np.isfinite(proposal["head9_logq"]).all() and np.isfinite(proposal["head7_logq"]).all()
                  and np.isfinite(proposal["student_logq"]).all(),
                  "proposal draws or component densities are not finite inside the prior box")
    stage.require(np.max(np.abs(mixture - proposal["logq"])) <= 1e-12,
                  "the proposal density is not the deterministic mixture of its components")
    stage.require(np.array_equal(np.bincount(proposal["component"], minlength=3).sum(), draws),
                  "component labels do not cover every draw")
    stage.numbers = {"draws": draws, "component_rows": np.bincount(proposal["component"], minlength=3).tolist(),
                     "flow_steps": int(proposal["flow_steps"]),
                     "largest_difference_mixture_log_q": float(np.max(np.abs(mixture - proposal["logq"])))}
    return proposal


def check_exact(stage: Stage, draws: int) -> None:
    exact = load(WORK / "exact/A1_dual_smoke_exact.npz")
    proposal = load(WORK / "proposal/A1_dual_smoke.npz")
    shaped(stage, exact, {"theta": (draws, 9), "index": (draws,), "logl": (draws,)}, "A1_dual_smoke_exact.npz")
    if stage.failures:
        return
    valid = exact["logl"] > -1.0e50
    stage.require(np.array_equal(exact["theta"], proposal["theta"]) and np.array_equal(exact["index"], np.arange(draws)),
                  "exact rows are not the proposal rows in order")
    stage.require(text(exact["nuclear_scenario"]) == "A1" and text(exact["source_scenario"]) == "A1"
                  and not bool(exact["reference_present"]), "exact file scenario or lineage is wrong")
    stage.require(np.isfinite(exact["logl"]).all() and valid.any(),
                  "exact log-likelihoods are not finite (rejected rows carry -1e100) or all rows were rejected")
    stage.numbers = {"rows": draws, "accepted_rows": int(valid.sum())}


def check_certificate(stage: Stage, draws: int) -> None:
    path = WORK / "certified/A1_dual_smoke_certificate.npz"
    certificate = load(path)
    report = json.loads(path.with_suffix(".json").read_text())
    shaped(stage, certificate, {"theta": (draws, 9), "logq": (draws,), "logl": (draws,), "normalized_weight": (draws,)},
           "A1_dual_smoke_certificate.npz")
    if stage.failures:
        return
    logl, logq = certificate["logl"], certificate["logq"]
    valid = logl > -1.0e50
    log_prior = -float(np.log(certificate["prior_high"] - certificate["prior_low"]).sum())
    log_weight = log_prior + logl[valid] - logq[valid]
    largest = float(np.max(log_weight))
    normalizer = largest + float(np.log(np.exp(log_weight - largest).sum()))
    weight = np.zeros(draws)
    weight[valid] = np.exp(log_weight - normalizer)
    ess = float(1.0 / np.sum(weight * weight))
    log_evidence = normalizer - math.log(draws)
    error = math.sqrt(max(draws / ess - 1.0, 0.0) / draws)
    gate = ess >= float(certificate["minimum_posterior_ess"]) and weight.max() <= float(certificate["maximum_normalized_weight_gate"])
    stage.require(float(certificate["minimum_posterior_ess"]) == 1000.0
                  and float(certificate["maximum_normalized_weight_gate"]) == 0.01,
                  "the certificate did not use the production gates")
    stage.require(np.max(np.abs(weight - certificate["normalized_weight"])) <= 1e-12
                  and abs(float(certificate["posterior_ess"]) - ess) <= 1e-12 * ess
                  and abs(float(certificate["log_evidence"]) - log_evidence) <= 1e-12
                  and abs(float(certificate["log_evidence_standard_error"]) - error) <= 1e-12,
                  "certificate weights, ESS or log Z differ from the recomputed values")
    stage.require(bool(certificate["gate_pass"]) == gate and report["status"] == ("PASS" if gate else "FAIL"),
                  "certificate gate decision is inconsistent with its ESS and largest weight")
    stage.require(np.isfinite(log_evidence) and 1.0 <= ess <= draws, "ESS or log Z is not finite")
    stage.numbers = {"rows": draws, "valid_rows": int(valid.sum()), "ess": ess,
                     "maximum_normalized_weight": float(weight.max()), "log_evidence": log_evidence,
                     "largest_differences": {
                         "normalized_weight": float(np.max(np.abs(weight - certificate["normalized_weight"]))),
                         "ess_relative": abs(float(certificate["posterior_ess"]) - ess) / ess,
                         "log_evidence": abs(float(certificate["log_evidence"]) - log_evidence),
                         "log_evidence_standard_error": abs(float(certificate["log_evidence_standard_error"]) - error)},
                     "production_gates": {"minimum_ess": 1000.0, "maximum_normalized_weight": 0.01,
                                          "passed": bool(gate), "expected_to_pass_at_this_size": False}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=3, help="CPU cores and threads to use (default 3)")
    arguments = parser.parse_args()
    started = time.time()
    cpus = pin_cpus(arguments.threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(arguments.threads)
    prepare_observations()
    use_hyperonic_pipeline()  # sets CERTIFIED_DDB_DIR / CERTIFIED_ASTRO_DIR / JAX settings for every stage
    from eos import get_eos

    plugin = get_eos("ddb-hyperonic")
    low = np.asarray(plugin.prior_low, dtype=np.float64)
    high = np.asarray(plugin.prior_high, dtype=np.float64)
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "logs").mkdir(parents=True)
    environment = stage_environment(arguments.threads, (PIPELINE,))
    environment["TOV_NW"] = "1"
    print(f"{TAG} CPU cores {','.join(map(str, cpus)) or 'unpinned'}; outputs in {rel(WORK)}", flush=True)

    bank_rows, screen_proposals, student_draws, dual_draws = 48, 40_000, 24, 64
    caches = [WORK / "uniform_cache.npz", WORK / "support_cache.npz", WORK / "proposal_cache.npz"]
    common = ["--data-root", OBSERVATIONS]
    # The support-cache shard must cover every screened row; its size is known only after the screen.
    stages = [
        (lambda: Stage("bank", [(PIPELINE / "generate_ddbhy_bank.py",
                                 ["--n", bank_rows, "--seed", 91, "--workers", 1, "--chunk-size", bank_rows,
                                  "--output", WORK / "uniform_bank.npz"])],
                       note="48 uniform rows, seed 91 (production 150,000 rows per part, seeds 91-94, 56 workers)"),
         lambda stage: check_bank(stage, low, high, bank_rows)),
        (lambda: Stage("uniform", [(PIPELINE / "build_clean_uniform_hyperonic_training_cache.py",
                                    ["--bank", WORK / "uniform_bank.npz", *common, "--output", caches[0],
                                     "--workers", 1])],
                       note="every forward-valid bank row (production: 62 workers per 150,000-row part)"),
         lambda stage: check_uniform_cache(stage, low, high)),
        (lambda: Stage("support", [
            (PIPELINE / "screen_clean_hyperonic_support_enrichment.py",
             ["--output", WORK / "support_screen.npz", "--proposals", screen_proposals, "--block", screen_proposals]),
            (PIPELINE / "evaluate_clean_hyperonic_support_cache.py",
             ["--screen", WORK / "support_screen.npz", "--template-bank", WORK / "uniform_bank.npz", *common,
              "--output", caches[1], "--workers", 1, "--start", 0, "--stop", "{screened}"])],
                       note="40,000 screened proposals (production 160,000,000 in blocks of 2,000,000; seed 20260729 "
                            "and cbox 3 as in production)"),
         lambda stage: check_support(stage, low, high, screen_proposals)),
        (lambda: Stage("student", [
            (PIPELINE / "clean_student_t_ensemble.py",
             ["sample", "--checkpoint", STUDENT, "--output", WORK / "student_draws.npz", "--draws", student_draws,
              "--seed", 20_260_992]),
            (PIPELINE / "evaluate_clean_hyperonic_proposal_cache.py",
             ["--proposal", WORK / "student_draws.npz", "--checkpoint", STUDENT, "--template-bank",
              WORK / "uniform_bank.npz", *common, "--output", caches[2], "--workers", 1, "--start", 0,
              "--stop", student_draws])],
                       note="24 draws from the frozen defensive Student (production 300,000, same seed)"),
         lambda stage: check_student(stage, low, high, student_draws)),
        (lambda: Stage("correction", [(PIPELINE / "build_clean_hyperonic_augmented_correction.py",
                                       ["--base-cache", caches[0], "--support-cache", caches[1], "--proposal-cache",
                                        caches[2], "--proposal-checkpoint", STUDENT, "--base-proposals", bank_rows,
                                        "--conditional-ess-gate", 1, "--output", WORK / "correction.npz"])],
                       note="--base-proposals 48 and --conditional-ess-gate 1 (production 600,000 and 2,000)"),
         lambda stage: check_correction(stage, caches)),
        (lambda: Stage("scenarios", [(PIPELINE / "build_clean_hyperonic_parity_scenarios.py",
                                      ["--cache", caches[0], "--cache", caches[1], "--cache", caches[2],
                                       "--counting-correction", WORK / "correction.npz", *common, "--output",
                                       WORK / "scenarios.npz", "--seed", 9, "--device", "cpu", "--smoke"])],
                       uses_torch=True, note="--smoke: 12 scenarios, 3 validation, 2 identity, at most 200 rows each"),
         lambda stage: check_scenarios(stage, low, high)),
        (lambda: Stage("training", [(PIPELINE / "train_clean_hyperonic_parity_anet.py",
                                     ["--scenarios", WORK / "scenarios.npz", "--base-cache", caches[0], "--output-dir",
                                      WORK / "head9", "--seed", 2, "--device", "cpu", "--smoke"])],
                       uses_torch=True, note="--smoke: 2 epochs of the nine-dimensional head (production 600)"),
         lambda stage: check_training(stage, low, high, arguments.threads)),
        (lambda: Stage("dual", [(PIPELINE / "sample_clean_hyperonic_dual_parity_mixture.py",
                                 [*[item for dimension in (9, 7) for item in (
                                     f"--head{dimension}-checkpoint", NETWORKS / f"head{dimension}/anet_net.pt",
                                     f"--head{dimension}-standardization", NETWORKS / f"head{dimension}/anet_std.npz",
                                     f"--head{dimension}-report", NETWORKS / f"head{dimension}/run_report.json")],
                                  "--student-checkpoint", STUDENT, *common, "--source-data-root", OBSERVATIONS,
                                  "--source-scenario", "A1", "--nuclear-scenario", "A1", "--output",
                                  WORK / "proposal/A1_dual_smoke.npz", "--draws", dual_draws, "--head9-weight", 0.35,
                                  "--head7-weight", 0.55, "--student-weight", 0.10, "--cloud-seed", 777_001,
                                  "--proposal-seed", 20_260_944, "--flow-steps", 1024, "--device", "cpu"])],
                       uses_torch=True, note="frozen production heads and Student; 64 draws (production 25,000 on a GPU)"),
         lambda stage: check_dual(stage, low, high, dual_draws)),
        (lambda: Stage("exact", [
            (PIPELINE / "build_portable_hyperonic_target_cache.py", [*common, "--output", WORK / "target/A1.joblib"]),
            (PIPELINE / "split_clean_proposal.py",
             ["--proposal", WORK / "proposal/A1_dual_smoke.npz", "--output", WORK / "exact_shards/shard0.npz"]),
            (PIPELINE / "fast_portable_hyperonic_likelihood.py",
             ["--input", WORK / "exact_shards/shard0.npz", "--output", WORK / "exact_shards/shard0_exact.npz",
              "--target-cache", WORK / "target/A1.joblib", "--workers", 1, "--nuclear-scenario", "A1",
              "--source-scenario", "A1"]),
            (PIPELINE / "assemble_likelihood_shards.py",
             ["--proposal", WORK / "proposal/A1_dual_smoke.npz", "--shard", WORK / "exact_shards/shard0_exact.npz",
              "--output", WORK / "exact/A1_dual_smoke_exact.npz"])],
                       note="one shard and 1 worker (production: two 12,500-row shards on 62-core cluster nodes)"),
         lambda stage: check_exact(stage, dual_draws)),
        (lambda: Stage("certificate", [(PIPELINE / "certify_clean_anet_scenario.py",
                                        ["--proposal", WORK / "proposal/A1_dual_smoke.npz", "--likelihood",
                                         WORK / "exact/A1_dual_smoke_exact.npz", "--output",
                                         WORK / "certified/A1_dual_smoke_certificate.npz",
                                         "--minimum-posterior-ess", 1000, "--maximum-normalized-weight", 0.01])],
                       note="production gates; not expected to pass with 64 draws"),
         lambda stage: check_certificate(stage, dual_draws)),
    ]
    records, failed = {}, False
    for build, check in stages:
        stage = build()
        try:
            if stage.name == "support":  # fill in the screened-row count once the screen exists
                screen_step, cache_step = stage.steps
                stage.steps = [screen_step]
                stage.run(environment, WORK / "logs")
                seconds = stage.seconds
                screened = len(load(WORK / "support_screen.npz")["theta"])
                stage.steps = [(cache_step[0], [str(screened) if value == "{screened}" else value
                                                for value in cache_step[1]])]
                stage.run(environment, WORK / "logs", first=1)
                stage.seconds += seconds
                stage.steps = [screen_step, stage.steps[0]]
            else:
                stage.run(environment, WORK / "logs")
            check(stage)
        except StageFailure as error:
            stage.failures.append(str(error))
        except Exception as error:  # an unreadable or malformed output is a failure of this stage
            stage.failures.append(f"output check raised {error!r}")
        records[stage.name] = stage.record()
        numbers = stage.numbers
        if stage.failures:
            summary = "; ".join(stage.failures)
        elif stage.name == "bank":
            summary = f"{numbers['rows']} uniform rows, {numbers['forward_valid_rows']} forward-valid"
        elif stage.name == "uniform":
            summary = f"{numbers['rows']} rows, {numbers['target_valid_rows']} target-valid"
        elif stage.name == "support":
            summary = (f"{numbers['support_rows']} of {numbers['screened_proposals']} proposals in the support box, "
                       f"{numbers['target_valid_rows']} target-valid")
        elif stage.name == "student":
            summary = (f"{numbers['draws']} draws, {numbers['target_valid_rows']} target-valid, density replay "
                       f"{numbers['density_replay_error']:.0e}")
        elif stage.name == "correction":
            summary = f"{numbers['rows']} rows, conditional ESS {numbers['conditional_ess_at_A1']:.1f}"
        elif stage.name == "scenarios":
            summary = (f"{numbers['scenarios']} scenarios ({numbers['training_scenarios']} for training), "
                       f"{numbers['posterior_rows']} posterior rows")
        elif stage.name == "training":
            summary = (f"{numbers['epochs']} epochs, best validation loss {numbers['best_validation_loss']:.3f}; "
                       f"new head samples in the production class")
        elif stage.name == "dual":
            summary = f"{numbers['draws']} draws (head9/head7/Student {numbers['component_rows']}), mixture density exact"
        elif stage.name == "exact":
            summary = f"{numbers['rows']} rows, {numbers['accepted_rows']} accepted by the exact target"
        else:
            gates = numbers["production_gates"]
            summary = (f"ESS {numbers['ess']:.1f}, largest weight {numbers['maximum_normalized_weight']:.3f}; "
                       f"production gates applied ({'passed' if gates['passed'] else 'not passed, as expected'})")
        print(f"{TAG} {stage.name:11s} {'PASS' if not stage.failures else 'FAIL'}  {summary}  ({stage.seconds:.0f} s)",
              flush=True)
        if stage.failures:
            failed = True
            break
    seconds = time.time() - started
    outputs = {rel(path): sha256(path) for path in sorted(WORK.rglob("*")) if path.is_file() and path.parent.name != "logs"}
    summary = {"status": "FAIL" if failed or len(records) != len(stages) else "PASS", "smoke": NAME,
               "seconds": round(seconds, 1), "cpu_cores": cpus, "threads": arguments.threads,
               "scientific_result": False, "stages": records, "outputs_sha256": outputs}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{TAG} {summary['status']}: {len(records)}/{len(stages)} stages in {seconds:.0f} s on "
          f"{len(cpus) or arguments.threads} CPU cores (report: {rel(REPORT)})", flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
