#!/usr/bin/env python3
"""Route 2 smoke run: nucleonic A-NET end to end at toy size, on CPU (about 2-3 minutes on 3 cores).

What runs. Each stage is the unchanged production program of the nucleonic A-NET pipeline (inference/anet/),
started as its own process with its production settings except the sizes:

  1. bank       build_bank.py --smoke: 64 + 64 uniform-prior rows (seeds 1234 and 2026) with EOS/TOV curves,
                pQCD and GW grids and the exact NICER/GW likelihood terms, then the nuclear-support enrichment
                stream (seed 20260729) with its exact counting correction. Production: 200,000 + 460,000 rows,
                160 million screened proposals, 22 workers.
  2. scenarios  build_scenarios.py --smoke --seed 9: 12 scenarios (3 for validation, 2 identity), at most 200
                posterior rows each. Production: 2,000 scenarios (100 validation, 200 identity), 6,000 rows.
  3. training   train.py --smoke --seed 21: 2 epochs of the DeepSets + flow-matching network (production 600).
  4. query+IS   run_is.py --proposals 300 --seed 0 --resample-seed 1: the new network is conditioned on the A1
                data and draws 300 proposals (64 Heun steps); each gets the exact full likelihood, then the
                importance weights, ESS, log Z and a 300-draw resample. As in production, the shared target
                is first checked against its frozen certificate (1,870 rows, tolerance 1e-4).
                Production: the frozen fmpe9c network and 8,000 proposals.

No program needs an edit; the stage settings are command-line options (the --smoke options are the programs'
own). The four command lines are written to the report.

What is asserted (no scientific claim is made; a 2-epoch network trained on 12 scenarios is a useless proposal,
and its ESS and log Z are printed only as numbers):
  - every output exists with the documented keys and shapes; parameters are finite and inside the prior box;
  - bank: the exact likelihood terms are finite on every row with a stable TOV branch, and the maximum mass of
    the exact table equals that of the bank there; the counting correction is finite on every row;
  - scenarios: the train/validation/identity split and posterior-row counts are consistent;
  - training: finite losses; the checkpoint and standardization match the hashes in the training report;
  - query+IS: the target certificate gate passed; the stored log-weights equal log L + log prior - log q, the
    weights their normalization, the ESS 1/sum(w^2) (within 1e-12), and the resample indices are the ones drawn
    with seed 1 (exact).

CPU use: the process and every stage are pinned to 3 CPU cores (the least busy ones), because the JAX CPU
backend ignores OMP_NUM_THREADS; OMP/MKL/OpenBLAS/numba and torch use 3 threads; CUDA_VISIBLE_DEVICES is empty.
Outputs: build/route2/smoke/anet_nucleonic/ (stage outputs and logs, replaced on every run) and the report
build/route2/smoke/anet_nucleonic_smoke.json. Exit status 0 (PASS) or 1 (FAIL).

Run from the repository root:  PYTHONPATH=$PWD python -m route2.smoke.anet_nucleonic_smoke
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
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from route2.common import OBSERVATIONS, ROOT, prepare_observations, sha256  # noqa: E402

NAME = "anet_nucleonic"
TAG = "[smoke-anet-nucleonic]"
WORK = ROOT / "build/route2/smoke" / NAME
REPORT = ROOT / "build/route2/smoke" / f"{NAME}_smoke.json"
ANET = ROOT / "inference/anet"
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
    """One smoke stage: runs its production program, then collects assertion failures and key numbers."""

    def __init__(self, name: str, script: Path, arguments: list, *, uses_torch: bool, note: str = ""):
        self.name, self.script, self.uses_torch, self.note = name, script, uses_torch, note
        self.arguments = [rel(value) if isinstance(value, Path) else str(value) for value in arguments]
        self.failures: list[str] = []
        self.numbers: dict = {}
        self.seconds = 0.0

    @property
    def command(self) -> str:
        return " ".join(["python", rel(self.script), *self.arguments])

    def run(self, environment: dict[str, str], logs: Path) -> None:
        log = logs / f"{self.name}.log"
        command = [sys.executable, "-c", BOOTSTRAP, str(self.script), "1" if self.uses_torch else "0", *self.arguments]
        started = time.time()
        with log.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT).returncode
        self.seconds = time.time() - started
        if code != 0:
            tail = "\n    ".join(log.read_text(errors="replace").splitlines()[-12:])
            raise StageFailure(f"{rel(self.script)} exited with status {code}; end of {rel(log)}:\n    {tail}")

    def require(self, condition, message: str) -> None:
        if not bool(condition):
            self.failures.append(message)

    def record(self) -> dict:
        return {"pass": not self.failures, "seconds": round(self.seconds, 1), "command": self.command,
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


# ----------------------------------------------------------------------------------------------------------

def check_bank(stage: Stage, low: np.ndarray, high: np.ndarray) -> None:
    folder = WORK / "bank"
    bank, exact = load(folder / "final_bank.npz"), load(folder / "final_exact.npz")
    report = json.loads((folder / "final_bank.json").read_text())
    rows = len(bank["theta"])
    grid = len(bank["MG"])
    shaped(stage, bank, {"theta": (rows, 7), "X": (rows, 7), "Rg": (rows, grid), "Lg": (rows, grid), "MM": (rows,),
                         "PQG": (rows, 5), "GWG": (rows, 9), "R14": (rows,), "logw_prior": (rows,), "logq": (rows,)},
           "final_bank.npz")
    exact_keys = ("exact_j0030", "exact_j0740", "exact_j0437", "exact_gw", "Rmm", "Mmax_fresh")
    shaped(stage, exact, {key: (rows,) for key in exact_keys}, "final_exact.npz")
    if stage.failures:
        return
    valid = np.isfinite(bank["MM"])
    stage.require(report.get("status") == "SMOKE_COMPLETED", "bank report status is not SMOKE_COMPLETED")
    stage.require(int(bank["N0"]) == 128 and rows == 128 + report["new_rows"], "bank row counts are inconsistent")
    stage.require(inside(bank["theta"], low, high), "bank parameters are not finite inside the prior box")
    stage.require(np.isfinite(bank["X"]).all(), "nuclear predictions are not finite")
    stage.require(valid.any(), "no bank row has a stable TOV branch")
    stage.require(all(np.isfinite(exact[key][valid]).all() for key in exact_keys),
                  "exact likelihood terms are not finite on every row with a stable branch")
    stage.require(np.array_equal(exact["Mmax_fresh"][valid], bank["MM"][valid]),
                  "maximum mass of the exact table differs from the bank")
    stage.require(np.isfinite(bank["logw_prior"]).all() and np.isfinite(bank["logq"]).all(),
                  "counting correction is not finite on every row")
    stage.require(np.isfinite(report.get("conditional_ess_at_A1", np.nan)), "conditional ESS is not finite")
    stage.numbers = {"rows": rows, "uniform_rows": int(bank["N0"]), "enriched_rows": int(report["new_rows"]),
                     "rows_with_stable_branch": int(valid.sum()),
                     "conditional_ess_at_A1": report.get("conditional_ess_at_A1"),
                     "conditional_ess_gate_applied": False}


def check_scenarios(stage: Stage, low: np.ndarray, high: np.ndarray) -> None:
    path = WORK / "scenarios.npz"
    archive = load(path)
    report = json.loads(path.with_suffix(".json").read_text())
    count = len(archive["clouds"])
    rows = len(archive["theta"])
    shaped(stage, archive, {"clouds": (12, 3, 256, 2), "dial": (12, 2), "transforms": (12, 3, 4), "ess": (12,),
                            "mode": (12,), "theta_len": (12,), "theta": (rows, 7), "gtheta": (rows, 7)},
           "scenarios.npz")
    if stage.failures:
        return
    lengths = archive["theta_len"]
    stage.require(report.get("status") == "SMOKE_COMPLETED", "scenario report status is not SMOKE_COMPLETED")
    stage.require(int(archive["ntr"]) == 9 and int(archive["nreal"]) == 2, "train/identity split is not 9/2")
    stage.require(int(lengths.sum()) == rows and np.all((lengths >= 1) & (lengths <= 200)),
                  "posterior-row counts are inconsistent")
    stage.require(np.isfinite(archive["clouds"]).all() and np.isfinite(archive["dial"]).all(),
                  "conditioning clouds or dials are not finite")
    stage.require(np.all(archive["mode"][:2] == 2) and np.all(archive["dial"][:2] == [0.0, 1.2]),
                  "the first two scenarios are not identity scenarios")
    stage.require(np.all(archive["ess"] >= 1.0), "a scenario is below the smoke bank-ESS floor of 1")
    stage.require(inside(archive["theta"], low, high), "scenario parameters are not finite inside the prior box")
    stage.require(np.isfinite(archive["gtheta"]).all(), "scenario nuclear predictions are not finite")
    stage.numbers = {"scenarios": count, "training_scenarios": int(archive["ntr"]), "posterior_rows": rows,
                     "smallest_bank_ess": float(archive["ess"].min())}


def check_training(stage: Stage) -> None:
    folder = WORK / "train"
    report = json.loads((folder / "run_report.json").read_text())
    standardization = load(folder / "anet_std.npz")
    shaped(stage, standardization, {"tmth": (7,), "tsth": (7,), "xm": (7,), "xs": (7,), "cmean": (2,), "cstd": (2,)},
           "anet_std.npz")
    if stage.failures:
        return
    losses = [value for entry in report["history"] for value in (entry["training_loss"], entry["validation_loss"])]
    stage.require(report.get("status") == "SMOKE_COMPLETED" and report.get("epochs") == 2 and report.get("seed") == 21,
                  "training report does not record the 2-epoch smoke run with seed 21")
    stage.require(len(report["history"]) == 2 and np.isfinite(losses).all() and np.isfinite(report["best_validation_loss"]),
                  "training or validation losses are not finite")
    stage.require(report["checkpoint_sha256"] == sha256(folder / "anet_net.pt")
                  and report["standardization_sha256"] == sha256(folder / "anet_std.npz"),
                  "checkpoint or standardization differs from its training report")
    stage.require(all(np.isfinite(standardization[key]).all() for key in ("tmth", "xm", "cmean"))
                  and all(np.all(standardization[key] > 0) for key in ("tsth", "xs", "cstd")),
                  "standardization is not finite or has a non-positive scale")
    stage.require(int(standardization["kpts"]) == 256 and int(standardization["heun_steps"]) == 64,
                  "standardization does not declare 256 cloud points and 64 Heun steps")
    stage.numbers = {"epochs": report["epochs"], "best_validation_loss": report["best_validation_loss"],
                     "posterior_training_rows": report["posterior_training_rows"]}


def check_query(stage: Stage, low: np.ndarray, high: np.ndarray, proposals: int) -> None:
    path = WORK / "query/A1_smoke.npz"
    archive = load(path)
    report = json.loads(path.with_suffix(".json").read_text())
    shaped(stage, archive, {"raw": (proposals, 7), "theta": (proposals, 7), "posterior_index": (proposals,),
                            "w": (proposals,), "logw": (proposals,), "logl": (proposals,), "logq": (proposals,)},
           "A1_smoke.npz")
    if stage.failures:
        return
    gate = report.get("target_gate", {})
    stage.require(report.get("status") == "SMOKE_COMPLETED", "IS report status is not SMOKE_COMPLETED")
    stage.require(gate.get("status") == "PASS" and gate.get("valid_mask_equal") is True
                  and gate.get("max_abs_log_likelihood_error", np.inf) <= gate.get("tolerance", 0.0),
                  "the shared target certificate gate did not pass")
    stage.require(np.isfinite(archive["raw"]).all() and np.isfinite(archive["logq"]).all(),
                  "proposal draws or proposal log densities are not finite")
    in_bounds = np.all((archive["raw"] >= low) & (archive["raw"] <= high), axis=1)
    log_prior = -float(np.sum(np.log(high - low)))
    log_weight = np.where(in_bounds, archive["logl"] + log_prior, -np.inf) - archive["logq"]
    finite = np.isfinite(log_weight)
    stage.require(finite.any(), "no proposal has a finite importance weight")
    if not finite.any():
        return
    weight = np.zeros(proposals)
    weight[finite] = np.exp(log_weight[finite] - log_weight[finite].max())
    weight /= weight.sum()
    ess = float(1.0 / np.sum(weight**2))
    stage.require(np.array_equal(np.isfinite(archive["logw"]), finite)
                  and np.array_equal(archive["logw"][finite], log_weight[finite]),
                  "stored log-weights differ from log L + log prior - log q")
    stage.require(np.max(np.abs(archive["w"] - weight)) <= 1e-12 and abs(float(archive["w"].sum()) - 1.0) <= 1e-12,
                  "stored weights are not the normalized importance weights")
    stage.require(abs(float(archive["ess"]) - ess) <= 1e-12 * ess and 1.0 <= ess <= proposals, "ESS is inconsistent")
    stage.require(np.isfinite(archive["logz"]) and np.isfinite(archive["logzerr"]), "log Z or its error is not finite")
    index = np.random.default_rng(1).choice(proposals, size=proposals, replace=True, p=archive["w"])
    stage.require(np.array_equal(index, archive["posterior_index"]) and np.array_equal(archive["theta"], archive["raw"][index]),
                  "the resample is not the seed-1 draw from the stored weights")
    stage.numbers = {"proposals": proposals, "in_prior": int(in_bounds.sum()), "finite_weights": int(finite.sum()),
                     "ess": ess, "log_evidence": float(archive["logz"]),
                     "largest_differences": {
                         "log_weight": float(np.max(np.abs(archive["logw"][finite] - log_weight[finite]))),
                         "normalized_weight": float(np.max(np.abs(archive["w"] - weight))),
                         "ess_relative": abs(float(archive["ess"]) - ess) / ess},
                     "target_gate": {key: gate.get(key) for key in ("status", "rows", "max_abs_log_likelihood_error",
                                                                     "tolerance")}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=3, help="CPU cores and threads to use (default 3)")
    parser.add_argument("--proposals", type=int, default=300, help="A-NET proposals of the final query (default 300)")
    arguments = parser.parse_args()
    started = time.time()
    cpus = pin_cpus(arguments.threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(arguments.threads)
    prepare_observations()
    from eos.ddb import PRIOR_HIGH, PRIOR_LOW  # the canonical seven-dimensional prior box

    low, high = np.asarray(PRIOR_LOW, dtype=np.float64), np.asarray(PRIOR_HIGH, dtype=np.float64)
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "logs").mkdir(parents=True)
    environment = stage_environment(arguments.threads)
    print(f"{TAG} CPU cores {','.join(map(str, cpus)) or 'unpinned'}; outputs in {rel(WORK)}", flush=True)

    stages = [
        (Stage("bank", ANET / "build_bank.py",
               ["--data-root", OBSERVATIONS, "--output-dir", WORK / "bank", "--workers", 1, "--smoke"],
               uses_torch=False, note="--smoke: 64 + 64 uniform rows, 128 screened enrichment proposals, cbox 100"),
         lambda stage: check_bank(stage, low, high)),
        (Stage("scenarios", ANET / "build_scenarios.py",
               ["--bank", WORK / "bank/final_bank.npz", "--exact", WORK / "bank/final_exact.npz", "--data-root",
                OBSERVATIONS, "--output", WORK / "scenarios.npz", "--seed", 9, "--device", "cpu", "--smoke"],
               uses_torch=True, note="--smoke: 12 scenarios, 3 validation, 2 identity, at most 200 rows each"),
         lambda stage: check_scenarios(stage, low, high)),
        (Stage("training", ANET / "train.py",
               ["--scenarios", WORK / "scenarios.npz", "--bank", WORK / "bank/final_bank.npz", "--output-dir",
                WORK / "train", "--seed", 21, "--device", "cpu", "--smoke"],
               uses_torch=True, note="--smoke: 2 epochs"),
         check_training),
        (Stage("query+IS", ANET / "run_is.py",
               ["--data-root", OBSERVATIONS, "--output", WORK / "query/A1_smoke.npz", "--checkpoint",
                WORK / "train/anet_net.pt", "--standardization", WORK / "train/anet_std.npz", "--scenario", "A1",
                "--source-scenario", "A1", "--proposals", arguments.proposals, "--seed", 0, "--resample-seed", 1,
                "--workers", 1, "--device", "cpu", "--smoke"],
               uses_torch=True, note="the network trained above; target certificate gate on (production default)"),
         lambda stage: check_query(stage, low, high, arguments.proposals)),
    ]
    records, failed = {}, False
    for stage, check in stages:
        try:
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
            summary = (f"{numbers['rows']} rows ({numbers['uniform_rows']} uniform + {numbers['enriched_rows']} "
                       f"support), {numbers['rows_with_stable_branch']} with a stable TOV branch; exact terms finite")
        elif stage.name == "scenarios":
            summary = (f"{numbers['scenarios']} scenarios ({numbers['training_scenarios']} for training), "
                       f"{numbers['posterior_rows']} posterior rows")
        elif stage.name == "training":
            summary = f"{numbers['epochs']} epochs, best validation loss {numbers['best_validation_loss']:.3f}"
        else:
            gate = numbers["target_gate"]
            summary = (f"target gate {gate['status']} ({gate['rows']} rows, max |d log L| "
                       f"{gate['max_abs_log_likelihood_error']:.1e}); {numbers['proposals']} proposals, "
                       f"{numbers['finite_weights']} with finite weight, ESS {numbers['ess']:.1f}")
        print(f"{TAG} {stage.name:10s} {'PASS' if not stage.failures else 'FAIL'}  {summary}  ({stage.seconds:.0f} s)",
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
