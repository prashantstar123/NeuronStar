#!/usr/bin/env python3
"""Route 2 smoke run: TSNPE + deterministic-mixture importance sampling at toy size, on CPU (about 2 minutes on
3 cores).

What runs. Both stages are the unchanged production programs of the nucleonic TSNPE pipeline (inference/tsnpe/),
started as their own processes with the production settings except the sizes:

  1. round  train.py --max-rounds 1: the frozen round-zero support flow (inference/tsnpe/checkpoints/
            flow_nucmm_tight.pt) defines the restricted prior (support quantile 1e-4, 2,000 support samples);
            32 pilot rows fix the acceptance reference; one round of 64 simulations trains the neural posterior
            estimator (the production architecture: neural spline flow, 256 hidden features, 10 transforms) for
            at most 2 epochs; 200 posterior samples. Production (the paper's training history): 30,000 then
            5 x 20,000 simulations in 6 rounds; the program defaults also use 2,000 pilot rows and 200,000 support
            samples. --smoke makes the simulator accept every physically valid row (the program's own documented
            smoke behaviour) and permits --skip-target-gate; the gate runs in stage 2.
  2. MIS    run_mis.py: 64 flow draws, then a broad (scale 2.5) and a mild (scale 1.5) box-truncated Gaussian
            with 32 draws each (normalizations from 20,000 draws), exact full likelihood on all 128 rows, the
            three-component deterministic-mixture proposal density, importance weights, ESS, log Z and a 200-row
            resample (seed 1). The shared target is first checked against its frozen certificate (1,870 rows,
            tolerance 1e-4), as in production. Production sizes (the program defaults): 60,000 + 15,000 + 30,000
            draws, normalizations from 2,000,000 draws, 20,000 resampled rows.

No program needs an edit; every setting above is a command-line option. The command lines are in the report.

What is asserted (no scientific claim is made; one tiny round gives a useless proposal, and its ESS and log Z
are printed only as numbers):
  - every output exists with the documented shape; parameters are finite and inside the prior box;
  - round: the round history, pilot gate and estimator hash are recorded; losses are not inspected;
  - MIS: the target certificate gate passed; the proposal density equals the deterministic mixture
    log sum_k (n_k/N) q_k of the three stored component densities, and the weights, ESS and resample indices
    are recomputed from log L, log prior and that density (1e-12; resample exact).

CPU use: the process and both stages are pinned to 3 CPU cores (the least busy ones), because the JAX CPU
backend ignores OMP_NUM_THREADS; OMP/MKL/OpenBLAS/numba and torch use 3 threads; CUDA_VISIBLE_DEVICES is empty.
Outputs: build/route2/smoke/tsnpe/ (stage outputs and logs, replaced on every run) and the report
build/route2/smoke/tsnpe_smoke.json. Exit status 0 (PASS) or 1 (FAIL).

Run from the repository root:  PYTHONPATH=$PWD python -m route2.smoke.tsnpe_smoke
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
from scipy.special import logsumexp  # noqa: E402

from route2.common import OBSERVATIONS, ROOT, prepare_observations, sha256  # noqa: E402

NAME = "tsnpe"
TAG = "[smoke-tsnpe]"
WORK = ROOT / "build/route2/smoke" / NAME
REPORT = ROOT / "build/route2/smoke" / f"{NAME}_smoke.json"
TSNPE = ROOT / "inference/tsnpe"
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

def check_round(stage: Stage, low, high, settings: dict) -> None:
    folder = WORK / "train"
    report = json.loads((folder / "run_report.json").read_text())
    history = json.loads((folder / "history.json").read_text())
    first = np.load(folder / "posterior_round_0.npy", allow_pickle=False)
    final = np.load(folder / "posterior_FINAL.npy", allow_pickle=False)
    stage.require(first.shape == (settings["posterior_samples"], 7) and final.shape == (settings["final_samples"], 7),
                  f"posterior samples have shapes {first.shape} and {final.shape}")
    if stage.failures:
        return
    pilot = report.get("pilot_gate", {})
    stage.require(report.get("status") == "SMOKE_COMPLETED" and report.get("rounds_completed") == 1
                  and report.get("smoke_accepts_all_physically_valid_rows") is True,
                  "training report does not record one smoke round")
    stage.require(report.get("target_gate", {}).get("status") == "SKIPPED", "the round's target gate was not skipped")
    stage.require(pilot.get("rows") == settings["pilot"] and pilot.get("valid_rows", 0) >= 1
                  and np.isfinite(pilot.get("log_reference", np.nan)),
                  "the pilot gate did not freeze a finite acceptance reference")
    stage.require(len(history) == 1 and history[0]["simulations"] == settings["n0"] and history[0]["accepted"] >= 1
                  and np.isfinite(history[0]["median"]).all() and np.isfinite(history[0]["width90"]).all(),
                  "round history is incomplete or not finite")
    stage.require(report["estimator_sha256"] == sha256(folder / "density_estimator.pt")
                  and report["posterior_sha256"] == sha256(folder / "posterior_FINAL.npy")
                  and report["seed_checkpoint_sha256"] == sha256(TSNPE / "checkpoints/flow_nucmm_tight.pt"),
                  "estimator, posterior or round-zero flow differs from the recorded hashes")
    stage.require(inside(first, low, high) and inside(final, low, high),
                  "posterior samples are not finite inside the prior box")
    stage.numbers = {"rounds": 1, "simulations": history[0]["simulations"], "accepted": history[0]["accepted"],
                     "pilot_rows": pilot.get("rows"), "posterior_samples": len(final)}


def check_mis(stage: Stage, low, high, settings: dict) -> None:
    folder = WORK / "mis"
    report = json.loads((folder / "run_report.json").read_text())
    archive = load(folder / "tsnpe_mis_corrected.npz")
    counts = [settings["flow"], settings["broad"], settings["mild"]]
    rows, resampled = sum(counts), settings["posterior_rows"]
    shaped(stage, archive, {key: (rows,) for key in ("logL", "logq", "logq_flow", "logq_broad", "logq_mild", "logw",
                                                      "w", "proposal_component")}
           | {"theta": (rows, 7), "posterior": (resampled, 7), "posterior_index": (resampled,)},
           "tsnpe_mis_corrected.npz")
    stage.require(len(load(folder / "stage_1_flow.npz")["theta"]) == counts[0]
                  and len(load(folder / "stage_2_flow_broad.npz")["theta"]) == counts[0] + counts[1],
                  "intermediate stage archives have the wrong row counts")
    if stage.failures:
        return
    gate = report.get("target_gate", {})
    stage.require(report.get("status") == "SMOKE_COMPLETED", "MIS report status is not SMOKE_COMPLETED")
    stage.require(gate.get("status") == "PASS" and gate.get("valid_mask_equal") is True
                  and gate.get("max_abs_log_likelihood_error", np.inf) <= gate.get("tolerance", 0.0),
                  "the shared target certificate gate did not pass")
    stage.require(report["components"] == {"flow": counts[0], "broad_truncated_gaussian": counts[1],
                                           "mild_truncated_gaussian": counts[2]}
                  and np.array_equal(np.bincount(archive["proposal_component"], minlength=3), counts),
                  "mixture component counts are wrong")
    normalizations = report["truncation_normalizations"]
    stage.require(all(0.0 < normalizations[key] <= 1.0 for key in ("broad", "mild")),
                  "a truncation normalization is outside (0, 1]")
    stage.require(inside(archive["theta"], low, high), "proposal rows are not finite inside the prior box")
    components = np.vstack([archive["logq_flow"], archive["logq_broad"], archive["logq_mild"]])
    stage.require(np.isfinite(components).all() and np.isfinite(archive["logq"]).all(),
                  "component or mixture proposal densities are not finite")
    mixture = logsumexp(np.log(np.asarray(counts, dtype=np.float64) / rows)[:, None] + components, axis=0)
    stage.require(np.max(np.abs(mixture - archive["logq"])) <= 1e-12,
                  "the proposal density is not the deterministic mixture of its three components")
    valid = np.isfinite(archive["logL"]) & (archive["logL"] > -1e50)
    log_prior = -float(np.sum(np.log(high - low)))
    log_weight = np.where(valid, archive["logL"] + log_prior, -np.inf) - archive["logq"]
    stage.require(valid.any(), "no proposal row is accepted by the exact target")
    if not valid.any():
        return
    weight = np.zeros(rows)
    weight[valid] = np.exp(log_weight[valid] - log_weight[valid].max())
    weight /= weight.sum()
    ess = float(1.0 / np.sum(weight**2))
    stage.require(np.array_equal(np.isfinite(archive["logw"]), valid)
                  and np.max(np.abs(archive["logw"][valid] - log_weight[valid])) <= 1e-12,
                  "stored log-weights differ from log L + log prior - log q")
    stage.require(np.max(np.abs(archive["w"] - weight)) <= 1e-12 and abs(float(archive["ESS"]) - ess) <= 1e-12 * ess
                  and 1.0 <= ess <= rows, "stored weights or ESS differ from the recomputed values")
    stage.require(np.isfinite(archive["logZ"]) and np.isfinite(archive["logZ_standard_error"]),
                  "log Z or its error is not finite")
    index = np.random.default_rng(1).choice(rows, size=resampled, replace=True, p=archive["w"])
    stage.require(np.array_equal(index, archive["posterior_index"])
                  and np.array_equal(archive["posterior"], archive["theta"][index]),
                  "the resample is not the seed-1 draw from the stored weights")
    stage.numbers = {"rows": rows, "components": counts, "accepted_rows": int(valid.sum()), "ess": ess,
                     "log_evidence": float(archive["logZ"]),
                     "largest_differences": {
                         "mixture_log_q": float(np.max(np.abs(mixture - archive["logq"]))),
                         "log_weight": float(np.max(np.abs(archive["logw"][valid] - log_weight[valid]))),
                         "normalized_weight": float(np.max(np.abs(archive["w"] - weight))),
                         "ess_relative": abs(float(archive["ESS"]) - ess) / ess},
                     "truncation_normalizations": normalizations,
                     "target_gate": {key: gate.get(key) for key in ("status", "rows", "max_abs_log_likelihood_error",
                                                                     "tolerance")}}


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
    from eos.ddb import PRIOR_HIGH, PRIOR_LOW  # the canonical seven-dimensional prior box

    low, high = np.asarray(PRIOR_LOW, dtype=np.float64), np.asarray(PRIOR_HIGH, dtype=np.float64)
    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / "logs").mkdir(parents=True)
    environment = stage_environment(arguments.threads)
    print(f"{TAG} CPU cores {','.join(map(str, cpus)) or 'unpinned'}; outputs in {rel(WORK)}", flush=True)

    round_settings = {"n0": 64, "pilot": 32, "posterior_samples": 200, "final_samples": 200}
    mis_settings = {"flow": 64, "broad": 32, "mild": 32, "posterior_rows": 200}
    stages = [
        (Stage("round", TSNPE / "train.py",
               ["--data-root", OBSERVATIONS, "--output-dir", WORK / "train", "--workers", 1, "--seed", 0,
                "--n0", round_settings["n0"], "--n", 32, "--pilot", round_settings["pilot"], "--max-rounds", 1,
                "--posterior-samples", round_settings["posterior_samples"], "--final-samples",
                round_settings["final_samples"], "--support-samples", 2000, "--max-training-epochs", 2,
                "--device", "cpu", "--smoke", "--skip-target-gate"],
               uses_torch=True, note="one round of 64 simulations; production 6 rounds (30,000 then 20,000 each)"),
         lambda stage: check_round(stage, low, high, round_settings)),
        (Stage("MIS", TSNPE / "run_mis.py",
               ["--data-root", OBSERVATIONS, "--estimator", WORK / "train/density_estimator.pt", "--output-dir",
                WORK / "mis", "--workers", 1, "--flow-draws", mis_settings["flow"], "--broad-draws",
                mis_settings["broad"], "--mild-draws", mis_settings["mild"], "--normalization-draws", 20000,
                "--seed", 0, "--resample-seed", 1, "--posterior-rows", mis_settings["posterior_rows"],
                "--device", "cpu", "--smoke"],
               uses_torch=True, note="64 + 32 + 32 draws (production 60,000 + 15,000 + 30,000); target gate on"),
         lambda stage: check_mis(stage, low, high, mis_settings)),
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
        elif stage.name == "round":
            summary = (f"1 round, {numbers['simulations']} simulations ({numbers['accepted']} accepted), "
                       f"{numbers['posterior_samples']} posterior samples")
        else:
            gate = numbers["target_gate"]
            summary = (f"target gate {gate['status']} ({gate['rows']} rows, max |d log L| "
                       f"{gate['max_abs_log_likelihood_error']:.1e}); {numbers['rows']} rows "
                       f"({'/'.join(map(str, numbers['components']))}), mixture density exact, ESS {numbers['ess']:.1f}")
        print(f"{TAG} {stage.name:6s} {'PASS' if not stage.failures else 'FAIL'}  {summary}  ({stage.seconds:.0f} s)",
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
