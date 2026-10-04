#!/usr/bin/env python3
"""Route 2 smoke run: the Green Evidence Network (EN) end to end on CPU, on a bank of a few hundred rows.

What runs (the production programs, unchanged):
  1. label builder   inference/evidence_network/conditional/build_source_weight_cache_v2.py
  2. trainer         green_en/solution/s2_green_en.py, with its helper green_en/train_onfly.py
  3. frozen query    inference/evidence_network/conditional/query_green_frozen_v5.py, the query engine behind
                     the v7/v8 query gates (the gates admit only checkpoints listed by SHA-256 in the frozen
                     manifests, so the network trained here is queried through the engine directly), on the five
                     nuclear settings and the three held-out NICER sources in data/observations.
Settings are the production settings of the chosen sector (docs/route2/STEPS_EVIDENCE_NETWORK.md)
except the training seed (1) and the sizes: 300 bank rows (production 645,084 hyperonic / 848,017 nucleonic),
a 21 x 25 kernel lattice (production 161 x 205 / 191 x 241) and 4 optimizer steps of 6 scenarios x 3 nuclear
designs (production 12,000 steps of 192 x 9). The 8,192 label scenarios are kept because the trainer hard-codes
its 7,168/512/512 split.

The bank. By default this script writes a synthetic bank in the production schema
(conditional-en-independent-physics-v2): nuclear observables within 4 sigma of the nuclear observation,
smooth mass-radius curves on the production 200-point mass grid with a square-root turnover at the maximum
mass, finite astrophysical and per-source NICER log-likelihood terms, plus invalid rows (no stable curve) and
target-rejected rows (log L = -inf) as in the real banks. With --bank PATH it instead takes --rows rows of a real
frozen EN bank (the rows with the largest base log-weight inside the 6-sigma nuclear support; a uniform draw of
so few rows would fail the unchanged ESS >= 20 label gate) and renormalizes the proposal denominator over them.
Either way the numbers mean nothing: the smoke only shows that the code runs.

CPU. The label builder and the trainer hard-code the device "cuda". The smoke runs their main() in this process
under a context that sends every CUDA request to the CPU: a torch function mode rewrites CUDA device arguments
(tensor constructors, .to()), torch.Generator is created on the CPU, and torch.cuda.synchronize,
max_memory_allocated and get_device_name are replaced by CPU stand-ins. The program files are not modified.
The query uses its own --device cpu option. Threads: at most 3 (OMP/MKL/OpenBLAS/numba and torch).
The trainer and its helper insert fixed paths into sys.path; the smoke sets DDB_REPO_ROOT to this repository,
removes any path outside it afterwards, and checks that every EN module in use was loaded from this repository.

Assertions: only finite values and correct shapes (label index and source-weight matrix; trainer checkpoint and
report; query output with 8 finite log-evidences), plus exit status 0 of each program.
Writes build/route2/en_smoke.json; exits 0 (PASS) or 1 (FAIL).

  PYTHONPATH=$PWD python -m route2.smoke.en_smoke                    # synthetic bank, hyperonic settings (~1 min)
  PYTHONPATH=$PWD python -m route2.smoke.en_smoke --sector nucleonic # (~1 min)
  PYTHONPATH=$PWD python -m route2.smoke.en_smoke --bank <frozen EN bank .npz>   # (1-3 min)
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402

from route2.common import OBSERVATIONS, ROOT, prepare_observations, sha256  # noqa: E402

# On the CPU the trainer's label matrix stays a read-only memory map (on a GPU it is copied); it is never written.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
REPORT = ROOT / "build/route2/en_smoke.json"
TRAINER = ROOT / "green_en/solution/s2_green_en.py"
OBSERVATION = np.asarray([0.153, -16.1, 230.0, 32.5, 0.505714285714279, 1.24142857142857, 2.4857142857143])
SIGMA = np.asarray([0.005, 0.2, 40.0, 1.8, 0.194285714285714, 0.608571428571429, 1.38285714285714])
MASS_GRID = np.linspace(0.5, 2.6, 200, dtype=np.float32)              # production bank mass grid
BASELINE_FILES = ("J0030_2spot_RM.txt", "J0740_NICERXMM_full_mr.txt", "J0437_post_equal_weights.dat")
SCENARIOS, VALIDATION = 8192, 1024                                     # fixed by the trainer's split
SMOKE = dict(rows=300, nm=21, nr=25, steps=4, batch_scenarios=6, designs_per_step=3, seed=1)
SECTORS = {  # production settings (docs/route2/STEPS_EVIDENCE_NETWORK.md), sizes excluded
    "hyperonic": {
        "label": ["--one-slot-fraction", "0.83", "--mass-centre-min", "0.85", "--mass-centre-max", "2.25",
                  "--radius-centre-min", "8.5", "--radius-centre-max", "16.5", "--mass-sigma-min", "0.015",
                  "--mass-sigma-max", "0.48", "--radius-sigma-min", "0.12", "--radius-sigma-max", "2.30",
                  "--correlation-limit", "0.95"],
        "domain": (1.0, 2.6, 8.2, 18.4), "model": "ddb-hyperonic", "maximum_mass": (1.9, 2.5)},
    "nucleonic": {
        "label": ["--one-slot-fraction", "0.98", "--mass-centre-min", "0.75", "--mass-centre-max", "2.35",
                  "--radius-centre-min", "8.0", "--radius-centre-max", "17.0", "--mass-sigma-min", "0.005",
                  "--mass-sigma-max", "0.5", "--radius-sigma-min", "0.05", "--radius-sigma-max", "2.5",
                  "--correlation-limit", "0.97"],
        "domain": (1.0, 2.9, 7.0, 19.0), "model": "ddb", "maximum_mass": (2.0, 2.8)},
}
COMMON_LABEL = ["--scenarios", str(SCENARIOS), "--validation-scenarios", str(VALIDATION), "--seed", "20261011",
                "--identity-fraction", "0.02", "--scenario-batch", "32", "--minimum-tilt-ess", "20",
                "--maximum-restriction-delta", "0.001", "--nuclear-support-sigma", "6", "--nuclear-designs", "64"]
PACKAGES = ("inference", "likelihoods", "workflows", "eos", "tov", "train_onfly", "s2_green_en")


# ---------------------------------------------------------------------------------------------- CUDA -> CPU
def _is_cuda(value) -> bool:
    return ((isinstance(value, str) and value.split(":")[0] == "cuda")
            or (isinstance(value, torch.device) and value.type == "cuda"))


class _CudaToCpu(TorchFunctionMode):
    """Rewrite every CUDA device argument of a torch call (constructors, Tensor.to, Module.to) to the CPU."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        cpu = torch.device("cpu")
        args = tuple(cpu if _is_cuda(value) else value for value in args)
        kwargs = {key: (cpu if _is_cuda(value) else value) for key, value in (kwargs or {}).items()}
        return func(*args, **kwargs)


class _CpuGenerator(torch.Generator):
    """torch.Generator that is always created on the CPU."""

    def __new__(cls, device="cpu"):
        return super().__new__(cls, "cpu" if _is_cuda(device) else device)

    def __init__(self, device="cpu"):
        pass


@contextlib.contextmanager
def cuda_on_cpu():
    saved = (torch.Generator, torch.cuda.synchronize, torch.cuda.max_memory_allocated, torch.cuda.get_device_name)
    precision = torch.get_float32_matmul_precision()
    torch.Generator = _CpuGenerator
    torch.cuda.synchronize = lambda *args, **kwargs: None
    torch.cuda.max_memory_allocated = lambda *args, **kwargs: 0
    torch.cuda.get_device_name = lambda *args, **kwargs: "cpu (Route 2 smoke: CUDA requests sent to the CPU)"
    try:
        with _CudaToCpu():
            yield
    finally:
        torch.Generator, torch.cuda.synchronize, torch.cuda.max_memory_allocated, torch.cuda.get_device_name = saved
        torch.set_float32_matmul_precision(precision)


def run_main(main, argv: list[str]):
    saved = sys.argv
    sys.argv = argv
    try:
        return main()
    finally:
        sys.argv = saved


# ---------------------------------------------------------------------------------------------- inputs
def synthetic_bank(path: Path, rows: int, sector: str, seed: int = 20260925) -> dict:
    """A small bank in the production schema; see the module docstring."""
    rng = np.random.default_rng(seed)
    low, high = SECTORS[sector]["maximum_mass"]
    prediction = (OBSERVATION + SIGMA * np.clip(rng.normal(0.0, 1.3, (rows, 7)), -4.0, 4.0)).astype(np.float32)
    maximum_mass = rng.uniform(low, high, rows)
    end_radius = rng.uniform(9.5, 11.5, rows)                        # radius at the maximum mass (km)
    radius_14 = end_radius + rng.uniform(0.8, 2.5, rows)             # radius at 1.4 Msun (km)
    slope = (radius_14 - end_radius) / np.sqrt(maximum_mass - 1.4)
    below = MASS_GRID[None, :] <= maximum_mass[:, None]
    radius = np.where(below, end_radius[:, None]
                      + slope[:, None] * np.sqrt(np.clip(maximum_mass[:, None] - MASS_GRID[None, :], 0.0, None)),
                      np.nan).astype(np.float32)
    nicer = rng.normal(-3.0, 1.0, (rows, 3))
    log_astro = nicer.sum(axis=1) + rng.normal(-18.0, 1.0, rows)
    invalid = rng.random(rows) < 0.07                                  # no stable curve: every field undefined
    rejected = ~invalid & (rng.random(rows) < 0.20)                    # curve exists, target rejects the row
    radius[invalid] = np.nan
    maximum_mass[invalid] = np.nan
    nicer[invalid] = np.nan
    log_astro[invalid | rejected] = -np.inf
    correction = np.zeros(rows)
    metadata = {"role": "Route 2 EN smoke: synthetic bank", "rows": rows, "seed": seed, "sector": sector,
                "forbidden_artifacts_used": [], "held_out_sources_used": []}
    np.savez(path, schema=np.asarray("conditional-en-independent-physics-v2"),
             model=np.asarray(SECTORS[sector]["model"]), X=prediction, R=radius, MG=MASS_GRID,
             MM=maximum_mass.astype(np.float32), LA=log_astro, LPC=correction, NIC=nicer,
             lpc_all_lse=np.float64(logsumexp(correction)), metadata=np.asarray(json.dumps(metadata)))
    return {"kind": "synthetic", "rows": rows, "invalid_rows": int(invalid.sum()), "rejected_rows": int(rejected.sum())}


def bank_subset(source: Path, path: Path, rows: int) -> tuple[dict, str]:
    """The --rows rows of a real frozen EN bank with the largest base log-weight LA + LPC inside the 6-sigma
    nuclear support. A uniform draw of a few hundred rows would keep almost no tilt effective sample size (the
    frozen banks have in-region ESS of about 100-500 over ~10^5 rows), so the unchanged production gate
    (ESS >= 20) would reject nearly every one-source label scenario."""
    with np.load(source, allow_pickle=False) as bank:
        model = str(bank["model"].item())
        prediction = np.asarray(bank["X"], dtype=np.float64)
        base = np.asarray(bank["LA"]) + np.asarray(bank["LPC"])
        support = np.max(np.abs((prediction - OBSERVATION) / SIGMA), axis=1) <= 6.0
        candidates = np.flatnonzero(np.isfinite(base) & support)
        chosen = np.sort(candidates[np.argsort(-base[candidates], kind="stable")[:rows]])
        fields = {key: np.asarray(bank[key])[chosen] for key in ("X", "R", "MM", "LA", "LPC", "NIC")}
        fields["MG"] = np.asarray(bank["MG"])
    metadata = {"role": "Route 2 EN smoke: subset of a frozen EN bank", "source_bank": source.name,
                "source_bank_sha256": sha256(source), "rows": rows,
                "selection": "largest LA + LPC inside the 6-sigma nuclear support",
                "forbidden_artifacts_used": [], "held_out_sources_used": []}
    np.savez(path, schema=np.asarray("conditional-en-independent-physics-v2"), model=np.asarray(model),
             lpc_all_lse=np.float64(logsumexp(fields["LPC"])), metadata=np.asarray(json.dumps(metadata)), **fields)
    sector = "hyperonic" if model == "ddb-hyperonic" else "nucleonic"
    return {"kind": "subset of a frozen EN bank", "source": str(source), "source_sha256": metadata["source_bank_sha256"],
            "rows": rows, "candidates": int(len(candidates))}, sector


def baseline_folder(work: Path) -> Path:
    """The three baseline NICER files only: the label builder and trainer refuse a folder with held-out files."""
    folder = work / "baseline"
    folder.mkdir()
    for name in BASELINE_FILES:
        (folder / name).symlink_to(OBSERVATIONS / name)
    return folder


def load_trainer():
    """Import s2_green_en.py from green_en/ with this repository first on the path."""
    os.environ["DDB_REPO_ROOT"] = str(ROOT)
    before = list(sys.path)
    spec = importlib.util.spec_from_file_location("s2_green_en", TRAINER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["s2_green_en"] = module
    spec.loader.exec_module(module)
    added = [entry for entry in sys.path if entry not in before]
    foreign = [entry for entry in added if not Path(entry).resolve().is_relative_to(ROOT)]
    for entry in foreign:                                              # e.g. the helper's fixed production path
        while entry in sys.path:
            sys.path.remove(entry)
    return module, foreign


def module_origins() -> dict:
    outside = {}
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in PACKAGES and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(ROOT):
                outside[name] = module.__file__
    return outside


# ---------------------------------------------------------------------------------------------- checks
def check(condition: bool, message: str, failures: list) -> None:
    if not condition:
        failures.append(message)


def check_labels(index_path: Path, rows_kept_max: int, failures: list) -> dict:
    receipt = json.loads(index_path.with_suffix(".json").read_text())
    with np.load(index_path, allow_pickle=False) as index:
        kept = int(len(index["kept_bank_rows"]))
        shapes = {"mixture_weight": (SCENARIOS, 3, 4), "mixture_mean": (SCENARIOS, 3, 4, 2),
                  "mixture_covariance": (SCENARIOS, 3, 4, 2, 2), "active_mask": (SCENARIOS, 3),
                  "family": (SCENARIOS, 3), "mode": (SCENARIOS,), "logC": (SCENARIOS,), "tilt_ess": (SCENARIOS,),
                  "logZ_full": (SCENARIOS, 64), "logZ_box": (SCENARIOS, 64), "nuclear_observations": (64, 7),
                  "OBS": (7,), "SIG": (7,)}
        for key, shape in shapes.items():
            value = index[key]
            check(value.shape == shape, f"label {key} shape {value.shape} != {shape}", failures)
            if value.dtype.kind == "f":
                check(bool(np.isfinite(value).all()), f"label {key} not finite", failures)
        check(int(index["ntr"]) == SCENARIOS - VALIDATION, "label ntr", failures)
        check(bool(np.isfinite(index["denominator"])), "label denominator not finite", failures)
        modes = {int(m): int(n) for m, n in zip(*np.unique(index["mode"], return_counts=True))}
    matrix = np.load(index_path.with_suffix(".sourcelogw.npy"), mmap_mode="r")
    check(0 < kept <= rows_kept_max, f"kept bank rows {kept}", failures)
    check(matrix.shape == (SCENARIOS, kept), f"source-weight matrix shape {matrix.shape}", failures)
    check(bool(np.isfinite(matrix).all()), "source-weight matrix not finite", failures)
    check(receipt["status"] == "PASS", f"label summary status {receipt['status']}", failures)
    return {"kept_bank_rows": kept, "attempted_scenarios": receipt["attempted"], "mode_counts": modes,
            "matrix_shape": list(matrix.shape), "label_seconds": round(receipt["label_seconds"], 1)}


def check_trainer(checkpoint: Path, nm: int, nr: int, steps: int, failures: list) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    report = json.loads(checkpoint.with_suffix(".json").read_text())
    check(payload["model_type"] == "green_function_en_v1", "checkpoint model type", failures)
    check(list(payload["lattice"]) == [nm, nr], f"checkpoint lattice {payload['lattice']}", failures)
    check(np.isfinite(payload["mean"]), "checkpoint mean not finite", failures)
    tensors = payload["state_dict"]
    check(all(bool(torch.isfinite(value).all()) for value in tensors.values()), "checkpoint weights not finite", failures)
    families = report["blind_synthetic_one_source"]
    check(set(families) == {"axis", "gauss", "box", "paper"}, f"blind families {sorted(families)}", failures)
    for family, entry in families.items():
        check(set(entry) == {"all", "j0030", "j0740", "j0437"}, f"blind {family} slots {sorted(entry)}", failures)
        check(all(np.isfinite(value["mae"]) for value in entry.values()), f"blind {family} mae not finite", failures)
    identity = np.asarray(report["identity_paper_residual"])
    check(identity.shape == (5,) and bool(np.isfinite(identity).all()), "identity residual", failures)
    clouds = report["real_baseline_clouds_no_encoding"]
    check(set(clouds) == {"j0030", "j0740", "j0437"}, f"baseline clouds {sorted(clouds)}", failures)
    for name, entry in clouds.items():
        values = np.asarray(entry["network_logZ"])
        check(values.shape == (5,) and bool(np.isfinite(values).all()), f"baseline cloud {name}", failures)
    check([item["step"] for item in report["history"]] == [1, steps], "training history steps", failures)
    check(all(np.isfinite(item["train_mse"]) for item in report["history"]), "training loss not finite", failures)
    return {"parameters": int(sum(value.numel() for value in tensors.values())),
            "loss_first_last": [report["history"][0]["train_mse"], report["history"][-1]["train_mse"]],
            "training_seconds": round(report["training_seconds"], 1)}


def check_query(output: Path, failures: list) -> dict:
    receipt = json.loads(output.read_text())
    results = receipt["results"]
    names = ["A1", "K0_200", "K0_260", "Jsym_29", "Jsym_36", "J0614", "J1231", "J1614"]
    check(sorted(results) == sorted(names), f"query configurations {sorted(results)}", failures)
    values = [results[name]["log_evidence"] for name in names if name in results]
    check(len(values) == 8 and bool(np.isfinite(values).all()), "query log-evidence not finite", failures)
    check(len(receipt["source_receipts"]) == 3, "query source entries", failures)
    check(receipt["checkpoint_sha256_before"] == receipt["checkpoint_sha256_after"], "checkpoint changed", failures)
    return {"log_evidence": {name: results[name]["log_evidence"] for name in names if name in results}}


# ---------------------------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sector", choices=tuple(SECTORS), default="hyperonic",
                        help="production settings to use with the synthetic bank (ignored with --bank)")
    parser.add_argument("--bank", type=Path, help="take --rows rows of this frozen EN bank instead of a synthetic bank")
    parser.add_argument("--rows", type=int, default=SMOKE["rows"])
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--keep", type=Path, help="keep the work files in this new folder")
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    started = time.perf_counter()
    failures: list[str] = []
    stages: dict[str, dict] = {}
    report = {"status": "FAIL", "device": "cpu", "torch": torch.__version__, "threads": torch.get_num_threads(),
              "sizes": {**SMOKE, "rows": arguments.rows, "scenarios": SCENARIOS}}
    work = Path(tempfile.mkdtemp(prefix="en_smoke_"))
    logs = {stage: work / f"{stage}.log" for stage in ("labels", "trainer", "query")}
    try:
        prepare_observations()
        bank = work / "bank.npz"
        if arguments.bank:
            stages["bank"], sector = bank_subset(arguments.bank, bank, arguments.rows)
        else:
            sector = arguments.sector
            stages["bank"] = synthetic_bank(bank, arguments.rows, sector)
        report["sector_settings"] = sector
        print(f"[en-smoke] bank: {stages['bank']['kind']}, {arguments.rows} rows, {sector} settings", flush=True)
        baseline = baseline_folder(work)
        domain = SECTORS[sector]["domain"]

        from inference.evidence_network.conditional import build_source_weight_cache_v2 as labels
        from inference.evidence_network.conditional import query_green_frozen_v5 as query
        trainer, removed_paths = load_trainer()
        outside = module_origins()
        check(not outside, f"modules loaded from outside this repository: {outside}", failures)
        report["sys_path_entries_removed"] = removed_paths

        index = work / "cache/green_cache_smoke.npz"
        index.parent.mkdir()
        clock = time.perf_counter()
        with open(logs["labels"], "w") as log, contextlib.redirect_stdout(log), cuda_on_cpu():
            code = run_main(labels.main, ["build_source_weight_cache_v2.py", "--bank", str(bank), "--baseline-root",
                                          str(baseline), "--output", str(index), *COMMON_LABEL,
                                          *SECTORS[sector]["label"]])
        check(code == 0, f"label builder exit status {code}", failures)
        stages["labels"] = {"seconds": round(time.perf_counter() - clock, 1),
                            **check_labels(index, arguments.rows, failures)}
        print(f"[en-smoke] label builder: {SCENARIOS} scenarios on {stages['labels']['kept_bank_rows']} kept rows "
              f"in {stages['labels']['seconds']} s", flush=True)

        checkpoint = work / "models/green_smoke.pt"
        checkpoint.parent.mkdir()
        clock = time.perf_counter()
        with open(logs["trainer"], "w") as log, contextlib.redirect_stdout(log), cuda_on_cpu():
            run_main(trainer.main, [
                "s2_green_en.py", "--cache", str(index), "--output", str(checkpoint), "--bank", str(bank),
                "--baseline-root", str(baseline), "--seed", str(SMOKE["seed"]), "--steps", str(SMOKE["steps"]),
                "--batch-scenarios", str(SMOKE["batch_scenarios"]), "--designs-per-step", str(SMOKE["designs_per_step"]),
                "--lr", "0.0006", "--nm", str(SMOKE["nm"]), "--nr", str(SMOKE["nr"]), "--mass-low", str(domain[0]),
                "--mass-high", str(domain[1]), "--radius-low", str(domain[2]), "--radius-high", str(domain[3])])
        stages["trainer"] = {"seconds": round(time.perf_counter() - clock, 1),
                             **check_trainer(checkpoint, SMOKE["nm"], SMOKE["nr"], SMOKE["steps"], failures)}
        print(f"[en-smoke] trainer: {SMOKE['steps']} optimizer steps, {SMOKE['nm']}x{SMOKE['nr']} lattice, "
              f"{stages['trainer']['parameters']} parameters, in {stages['trainer']['seconds']} s", flush=True)

        output = work / "query.json"
        clock = time.perf_counter()
        with open(logs["query"], "w") as log, contextlib.redirect_stdout(log):
            code = run_main(query.main, ["query_green_frozen_v5.py", "--checkpoint", str(checkpoint), "--heldout-root",
                                         str(OBSERVATIONS), "--output", str(output), "--device", "cpu",
                                         "--sector", sector])
        check(code == 0, f"query exit status {code}", failures)
        stages["query"] = {"seconds": round(time.perf_counter() - clock, 1), **check_query(output, failures)}
        print(f"[en-smoke] frozen query: 5 nuclear settings + 3 held-out NICER sources, all finite, "
              f"in {stages['query']['seconds']} s", flush=True)
        outside = module_origins()
        check(not outside, f"modules loaded from outside this repository: {outside}", failures)
        if arguments.keep:
            shutil.copytree(work, arguments.keep)
    except Exception:                                                  # report the failure instead of hiding it
        failures.append(traceback.format_exc(limit=6))
        for stage, log in logs.items():
            if log.is_file():
                failures.append(f"last lines of the {stage} output:\n" + "\n".join(log.read_text().splitlines()[-15:]))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    report.update({"status": "PASS" if not failures else "FAIL", "stages": stages, "failures": failures,
                   "seconds": round(time.perf_counter() - started, 1)})
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=1) + "\n")
    for failure in failures:
        print(f"[en-smoke] problem: {failure}", flush=True)
    print(f"[en-smoke] {report['status']}: label builder -> trainer -> frozen query ran on CPU in "
          f"{report['seconds']} s with {report['threads']} threads; all outputs finite and correctly shaped"
          if not failures else f"[en-smoke] FAIL after {report['seconds']} s", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
