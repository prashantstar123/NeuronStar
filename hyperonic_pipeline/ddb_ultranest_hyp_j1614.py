#!/usr/bin/env python3
"""UltraNest runner for hyperonic DDB with J1614 replacing J0740.

This is the hyperonic counterpart of the certified nucleonic
``ddb_ultranest_sub.py``.  It retains the published A1 likelihood assembly:
the soft maximum-mass gate, J0030/J0740/J0437 NICER KDEs, GW170817 KDE,
pQCD at 1.2 fm^-3, and the seven-dimensional nuclear Gaussian kernel.

The nuclear forward is the certified nucleonic ``nmp_batch`` because the two
hyperon ratios do not enter saturation properties.  Core EOS tables and
crust/Love/TOV curves come only from ``generate_ddbhy_bank``'s shared row
forward.  One loky pool parallelizes complete hyperonic rows; no worker starts
another process pool.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, NamedTuple

# These must be fixed before importing JAX, h5py, or the certified forwards.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("CUDA_ROOT", "/tmp")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent
CERTIFIED_DDB_DIR = Path(
    os.environ.get(
        "CERTIFIED_DDB_DIR",
        "/home/nucleartheory/Desktop/validated_code_DDB",
    )
).resolve()
ASTRO_DIR = Path(
    os.environ.get("CERTIFIED_ASTRO_DIR", "/home/nucleartheory/ddb_astro_mod")
).resolve()
HYPERON_PROJECT_DIR = Path(
    os.environ.get("HYPERON_PROJECT_DIR", "/home/nucleartheory/ddb_hyperon_project")
).resolve()

for path in (Path.home(), CERTIFIED_DDB_DIR, ASTRO_DIR, HYPERON_PROJECT_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import numpy as np  # noqa: E402

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

# pqcd_like imports pQCD by its top-level name.  The certified desktop stack
# resolves that module from the installed InferenceWorkflow package.
try:  # noqa: E402
    import InferenceWorkflow as _IW
except ImportError:  # pragma: no cover - the certified environment supplies it.
    _IW = None
else:
    inference_path = os.path.dirname(_IW.__file__)
    if inference_path not in sys.path:
        sys.path.insert(0, inference_path)

import jax.numpy as jnp  # noqa: E402
import gw_like as GW  # noqa: E402
import nicer_like as NL  # noqa: E402
import pqcd_like as PQ_  # noqa: E402
import ddb_certified_forward as rn  # noqa: E402
from joblib import Parallel, delayed, parallel_config  # noqa: E402

from generate_ddbhy_bank import (  # noqa: E402
    HYPERON_HIGH,
    HYPERON_LOW,
    _stable_mrl_curve,
    _workers_from_environment,
    solve_hyperonic_core,
)


FloatArray = np.ndarray
LOG_LIKELIHOOD_FLOOR = -1.0e100
CERT_VALID_FLOOR = -1.0e50
CERT_TOLERANCE = 1.0e-4
CHUNK_SIZE = 2500
JAX_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 2500)

PARAMETER_NAMES = [
    "a_sig",
    "a_om",
    "a_rho",
    "G_sig",
    "G_om",
    "G_rho",
    "rho0",
    "x_sig_Lambda",
    "x_sig_Xi",
]
PRIOR_LOW = np.concatenate(
    [np.asarray(rn.THETA_LOW, dtype=np.float64), HYPERON_LOW]
)
PRIOR_HIGH = np.concatenate(
    [np.asarray(rn.THETA_HIGH, dtype=np.float64), HYPERON_HIGH]
)

SWAP = os.environ.get("DDB_SWAP", "J1614")
SWAP_FILES = {
    "J0614": (os.environ.get("DDB_J0614_FILE",
              "/home/nucleartheory/Downloads/MR_samples_and_contours/J0614_ST_PDT_20kLP_0p05SE_0p1ET_mrsamples_post_equal_weights.dat"), 0, 1, None, 2),
    "J1231": (os.environ.get("DDB_J1231_FILE",
              "/home/nucleartheory/Downloads/NICER_2026_zenodo/mr_samples_and_contours/PDTU_H_R1014/J1231_R1014_wmrsamples.txt"), 1, 2, 0, 0),
    "J1614": (os.environ.get("DDB_J1614_FILE",
              "/home/nucleartheory/ddb_j1614_validation_20260913/data/J1614_STU_mrsamples_post_equal_weights.dat"), 0, 1, None, 1),
}
OBS_DATA_DIR = Path(os.environ.get("DDB_OBS_DATA_DIR", "/home/nucleartheory/ddb_obs_data"))
J0437_FILE = Path(os.environ.get(
    "DDB_J0437_FILE",
    "/home/nucleartheory/Desktop/ROTATION_SINGLE_FLUID_PROTOSTRANGESTAR/"
    "observational_data/nlive20000_expf3.3_noCONST_noMM_tol0.1post_equal_weights.dat",
))
_srcs = [
    [OBS_DATA_DIR / "J0030_2spot_RM.txt", 1, 0, None],
    [OBS_DATA_DIR / "J0740_NICERXMM_full_mr.txt", 1, 0, 2],
    [J0437_FILE, 0, 1, None],
]
if SWAP:
    _p, _mc, _rc, _wc, _slot = SWAP_FILES[SWAP]
    _srcs[_slot] = [Path(_p), _mc, _rc, _wc]
NICER_SOURCES = tuple(tuple(x) for x in _srcs)
GW_FILE = Path(os.environ.get(
    "DDB_GW_FILE",
    "/home/nucleartheory/gw_eos_tmnre/data/real_events/GW170817/GW170817_GWTC-1.hdf5",
))
NUCLEAR_BANK = Path(os.environ.get(
    "DDB_NUCLEAR_BANK",
    "/home/nucleartheory/sbi-ddb-paper/FINAL_DDB_sig10/joint/joint_prior_bank4_20260729.npz",
))


class A1LikelihoodData(NamedTuple):
    """Read-only observational objects shared by parent and row workers."""

    nicer_grids: tuple[Any, Any, Any]
    gw_kernel: Any
    gw_chirp_mass: float
    nuclear_observations: FloatArray
    nuclear_sigmas: FloatArray


class HyperonicRowForward(NamedTuple):
    """Core table and stable curve returned by one row worker."""

    density: FloatArray
    energy: FloatArray
    pressure: FloatArray
    mass: FloatArray
    radius: FloatArray
    tidal_lambda: FloatArray
    maximum_mass: float


def build_a1_likelihood_data(*, verbose: bool = True) -> A1LikelihoodData:
    """Build the exact certified A1 NICER/GW objects and nuclear kernel."""

    for path, *_ in NICER_SOURCES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing certified NICER input: {path}")
    if not GW_FILE.is_file():
        raise FileNotFoundError(f"Missing certified GW170817 HDF5 file: {GW_FILE}")
    if not NUCLEAR_BANK.is_file():
        raise FileNotFoundError(f"Missing certified nuclear bank: {NUCLEAR_BANK}")

    if verbose:
        print("[UN:HYP-A1] building certified NICER and GW grids ...", flush=True)
    started = time.perf_counter()
    grids = tuple(
        NL.build_nicer_grid(
            str(path),
            mcol=mass_column,
            rcol=radius_column,
            wcol_or_None=weight_column,
            nM=150,
            nR=150,
            max_rows=20000,
            bw=0.08,
        )
        for path, mass_column, radius_column, weight_column in NICER_SOURCES
    )
    gw_kernel, gw_chirp_mass = GW.load_gw_kde(
        str(GW_FILE),
        subsample=4000,
    )
    with np.load(NUCLEAR_BANK, allow_pickle=False) as bank:
        observations = np.asarray(bank["OBS"], dtype=np.float64).copy()
        sigmas = np.asarray(bank["SIG"], dtype=np.float64).copy()
    if observations.shape != (7,) or sigmas.shape != (7,):
        raise RuntimeError("Certified nuclear OBS/SIG arrays must each have shape (7,).")
    if verbose:
        print(
            f"[UN:HYP-A1] grids + GW built in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return A1LikelihoodData(
        nicer_grids=grids,
        gw_kernel=gw_kernel,
        gw_chirp_mass=float(gw_chirp_mass),
        nuclear_observations=observations,
        nuclear_sigmas=sigmas,
    )


def _hyperonic_forward_worker(theta: FloatArray) -> HyperonicRowForward | None:
    """Solve one shared hyperonic core plus certified crust/Love/TOV curve."""

    # Defensive guard against nested pools in any imported helper.
    os.environ["TOV_NW"] = "1"
    try:
        core = solve_hyperonic_core(theta)
        curve = _stable_mrl_curve(core.energy, core.pressure)
    except Exception:
        return None
    return HyperonicRowForward(
        density=core.density,
        energy=core.energy,
        pressure=core.pressure,
        mass=curve.mass,
        radius=curve.radius,
        tidal_lambda=curve.tidal_lambda,
        maximum_mass=curve.maximum_mass,
    )


def _astro_loglike_from_forward(
    forward: HyperonicRowForward | None,
    data: A1LikelihoodData,
) -> float:
    """Assemble the certified non-nuclear A1 terms in the parent process."""

    if forward is None:
        return LOG_LIKELIHOOD_FLOOR
    try:
        value = -np.logaddexp(0.0, -(forward.maximum_mass - 2.0) / 0.05)
        for grid in data.nicer_grids:
            value += NL.logL_nicer_one(
                forward.mass,
                forward.radius,
                forward.maximum_mass,
                grid,
            )
        value += GW.logL_gw(
            forward.mass,
            forward.tidal_lambda,
            forward.maximum_mass,
            data.gw_kernel,
            data.gw_chirp_mass,
        )
        value += PQ_.logL_pqcd(
            forward.density,
            forward.energy,
            forward.pressure,
            1.2,
        )
    except Exception:
        return LOG_LIKELIHOOD_FLOOR
    return float(value) if np.isfinite(value) else LOG_LIKELIHOOD_FLOOR


class HyperonA1Likelihood:
    """Bucketed certified-NMP forward plus one non-nested row process pool."""

    def __init__(self, data: A1LikelihoodData, *, workers: int) -> None:
        if workers == 0 or workers < -1:
            raise ValueError("workers must be -1 or a positive integer.")
        self.data = data
        self.workers = workers
        self.calls = 0
        self.started = time.perf_counter()
        self._parallel_config: Any | None = None
        self._parallel: Parallel | None = None

    def __enter__(self) -> "HyperonA1Likelihood":
        if self.workers != 1:
            self._parallel_config = parallel_config(
                backend="loky",
                inner_max_num_threads=1,
            )
            self._parallel_config.__enter__()
            self._parallel = Parallel(
                n_jobs=self.workers,
                batch_size=1,
                pre_dispatch="2*n_jobs",
            )
            self._parallel.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._parallel is not None:
            self._parallel.__exit__(exc_type, exc, traceback)
            self._parallel = None
        if self._parallel_config is not None:
            self._parallel_config.__exit__(exc_type, exc, traceback)
            self._parallel_config = None

    def reset_counters(self) -> None:
        """Reset UltraNest progress accounting after startup checks."""

        self.calls = 0
        self.started = time.perf_counter()

    def _evaluate_forward_rows(
        self,
        theta: FloatArray,
    ) -> list[HyperonicRowForward | None]:
        if self.workers == 1:
            values = [_hyperonic_forward_worker(row) for row in theta]
        else:
            if self._parallel is None:
                raise RuntimeError("Use HyperonA1Likelihood as a context manager.")
            values = self._parallel(
                delayed(_hyperonic_forward_worker)(row) for row in theta
            )
        return list(values)

    def loglike_vec(self, theta: FloatArray) -> FloatArray:
        """Evaluate the exact A1 likelihood on a vector of nine-D parameters."""

        parameters = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        if parameters.ndim != 2 or parameters.shape[1] != 9:
            raise ValueError(f"Expected theta shape (N, 9), got {parameters.shape}.")
        count = len(parameters)
        output = np.full(count, LOG_LIKELIHOOD_FLOOR, dtype=np.float64)
        for start in range(0, count, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, count)
            selected = parameters[start:stop]
            size = len(selected)
            bucket = next(value for value in JAX_BUCKETS if value >= size)
            padded = (
                selected
                if size == bucket
                else np.vstack(
                    [selected, np.repeat(selected[-1:], bucket - size, axis=0)]
                )
            )
            nuclear_forward = np.asarray(
                rn.nmp_batch(
                    jnp.asarray(padded[:, :6]),
                    jnp.asarray(padded[:, 6]),
                ),
                dtype=np.float64,
            )[:size]
            nuclear_properties = np.column_stack(
                [selected[:, 6], nuclear_forward]
            )
            nuclear_loglike = -0.5 * np.sum(
                (
                    (
                        nuclear_properties
                        - self.data.nuclear_observations[None, :]
                    )
                    / self.data.nuclear_sigmas[None, :]
                )
                ** 2,
                axis=1,
            )
            forwards = self._evaluate_forward_rows(selected)
            astro_loglike = np.asarray(
                [
                    _astro_loglike_from_forward(forward, self.data)
                    for forward in forwards
                ],
                dtype=np.float64,
            )
            combined = astro_loglike + nuclear_loglike
            finite = np.isfinite(combined)
            output[start:stop][finite] = combined[finite]

        self.calls += count
        if self.calls % 20000 < count:
            elapsed = time.perf_counter() - self.started
            print(
                f"    [{elapsed / 60.0:.1f}min] {self.calls} calls "
                f"({self.calls / max(elapsed, 1.0e-12):.1f}/s)",
                flush=True,
            )
        return output


def transform(unit_cube: FloatArray) -> FloatArray:
    """Map a vectorized unit cube to the exact nine-dimensional prior box."""

    unit = np.atleast_2d(np.asarray(unit_cube, dtype=np.float64))
    if unit.ndim != 2 or unit.shape[1] != 9:
        raise ValueError(f"Expected unit-cube shape (N, 9), got {unit.shape}.")
    return PRIOR_LOW[None, :] + unit * (PRIOR_HIGH - PRIOR_LOW)[None, :]


def _certify(likelihood: HyperonA1Likelihood, path: Path) -> bool:
    """Replay a desktop reference NPZ and apply the certified comparison gate."""

    with np.load(path, allow_pickle=False) as reference_file:
        if not {"theta", "logl"}.issubset(reference_file.files):
            raise KeyError(f"{path} must contain theta and logl arrays.")
        theta = np.asarray(reference_file["theta"], dtype=np.float64)
        reference = np.asarray(reference_file["logl"], dtype=np.float64)
    started = time.perf_counter()
    actual = likelihood.loglike_vec(theta)
    elapsed = time.perf_counter() - started
    if actual.shape != reference.shape:
        raise ValueError(
            f"Certification shapes differ: {actual.shape} versus {reference.shape}."
        )
    valid_actual = actual > CERT_VALID_FLOOR
    valid_reference = reference > CERT_VALID_FLOOR
    pattern_match = bool(np.array_equal(valid_actual, valid_reference))
    common = valid_actual & valid_reference
    difference = actual[common] - reference[common]
    maximum = float(np.max(np.abs(difference))) if common.any() else np.inf
    median = float(np.median(difference)) if common.any() else np.nan
    passed = pattern_match and maximum <= CERT_TOLERANCE
    print(
        f"[CERT] valid-pattern match: {pattern_match} | rows {int(common.sum())} | "
        f"max|diff| {maximum:.2e} | median {median:+.2e}",
        flush=True,
    )
    print(
        f"[CERT] {'PASS' if passed else 'FAIL'} (tol {CERT_TOLERANCE:.0e}) | "
        f"{len(theta)} rows in {elapsed:.3f}s "
        f"({len(theta) / max(elapsed, 1.0e-12):.3f} rows/s)",
        flush=True,
    )
    return passed


def _smoke(likelihood: HyperonA1Likelihood, path: Path) -> bool:
    """Measure cold and source-matched warm likelihood throughput on 64 rows."""

    with np.load(path, allow_pickle=False) as smoke_file:
        theta = np.asarray(smoke_file["theta"], dtype=np.float64)
    if theta.shape != (64, 9):
        raise ValueError(f"Smoke input must have shape (64, 9), got {theta.shape}.")

    cold_started = time.perf_counter()
    cold = likelihood.loglike_vec(theta)
    cold_seconds = time.perf_counter() - cold_started
    warm_started = time.perf_counter()
    warm = likelihood.loglike_vec(theta)
    warm_seconds = time.perf_counter() - warm_started
    valid_match = np.array_equal(cold > CERT_VALID_FLOOR, warm > CERT_VALID_FLOOR)
    common = (cold > CERT_VALID_FLOOR) & (warm > CERT_VALID_FLOOR)
    maximum = (
        float(np.max(np.abs(cold[common] - warm[common])))
        if common.any()
        else np.inf
    )
    passed = valid_match and maximum <= CERT_TOLERANCE
    print(
        f"[SMOKE] cold: 64 rows in {cold_seconds:.3f}s "
        f"({64.0 / cold_seconds:.3f} rows/s)",
        flush=True,
    )
    print(
        f"[SMOKE] warm: 64 rows in {warm_seconds:.3f}s "
        f"({64.0 / warm_seconds:.3f} rows/s) | "
        f"repeat max|diff|={maximum:.2e} | {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return passed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--cert",
        type=Path,
        help="Evaluate a reference NPZ containing theta and logl, compare, and exit.",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Time two loglike_vec evaluations of the 64-row cert table and exit.",
    )
    parser.add_argument(
        "--smoke-file",
        type=Path,
        default=PROJECT_DIR / "hyp_ultranest_cert_64.npz",
    )
    parser.add_argument(
        "--nw",
        type=int,
        default=None,
        help="Row workers; default reads TOV_NW, then min(20, CPU count).",
    )
    parser.add_argument("--minlive", type=int, default=400)
    parser.add_argument(
        "--ndraw",
        type=int,
        default=2048,
        help="UltraNest minimum vectorized draw batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed numpy RNG before sampler construction (replica diversity).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "un_hyp_A1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "ultranest_hyp_A1.npz",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for certification, throughput smoke, or UltraNest."""

    arguments = _parser().parse_args(argv)
    workers = (
        _workers_from_environment() if arguments.nw is None else arguments.nw
    )
    if workers == 0 or workers < -1:
        raise ValueError("--nw must be -1 or a positive integer.")
    if arguments.ndraw < 1 or arguments.minlive < 1:
        raise ValueError("--ndraw and --minlive must be positive.")

    data = build_a1_likelihood_data()
    with HyperonA1Likelihood(data, workers=workers) as likelihood:
        if arguments.cert is not None:
            return 0 if _certify(likelihood, arguments.cert.resolve()) else 1
        if arguments.smoke:
            return 0 if _smoke(likelihood, arguments.smoke_file.resolve()) else 1

        center = 0.5 * (PRIOR_LOW + PRIOR_HIGH)
        sanity = float(likelihood.loglike_vec(center)[0])
        print(
            f"[UN:HYP-A1] sanity logL(prior center)={sanity:.6f}; "
            f"workers={workers}",
            flush=True,
        )

        import ultranest  # Imported only for a real sampler run.

        arguments.log_dir.mkdir(parents=True, exist_ok=True)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        likelihood.reset_counters()
        if arguments.seed:
            np.random.seed(arguments.seed)

        sampler = ultranest.ReactiveNestedSampler(
            PARAMETER_NAMES,
            likelihood.loglike_vec,
            transform,
            vectorized=True,
            log_dir=str(arguments.log_dir.resolve()),
            resume="resume",
            draw_multiple=True,
            num_bootstraps=30,
            ndraw_min=arguments.ndraw,
            storage_backend="hdf5",
        )
        started = time.perf_counter()
        result = sampler.run(
            min_num_live_points=arguments.minlive,
            show_status=False,
        )
        wall_seconds = time.perf_counter() - started
        samples = np.asarray(result["samples"], dtype=np.float64)
        np.savez(
            arguments.output,
            theta=samples,
            logZ=float(result["logz"]),
            logZerr=float(result["logzerr"]),
            ncall=likelihood.calls,
            wall_s=wall_seconds,
            OBS_MU=data.nuclear_observations,
            OBS_SIG=data.nuclear_sigmas,
            THETA_LOW=PRIOR_LOW,
            THETA_HIGH=PRIOR_HIGH,
            workers=workers,
            ndraw_min=arguments.ndraw,
            storage_backend="hdf5",
        )
        print(
            f"[UN:HYP-A1] wall={wall_seconds / 60.0:.1f}min "
            f"calls={likelihood.calls} "
            f"logZ={float(result['logz']):.3f}+-{float(result['logzerr']):.3f} "
            f"samples={len(samples)}",
            flush=True,
        )
        print(f"saved {arguments.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    # Make loky tasks resolve through the import-named module rather than a
    # cloudpickled ``__main__`` copy with an independent module state.
    from ddb_ultranest_hyp_j1614 import main as _module_main

    raise SystemExit(_module_main())
