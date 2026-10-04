#!/usr/bin/env python3
"""Fast drop-in replacement for evaluate_clean_hyperonic_support_cache.py (same CLI, same output keys).

Pipeline per shard: compiled hyperonic beta-equilibrium solver (hyp_solver_cpu, all cores) with the
certified Python solver as fallback for flagged rows; the unchanged exact TOV (tov.solve_stable_branch,
process-parallel); the batched arithmetic-identical likelihood (fast_likelihood).  With --eos-backend
certified the output is bitwise identical to the reference Python solver.  Extra provenance keys are appended; every
original key keeps its meaning.
"""
from __future__ import annotations
import argparse, hashlib, json, os, socket, sys, time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eos import get_eos
from eos.ddb_hyperonic.model import CORE_DENSITY_GRID
from likelihoods.nuclear import A1_SIGMA
from likelihoods.nicer import log_likelihood_one
from tov import solve_stable_branch
from workflows.a1_problem import A1Problem
import hyp_solver_cpu as HS
import fast_likelihood as FL


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tov_chunk(E, P, ok, mono):
    out = []
    for e, p, o in zip(E, P, ok):
        if not o:
            out.append(None); continue
        try:
            out.append(solve_stable_branch(e, p, require_monotonic_graft=mono))
        except ValueError:
            out.append(None)
    return out


def compiled_hyperonic_tables(theta, plugin, workers, backend="compiled"):
    """Return energy, pressure (MeV/fm^3), ok mask and diagnostics.

    backend="compiled": hyp_solver_cpu for every row, certified Python solver for flagged rows
    (agreement <= 3.6e-12 rel in E, 4.2e-9 rel in P on 6,400 rows; end-to-end log L within 1.3e-7).
    backend="certified": plugin.core_eos for every row (bitwise identical to the original shard)."""
    n = len(theta); grid = CORE_DENSITY_GRID.copy()
    E = np.zeros((n, grid.size)); P = np.zeros_like(E); st = np.zeros(n, np.int64); fi = np.zeros(n, np.int64)
    aset = np.zeros((n, grid.size), np.int64); nfev = np.zeros((n, 2), np.int64)
    th = np.ascontiguousarray(theta, dtype=np.float64)
    if backend == "certified":
        st[:] = 1  # every row goes through the certified solver below
    else:
        HS.solve_rows(th, grid, E, P, st, fi, aset, nfev)
    ok = st == 0
    fallback = np.nonzero(~ok)[0]
    fallback_ok = 0
    if fallback.size:
        def cert(rows):
            res = []
            for i in rows:
                try:
                    d, e, p = plugin.core_eos(th[i]); res.append((e, p))
                except Exception:
                    res.append(None)
            return res
        chunks = np.array_split(fallback, max(1, min(workers, fallback.size)))
        parts = Parallel(n_jobs=max(1, min(workers, fallback.size)))(delayed(cert)(c) for c in chunks if c.size)
        for c, part in zip([c for c in chunks if c.size], parts):
            for i, r in zip(c, part):
                if r is not None:
                    E[i], P[i] = r; ok[i] = True; fallback_ok += 1
    diag = dict(compiled_ok=int((st == 0).sum()), fallback_rows=int(fallback.size), fallback_recovered=int(fallback_ok),
                rows_with_emulated_exit=int((nfev[:, 1] > 0).sum()), status_counts={int(k): int(v) for k, v in zip(*np.unique(st, return_counts=True))})
    return E, P, ok, diag


def evaluate_rows(theta, prediction_nmp, target, mass_grid, plugin, workers, chunk=2500, eos_backend="compiled"):
    """Vectorised equivalent of build_clean_uniform_hyperonic_training_cache.evaluate_training_row."""
    n = len(theta); grid = CORE_DENSITY_GRID.copy()
    log_astro = np.full(n, np.nan); components = np.full((n, 4), np.nan); source_components = np.full((n, 3), np.nan)
    radius = np.full((n, len(mass_grid)), np.nan); tidal = np.full_like(radius, np.nan); maximum_mass = np.full(n, np.nan)
    t = {}
    t0 = time.perf_counter(); E, P, ok, diag = compiled_hyperonic_tables(theta, plugin, workers, eos_backend); t["eos_s"] = time.perf_counter() - t0; diag["eos_backend"] = eos_backend
    mono = plugin.metadata.get("require_monotonic_tov_graft", False)
    t0 = time.perf_counter()
    idx = np.arange(n); chunks = np.array_split(idx, max(1, min(workers * 8, n)))
    parts = Parallel(n_jobs=min(workers, n), batch_size=1, pre_dispatch="2*n_jobs")(delayed(_tov_chunk)(E[c], P[c], ok[c], mono) for c in chunks if c.size)
    branches = [b for part in parts for b in part]
    t["tov_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    bk = FL.BatchedGWKernel(target.gw_kernel) if target.gw_kernel is not None else None
    res = FL.evaluate_batch(target, theta, prediction_nmp, grid, E, P, branches, bk)
    t["likelihood_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    nsrc = len(target.nicer_interpolators)
    for i in range(n):
        b = branches[i]
        if b is None or not np.isfinite(res["total"][i]):
            continue  # shard: ValueError / FloatingPointError -> NaN row
        comp = np.asarray([res["maximum_mass"][i], res["nicer"][i], res["gw170817"][i], res["pqcd"][i]], dtype=np.float64)
        src = np.asarray(res["nicer_sources"][i], dtype=np.float64)
        la = float(np.sum(comp))
        if not np.isfinite(la) or not np.all(np.isfinite(src)):
            continue  # FloatingPointError("non-finite corrected target")
        if abs(float(src.sum()) - float(res["nicer"][i])) > 1.0e-10:
            continue  # RuntimeError("per-source NICER terms do not sum to target")
        covered = (mass_grid >= float(b.mass.min())) & (mass_grid <= float(b.mass.max())) & (mass_grid <= float(b.maximum_mass))
        r = np.full(len(mass_grid), np.nan); l = r.copy()
        r[covered] = np.interp(mass_grid[covered], b.mass, b.radius)
        l[covered] = np.exp(np.interp(mass_grid[covered], b.mass, np.log(np.clip(b.tidal_lambda, 1.0e-300, None))))
        log_astro[i] = la; components[i] = comp; source_components[i] = src; radius[i] = r; tidal[i] = l; maximum_mass[i] = float(b.maximum_mass)
    t["curves_s"] = time.perf_counter() - t0
    diag.update(t)
    return log_astro, components, source_components, radius, tidal, maximum_mass, diag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--template-bank", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--eos-backend", choices=("compiled", "certified"), default="compiled",
                        help="compiled: numba port of the certified Newton solver (default); certified: original Python solver, bitwise-identical output")
    arguments = parser.parse_args()
    if arguments.workers < 1:
        raise ValueError("workers must be positive")
    started_unix = time.time()
    with np.load(arguments.screen, allow_pickle=False) as screen:
        required = {"schema", "theta", "prediction", "total_proposals", "cbox", "seed", "stream_id", "reference_present", "forbidden_artifacts_used"}
        missing = required - set(screen.files)
        if missing:
            raise RuntimeError(f"support screen is missing {sorted(missing)}")
        if str(screen["schema"].item()) != "ddb-hyperonic-clean-support-screen-v1":
            raise RuntimeError("unexpected support-screen schema")
        if bool(screen["reference_present"]) or screen["forbidden_artifacts_used"].size:
            raise RuntimeError("support screen declares forbidden ancestry")
        all_theta = np.asarray(screen["theta"], dtype=np.float64)
        all_prediction = np.asarray(screen["prediction"], dtype=np.float64)
        total_proposals = int(screen["total_proposals"]); cbox = float(screen["cbox"]); seed = int(screen["seed"]); stream_id = int(screen["stream_id"])
    if not 0 <= arguments.start < arguments.stop <= len(all_theta):
        raise ValueError(f"invalid shard [{arguments.start},{arguments.stop}) for {len(all_theta)} rows")
    theta = all_theta[arguments.start:arguments.stop]
    stored_prediction = all_prediction[arguments.start:arguments.stop]
    with np.load(arguments.template_bank, allow_pickle=False) as template:
        if str(template["schema_version"].item()) != "ddbhy-bank-v1":
            raise RuntimeError("unexpected template-bank schema")
        mass_grid = np.asarray(template["MG"], dtype=np.float64)
        prior_low = np.asarray(template["THETA_LOW"], dtype=np.float64)
        prior_high = np.asarray(template["THETA_HIGH"], dtype=np.float64)
    problem = A1Problem.from_data_root(arguments.data_root, workers=arguments.workers, verify_data=True, nuclear_scenario="A1", source_scenario="A1", model="ddb-hyperonic")
    plugin = get_eos("ddb-hyperonic")
    prediction = np.empty_like(stored_prediction)
    prediction[:, 0] = theta[:, plugin.parameter_names.index("rho0")]
    prediction[:, 1:] = plugin.nuclear_observables_batch(theta)
    prediction_delta = np.abs(prediction - stored_prediction)
    maximum_prediction_delta = float(np.max(prediction_delta))
    maximum_normalized_prediction_delta = float(np.max(prediction_delta / A1_SIGMA))
    prediction_tolerance_sigma = 1.0e-4
    if maximum_normalized_prediction_delta > prediction_tolerance_sigma:
        raise RuntimeError(f"screen and evaluator predictions differ by {maximum_normalized_prediction_delta:.3e} nuclear sigma")
    rows = len(theta)
    log_astrophysical, components, source_components, radius, tidal, maximum_mass, diag = evaluate_rows(theta, prediction[:, 1:], problem.target, mass_grid, plugin, arguments.workers, eos_backend=arguments.eos_backend)
    print(f"[fast-support-cache] {rows}/{rows} exact rows; {json.dumps(diag)}", flush=True)
    target_valid = (np.isfinite(log_astrophysical) & (log_astrophysical > -1.0e29) & np.isfinite(components).all(axis=1) & np.isfinite(source_components).all(axis=1)
                    & np.isfinite(maximum_mass) & np.isfinite(radius).any(axis=1) & np.isfinite(tidal).any(axis=1))
    if not target_valid.any():
        raise RuntimeError("corrected target rejected every support row")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    template_sha = hashlib.sha256((HERE / "hyp_solver_template.py").read_bytes()).hexdigest()
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray("ddb-hyperonic-clean-support-training-cache-v1"),
            source_index=np.arange(arguments.start, arguments.stop, dtype=np.int64),
            theta=theta, prediction=prediction, log_astrophysical=log_astrophysical, astrophysical_components=components,
            nicer_source_components=source_components, target_valid=target_valid, mass_grid=mass_grid, radius=radius, tidal_lambda=tidal,
            maximum_mass=maximum_mass, prior_low=prior_low, prior_high=prior_high,
            screen_sha256=np.asarray(sha256(arguments.screen)), template_bank_sha256=np.asarray(sha256(arguments.template_bank)),
            total_proposals=np.int64(total_proposals), cbox=np.float64(cbox), seed=np.int64(seed), stream_id=np.int64(stream_id),
            shard_start=np.int64(arguments.start), shard_stop=np.int64(arguments.stop),
            prediction_tolerance_sigma=np.float64(prediction_tolerance_sigma), maximum_prediction_delta=np.float64(maximum_prediction_delta),
            maximum_normalized_prediction_delta=np.float64(maximum_normalized_prediction_delta),
            workers=np.int64(arguments.workers), hostname=np.asarray(socket.gethostname()), reference_present=np.bool_(False),
            forbidden_artifacts_used=np.asarray([], dtype="U1"), started_unix=np.float64(started_unix), finished_unix=np.float64(time.time()),
            evaluator=np.asarray("fastsolver/fast_exact_rows.py"), hyp_solver_template_sha256=np.asarray(template_sha), eos_backend=np.asarray(arguments.eos_backend),
            fast_fallback_rows=np.int64(diag["fallback_rows"]), fast_rows_with_emulated_exit=np.int64(diag["rows_with_emulated_exit"]),
        )
    os.replace(temporary, arguments.output)
    report = {"status": "PASS", "schema": "ddb-hyperonic-clean-support-training-cache-v1", "evaluator": "fastsolver/fast_exact_rows.py",
              "hyp_solver_template_sha256": template_sha, "eos_backend": arguments.eos_backend, "screen": str(arguments.screen.resolve()), "screen_sha256": sha256(arguments.screen),
              "template_bank": str(arguments.template_bank.resolve()), "template_bank_sha256": sha256(arguments.template_bank), "total_screen_rows": int(len(all_theta)),
              "total_proposals": total_proposals, "shard_start": arguments.start, "shard_stop": arguments.stop, "shard_rows": rows, "target_valid_rows": int(target_valid.sum()),
              "maximum_prediction_delta": maximum_prediction_delta, "maximum_normalized_prediction_delta": maximum_normalized_prediction_delta,
              "prediction_tolerance_sigma": prediction_tolerance_sigma, "workers": arguments.workers, "hostname": socket.gethostname(), "output": str(arguments.output.resolve()),
              "output_sha256": sha256(arguments.output), "reference_present": False, "forbidden_artifacts_used": [], "wall_seconds": time.time() - started_unix, "stage_diagnostics": diag}
    arguments.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
