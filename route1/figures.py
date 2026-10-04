#!/usr/bin/env python3
"""Redraw the data figures of the paper (Figs. 2-11, 14 figure files) from the saved results into build/figures/.
Fig. 1 is the workflow diagram; route1.compare checks it by its SHA-256."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA, OVERLAYS = "data/observations", "data/plotting_overlays"
J1614_DATA = f"{DATA}/J1614_STU_mrsamples_post_equal_weights.dat"
# File names in paper/figures: the plotting modules write the first name, the paper uses the second
# (the _j0740v2 figures show the 2024 J0740+6620 contour).
RENAMED = {f"{stem}.pdf": f"{stem}_j0740v2.pdf" for stem in (
    "FINAL_A1_methods_with_observations", "FINAL_J0614_methods_with_observations", "FINAL_J1231_methods_with_observations",
    "FINAL_hyp_A1_methods_with_observations")}
RENAMED["FINAL_hyp_A1_mass_tidal_three_methods.pdf"] = "FINAL_hyp_A1_mass_tidal_three_methods_20260924.pdf"


def jobs(out: Path, work: Path) -> dict[str, tuple[list[str], list[tuple[Path, str]]]]:
    """name -> (command, [(produced file, paper file name)]) ; empty list = produced directly under its paper name."""
    py = [sys.executable, "-m"]
    result = {
        "Fig. 2 (A1 parameter corner)": (py + ["plotting.parameter_corner", "--output-stem",
                                              str(out / "FINAL_A1_parameter_corner_methods")], []),
        "Figs. 3a, 5 (nucleonic mass-radius)": (py + ["plotting.mass_radius", "--mr-data-root", OVERLAYS,
                                                      "--pulsar-data-root", DATA, "--output-dir", str(out)], []),
        "Fig. 3b (A1 mass-tidal)": (py + ["plotting.mass_tidal", "--output-stem",
                                          str(out / "FINAL_A1_mass_tidal_three_methods")], []),
        "Fig. 4 (nuclear-data changes)": (py + ["plotting.nuclear_amortization", "--mr-data-root", OVERLAYS,
                                                "--output-stem", str(out / "FINAL_nuclear_amortization_four_scenarios_j0740v2")], []),
        "Fig. 6 (nucleonic J1614)": (py + ["route1.fig_nucleonic_j1614", "--j1614", J1614_DATA, "--overlay-root", OVERLAYS,
                                           "--output-dir", str(work / "j1614")],
                                     [(work / "j1614/J1614_mass_radius_paper_figure.pdf",
                                       "FINAL_J1614_methods_with_observations.pdf")]),
        "Fig. 7 (hyperonic A1)": (py + ["route1.fig_hyperonic_a1", "--output-dir", str(out)], []),
        "Fig. 10 (hyperonic nuclear-data changes)": (py + [
            "plotting.nuclear_amortization", "--input", "results/appendix/hyperonic_nuclear_amortization_bands.npz",
            "--report", "results/appendix/hyperonic_nuclear_amortization_report.json", "--mr-data-root", OVERLAYS,
            "--output-stem", str(out / "FINAL_hyp_nuclear_amortization_four_scenarios_j0740v2")], []),
        "Fig. 11 (P-P calibration)": (py + ["route1.fig_pp_calibration", "--output-dir", str(out)], []),
    }
    for source, label in (("J0614", "Fig. 8 left (hyperonic J0614)"), ("J1231", "Fig. 8 right (hyperonic J1231)"),
                          ("J1614", "Fig. 9 (hyperonic J1614)")):
        if source == "J1614":
            ultranest, prefix = "results/hyperonic/J1614/ultranest_mass_radius.npz", ""
        else:
            ultranest, prefix = "results/hyperonic/hyperonic_mass_radius_tail_support.npz", f"{source}_UltraNest"
        stem = work / f"hyperonic_{source}"
        result[label] = (py + ["route1.fig_hyperonic_sources", "--scenario", source,
                               "--certificate", f"results/hyperonic/{source}/anet_certificate.npz",
                               "--resample", f"results/hyperonic/{source}/anet_resample6000.npz",
                               "--curves", f"results/hyperonic/{source}/anet_curves.npz",
                               "--ultranest", ultranest, "--ultranest-prefix", prefix,
                               "--mr-data", OVERLAYS, "--pulsar-data", DATA, "--j1614-data", J1614_DATA,
                               "--output-stem", str(stem)],
                         [(stem.with_suffix(".pdf"), f"FINAL_hyp_{source}_methods_with_observations_"
                                                      f"{'20260924' if source == 'J1614' else 'j0740v2'}.pdf")])
    return result


def run(label: str, command: list[str], log_dir: Path) -> tuple[str, int]:
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    log = log_dir / (label.split(" (")[0].replace(" ", "_").replace(".", "").replace(",", "") + ".log")
    with log.open("w") as stream:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    return label, code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/figures")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    arguments = parser.parse_args()
    out, work, logs = arguments.output_dir, ROOT / "build/work", ROOT / "build/logs"
    for folder in (out, work, logs):
        folder.mkdir(parents=True, exist_ok=True)
    todo = jobs(out, work)
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        futures = [pool.submit(run, label, command, logs) for label, (command, _) in todo.items()]
        for future in concurrent.futures.as_completed(futures):
            label, code = future.result()
            print(f"[figures] {'PASS' if code == 0 else 'FAIL'} {label}", flush=True)
            if code:
                failed.append(label)
    if failed:
        print(f"[figures] FAIL: {len(failed)} figure job(s) failed; see build/logs/")
        return 1
    for label, (_, copies) in todo.items():
        for produced, name in copies:
            shutil.copyfile(produced, out / name)
    for produced, name in RENAMED.items():  # figures that show the 2024 J0740+6620 contour
        os.replace(out / produced, out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
