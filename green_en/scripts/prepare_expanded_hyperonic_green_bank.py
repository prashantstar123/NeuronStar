#!/usr/bin/env python3
"""Build a count-corrected full-prior bank for the hyperonic Green EN.

The construction combines one uniform-prior stream with one or more
independent, deterministic nuclear-support screens.  Every retained support
row represents an original full-prior proposal; the known proposal counts are
used in a deterministic-mixture correction.  Posterior samples and outputs
from UltraNest, A-NET, TSNPE, or another EN are not accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


OBSERVATION = np.asarray(
    [0.153, -16.1, 230.0, 32.5, 0.505714285714279,
     1.24142857142857, 2.4857142857143],
    dtype=np.float64,
)
SIGMA = np.asarray(
    [0.005, 0.2, 40.0, 1.8, 0.194285714285714,
     0.608571428571429, 1.38285714285714],
    dtype=np.float64,
)
CASES = {
    "base": {},
    "K0_200": {2: 200.0},
    "K0_260": {2: 260.0},
    "Jsym_29": {3: 29.0},
    "Jsym_36": {3: 36.0},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_part(path: Path, role: str) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "theta", "prediction", "log_astrophysical",
            "astrophysical_components", "nicer_source_components",
            "target_valid", "mass_grid", "radius", "maximum_mass",
            "reference_present", "forbidden_artifacts_used",
        }
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"{path} lacks fields {sorted(missing)}")
        if bool(archive["reference_present"]) or archive["forbidden_artifacts_used"].size:
            raise RuntimeError(f"{path} declares forbidden/reference ancestry")
        if role == "support":
            extra = {"screen_sha256", "total_proposals", "cbox", "seed", "source_index"}
            missing = extra - set(archive.files)
            if missing:
                raise RuntimeError(f"{path} lacks support fields {sorted(missing)}")
        result: dict[str, object] = {
            key: np.asarray(archive[key]) for key in required
            if key not in {"reference_present", "forbidden_artifacts_used"}
        }
        if role == "support":
            for key in ("screen_sha256", "total_proposals", "cbox", "seed", "source_index"):
                result[key] = np.asarray(archive[key])
    result["path"] = path
    result["sha256"] = sha256(path)
    return result


def direct_metrics(
    prediction: np.ndarray,
    log_astro: np.ndarray,
    correction: np.ndarray,
    denominator: float,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, changes in CASES.items():
        observation = OBSERVATION.copy()
        for column, value in changes.items():
            observation[column] = value
        standardized = (prediction - observation) / SIGMA
        log_weight = correction + log_astro - 0.5 * np.sum(standardized**2, axis=1)
        finite = np.isfinite(log_weight)
        normalization = float(logsumexp(log_weight[finite]))
        weight = np.exp(log_weight[finite] - normalization)
        result[name] = {
            "log_evidence": normalization - denominator,
            "effective_sample_size": float(1.0 / np.sum(weight**2)),
            "naive_monte_carlo_error": float(np.sqrt(np.sum(weight**2))),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform-part", type=Path, action="append", required=True)
    parser.add_argument("--support-part", type=Path, action="append", required=True)
    parser.add_argument("--base-proposals", type=int, default=600_000)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    if arguments.base_proposals < 1:
        raise ValueError("base proposal count must be positive")

    uniform = [load_part(path, "uniform") for path in arguments.uniform_part]
    support = [load_part(path, "support") for path in arguments.support_part]
    parts = uniform + support
    reference_grid = np.asarray(parts[0]["mass_grid"], dtype=np.float64)
    for part in parts:
        if not np.array_equal(np.asarray(part["mass_grid"], dtype=np.float64), reference_grid):
            raise RuntimeError("input parts use different mass grids")

    streams: dict[str, dict[str, object]] = {}
    for part in support:
        identity = str(np.asarray(part["screen_sha256"]).item())
        count = int(np.asarray(part["total_proposals"]).item())
        radius = float(np.asarray(part["cbox"]).item())
        seed = int(np.asarray(part["seed"]).item())
        stream = streams.setdefault(
            identity,
            {"total_proposals": count, "cbox": radius, "seed": seed,
             "indices": [], "parts": []},
        )
        if (stream["total_proposals"], stream["cbox"], stream["seed"]) != (count, radius, seed):
            raise RuntimeError(f"inconsistent metadata within support stream {identity}")
        stream["indices"].append(np.asarray(part["source_index"], dtype=np.int64))
        stream["parts"].append(str(Path(part["path"]).resolve()))
    support_radius = {float(stream["cbox"]) for stream in streams.values()}
    if support_radius != {3.0}:
        raise RuntimeError(f"support streams do not all use cbox=3: {support_radius}")
    for identity, stream in streams.items():
        index = np.concatenate(stream["indices"])
        if len(np.unique(index)) != len(index):
            raise RuntimeError(f"support stream {identity} has duplicate selected rows")
        if index.min() != 0 or index.max() + 1 != len(index):
            raise RuntimeError(f"support stream {identity} is incomplete")

    prediction64 = np.concatenate(
        [np.asarray(part["prediction"], dtype=np.float64) for part in parts]
    )
    theta = np.concatenate([np.asarray(part["theta"], dtype=np.float64) for part in parts])
    radius = np.concatenate([np.asarray(part["radius"], dtype=np.float32) for part in parts])
    maximum_mass = np.concatenate(
        [np.asarray(part["maximum_mass"], dtype=np.float32) for part in parts]
    )
    target_valid = np.concatenate(
        [np.asarray(part["target_valid"], dtype=bool) for part in parts]
    )
    raw_log_astro = np.concatenate(
        [np.asarray(part["log_astrophysical"], dtype=np.float64) for part in parts]
    )
    astro_components = np.concatenate(
        [np.asarray(part["astrophysical_components"], dtype=np.float64) for part in parts]
    )
    nicer_components = np.concatenate(
        [np.asarray(part["nicer_source_components"], dtype=np.float64) for part in parts]
    )
    log_astro = np.where(target_valid, raw_log_astro, -np.inf)
    finite = np.isfinite(log_astro)
    component_error = float(np.max(np.abs(astro_components[finite].sum(1) - log_astro[finite])))
    nicer_error = float(np.max(np.abs(nicer_components[finite].sum(1) - astro_components[finite, 1])))
    if component_error > 1.0e-8 or nicer_error > 1.0e-8:
        raise RuntimeError(
            f"likelihood-component gate failed: total={component_error:.3e}, NICER={nicer_error:.3e}"
        )

    support_indicator = np.all(
        np.abs((prediction64 - OBSERVATION) / SIGMA) <= 3.0,
        axis=1,
    )
    uniform_rows = sum(len(np.asarray(part["theta"])) for part in uniform)
    if not support_indicator[uniform_rows:].all():
        raise RuntimeError("a retained support row lies outside the declared box")
    total_support_proposals = sum(int(stream["total_proposals"]) for stream in streams.values())
    correction = math.log(arguments.base_proposals) - np.log(
        arguments.base_proposals
        + total_support_proposals * support_indicator.astype(np.float64)
    )
    denominator = math.log(arguments.base_proposals)

    parents = [
        {"path": str(Path(part["path"]).resolve()), "sha256": part["sha256"],
         "role": "uniform" if number < len(uniform) else "support"}
        for number, part in enumerate(parts)
    ]
    stream_receipts = [
        {"screen_sha256": identity, "total_proposals": int(stream["total_proposals"]),
         "cbox": float(stream["cbox"]), "seed": int(stream["seed"]),
         "selected_rows": int(sum(len(index) for index in stream["indices"])),
         "parts": stream["parts"]}
        for identity, stream in sorted(streams.items())
    ]
    metadata = {
        "lineage_class": "independent_conditional_evidence_network",
        "model": "ddb-hyperonic",
        "role": "expanded exact full-prior Green-EN physics bank",
        "base_total_proposals": arguments.base_proposals,
        "support_total_proposals": total_support_proposals,
        "support_streams": stream_receipts,
        "parents": parents,
        "forbidden_artifacts_used": [],
        "held_out_sources_used": [],
    }
    atomic_savez(
        arguments.output,
        schema=np.asarray("conditional-en-independent-physics-v2"),
        model=np.asarray("ddb-hyperonic"),
        X=prediction64.astype(np.float32),
        theta=theta.astype(np.float32),
        R=radius,
        MG=reference_grid.astype(np.float32),
        MM=maximum_mass,
        LA=log_astro,
        LPC=correction,
        AC=astro_components,
        NIC=nicer_components,
        lpc_all_lse=np.float64(denominator),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = {
        "status": "PASS",
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "rows": int(len(prediction64)),
        "finite_rows": int(finite.sum()),
        "uniform_retained_rows": int(uniform_rows),
        "support_retained_rows": int(len(prediction64) - uniform_rows),
        "base_total_proposals": arguments.base_proposals,
        "support_total_proposals": total_support_proposals,
        "likelihood_component_max_abs_error": component_error,
        "nicer_component_max_abs_error": nicer_error,
        "direct_evidence": direct_metrics(prediction64, log_astro, correction, denominator),
        "forbidden_artifacts_used": [],
        "held_out_sources_used": [],
        "support_streams": stream_receipts,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
