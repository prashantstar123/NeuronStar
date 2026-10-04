#!/usr/bin/env python3
"""Train the portable sequential neural posterior estimator for A1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eos.ddb import PARAMETER_NAMES, PRIOR_HIGH, PRIOR_LOW
from likelihoods.nuclear import A1_OBSERVATION, A1_SIGMA
from workflows import A1Problem
from workflows.resources import DEFAULT_A1_CERTIFICATE

CHECKPOINT_DIR = ROOT / "inference" / "tsnpe" / "checkpoints"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def width90(sample: np.ndarray) -> np.ndarray:
    return np.percentile(sample, 95.0, axis=0) - np.percentile(
        sample, 5.0, axis=0
    )


class NuclearMmaxSeedPosterior:
    """Adapter around the frozen nuclear+Mmax NSF used for round-0 support."""

    def __init__(self, checkpoint: Path, standardization: Path, device: str):
        import zuko

        self.device = device
        self.low = torch.as_tensor(PRIOR_LOW, dtype=torch.float32, device=device)
        self.high = torch.as_tensor(PRIOR_HIGH, dtype=torch.float32, device=device)
        with np.load(standardization, allow_pickle=False) as archive:
            self.theta_mean, self.theta_scale, self.x_mean, self.x_scale = [
                torch.as_tensor(archive[key], dtype=torch.float32, device=device)
                for key in ("tm", "ts", "xm", "xs")
            ]
        self.flow = zuko.flows.NSF(
            7,
            7,
            transforms=8,
            hidden_features=[192, 192, 192],
            bins=10,
        ).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        # Zuko renamed the two Normal base-distribution buffers after this
        # checkpoint was frozen.  Migrate only those known names, then retain
        # strict loading for every learned parameter and remaining buffer.
        if "base._0" in state or "base._1" in state:
            if set(("base._0", "base._1")) - set(state):
                raise RuntimeError("incomplete frozen-checkpoint Zuko base buffers")
            if "base.loc" in state or "base.scale" in state:
                raise RuntimeError("ambiguous old/new Zuko base buffers")
            state["base.loc"] = state.pop("base._0")
            state["base.scale"] = state.pop("base._1")
        self.flow.load_state_dict(state, strict=True)
        self.flow.eval()
        observation = torch.as_tensor(
            A1_OBSERVATION, dtype=torch.float32, device=device
        )
        self.standardized_observation = (
            (observation - self.x_mean) / self.x_scale
        )[None]

    def sample(self, shape, x=None, show_progress_bars=False):
        count = int(np.prod(shape))
        with torch.no_grad():
            sample = self.flow(self.standardized_observation).sample((count,))
            sample = sample.reshape(count, 7) * self.theta_scale + self.theta_mean
        return torch.clamp(sample, self.low + 1e-6, self.high - 1e-6)

    def log_prob(self, theta, x=None, norm_posterior=False):
        theta = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        standardized = (theta - self.theta_mean) / self.theta_scale
        with torch.no_grad():
            density = self.flow(self.standardized_observation).log_prob(
                standardized
            ) - torch.log(self.theta_scale).sum()
        inside = ((theta > self.low) & (theta < self.high)).all(dim=-1)
        return torch.where(
            inside, density, torch.full_like(density, -1e30)
        )

    def set_default_x(self, x):
        return self


class A1AcceptRejectSimulator:
    """TSNPE simulator whose acceptance product is the shared A1 target."""

    def __init__(self, problem: A1Problem, seed: int = 0, smoke: bool = False):
        self.problem = problem
        self.rng = np.random.default_rng(seed)
        self.log_reference: float | None = None
        self.smoke = bool(smoke)

    def terms(self, theta: np.ndarray):
        prediction, _log_astro, components = (
            self.problem.evaluate_evidence_cache_batch(theta)
        )
        # columns: maximum_mass, NICER, GW170817, pQCD
        log_mmax = components[:, 0]
        log_other = np.sum(components[:, 1:], axis=1)
        return prediction, log_mmax, log_other

    def freeze_reference(self, theta: np.ndarray) -> dict:
        _prediction, _log_mmax, log_other = self.terms(theta)
        finite = np.isfinite(log_other)
        self.log_reference = (
            float(np.max(log_other[finite]) + 0.5) if finite.any() else 0.0
        )
        return {
            "rows": int(len(theta)),
            "valid_rows": int(finite.sum()),
            "log_reference": self.log_reference,
        }

    def __call__(self, theta: torch.Tensor):
        if self.log_reference is None:
            raise RuntimeError("TSNPE acceptance reference has not been frozen")
        values = np.asarray(theta.detach().cpu(), dtype=np.float64)
        prediction, log_mmax, log_other = self.terms(values)
        valid = (
            np.isfinite(prediction).all(axis=1)
            & np.isfinite(log_mmax)
            & np.isfinite(log_other)
        )
        log_acceptance = log_mmax + np.minimum(
            log_other - self.log_reference, 0.0
        )
        probability = np.where(valid, np.exp(log_acceptance), 0.0)
        # A smoke run checks simulator/trainer plumbing with tiny samples.  It
        # accepts every physically valid row so that a random 32-row wiring
        # test cannot fail merely because the production rejection target is
        # intentionally sharp.  SMOKE_COMPLETED is never a scientific gate.
        accepted = (
            valid
            if self.smoke
            else valid & (self.rng.random(len(values)) < probability)
        )
        simulated = prediction.copy()
        simulated[~accepted] = np.nan
        simulated += self.rng.normal(0.0, 1.0, simulated.shape) * (
            A1_SIGMA[None, :]
        )
        return torch.as_tensor(simulated, dtype=torch.float32), int(accepted.sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n0", type=int, default=30_000)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--pilot", type=int, default=2_000)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--posterior-samples", type=int, default=4_000)
    parser.add_argument("--final-samples", type=int, default=20_000)
    parser.add_argument("--support-samples", type=int, default=200_000)
    parser.add_argument("--support-quantile", type=float, default=1e-4)
    parser.add_argument("--hidden-features", type=int, default=256)
    parser.add_argument("--transforms", type=int, default=10)
    parser.add_argument("--max-training-epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--seed-checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "flow_nucmm_tight.pt",
    )
    parser.add_argument(
        "--seed-standardization",
        type=Path,
        default=CHECKPOINT_DIR / "flow_nucmm_tight_std.npz",
    )
    parser.add_argument("--certificate", type=Path, default=DEFAULT_A1_CERTIFICATE)
    parser.add_argument("--skip-target-gate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.skip_target_gate and not arguments.smoke:
        raise ValueError("--skip-target-gate is permitted only with --smoke")

    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform, RestrictedPrior, get_density_thresholder

    started = time.time()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    device = arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    problem = A1Problem.from_data_root(arguments.data_root, workers=arguments.workers)
    target_gate = (
        {"status": "SKIPPED"}
        if arguments.skip_target_gate
        else problem.certify(arguments.certificate)
    )
    if target_gate["status"] not in {"PASS", "SKIPPED"}:
        raise RuntimeError(f"shared target gate failed: {target_gate}")

    low = torch.as_tensor(PRIOR_LOW, dtype=torch.float32, device=device)
    high = torch.as_tensor(PRIOR_HIGH, dtype=torch.float32, device=device)
    prior = BoxUniform(low=low, high=high)
    observed = torch.as_tensor(
        A1_OBSERVATION, dtype=torch.float32, device=device
    )
    seed_posterior = NuclearMmaxSeedPosterior(
        arguments.seed_checkpoint, arguments.seed_standardization, device
    )
    accept = get_density_thresholder(
        seed_posterior,
        quantile=arguments.support_quantile,
        num_samples_to_estimate_support=arguments.support_samples,
    )
    proposal = RestrictedPrior(
        prior,
        accept,
        posterior=seed_posterior,
        sample_with="sir",
        device=device,
    )
    simulator = A1AcceptRejectSimulator(
        problem, seed=arguments.seed, smoke=arguments.smoke
    )
    pilot_theta = (
        proposal.sample((arguments.pilot,)).detach().cpu().numpy().astype(np.float64)
    )
    pilot_gate = simulator.freeze_reference(pilot_theta)

    estimator_builder = posterior_nn(
        model="nsf",
        hidden_features=arguments.hidden_features,
        num_transforms=arguments.transforms,
    )
    inference = NPE(
        prior=prior, density_estimator=estimator_builder, device=device
    )
    history = []
    posterior = None
    estimator = None
    for round_index in range(arguments.max_rounds):
        round_started = time.time()
        simulations = arguments.n0 if round_index == 0 else arguments.n
        theta = proposal.sample((simulations,))
        x, accepted = simulator(theta.cpu())
        theta = theta.to(device)
        x = x.to(device)
        train_kwargs = {
            "force_first_round_loss": True,
            "show_train_summary": False,
        }
        if arguments.max_training_epochs is not None:
            train_kwargs["max_num_epochs"] = arguments.max_training_epochs
        estimator = inference.append_simulations(
            theta, x, exclude_invalid_x=True
        ).train(**train_kwargs)
        posterior = inference.build_posterior(estimator).set_default_x(observed)
        sample = (
            posterior.sample(
                (arguments.posterior_samples,),
                x=observed,
                show_progress_bars=False,
            )
            .detach()
            .cpu()
            .numpy()
        )
        np.save(arguments.output_dir / f"posterior_round_{round_index}.npy", sample)
        median = np.median(sample, axis=0)
        width = width90(sample)
        entry = {
            "round": round_index,
            "simulations": simulations,
            "accepted": accepted,
            "median": median.tolist(),
            "width90": width.tolist(),
            "wall_seconds": time.time() - round_started,
        }
        history.append(entry)
        (arguments.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        if round_index >= 2:
            previous_median = np.asarray(history[-2]["median"])
            previous_width = np.asarray(history[-2]["width90"])
            median_change = np.max(
                np.abs(median - previous_median) / (previous_width + 1e-12)
            )
            width_change = np.max(
                np.abs(width - previous_width) / (previous_width + 1e-12)
            )
            if median_change < 0.05 and width_change < 0.05:
                entry["self_converged"] = True
                break
        accept = get_density_thresholder(
            posterior,
            quantile=arguments.support_quantile,
            num_samples_to_estimate_support=arguments.support_samples,
        )
        proposal = RestrictedPrior(
            prior,
            accept,
            posterior=posterior,
            sample_with="sir",
            device=device,
        )
    if posterior is None or estimator is None:
        raise RuntimeError("TSNPE completed no training rounds")
    final = (
        posterior.sample(
            (arguments.final_samples,), x=observed, show_progress_bars=False
        )
        .detach()
        .cpu()
        .numpy()
    )
    posterior_path = arguments.output_dir / "posterior_FINAL.npy"
    estimator_path = arguments.output_dir / "density_estimator.pt"
    np.save(posterior_path, final)
    torch.save(estimator, estimator_path)
    report = {
        "status": (
            "SMOKE_COMPLETED" if arguments.smoke else "PROPOSAL_READY_FOR_MIS"
        ),
        "method": "TSNPE",
        "scientific_posterior_certified": False,
        "certification_stage": "run corrected TSNPE+MIS with its ESS gate",
        "parameter_names": list(PARAMETER_NAMES),
        "target_gate": target_gate,
        "pilot_gate": pilot_gate,
        "smoke_accepts_all_physically_valid_rows": bool(arguments.smoke),
        "rounds_completed": len(history),
        "simulations_total": arguments.pilot
        + sum(item["simulations"] for item in history),
        "history": history,
        "device": device,
        "seed_checkpoint_sha256": sha256(arguments.seed_checkpoint),
        "seed_standardization_sha256": sha256(arguments.seed_standardization),
        "posterior_sha256": sha256(posterior_path),
        "estimator_sha256": sha256(estimator_path),
        "wall_seconds": time.time() - started,
    }
    (arguments.output_dir / "run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
