#!/usr/bin/env python3
"""Generate one clean 9D hyperonic TSNPE restricted-prior round.

Round zero is restricted by the TSNPE-owned nuclear-plus-maximum-mass seed
flow.  Every later round is restricted only by the preceding TSNPE density
estimator.  This script has no import or path to any A-NET, Evidence-Network,
classical-sampler, or historical hyperonic TSNPE artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


A1_OBSERVATION = np.asarray(
    [
        0.153,
        -16.1,
        230.0,
        32.5,
        0.505714285714279,
        1.24142857142857,
        2.4857142857143,
    ],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_clean_report(path: Path, expected_role: str) -> dict:
    report = json.loads(path.read_text())
    if report.get("forbidden_artifacts_used"):
        raise RuntimeError(f"forbidden ancestry declared by {path}")
    if report.get("lineage_class") != "independent_tsnpe":
        raise RuntimeError(f"wrong lineage class in {path}")
    if report.get("role") != expected_role:
        raise RuntimeError(f"wrong artifact role in {path}")
    return report


class SeedPosterior:
    """Adapter exposing the frozen 9D seed flow to sbi's support tools."""

    def __init__(self, checkpoint: Path, standardization: Path, device: torch.device):
        import zuko

        self.device = device
        with np.load(standardization, allow_pickle=False) as archive:
            self.theta_mean, self.theta_scale, self.x_mean, self.x_scale = [
                torch.as_tensor(archive[key], dtype=torch.float32, device=device)
                for key in ("tm", "ts", "xm", "xs")
            ]
            self.low = torch.as_tensor(
                archive["prior_low"], dtype=torch.float32, device=device
            )
            self.high = torch.as_tensor(
                archive["prior_high"], dtype=torch.float32, device=device
            )
        self.flow = zuko.flows.NSF(
            9, 7, transforms=8, hidden_features=[192, 192, 192], bins=10
        ).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        self.flow.load_state_dict(state, strict=True)
        self.flow.eval()
        observation = torch.as_tensor(
            A1_OBSERVATION, dtype=torch.float32, device=device
        )
        self.context = ((observation - self.x_mean) / self.x_scale)[None]

    def sample(self, shape, **_kwargs):
        count = int(np.prod(shape))
        with torch.no_grad():
            sample = self.flow(self.context).sample((count,)).reshape(count, 9)
        sample = sample * self.theta_scale + self.theta_mean
        # Match the locked nucleonic seed adapter: the seed flow is only a
        # proposal accelerator, while the canonical prior supplies the true
        # support.  Clamping prevents raw-flow leakage from turning the very
        # low support-density quantile into ``-inf``.
        return torch.clamp(sample, self.low + 1.0e-6, self.high - 1.0e-6)

    def log_prob(self, theta, **_kwargs):
        theta = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        standardized = (theta - self.theta_mean) / self.theta_scale
        with torch.no_grad():
            value = self.flow(self.context).log_prob(standardized)
            value = value - torch.log(self.theta_scale).sum()
        inside = ((theta >= self.low) & (theta <= self.high)).all(dim=-1)
        return torch.where(inside, value, torch.full_like(value, -torch.inf))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--support-quantile", type=float, default=1.0e-4)
    parser.add_argument("--support-samples", type=int, default=200_000)
    parser.add_argument("--sir-oversampling", type=int, default=1024)
    parser.add_argument("--seed-checkpoint", type=Path)
    parser.add_argument("--seed-standardization", type=Path)
    parser.add_argument("--seed-report", type=Path)
    parser.add_argument("--estimator", type=Path)
    parser.add_argument("--estimator-report", type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.round_index < 0 or arguments.count < 1 or arguments.pilot < 0:
        raise ValueError("round, count, and pilot must be non-negative")
    if arguments.round_index != 0 and arguments.pilot:
        raise ValueError("pilot rows are permitted only in round zero")
    if not 0.0 < arguments.support_quantile < 1.0:
        raise ValueError("support quantile must lie strictly between zero and one")
    if arguments.support_samples < 10_000 or arguments.sir_oversampling < 2:
        raise ValueError("support sample count or SIR oversampling is too small")
    if arguments.output.exists():
        raise FileExistsError(f"refusing to replace {arguments.output}")

    from sbi.inference.posteriors import DirectPosterior
    from sbi.samplers.importance.sir import sampling_importance_resampling
    from sbi.utils import BoxUniform, get_density_thresholder

    started = time.time()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for production round generation")
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(arguments.seed)

    parents = []
    if arguments.round_index == 0:
        required = (
            arguments.seed_checkpoint,
            arguments.seed_standardization,
            arguments.seed_report,
        )
        if any(path is None for path in required):
            raise ValueError("round zero requires the three seed-flow artifacts")
        seed_report = json.loads(arguments.seed_report.read_text())
        if seed_report.get("forbidden_artifacts_used"):
            raise RuntimeError("seed report declares forbidden ancestry")
        if seed_report.get("bank_lineage_class") != "independent_tsnpe_uniform_prior":
            raise RuntimeError("seed report lacks clean full-prior lineage")
        if seed_report.get("checkpoint_sha256") != sha256(arguments.seed_checkpoint):
            raise RuntimeError("seed checkpoint hash does not match its report")
        if seed_report.get("standardization_sha256") != sha256(
            arguments.seed_standardization
        ):
            raise RuntimeError("seed standardization hash does not match its report")
        proposal_posterior = SeedPosterior(
            arguments.seed_checkpoint, arguments.seed_standardization, device
        )
        prior_low = proposal_posterior.low
        prior_high = proposal_posterior.high
        parents = [
            {
                "role": "tsnpe_owned_uniform_prior_seed",
                "path": str(arguments.seed_checkpoint.resolve()),
                "sha256": sha256(arguments.seed_checkpoint),
            }
        ]
    else:
        if arguments.estimator is None or arguments.estimator_report is None:
            raise ValueError("later rounds require the preceding TSNPE estimator")
        estimator_report = require_clean_report(
            arguments.estimator_report, "tsnpe_round_density_estimator"
        )
        if estimator_report.get("round_index") != arguments.round_index - 1:
            raise RuntimeError("estimator report is not from the preceding round")
        if estimator_report.get("estimator_sha256") != sha256(arguments.estimator):
            raise RuntimeError("preceding estimator hash does not match its report")
        prior_low = torch.as_tensor(
            estimator_report["prior_low"], dtype=torch.float32, device=device
        )
        prior_high = torch.as_tensor(
            estimator_report["prior_high"], dtype=torch.float32, device=device
        )
        prior = BoxUniform(low=prior_low, high=prior_high)
        estimator = torch.load(
            arguments.estimator, map_location=device, weights_only=False
        )
        estimator.eval()
        proposal_posterior = DirectPosterior(
            posterior_estimator=estimator, prior=prior, device=str(device)
        ).set_default_x(
            torch.as_tensor(A1_OBSERVATION, dtype=torch.float32, device=device)
        )
        parents = [
            {
                "role": "preceding_tsnpe_round",
                "path": str(arguments.estimator.resolve()),
                "sha256": sha256(arguments.estimator),
            }
        ]

    prior = BoxUniform(low=prior_low, high=prior_high)
    accept = get_density_thresholder(
        proposal_posterior,
        quantile=arguments.support_quantile,
        num_samples_to_estimate_support=arguments.support_samples,
    )
    # Draw the prior restricted to the TSNPE-owned high-density region.  Pass
    # an actual *log target* to SIR: uniform-prior log density inside the
    # region and -inf outside.  This explicit form avoids treating a Boolean
    # support indicator as though it were already a log density.
    def restricted_log_target(theta):
        accepted = accept(theta)
        log_prior = prior.log_prob(theta)
        return torch.where(
            accepted,
            log_prior,
            torch.full_like(log_prior, -torch.inf),
        )

    total = arguments.pilot + arguments.count
    with torch.no_grad():
        theta = sampling_importance_resampling(
            potential_fn=restricted_log_target,
            proposal=proposal_posterior,
            num_samples=total,
            num_candidate_samples=arguments.sir_oversampling,
            # This is the number of selected rows per SIR batch; each selected
            # row evaluates ``sir_oversampling`` candidates.  Keep the product
            # bounded so the conditional flow never needs an oversized CUDA
            # activation batch.
            max_sampling_batch_size=min(512, total),
            show_progress_bars=False,
            device=str(device),
        )
    theta = theta.detach().cpu().numpy().astype(np.float64)
    low = prior_low.detach().cpu().numpy().astype(np.float64)
    high = prior_high.detach().cpu().numpy().astype(np.float64)
    if theta.shape != (total, 9) or not np.isfinite(theta).all():
        raise RuntimeError("restricted-prior sampler returned invalid rows")
    if not np.all((theta >= low) & (theta <= high)):
        raise RuntimeError("restricted-prior sampler returned an out-of-prior row")

    metadata = {
        "schema_version": 1,
        "lineage_class": "independent_tsnpe",
        "role": "tsnpe_round_proposal",
        "model": "ddb-hyperonic",
        "round_index": arguments.round_index,
        "pilot_rows": arguments.pilot,
        "training_candidate_rows": arguments.count,
        "full_prior_dimension": 9,
        "support_quantile": arguments.support_quantile,
        "support_samples": arguments.support_samples,
        "sir_oversampling": arguments.sir_oversampling,
        "restricted_prior_sampler": "explicit_log_target_SIR",
        "seed": arguments.seed,
        "parents": parents,
        "forbidden_artifacts_used": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        theta=theta,
        prior_low=low,
        prior_high=high,
        pilot_rows=np.asarray(arguments.pilot, dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        forbidden_artifacts_used=np.asarray([], dtype="U1"),
    )
    report = {
        "status": "PASS",
        **metadata,
        "role": "tsnpe_round_proposal",
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "wall_seconds": time.time() - started,
    }
    report_path = arguments.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
