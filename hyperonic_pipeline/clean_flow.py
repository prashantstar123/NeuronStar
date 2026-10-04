#!/usr/bin/env python3
"""Train and sample a self-tempered normalized proposal for the clean branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from zuko.flows import NSF


ROOT = Path(__file__).resolve().parent
LEARNED = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
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


def load_prior(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    payload = json.loads(path.read_text())
    lower = np.asarray(payload["lower"], dtype=np.float64)
    upper = np.asarray(payload["upper"], dtype=np.float64)
    if lower.shape != (9,) or upper.shape != (9,) or np.any(lower >= upper):
        raise RuntimeError("invalid prior manifest")
    return lower, upper, payload


def to_unconstrained(theta: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    box = 2.0 * (theta[:, :LEARNED] - lower[:LEARNED]) / (
        upper[:LEARNED] - lower[:LEARNED]
    ) - 1.0
    if np.any((box <= -1.0) | (box >= 1.0)):
        box = np.clip(box, -1.0 + 1.0e-9, 1.0 - 1.0e-9)
    return np.arctanh(box)


def from_unconstrained(value: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return lower[:LEARNED] + 0.5 * (np.tanh(value) + 1.0) * (
        upper[:LEARNED] - lower[:LEARNED]
    )


def log_box_jacobian(
    theta7: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    box = np.clip(
        2.0 * (theta7 - lower[:LEARNED]) / (upper[:LEARNED] - lower[:LEARNED]) - 1.0,
        -1.0 + 1.0e-12,
        1.0 - 1.0e-12,
    )
    return (
        np.log(2.0 / (upper[:LEARNED] - lower[:LEARNED]))[None, :]
        - np.log1p(-(box * box))
    ).sum(axis=1)


def normalized_weights(logweight: np.ndarray) -> tuple[np.ndarray, float]:
    finite = np.isfinite(logweight)
    if not finite.any():
        raise RuntimeError("all training weights are non-finite")
    shifted = np.full_like(logweight, -np.inf, dtype=np.float64)
    shifted[finite] = logweight[finite] - logsumexp(logweight[finite])
    weight = np.exp(shifted)
    ess = float(1.0 / np.sum(weight * weight))
    return weight, ess


def build_flow(transforms: int, hidden: int, layers: int, bins: int) -> NSF:
    return NSF(
        features=LEARNED,
        transforms=transforms,
        hidden_features=[hidden] * layers,
        bins=bins,
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def train(arguments: argparse.Namespace) -> None:
    started = time.time()
    lower, upper, _ = load_prior(arguments.prior)
    with np.load(arguments.input, allow_pickle=False) as source:
        theta = np.asarray(source["theta"], dtype=np.float64)
        if "logl_stage1" in source.files:
            source_kind = "uniform-prior-seed"
            loglike = np.asarray(source["logl_stage1"], dtype=np.float64)
            logweight = arguments.beta * loglike
            required_schema = "ddb-hyperonic-clean-seed-v1"
            if str(source["schema"].item()) != required_schema:
                raise RuntimeError("unrecognized clean-seed schema")
            if "source_part" not in source.files or np.any(source["source_part"] >= 4):
                raise RuntimeError("clean seed contains a non-prior stream")
        elif {"logl", "logq"}.issubset(source.files):
            source_kind = "self-proposal-exact-evaluation"
            loglike = np.asarray(source["logl"], dtype=np.float64)
            logq = np.asarray(source["logq"], dtype=np.float64)
            logprior = -float(np.log(upper - lower).sum())
            logweight = logprior + arguments.beta * loglike - logq
            if str(source["schema"].item()) != "ddb-hyperonic-clean-eval-v1":
                raise RuntimeError("unrecognized exact-evaluation schema")
        else:
            raise RuntimeError("training input lacks a recognized exact-weight schema")
    if theta.ndim != 2 or theta.shape[1] != 9 or loglike.shape != (len(theta),):
        raise RuntimeError("invalid training-array shapes")
    if not np.all((theta >= lower) & (theta <= upper)):
        raise RuntimeError("training row outside canonical prior")
    weight, input_ess = normalized_weights(logweight)
    if input_ess < arguments.minimum_input_ess:
        raise RuntimeError(
            f"input ESS {input_ess:.1f} is below gate {arguments.minimum_input_ess:.1f}"
        )

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)
    rng = np.random.default_rng(arguments.seed)
    index = rng.choice(len(theta), size=arguments.training_draws, replace=True, p=weight)
    unconstrained = to_unconstrained(theta[index], lower, upper)
    if arguments.jitter > 0.0:
        unconstrained += rng.normal(0.0, arguments.jitter, size=unconstrained.shape)
    mean = unconstrained.mean(axis=0)
    std = unconstrained.std(axis=0)
    if np.any(~np.isfinite(std)) or np.any(std <= 1.0e-8):
        raise RuntimeError("degenerate transformed training scale")
    standardized = ((unconstrained - mean) / std).astype(np.float32)
    order = rng.permutation(len(standardized))
    validation_rows = min(arguments.validation_rows, max(1000, len(order) // 10))
    validation = torch.from_numpy(standardized[order[:validation_rows]])
    training = torch.from_numpy(standardized[order[validation_rows:]])

    device = resolve_device(arguments.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(arguments.cpu_threads, os.cpu_count() or 1)))
    flow = build_flow(
        arguments.transforms, arguments.hidden, arguments.layers, arguments.bins
    ).to(device)
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=arguments.learning_rate, weight_decay=1.0e-6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=arguments.epochs
    )
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(arguments.seed + 10)
    for epoch in range(arguments.epochs):
        permutation = torch.randperm(len(training), generator=generator)
        flow.train()
        total = 0.0
        batches = 0
        for start in range(0, len(training), arguments.batch_size):
            batch = training[permutation[start : start + arguments.batch_size]].to(device)
            loss = -flow().log_prob(batch).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite flow loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        scheduler.step()
        flow.eval()
        with torch.no_grad():
            pieces = []
            for start in range(0, len(validation), arguments.batch_size):
                batch = validation[start : start + arguments.batch_size].to(device)
                pieces.append(float(-flow().log_prob(batch).mean()))
            validation_loss = float(np.mean(pieces))
        if validation_loss < best_loss - arguments.minimum_improvement:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu() for key, value in flow.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 10 == 0 or epoch + 1 == arguments.epochs:
            print(
                f"epoch={epoch:03d} train_nll={total / batches:.5f} "
                f"validation_nll={validation_loss:.5f} best={best_loss:.5f}",
                flush=True,
            )
        if epoch + 1 >= arguments.minimum_epochs and stale >= arguments.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")

    checkpoint = {
        "schema": "ddb-hyperonic-clean-flow-v1",
        "state_dict": best_state,
        "mean": mean,
        "std": std,
        "prior_low": lower,
        "prior_high": upper,
        "beta": float(arguments.beta),
        "seed": int(arguments.seed),
        "source_kind": source_kind,
        "input_sha256": sha256(arguments.input),
        "input_ess": input_ess,
        "best_validation_nll": best_loss,
        "architecture": {
            "transforms": arguments.transforms,
            "hidden": arguments.hidden,
            "layers": arguments.layers,
            "bins": arguments.bins,
        },
        "training": {
            "draws": arguments.training_draws,
            "jitter": arguments.jitter,
            "epochs_requested": arguments.epochs,
            "wall_seconds": time.time() - started,
        },
    }
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.checkpoint.with_name(
        f".{arguments.checkpoint.name}.tmp-{os.getpid()}"
    )
    torch.save(checkpoint, temporary)
    os.replace(temporary, arguments.checkpoint)
    report = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"state_dict", "mean", "std", "prior_low", "prior_high"}
    }
    report.update(
        {
            "status": "PASS",
            "checkpoint": str(arguments.checkpoint.resolve()),
            "checkpoint_sha256": sha256(arguments.checkpoint),
        }
    )
    arguments.checkpoint.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def sample(arguments: argparse.Namespace) -> None:
    started = time.time()
    device = resolve_device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "ddb-hyperonic-clean-flow-v1":
        raise RuntimeError("unrecognized flow checkpoint")
    architecture = checkpoint["architecture"]
    flow = build_flow(**architecture).to(device)
    flow.load_state_dict(checkpoint["state_dict"])
    flow.eval()
    lower = np.asarray(checkpoint["prior_low"], dtype=np.float64)
    upper = np.asarray(checkpoint["prior_high"], dtype=np.float64)
    mean = np.asarray(checkpoint["mean"], dtype=np.float64)
    std = np.asarray(checkpoint["std"], dtype=np.float64)
    if not 0.0 <= arguments.defensive_fraction < 1.0:
        raise ValueError("defensive fraction must be in [0, 1)")
    coupling_rng = np.random.default_rng(arguments.seed + 1)
    theta = np.empty((arguments.draws, 9), dtype=np.float64)
    component = np.zeros(arguments.draws, dtype=np.uint8)
    uniform_rows = int(round(arguments.draws * arguments.defensive_fraction))
    flow_rows = arguments.draws - uniform_rows
    uniform_logq = -float(np.log(upper[LEARNED:] - lower[LEARNED:]).sum())
    with torch.no_grad():
        for start in range(0, flow_rows, arguments.batch_size):
            stop = min(start + arguments.batch_size, flow_rows)
            count = stop - start
            # Zuko's distribution delegates base draws through torch; fork_rng keeps
            # this command deterministic without altering the caller's global state.
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(arguments.seed + start)
                standardized = flow().sample((count,))
            unconstrained = standardized.cpu().numpy() * std + mean
            learned = from_unconstrained(unconstrained, lower, upper)
            coupling = coupling_rng.uniform(lower[LEARNED:], upper[LEARNED:], (count, 2))
            theta[start:stop] = np.column_stack([learned, coupling])
    if uniform_rows:
        theta[flow_rows:] = coupling_rng.uniform(
            lower, upper, size=(uniform_rows, len(lower))
        )
        component[flow_rows:] = 1
    permutation = coupling_rng.permutation(arguments.draws)
    theta = theta[permutation]
    component = component[permutation]

    # Evaluate every component density on every row and use the normalized
    # defensive-mixture density in all downstream importance weights.
    flow_logq = np.empty(arguments.draws, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, arguments.draws, arguments.batch_size):
            stop = min(start + arguments.batch_size, arguments.draws)
            unconstrained = to_unconstrained(theta[start:stop], lower, upper)
            standardized = torch.as_tensor(
                ((unconstrained - mean) / std).astype(np.float32), device=device
            )
            base_logq = flow().log_prob(standardized).cpu().numpy()
            flow_logq[start:stop] = (
                base_logq
                - np.log(std).sum()
                + log_box_jacobian(theta[start:stop, :LEARNED], lower, upper)
                + uniform_logq
            )
    prior_logq = -float(np.log(upper - lower).sum())
    if arguments.defensive_fraction:
        logq = np.logaddexp(
            math.log1p(-arguments.defensive_fraction) + flow_logq,
            math.log(arguments.defensive_fraction) + prior_logq,
        )
    else:
        logq = flow_logq
    if not np.isfinite(theta).all() or not np.isfinite(logq).all():
        raise RuntimeError("non-finite proposal output")
    if not np.all((theta >= lower) & (theta <= upper)):
        raise RuntimeError("proposal draw outside canonical prior")
    atomic_savez(
        arguments.output,
        schema=np.asarray("ddb-hyperonic-clean-proposal-v1"),
        theta=theta,
        logq=logq,
        prior_low=lower,
        prior_high=upper,
        beta=np.float64(checkpoint["beta"]),
        seed=np.int64(arguments.seed),
        checkpoint_sha256=np.asarray(sha256(arguments.checkpoint)),
        source=np.asarray("self-tempered normalized flow"),
        proposal_component=component,
        defensive_fraction=np.float64(arguments.defensive_fraction),
    )
    report = {
        "status": "PASS",
        "draws": arguments.draws,
        "beta": checkpoint["beta"],
        "seed": arguments.seed,
        "defensive_fraction": arguments.defensive_fraction,
        "defensive_rows": uniform_rows,
        "wall_seconds": time.time() - started,
        "checkpoint_sha256": sha256(arguments.checkpoint),
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "logq_min": float(logq.min()),
        "logq_max": float(logq.max()),
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def uniform(arguments: argparse.Namespace) -> None:
    """Draw an exact full-support proposal from the canonical prior."""

    started = time.time()
    lower, upper, _ = load_prior(arguments.prior)
    rng = np.random.default_rng(arguments.seed)
    theta = rng.uniform(lower, upper, size=(arguments.draws, len(lower)))
    logq = np.full(arguments.draws, -float(np.log(upper - lower).sum()))
    atomic_savez(
        arguments.output,
        schema=np.asarray("ddb-hyperonic-clean-proposal-v1"),
        theta=theta,
        logq=logq,
        prior_low=lower,
        prior_high=upper,
        beta=np.float64(0.0),
        seed=np.int64(arguments.seed),
        checkpoint_sha256=np.asarray("none-uniform-prior"),
        source=np.asarray("canonical uniform prior"),
        proposal_component=np.ones(arguments.draws, dtype=np.uint8),
        defensive_fraction=np.float64(1.0),
    )
    report = {
        "status": "PASS",
        "draws": arguments.draws,
        "beta": 0.0,
        "seed": arguments.seed,
        "wall_seconds": time.time() - started,
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256(arguments.output),
        "logq": float(logq[0]),
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def suggest(arguments: argparse.Namespace) -> None:
    with np.load(arguments.input, allow_pickle=False) as source:
        if str(source["schema"].item()) != "ddb-hyperonic-clean-eval-v1":
            raise RuntimeError("unrecognized exact-evaluation schema")
        loglike = np.asarray(source["logl"], dtype=np.float64)
        logq = np.asarray(source["logq"], dtype=np.float64)
        lower = np.asarray(source["prior_low"], dtype=np.float64)
        upper = np.asarray(source["prior_high"], dtype=np.float64)
    logprior = -float(np.log(upper - lower).sum())

    def ess(beta: float) -> float:
        return normalized_weights(logprior + beta * loglike - logq)[1]

    current_ess = ess(arguments.current_beta)
    final_ess = ess(1.0)
    if final_ess >= arguments.final_ess_gate:
        next_beta = 1.0
    else:
        low = arguments.current_beta
        high = 1.0
        if current_ess < arguments.target_ess:
            raise RuntimeError(
                f"current-temperature ESS {current_ess:.1f} is below bridge gate"
            )
        for _ in range(60):
            middle = 0.5 * (low + high)
            if ess(middle) >= arguments.target_ess:
                low = middle
            else:
                high = middle
        next_beta = low
        if next_beta <= arguments.current_beta + arguments.minimum_step:
            raise RuntimeError("temperature bridge cannot make the required minimum step")
    payload = {
        "schema": "ddb-hyperonic-clean-temperature-v1",
        "input_sha256": sha256(arguments.input),
        "current_beta": arguments.current_beta,
        "next_beta": next_beta,
        "current_ess": current_ess,
        "next_ess": ess(next_beta),
        "posterior_ess": final_ess,
        "target_ess": arguments.target_ess,
        "final_ess_gate": arguments.final_ess_gate,
        "status": "POSTERIOR_READY" if next_beta == 1.0 else "BRIDGE_CONTINUE",
    }
    if arguments.output:
        arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prior", type=Path, default=ROOT / "prior_manifest.json")
    commands = result.add_subparsers(dest="command", required=True)
    p_train = commands.add_parser("train")
    p_train.add_argument("--input", type=Path, required=True)
    p_train.add_argument("--checkpoint", type=Path, required=True)
    p_train.add_argument("--beta", type=float, required=True)
    p_train.add_argument("--training-draws", type=int, default=200_000)
    p_train.add_argument("--validation-rows", type=int, default=20_000)
    p_train.add_argument("--minimum-input-ess", type=float, default=500.0)
    p_train.add_argument("--jitter", type=float, default=0.003)
    p_train.add_argument("--transforms", type=int, default=10)
    p_train.add_argument("--hidden", type=int, default=256)
    p_train.add_argument("--layers", type=int, default=3)
    p_train.add_argument("--bins", type=int, default=8)
    p_train.add_argument("--epochs", type=int, default=160)
    p_train.add_argument("--minimum-epochs", type=int, default=50)
    p_train.add_argument("--patience", type=int, default=25)
    p_train.add_argument("--minimum-improvement", type=float, default=1.0e-4)
    p_train.add_argument("--batch-size", type=int, default=4096)
    p_train.add_argument("--learning-rate", type=float, default=5.0e-4)
    p_train.add_argument("--seed", type=int, default=20260919)
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("--cpu-threads", type=int, default=16)
    p_train.set_defaults(function=train)

    p_sample = commands.add_parser("sample")
    p_sample.add_argument("--checkpoint", type=Path, required=True)
    p_sample.add_argument("--output", type=Path, required=True)
    p_sample.add_argument("--draws", type=int, default=60_000)
    p_sample.add_argument("--batch-size", type=int, default=20_000)
    p_sample.add_argument("--seed", type=int, default=20260920)
    p_sample.add_argument("--defensive-fraction", type=float, default=0.05)
    p_sample.add_argument("--device", default="auto")
    p_sample.set_defaults(function=sample)

    p_uniform = commands.add_parser("uniform")
    p_uniform.add_argument("--output", type=Path, required=True)
    p_uniform.add_argument("--draws", type=int, default=100_000)
    p_uniform.add_argument("--seed", type=int, default=20260921)
    p_uniform.set_defaults(function=uniform)

    p_suggest = commands.add_parser("suggest")
    p_suggest.add_argument("--input", type=Path, required=True)
    p_suggest.add_argument("--current-beta", type=float, required=True)
    p_suggest.add_argument("--target-ess", type=float, default=3000.0)
    p_suggest.add_argument("--final-ess-gate", type=float, default=1000.0)
    # The first bridge away from a broad prior can be much smaller than later
    # posterior-temperature steps; the ESS gate, not an arbitrary step floor,
    # is the safety criterion.
    p_suggest.add_argument("--minimum-step", type=float, default=1.0e-7)
    p_suggest.add_argument("--output", type=Path)
    p_suggest.set_defaults(function=suggest)
    return result


def main() -> None:
    arguments = parser().parse_args()
    if hasattr(arguments, "beta") and not 0.0 < arguments.beta <= 1.0:
        raise ValueError("beta must be in (0, 1]")
    arguments.function(arguments)


if __name__ == "__main__":
    main()
