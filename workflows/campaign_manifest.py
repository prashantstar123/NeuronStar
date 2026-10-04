"""Validation and concise summaries for the paper inference campaign."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ALLOWED_STATES = {
    "portable_gated",
    "artifact_verified_runner_pending",
}
ALLOWED_PRODUCTS = {"posterior", "evidence"}


def load_campaign(path: str | Path) -> dict:
    path = Path(path)
    campaign = json.loads(path.read_text())
    if campaign.get("schema_version") != 1:
        raise ValueError("campaign manifest schema_version must be 1")
    models = campaign.get("models", {})
    scenarios = campaign.get("scenarios", {})
    configurations = campaign.get("configurations", [])
    if not models or not scenarios or not configurations:
        raise ValueError("campaign manifest has an empty model/scenario matrix")
    seen_configurations: set[tuple[str, str]] = set()
    for configuration in configurations:
        model = configuration.get("model")
        scenario = configuration.get("scenario")
        identity = (model, scenario)
        if model not in models or scenario not in scenarios:
            raise ValueError(f"unknown campaign configuration {identity}")
        if identity in seen_configurations:
            raise ValueError(f"duplicate campaign configuration {identity}")
        seen_configurations.add(identity)
        backends = configuration.get("backends", [])
        if not backends:
            raise ValueError(f"campaign configuration {identity} has no backends")
        names: set[str] = set()
        for backend in backends:
            name = backend.get("name")
            state = backend.get("state")
            products = backend.get("products", [])
            if not name or name in names:
                raise ValueError(f"bad or duplicate backend in {identity}: {name!r}")
            names.add(name)
            if state not in ALLOWED_STATES:
                raise ValueError(f"unknown state {state!r} for {identity}/{name}")
            if not products or not set(products) <= ALLOWED_PRODUCTS:
                raise ValueError(f"bad products for {identity}/{name}: {products}")
            if state == "portable_gated" and not backend.get("runner"):
                raise ValueError(f"portable task {identity}/{name} has no runner")
            if state != "portable_gated" and not backend.get("blocker"):
                raise ValueError(f"pending task {identity}/{name} has no blocker")
    return campaign


def summarize_campaign(campaign: dict) -> dict:
    states: Counter[str] = Counter()
    products: Counter[str] = Counter()
    models: Counter[str] = Counter()
    pending = []
    for configuration in campaign["configurations"]:
        model = configuration["model"]
        scenario = configuration["scenario"]
        for backend in configuration["backends"]:
            states[backend["state"]] += 1
            models[model] += 1
            products.update(backend["products"])
            if backend["state"] != "portable_gated":
                pending.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "backend": backend["name"],
                        "blocker": backend["blocker"],
                    }
                )
    total = sum(states.values())
    portable = states["portable_gated"]
    return {
        "artifact_level_status": campaign["artifact_level_status"],
        "full_inference_status": campaign["full_inference_status"],
        "configurations": len(campaign["configurations"]),
        "backend_tasks": total,
        "portable_gated_tasks": portable,
        "runner_pending_tasks": total - portable,
        "state_counts": dict(sorted(states.items())),
        "product_counts": dict(sorted(products.items())),
        "model_task_counts": dict(sorted(models.items())),
        "pending": pending,
    }
