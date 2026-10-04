"""Named NICER-source substitutions used by the paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class NicerSourceSpec:
    """Column schema and target slot for one NICER source."""

    name: str
    filename: str
    mass_column: int
    radius_column: int
    weight_column: int | None
    slot: int
    replaces: str | None = None

    def columns(self) -> tuple[int, int, int | None]:
        return self.mass_column, self.radius_column, self.weight_column


BASE_NICER_SOURCES = (
    NicerSourceSpec("J0030", "J0030_2spot_RM.txt", 1, 0, None, 0),
    NicerSourceSpec("J0740", "J0740_NICERXMM_full_mr.txt", 1, 0, 2, 1),
    NicerSourceSpec("J0437", "J0437_post_equal_weights.dat", 0, 1, None, 2),
)

SOURCE_SUBSTITUTIONS = MappingProxyType(
    {
        "J0614": NicerSourceSpec(
            "J0614", "J0614_mrsamples.dat", 0, 1, None, 2, "J0437"
        ),
        "J1231": NicerSourceSpec(
            "J1231", "J1231_wmrsamples.txt", 1, 2, 0, 0, "J0030"
        ),
        "J1614": NicerSourceSpec(
            "J1614",
            "J1614_STU_mrsamples_post_equal_weights.dat",
            0,
            1,
            None,
            1,
            "J0740",
        ),
    }
)
SOURCE_SCENARIO_NAMES = ("A1", *SOURCE_SUBSTITUTIONS)


def source_substitution(scenario: str) -> NicerSourceSpec | None:
    """Return the declared replacement, or ``None`` for baseline A1."""

    if scenario == "A1":
        return None
    try:
        return SOURCE_SUBSTITUTIONS[scenario]
    except KeyError as error:
        available = ", ".join(SOURCE_SCENARIO_NAMES)
        raise KeyError(
            f"unknown source scenario {scenario!r}; available: {available}"
        ) from error


def nicer_source_paths(
    base_paths: dict[str, Path],
    scenario: str,
    substituted_paths: dict[str, Path] | None = None,
) -> tuple[tuple[Path, NicerSourceSpec], ...]:
    """Return the three ordered likelihood/conditioning source records."""

    records = [
        (base_paths[spec.filename], spec) for spec in BASE_NICER_SOURCES
    ]
    replacement = source_substitution(scenario)
    if replacement is not None:
        if substituted_paths is None or replacement.filename not in substituted_paths:
            raise ValueError(
                f"source scenario {scenario} requires {replacement.filename}"
            )
        records[replacement.slot] = (
            substituted_paths[replacement.filename],
            replacement,
        )
    return tuple(records)
