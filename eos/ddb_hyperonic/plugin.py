"""Registry adapter for the certified DDB Lambda-Xi-minus extension."""

from __future__ import annotations

from eos.base import EOSPlugin

from .model import (
    PARAMETER_NAMES,
    PRIOR_HIGH,
    PRIOR_LOW,
    core_eos,
    core_eos_batch,
    nuclear_matter_observables,
    nuclear_matter_observables_batch,
)


DDB_HYPERONIC_PLUGIN = EOSPlugin(
    key="ddb-hyperonic",
    display_name="Nine-parameter DDB Lambda-Xi-minus",
    parameter_names=PARAMETER_NAMES,
    prior_low=PRIOR_LOW,
    prior_high=PRIOR_HIGH,
    core_eos_fn=core_eos,
    core_eos_batch_fn=core_eos_batch,
    nuclear_observables_fn=nuclear_matter_observables,
    nuclear_observables_batch_fn=nuclear_matter_observables_batch,
    metadata={
        "family": "density-dependent relativistic mean field",
        "composition": "n,p,Lambda,Xi-,e-,mu-",
        "core_table_units": "MeV/fm^3",
        "row_parallel_forward": True,
        "require_monotonic_tov_graft": True,
        "physics_reference": "Malik and Providencia, Phys. Rev. D 106, 063024 (2022)",
        "validation": "certified hyperonic physics and forward regression",
    },
)
