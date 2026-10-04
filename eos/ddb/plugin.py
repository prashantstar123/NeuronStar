"""Registry adapter for the certified seven-parameter nucleonic DDB EOS."""

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


DDB_PLUGIN = EOSPlugin(
    key="ddb",
    display_name="Seven-parameter nucleonic DDB",
    parameter_names=PARAMETER_NAMES,
    prior_low=PRIOR_LOW,
    prior_high=PRIOR_HIGH,
    core_eos_fn=core_eos,
    core_eos_batch_fn=core_eos_batch,
    nuclear_observables_fn=nuclear_matter_observables,
    nuclear_observables_batch_fn=nuclear_matter_observables_batch,
    metadata={
        "family": "density-dependent relativistic mean field",
        "composition": "nucleonic",
        "core_table_units": "MeV/fm^3",
        "validation": "bitwise frozen forward regression",
    },
)
