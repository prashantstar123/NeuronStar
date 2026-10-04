"""Equation-of-state plug-ins and registry."""

from .base import EOSPlugin
from .ddb.plugin import DDB_PLUGIN
from .ddb_hyperonic.plugin import DDB_HYPERONIC_PLUGIN
from .registry import EOS_MODELS, EOSRegistry, available_eos, get_eos, register_eos

register_eos(DDB_PLUGIN)
register_eos(DDB_HYPERONIC_PLUGIN)

__all__ = [
    "DDB_PLUGIN",
    "DDB_HYPERONIC_PLUGIN",
    "EOS_MODELS",
    "EOSPlugin",
    "EOSRegistry",
    "available_eos",
    "get_eos",
    "register_eos",
]
