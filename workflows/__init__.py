"""Portable integration workflows for the paper calculations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .a1_problem import A1Problem

__all__ = ["A1Problem"]


def __getattr__(name: str):
    if name == "A1Problem":
        from .a1_problem import A1Problem

        return A1Problem
    raise AttributeError(name)
