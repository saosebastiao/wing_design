"""wing-design: structural design of unidirectional CFRP wingsails.

See docs/plan.md for the phased development plan and docs/glossary.md
for terminology.

The single source of truth for design parameters is
`wing_design.scenario.DesignParameters` — use `small_scenario()` to
load the 5 m demonstration wingsail scenario.
"""
from __future__ import annotations

from .scenario import (
    DesignParameters,
    small_scenario, medium_scenario, large_scenario,
)
from .viz import show_in_viewer

__all__ = ["DesignParameters", "show_in_viewer",
           "small_scenario", "medium_scenario", "large_scenario"]


def main() -> None:
    print("wing-design — run examples with `just example 01_wing_solid` (see docs/plan.md).")
