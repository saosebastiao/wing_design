"""wing-design: structural design of unidirectional CFRP wingsails.

See docs/plan.md for the phased development plan and docs/glossary.md
for terminology.

The single source of truth for design parameters is
`wing_design.scenario.DesignParameters` — use `default_scenario()` to
load the 5 m demonstration wingsail scenario.
"""
from __future__ import annotations

from .scenario import DesignParameters, default_scenario
from .viz import show_in_viewer

__all__ = ["DesignParameters", "default_scenario", "show_in_viewer"]


def main() -> None:
    print("wing-design — run examples with `just example 01_wing_solid` (see docs/plan.md).")
