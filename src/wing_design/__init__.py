"""wing-design: structural design of unidirectional CFRP wingsails.

See docs/plan.md for the phased development plan.
"""
from __future__ import annotations

from .viz import show_in_viewer

__all__ = ["show_in_viewer"]


def main() -> None:
    print("wing-design — run examples with `just example 01_wing_solid` (see docs/plan.md).")
