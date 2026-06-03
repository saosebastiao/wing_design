"""Deflection-driven selection loop (M2-core).

The frame gate is cheap (a tiny 1D solve; the slow FEA runs once in the menu
build), so the loop enumerates CP-SAT designs in ascending mass and gates each
against the load envelope, returning the lightest design that survives. See
docs/superpowers/specs/2026-06-03-m2-core-deflection-driven-loop-design.md.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .gate import build_frame, solve_frame
from .loads import lump_spanwise_force_to_nodes
from .menu import GateResult, WingCandidate
from .model import solve_designs


@dataclass(frozen=True)
class GatedDesign:
    """One enumerated design and its worst-case gate verdict."""
    design: WingCandidate
    verdict: GateResult


@dataclass(frozen=True)
class TrussResult:
    """Outcome of the selection loop.

    `chosen`/`verdict` are None when no enumerated design survived the envelope.
    `frontier` is every design gated up to (and including) the chosen one, in the
    order tried (ascending mass).
    """
    chosen: WingCandidate | None
    verdict: GateResult | None
    frontier: tuple[GatedDesign, ...]


def select_lightest_feasible(designs, gate_fn):
    """Return (design, verdict) for the first design whose `gate_fn` is feasible.

    `designs` is an iterable already in ascending mass; `gate_fn(design)` returns
    a GateResult. Returns None if no design is feasible.
    """
    for d in designs:
        verdict = gate_fn(d)
        if verdict.feasible:
            return d, verdict
    return None
