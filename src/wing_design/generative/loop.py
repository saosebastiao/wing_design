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


def _severity(verdict, tip_limit_m):
    """Scalar demand of a verdict: max of stress ratio and normalized deflection.
    The case with the highest severity is the governing (worst) case."""
    tip_term = verdict.tip_deflection_m / tip_limit_m if tip_limit_m > 0 else 0.0
    return max(verdict.max_stress_ratio, tip_term)


def _worst_over_cases(frame, params, case_load_fns, load_direction):
    """Gate a frame against every case; return the highest-severity GateResult."""
    spec = params.geometry
    tip_limit = params.generative.tip_deflection_limit_m
    worst = None
    for case_name, density_fn in case_load_fns.items():
        loads = lump_spanwise_force_to_nodes(
            frame, density_fn, z_min=spec.z_wing_root, z_max=spec.z_wing_tip,
            direction=load_direction,
        )
        verdict = solve_frame(frame, params, loads, governing_case=case_name)
        if worst is None or _severity(verdict, tip_limit) > _severity(worst, tip_limit):
            worst = verdict
    return worst


def generate_truss(menu, params, case_load_fns, *,
                   load_direction=(0.0, 1.0, 0.0), max_candidates=200):
    """Enumerate designs by mass and return the lightest that survives the envelope.

    `case_load_fns` maps a case name to a spanwise normal-force density function
    density(z) -> N/m (the caller builds these from the aero results, keeping this
    loop independent of AeroSandbox). Returns a TrussResult; `chosen` is None if no
    enumerated design survives every case.
    """
    designs = solve_designs(menu, params.generative, top_n=max_candidates,
                            enforce_coverage=False)
    frontier = []

    def gate_fn(design):
        frame = build_frame(
            design, menu,
            max_element_length_m=params.generative.frame_max_element_length_m,
        )
        verdict = _worst_over_cases(frame, params, case_load_fns, load_direction)
        frontier.append(GatedDesign(design=design, verdict=verdict))
        return verdict

    result = select_lightest_feasible(designs, gate_fn)
    chosen, verdict = result if result is not None else (None, None)
    return TrussResult(chosen=chosen, verdict=verdict, frontier=tuple(frontier))
