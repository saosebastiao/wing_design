"""Constraint-based generative wing-truss stack (Milestone 1).

CP-SAT selects/assembles internal beams from a precomputed candidate menu;
spline geometry and FEM live outside the solver. See
docs/superpowers/specs/2026-06-02-constraint-based-generation-design.md.
"""
from .menu import (
    BeamWrap,
    CandidateBeam,
    CandidateMenu,
    CandidateNode,
    ConflictTable,
    CoverageTarget,
    CrossSectionOption,
    CrossSectionShape,
    GateResult,
    NodeKind,
    WingCandidate,
    WingWrap,
    validate_menu,
)
from .model import build_cp_model, solve_designs
from .gate import build_frame, solve_frame
from .build import wing_candidate_to_part
from .candidates import build_beam_library, build_candidate_menu
from .loads import lump_spanwise_force_to_nodes
from .loop import GatedDesign, TrussResult, generate_truss, select_lightest_feasible

__all__ = [
    "BeamWrap",
    "CandidateBeam",
    "CandidateMenu",
    "CandidateNode",
    "ConflictTable",
    "CoverageTarget",
    "CrossSectionOption",
    "CrossSectionShape",
    "GateResult",
    "NodeKind",
    "WingCandidate",
    "WingWrap",
    "GatedDesign",
    "TrussResult",
    "build_beam_library",
    "build_candidate_menu",
    "build_cp_model",
    "build_frame",
    "generate_truss",
    "lump_spanwise_force_to_nodes",
    "select_lightest_feasible",
    "solve_designs",
    "solve_frame",
    "validate_menu",
    "wing_candidate_to_part",
]
