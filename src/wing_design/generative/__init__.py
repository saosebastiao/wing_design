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
from .candidates import build_beam_library
from .loads import lump_spanwise_force_to_nodes

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
    "build_beam_library",
    "build_cp_model",
    "build_frame",
    "lump_spanwise_force_to_nodes",
    "solve_designs",
    "solve_frame",
    "validate_menu",
    "wing_candidate_to_part",
]
