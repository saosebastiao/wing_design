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
)
from .model import build_cp_model, solve_designs
from .gate import build_frame, solve_frame

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
    "build_cp_model",
    "build_frame",
    "solve_designs",
    "solve_frame",
]
