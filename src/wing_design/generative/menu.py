"""Immutable data model for the generative truss stack.

A `CandidateMenu` is the precomputed discrete input to the CP-SAT model: a
library of candidate beams, a cross-section catalog, a pairwise conflict table,
and stress-coverage targets. CP-SAT selects a subset of beams and assigns each
a cross-section bucket; the result is a `WingCandidate`, judged by the frame
gate into a `GateResult`.

All dataclasses are frozen so a menu can't be mutated mid-solve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class NodeKind(str, Enum):
    KEEL_STEP = "keel_step"
    DECK_STEP = "deck_step"
    TIP = "tip"
    MESH_CENTROID = "mesh_centroid"
    ON_BEAM = "on_beam"


class CrossSectionShape(str, Enum):
    CIRCLE = "circle"
    SEMICIRCLE = "semicircle"
    VORONOI = "voronoi"


@dataclass(frozen=True)
class CrossSectionOption:
    """One discrete cross-section choice in the catalog."""
    bucket: int
    shape: CrossSectionShape
    area_m2: float

    @property
    def radius_m(self) -> float:
        """Equivalent-circle radius for the cross-sectional area."""
        return math.sqrt(self.area_m2 / math.pi)
