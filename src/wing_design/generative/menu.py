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


@dataclass(frozen=True)
class CandidateNode:
    """A discrete point CP-SAT may build beams from."""
    id: int
    xyz: tuple[float, float, float]
    kind: NodeKind
    z_layer: int
    # Background-FEM principal frame (filled by the candidate generator, Plan 1C).
    principal_dirs: tuple[tuple[float, float, float], ...] = ()
    principal_mags: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CandidateBeam:
    """A complete pre-traced beam centerline (monotonic-increasing z).

    `host_id` is the beam this one starts or ends *on* (for an ON_BEAM
    endpoint), or None. Per the valid endpoint rules (keel->tip, keel->on-beam,
    on-beam->tip) at most one endpoint is ever ON_BEAM, so a single host
    suffices. `mirror_id` is the reflected partner across the chord plane, or
    None when the beam lies on the chord plane (`on_chord_plane=True`).
    """
    id: int
    control_points: tuple[tuple[float, float, float], ...]
    start_kind: NodeKind
    end_kind: NodeKind
    start_node: int
    end_node: int
    length_m: float
    min_radius_m: float
    on_chord_plane: bool
    mirror_id: int | None
    host_id: int | None
    covers: tuple[int, ...]


@dataclass(frozen=True)
class ConflictTable:
    """Precomputed pairwise incompatibilities.

    Each entry `(beam_i, bucket_a, beam_j, bucket_b)` forbids selecting beam_i
    at bucket_a together with beam_j at bucket_b (their centerlines pass closer
    than the sum of radii at those buckets). Legitimate shared nodes are
    excluded when the table is built, so beams may touch at junctions.
    """
    forbidden: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class CoverageTarget:
    """A high-stress region that must be served by an adequately-sized beam."""
    id: int
    centroid: tuple[float, float, float]
    required_min_area_m2: float
    candidate_beams: tuple[int, ...]


@dataclass(frozen=True)
class BeamWrap:
    """Filament-wound shell binding a span of beams (inert until the wrap milestone)."""
    beam_ids: tuple[int, ...]
    thickness_m: float


@dataclass(frozen=True)
class WingWrap:
    """Single airfoil-forming shell (inert until the wrap milestone)."""
    thickness_m: float


@dataclass(frozen=True)
class CandidateMenu:
    """The precomputed discrete input to the CP-SAT model."""
    nodes: tuple[CandidateNode, ...]
    beams: tuple[CandidateBeam, ...]
    cross_sections: tuple[CrossSectionOption, ...]
    conflicts: ConflictTable
    coverage_targets: tuple[CoverageTarget, ...]
    rho_kgm3: float

    def beam_by_id(self, beam_id: int) -> CandidateBeam:
        for b in self.beams:
            if b.id == beam_id:
                return b
        raise KeyError(f"no candidate beam with id {beam_id}")


@dataclass(frozen=True)
class WingCandidate:
    """A CP-SAT-selected design: (beam_id, cross-section bucket) pairs + mass."""
    beam_sections: tuple[tuple[int, int], ...]
    mass_kg: float

    @property
    def beam_ids(self) -> tuple[int, ...]:
        return tuple(bid for bid, _ in self.beam_sections)


@dataclass(frozen=True)
class GateResult:
    """The frame solver's verdict on a WingCandidate (filled by Plan 1B)."""
    feasible: bool
    max_stress_ratio: float
    tip_deflection_m: float
    governing_case: str
    mass_kg: float
