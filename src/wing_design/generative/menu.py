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

    Generator contract (the candidate generator, Plan 1C, must uphold these;
    the CP-SAT model's correctness depends on them but does NOT re-validate):
      * z is monotonic-increasing along `control_points`, so the tip is always
        the high-z *end* (never a start) — relied on by the reach-tip
        constraint.
      * the `host_id` graph is acyclic and every host chain terminates at a
        keel-rooted beam (`start_kind == KEEL_STEP`, `host_id is None`) — this
        is what makes `select[b] <= select[host]` transitively ground every
        selected beam to the keel-step.
      * mirror pairs are reciprocal: if beam A has `mirror_id == B` then beam B
        has `mirror_id == A` — relied on by the symmetry tie.
    A future `validate_menu(menu)` guard (Plan 1C) should assert these before
    solving.
    """
    id: int
    control_points: tuple[tuple[float, float, float], ...]
    start_kind: NodeKind
    end_kind: NodeKind
    start_node: int
    end_node: int
    length_m: float
    min_radius_m: float  # minimum curvature radius along the centerline (not a section radius)
    on_chord_plane: bool
    mirror_id: int | None
    host_id: int | None
    covers: tuple[int, ...]  # CoverageTarget ids this beam can satisfy


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


def _coord_key(xyz, tol=1e-6):
    return (round(xyz[0] / tol), round(xyz[1] / tol), round(xyz[2] / tol))


def validate_menu(menu: "CandidateMenu") -> None:
    """Assert the contracts the CP-SAT model and frame gate rely on but do not
    re-check. Raises ValueError on the first violation.

    Contracts (see the design spec, generator-contracts section):
      * each beam's control points are strictly increasing in z (monotonic-z);
      * every beam endpoint that is a landmark kind (keel/deck/tip) coincides
        with a CandidateNode of that kind at the same coordinate;
      * the host_id graph is acyclic;
      * mirror_id is reciprocal (A.mirror == B  =>  B.mirror == A);
      * an ON_BEAM endpoint names a host that exists.
    """
    node_kind_by_key = {_coord_key(n.xyz): n.kind for n in menu.nodes}
    beam_by_id = {b.id: b for b in menu.beams}

    for b in menu.beams:
        zs = [p[2] for p in b.control_points]
        if any(z1 <= z0 for z0, z1 in zip(zs[:-1], zs[1:])):
            raise ValueError(f"beam {b.id} control points are not monotonic in z: {zs}")

        start_key = _coord_key(b.control_points[0])
        end_key = _coord_key(b.control_points[-1])
        if b.start_kind in (NodeKind.KEEL_STEP, NodeKind.DECK_STEP, NodeKind.TIP):
            if node_kind_by_key.get(start_key) != b.start_kind:
                raise ValueError(
                    f"beam {b.id} start landmark {b.start_kind} has no matching node"
                )
        if b.end_kind in (NodeKind.KEEL_STEP, NodeKind.DECK_STEP, NodeKind.TIP):
            if node_kind_by_key.get(end_key) != b.end_kind:
                raise ValueError(
                    f"beam {b.id} end landmark {b.end_kind} has no matching node"
                )

        if b.host_id is not None and b.host_id not in beam_by_id:
            raise ValueError(f"beam {b.id} names a host {b.host_id} that does not exist")

        if b.mirror_id is not None:
            partner = beam_by_id.get(b.mirror_id)
            if partner is None or partner.mirror_id != b.id:
                raise ValueError(f"beam {b.id} mirror_id {b.mirror_id} is not reciprocal")

    # Host-graph acyclicity: walk each chain, bail if it revisits a beam.
    for b in menu.beams:
        seen = set()
        cur = b
        while cur.host_id is not None:
            if cur.id in seen:
                raise ValueError(f"host_id graph has a cycle involving beam {cur.id}")
            seen.add(cur.id)
            cur = beam_by_id[cur.host_id]
