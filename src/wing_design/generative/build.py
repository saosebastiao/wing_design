"""Export a selected WingCandidate to build123d geometry.

Each selected beam's circular cross-section is swept along its centerline
polyline; the beams are collected into a single Compound (no boolean union, so
the export is robust). Use `export_step` / `export_stl` from build123d on the
result.
"""
from __future__ import annotations

from build123d import (
    BuildLine,
    BuildPart,
    BuildSketch,
    Circle,
    Compound,
    Plane,
    Polyline,
    Vector,
    sweep,
)


def _beam_solid(control_points, radius_m):
    pts = [tuple(float(c) for c in p) for p in control_points]
    start = Vector(*pts[0])
    nxt = Vector(*pts[1])
    tangent = (nxt - start).normalized()
    with BuildPart() as bp:
        with BuildLine():
            Polyline(*pts)
        with BuildSketch(Plane(origin=start, z_dir=tangent)):
            Circle(radius_m)
        sweep()
    return bp.part


def wing_candidate_to_part(candidate, menu) -> Compound:
    """Return a Compound with one swept circular beam per selected (beam, bucket)."""
    radius_by_bucket = {cs.bucket: cs.radius_m for cs in menu.cross_sections}
    solids = []
    for beam_id, bucket in candidate.beam_sections:
        beam = menu.beam_by_id(beam_id)
        solids.append(_beam_solid(beam.control_points, radius_by_bucket[bucket]))
    return Compound(children=solids)
