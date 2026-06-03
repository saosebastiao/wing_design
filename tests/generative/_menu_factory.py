"""Builders for small synthetic CandidateMenus used in CP-SAT unit tests.

Every beam built here defaults to a globally-valid configuration (ends at the
tip, on the chord plane, no host, no coverage), so a menu stays feasible under
the *full* CP-SAT model even when an individual test only exercises one
constraint.
"""
from __future__ import annotations

from wing_design.generative.menu import (
    CandidateBeam,
    CandidateMenu,
    ConflictTable,
    CoverageTarget,
    CrossSectionOption,
    CrossSectionShape,
    NodeKind,
)


def cs_catalog(areas):
    """Catalog of circular cross-sections, one bucket per area (m^2)."""
    return tuple(
        CrossSectionOption(bucket=i, shape=CrossSectionShape.CIRCLE, area_m2=a)
        for i, a in enumerate(areas)
    )


def beam(
    beam_id,
    length=1.0,
    start_kind=NodeKind.KEEL_STEP,
    end_kind=NodeKind.TIP,
    on_chord_plane=True,
    mirror_id=None,
    host_id=None,
    covers=(),
):
    """A globally-valid CandidateBeam with sensible defaults."""
    return CandidateBeam(
        id=beam_id,
        control_points=((0.0, 0.0, 0.0), (0.0, 0.0, length)),
        start_kind=start_kind,
        end_kind=end_kind,
        start_node=0,
        end_node=1,
        length_m=length,
        min_radius_m=10.0,
        on_chord_plane=on_chord_plane,
        mirror_id=mirror_id,
        host_id=host_id,
        covers=covers,
    )


def menu(beams, cross_sections=None, conflicts=(), coverage=(), rho=1550.0):
    if cross_sections is None:
        cross_sections = cs_catalog([1.0e-3, 2.0e-3])
    return CandidateMenu(
        nodes=(),
        beams=tuple(beams),
        cross_sections=tuple(cross_sections),
        conflicts=ConflictTable(forbidden=tuple(conflicts)),
        coverage_targets=tuple(coverage),
        rho_kgm3=rho,
    )


def target(target_id, required_min_area_m2, candidate_beams):
    return CoverageTarget(
        id=target_id,
        centroid=(0.0, 0.0, 0.0),
        required_min_area_m2=required_min_area_m2,
        candidate_beams=tuple(candidate_beams),
    )
