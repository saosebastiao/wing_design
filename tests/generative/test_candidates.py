import math

from wing_design.generative.candidates import build_beam_library, build_candidate_menu
from wing_design.generative.menu import (
    CandidateMenu,
    ConflictTable,
    NodeKind,
    validate_menu,
)
from wing_design.scenario import default_scenario


def _wrap_for_validation(nodes, beams, cross_sections):
    return CandidateMenu(
        nodes=tuple(nodes), beams=tuple(beams), cross_sections=tuple(cross_sections),
        conflicts=ConflictTable(forbidden=()), coverage_targets=(), rho_kgm3=1550.0,
    )


def test_build_beam_library_is_contract_valid():
    params = default_scenario()
    nodes, beams, cross_sections = build_beam_library(params)
    # landmarks present
    kinds = {n.kind for n in nodes}
    assert NodeKind.KEEL_STEP in kinds
    assert NodeKind.DECK_STEP in kinds
    assert NodeKind.TIP in kinds
    # at least the central spar + a few branches
    assert len(beams) >= 3
    # exactly one keel-rooted spar; it carries keel->deck->tip
    spars = [b for b in beams if b.start_kind == NodeKind.KEEL_STEP]
    assert len(spars) == 1
    spar = spars[0]
    assert spar.control_points[0][2] < spar.control_points[1][2] < spar.control_points[-1][2]
    # every other beam is hosted on the spar at the deck node
    for b in beams:
        if b is spar:
            continue
        assert b.host_id == spar.id
        assert b.start_kind == NodeKind.ON_BEAM
    # the whole library satisfies validate_menu
    validate_menu(_wrap_for_validation(nodes, beams, cross_sections))


def test_build_beam_library_cross_sections_within_max():
    params = default_scenario()
    _nodes, _beams, cross_sections = build_beam_library(params)
    assert len(cross_sections) == params.generative.n_area_buckets
    for cs in cross_sections:
        assert 0 < cs.area_m2 <= params.generative.cross_section_area_max_m2


def test_build_beam_library_has_a_reciprocal_mirror_pair():
    params = default_scenario()
    _nodes, beams, _cs = build_beam_library(params)
    mirrored = [b for b in beams if b.mirror_id is not None]
    assert len(mirrored) >= 2
    by_id = {b.id: b for b in beams}
    for b in mirrored:
        assert by_id[b.mirror_id].mirror_id == b.id


def test_build_candidate_menu_runs_and_validates():
    # Uses the real shell FEA (~40 s). Smaller mesh via a coarser element size
    # keeps it fast enough for a test.
    params = default_scenario()
    menu = build_candidate_menu(params)
    assert isinstance(menu, CandidateMenu)
    assert len(menu.beams) >= 3
    assert len(menu.coverage_targets) >= 1
    # rho carried from the material
    assert math.isclose(menu.rho_kgm3, params.material.rho_kgm3, rel_tol=1e-9)
    # every coverage target is satisfiable by some beam+bucket in the menu
    max_area = max(cs.area_m2 for cs in menu.cross_sections)
    for tgt in menu.coverage_targets:
        assert tgt.required_min_area_m2 <= max_area + 1e-12
        assert len(tgt.candidate_beams) >= 1
    # the produced menu obeys all generator contracts
    validate_menu(menu)
