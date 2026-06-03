import math

from wing_design.generative.menu import (
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
)


def test_cross_section_radius_is_equivalent_circle():
    cs = CrossSectionOption(bucket=0, shape=CrossSectionShape.CIRCLE, area_m2=math.pi)
    assert math.isclose(cs.radius_m, 1.0, rel_tol=1e-9)


def test_candidate_menu_lookup_and_wing_candidate_ids():
    nodes = (
        CandidateNode(id=0, xyz=(0.0, 0.0, -0.95), kind=NodeKind.KEEL_STEP, z_layer=0),
        CandidateNode(id=1, xyz=(0.0, 0.0, 5.0), kind=NodeKind.TIP, z_layer=11),
    )
    beam = CandidateBeam(
        id=7,
        control_points=((0.0, 0.0, -0.95), (0.0, 0.0, 5.0)),
        start_kind=NodeKind.KEEL_STEP,
        end_kind=NodeKind.TIP,
        start_node=0,
        end_node=1,
        length_m=5.95,
        min_radius_m=10.0,
        on_chord_plane=True,
        mirror_id=None,
        host_id=None,
        covers=(2,),
    )
    cs = (CrossSectionOption(bucket=0, shape=CrossSectionShape.CIRCLE, area_m2=1e-3),)
    menu = CandidateMenu(
        nodes=nodes,
        beams=(beam,),
        cross_sections=cs,
        conflicts=ConflictTable(forbidden=()),
        coverage_targets=(
            CoverageTarget(id=2, centroid=(0.0, 0.0, 2.5),
                           required_min_area_m2=1e-3, candidate_beams=(7,)),
        ),
        rho_kgm3=1550.0,
    )
    assert menu.beam_by_id(7) is beam
    cand = WingCandidate(beam_sections=((7, 0),), mass_kg=9.22)
    assert cand.beam_ids == (7,)
    verdict = GateResult(feasible=True, max_stress_ratio=0.8,
                         tip_deflection_m=0.1, governing_case="nominal", mass_kg=9.22)
    assert verdict.feasible
