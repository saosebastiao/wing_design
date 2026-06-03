from build123d import Compound

from wing_design.generative.build import wing_candidate_to_part
from wing_design.generative.candidates import build_beam_library
from wing_design.generative.menu import (
    CandidateMenu,
    ConflictTable,
    WingCandidate,
)
from wing_design.scenario import default_scenario


def _menu():
    params = default_scenario()
    nodes, beams, cross_sections = build_beam_library(params)
    return CandidateMenu(
        nodes=tuple(nodes), beams=tuple(beams), cross_sections=tuple(cross_sections),
        conflicts=ConflictTable(forbidden=()), coverage_targets=(),
        rho_kgm3=params.material.rho_kgm3,
    )


def test_wing_candidate_to_part_builds_a_solid_per_beam():
    menu = _menu()
    # select the spar (id 0) and one branch (id 1), each at the largest bucket
    big = menu.cross_sections[-1].bucket
    candidate = WingCandidate(beam_sections=((0, big), (1, big)), mass_kg=1.0)
    part = wing_candidate_to_part(candidate, menu)
    assert isinstance(part, Compound)
    # one swept solid per selected beam
    assert len(part.solids()) == 2
    assert part.volume > 0.0
