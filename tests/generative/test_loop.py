from wing_design.generative.loop import (
    GatedDesign,
    TrussResult,
    select_lightest_feasible,
)
from wing_design.generative.menu import GateResult, WingCandidate


def _design(mass):
    return WingCandidate(beam_sections=((0, 0),), mass_kg=mass)


def _verdict(feasible, mass):
    return GateResult(feasible=feasible, max_stress_ratio=0.5,
                      tip_deflection_m=0.1, governing_case="x", mass_kg=mass)


def test_select_returns_first_feasible():
    # designs ascending by mass; the lightest fails, the next passes.
    a, b, c = _design(1.0), _design(2.0), _design(3.0)
    verdicts = {1.0: _verdict(False, 1.0), 2.0: _verdict(True, 2.0),
                3.0: _verdict(True, 3.0)}
    result = select_lightest_feasible([a, b, c], lambda d: verdicts[d.mass_kg])
    assert result is not None
    chosen, verdict = result
    assert chosen is b
    assert verdict.feasible


def test_select_returns_none_when_all_fail():
    a, b = _design(1.0), _design(2.0)
    result = select_lightest_feasible([a, b], lambda d: _verdict(False, d.mass_kg))
    assert result is None


from wing_design.generative.candidates import build_beam_library
from wing_design.generative.loop import generate_truss
from wing_design.generative.menu import CandidateMenu, ConflictTable
from wing_design.scenario import default_scenario


def _library_menu(params):
    nodes, beams, cross_sections = build_beam_library(params)
    return CandidateMenu(
        nodes=tuple(nodes), beams=tuple(beams), cross_sections=tuple(cross_sections),
        conflicts=ConflictTable(forbidden=()), coverage_targets=(),
        rho_kgm3=params.material.rho_kgm3,
    )


def test_generate_truss_picks_lightest_feasible_under_envelope():
    # No FEA: a real beam library + simple analytic per-case load densities.
    params = default_scenario()
    menu = _library_menu(params)
    # A gentle case everything passes, and a severe case that sizes the design.
    # 50 N/m over 5 m span: lightest design (mass ~7.7 kg) fails on tip deflection
    # (~731 mm >> 250 mm limit) while the next-lightest (~15.4 kg) passes (~183 mm).
    cases = {
        "gentle": (lambda z: 5.0),
        "severe": (lambda z: 50.0),
    }
    result = generate_truss(menu, params, cases, max_candidates=200)
    assert result.chosen is not None
    assert result.verdict.feasible
    # the chosen design survives the worst case
    assert result.verdict.governing_case in cases
    # frontier records what was tried, ascending in mass
    masses = [g.design.mass_kg for g in result.frontier]
    assert masses == sorted(masses)
    assert result.frontier[-1].design is result.chosen


def test_generate_truss_returns_none_when_envelope_unsurvivable():
    params = default_scenario()
    menu = _library_menu(params)
    # An absurd load no catalog section can survive.
    cases = {"impossible": (lambda z: 1.0e9)}
    result = generate_truss(menu, params, cases, max_candidates=50)
    assert result.chosen is None
    assert result.verdict is None
    assert len(result.frontier) > 0  # designs were tried and all failed
