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
