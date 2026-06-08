import math

from wing_design.scenario import small_scenario, medium_scenario, large_scenario
from wing_design import DesignParameters


def test_builders_return_design_parameters():
    for f in (small_scenario, medium_scenario, large_scenario):
        assert isinstance(f(), DesignParameters)


def test_scenario_spans():
    assert small_scenario().geometry.span == 5.0
    assert medium_scenario().geometry.span == 22.0
    assert large_scenario().geometry.span == 45.0


def test_derived_properties_resolve():
    P = medium_scenario()
    for v in (P.E_iso_Pa, P.nu_iso, P.G_iso_Pa, P.sigma_allow_Pa, P.rho_kgm3):
        assert math.isfinite(v) and v > 0.0


def test_shared_params_equal_across_sizes():
    a, b = small_scenario(), large_scenario()
    assert a.material_iso == b.material_iso
    assert a.aero == b.aero
    assert a.skin_sizing == b.skin_sizing
