import math

from wing_design.generative.menu import CrossSectionOption, CrossSectionShape


def test_cross_section_radius_is_equivalent_circle():
    cs = CrossSectionOption(bucket=0, shape=CrossSectionShape.CIRCLE, area_m2=math.pi)
    assert math.isclose(cs.radius_m, 1.0, rel_tol=1e-9)
