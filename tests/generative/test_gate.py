import math

import numpy as np

from wing_design.generative.gate import section_properties


def test_section_properties_circle():
    area = 4.0e-3
    A, I, J = section_properties(area)
    assert math.isclose(A, area, rel_tol=1e-12)
    # circle: I = A^2 / (4 pi), J = 2 I
    assert math.isclose(I, area**2 / (4.0 * math.pi), rel_tol=1e-12)
    assert math.isclose(J, 2.0 * I, rel_tol=1e-12)
