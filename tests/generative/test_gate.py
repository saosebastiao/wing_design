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


from wing_design.generative.gate import local_beam_stiffness


def test_local_beam_stiffness_symmetric_and_axial():
    E, G, A, I, J, L = 200e9, 80e9, 1e-3, 1e-6, 2e-6, 2.0
    k = local_beam_stiffness(E, G, A, I, J, L)
    assert k.shape == (12, 12)
    # symmetric
    assert np.allclose(k, k.T)
    # axial sub-terms: k[0,0] = EA/L, k[0,6] = -EA/L
    assert math.isclose(k[0, 0], E * A / L, rel_tol=1e-12)
    assert math.isclose(k[0, 6], -E * A / L, rel_tol=1e-12)
    # torsion: k[3,3] = GJ/L
    assert math.isclose(k[3, 3], G * J / L, rel_tol=1e-12)
    # bending: k[1,1] = 12 EI / L^3
    assert math.isclose(k[1, 1], 12.0 * E * I / L**3, rel_tol=1e-12)


from wing_design.generative.gate import beam_transform


def test_beam_transform_length_and_orthonormal():
    T, L = beam_transform((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    assert math.isclose(L, 3.0, rel_tol=1e-12)
    assert T.shape == (12, 12)
    R = T[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.allclose(R[0], [1.0, 0.0, 0.0], atol=1e-12)


def test_beam_transform_handles_vertical_beam():
    T, L = beam_transform((0.0, 0.0, 0.0), (0.0, 0.0, 5.0))
    assert math.isclose(L, 5.0, rel_tol=1e-12)
    R = T[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.allclose(R[0], [0.0, 0.0, 1.0], atol=1e-12)
