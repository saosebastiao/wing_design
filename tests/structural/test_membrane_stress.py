import numpy as np

from wing_design.structural.shell import recover_membrane_stress, membrane_von_mises

E, NU = 70e9, 0.3


def test_uniaxial_stretch_recovers_known_stress():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    tris = np.array([[0, 1, 2]], dtype=int)
    eps = 1e-3
    disp = np.zeros((3, 6))
    disp[:, 0] = eps * nodes[:, 0]      # u = eps*x, v=0
    s = recover_membrane_stress(nodes, tris, disp, E=E, nu=NU)
    sxx_expected = E / (1 - NU**2) * eps
    assert s.shape == (1, 3)
    assert abs(s[0, 0] - sxx_expected) / sxx_expected < 1e-9
    assert abs(s[0, 1] - NU * sxx_expected) / (NU * sxx_expected) < 1e-9
    assert abs(s[0, 2]) < 1e-3


def test_membrane_von_mises_matches_formula():
    stress = np.array([[100.0, 40.0, 30.0]])
    vm = membrane_von_mises(stress)
    expected = np.sqrt(100**2 - 100 * 40 + 40**2 + 3 * 30**2)
    assert abs(vm[0] - expected) < 1e-9
