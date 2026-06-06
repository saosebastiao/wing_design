import numpy as np

from wing_design.structural.shell import (
    tri_element_stiffness, tri_element_stiffness_laminate,
    recover_membrane_stress, recover_membrane_stress_C,
)

E, NU, T = 70e9, 0.3, 0.003


def _iso_A_D(E, nu, t):
    Dm = (E / (1 - nu**2)) * np.array([[1, nu, 0.0], [nu, 1, 0.0], [0, 0, (1 - nu) / 2]])
    return t * Dm, (t**3 / 12.0) * Dm


def test_laminate_element_reduces_to_isotropic():
    p1, p2, p3 = np.array([0, 0, 0.]), np.array([1, 0, 0.2]), np.array([0, 1, 0.1])
    A, D = _iso_A_D(E, NU, T)
    K_iso = tri_element_stiffness(p1, p2, p3, E=E, nu=NU, t=T)
    K_lam = tri_element_stiffness_laminate(p1, p2, p3, A=A, D=D)
    assert np.allclose(K_iso, K_lam, rtol=1e-9, atol=1e-3)


def test_recover_membrane_stress_C_matches_isotropic():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    tris = np.array([[0, 1, 2]], dtype=int)
    disp = np.zeros((3, 6)); disp[:, 0] = 1e-3 * nodes[:, 0]
    Dm = (E / (1 - NU**2)) * np.array([[1, NU, 0.0], [NU, 1, 0.0], [0, 0, (1 - NU) / 2]])
    s_iso = recover_membrane_stress(nodes, tris, disp, E=E, nu=NU)
    s_C = recover_membrane_stress_C(nodes, tris, disp, C=Dm)
    assert np.allclose(s_iso, s_C, rtol=1e-9)
