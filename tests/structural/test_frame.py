import numpy as np

from wing_design.structural.frame import BeamSection, solve_frame

E = 70e9
NU = 0.3
G = E / (2 * (1 + NU))


def _two_node(p0, p1, sec):
    nodes = np.array([p0, p1], dtype=float)
    elements = np.array([[0, 1]], dtype=int)
    return nodes, elements, [sec]


def test_cantilever_tip_bending():
    sec = BeamSection.circular(0.02)
    nodes, elements, sections = _two_node([0, 0, 0], [1, 0, 0], sec)
    loads = np.zeros((2, 6))
    P = 100.0
    loads[1, 2] = P  # transverse Fz at the tip
    res = solve_frame(nodes, elements, sections, E=E, G=G,
                      fixed_nodes=np.array([0]), loads=loads)
    expected = P * 1.0**3 / (3 * E * sec.Iy)  # PL^3 / 3EI
    assert abs(res.displacements[1, 2] - expected) / expected < 1e-6


def test_axial_bar():
    sec = BeamSection.circular(0.02)
    nodes, elements, sections = _two_node([0, 0, 0], [2, 0, 0], sec)
    loads = np.zeros((2, 6))
    P = 1000.0
    loads[1, 0] = P
    res = solve_frame(nodes, elements, sections, E=E, G=G,
                      fixed_nodes=np.array([0]), loads=loads)
    expected = P * 2.0 / (E * sec.A)  # PL / EA
    assert abs(res.displacements[1, 0] - expected) / expected < 1e-9
    assert abs(res.axial_force[0] - P) / P < 1e-9  # tension positive


def test_torsion_bar():
    sec = BeamSection.circular(0.02)
    nodes, elements, sections = _two_node([0, 0, 0], [1, 0, 0], sec)
    loads = np.zeros((2, 6))
    T = 50.0
    loads[1, 3] = T  # Mx (torque) at the tip
    res = solve_frame(nodes, elements, sections, E=E, G=G,
                      fixed_nodes=np.array([0]), loads=loads)
    expected = T * 1.0 / (G * sec.J)  # TL / GJ
    assert abs(res.displacements[1, 3] - expected) / expected < 1e-9
    assert abs(res.torsion[0] - T) / T < 1e-6
