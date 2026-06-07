import numpy as np

from wing_design.structural.buckling import beam_euler_utilization, panel_buckling_utilization


def test_euler_utilization_against_closed_form():
    E, r, L = 70e9, 0.02, 1.0
    I = np.pi * r**4 / 4.0
    Pcr = np.pi**2 * E * I / L**2          # K=1
    axial = np.array([-Pcr / 2.0])         # negative = compression; with SF=2 → util 1.0
    u = beam_euler_utilization(axial, np.array([r]), np.array([L]), E=E, K=1.0, safety_factor=2.0)
    assert abs(u[0] - 1.0) < 1e-9
    ut = beam_euler_utilization(np.array([+Pcr]), np.array([r]), np.array([L]), E=E, safety_factor=2.0)
    assert ut[0] == 0.0   # tension never buckles


def test_panel_utilization_against_closed_form():
    E, nu, t, area, kc = 70e9, 0.3, 0.003, 0.04, 4.0
    D11 = E * t**3 / (12 * (1 - nu**2))
    b = np.sqrt(area)
    sigma_cr = kc * np.pi**2 * D11 / (b**2 * t)
    stress = np.array([[-sigma_cr, 0.0, 0.0]])   # uniaxial compression at σcr, SF=1 → util 1
    u = panel_buckling_utilization(stress, np.array([area]), D11=D11, t=t, kc=kc, safety_factor=1.0)
    assert abs(u[0] - 1.0) < 1e-9
    ut = panel_buckling_utilization(np.array([[sigma_cr, 0.0, 0.0]]), np.array([area]), D11=D11, t=t, kc=kc)
    assert ut[0] == 0.0   # tension → no buckling
