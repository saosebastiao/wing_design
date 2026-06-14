import numpy as np
import pytest

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


def test_euler_I_matches_solid_rod_form():
    import numpy as np
    from wing_design.structural.buckling import beam_euler_utilization, beam_euler_utilization_I
    r = np.array([0.01, 0.02]); L = np.array([1.0, 1.5]); N = np.array([-500.0, -2000.0])
    I = np.pi * r**4 / 4.0
    a = beam_euler_utilization(N, r, L, E=70e9, K=1.0, safety_factor=1.5)
    b = beam_euler_utilization_I(N, I, L, E=70e9, K=1.0, safety_factor=1.5)
    assert np.allclose(a, b)


def test_tube_wall_utilization_scales():
    import numpy as np
    from wing_design.structural.frame import BeamSection
    from wing_design.structural.buckling import tube_wall_utilization
    sec = BeamSection.annular(0.05, 0.002)
    u = tube_wall_utilization(np.array([-1.0e4]), np.array([0.05]), np.array([0.002]),
                              np.array([sec.A]), E=70e9, safety_factor=1.5)
    # doubling the wall halves stress AND doubles sigma_cr -> ~4x lower util
    sec2 = BeamSection.annular(0.05, 0.004)
    u2 = tube_wall_utilization(np.array([-1.0e4]), np.array([0.05]), np.array([0.004]),
                               np.array([sec2.A]), E=70e9, safety_factor=1.5)
    assert u[0] > 0
    assert u[0] / u2[0] == pytest.approx(4.0, rel=0.05)
    # tension never crimps
    ut = tube_wall_utilization(np.array([+1.0e4]), np.array([0.05]), np.array([0.002]),
                               np.array([sec.A]), E=70e9)
    assert ut[0] == 0.0


def _central_diff(f, x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


def test_k_ring_quartic_and_gradient_fd():
    E = 7.0e10
    L_ring = np.array([0.45, 0.50, 0.40])
    s = np.array([0.9, 0.9, 1.1])
    alpha = 3.0
    r = 0.012
    from wing_design.structural.buckling import k_ring_stiffness, dk_ring_dr
    k = k_ring_stiffness(r, E, L_ring, s, alpha)
    expect = alpha * E * (np.pi * r**4 / 4.0) / (L_ring**3 * s)
    assert np.allclose(k, expect, rtol=1e-12)
    ana = dk_ring_dr(r, E, L_ring, s, alpha)
    fd = _central_diff(lambda rr: k_ring_stiffness(rr, E, L_ring, s, alpha), r, 1e-7)
    assert np.allclose(ana, fd, rtol=1e-6, atol=1e-6 * np.abs(fd).max())
    assert np.allclose(ana, 4.0 * k / r, rtol=1e-12)


def test_k_ring_finite_at_degenerate_inputs():
    import numpy as np
    from wing_design.structural.buckling import k_ring_stiffness, dk_ring_dr
    # zero radius / zero span must not produce nan/inf (bounded DV never hits 0, but
    # boundary probes and degenerate mesh elements must stay finite for the optimizer)
    assert np.isfinite(dk_ring_dr(0.0, 7e10, np.array([0.4]), np.array([0.9])))
    assert np.isfinite(k_ring_stiffness(0.012, 7e10, np.array([0.0]), np.array([0.0]))).all()
