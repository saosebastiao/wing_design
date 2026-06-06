import numpy as np

from wing_design.materials.unidir import T700_EPOXY, reduced_stiffness_Q, laminate_stiffness


def test_Q_symmetric_and_positive():
    Q = reduced_stiffness_Q(T700_EPOXY)
    assert Q.shape == (3, 3)
    assert np.allclose(Q, Q.T)
    assert Q[0, 0] > Q[1, 1] > 0
    assert Q[2, 2] > 0


def test_balanced_laminate_has_no_extension_shear_coupling():
    A, D, Qeff = laminate_stiffness(T700_EPOXY, f0=0.5, f45=0.25, f90=0.25, thickness=0.003)
    assert A.shape == (3, 3) and D.shape == (3, 3)
    assert abs(A[0, 2]) < 1e-3 * abs(A[0, 0])
    assert abs(A[1, 2]) < 1e-3 * abs(A[1, 1])
    assert np.allclose(D, (0.003**2 / 12.0) * A, rtol=1e-9)


def test_pm45_maximizes_shear_stiffness():
    _, _, q_all0 = laminate_stiffness(T700_EPOXY, f0=1.0, f45=0.0, f90=0.0, thickness=0.003)
    _, _, q_45 = laminate_stiffness(T700_EPOXY, f0=0.0, f45=1.0, f90=0.0, thickness=0.003)
    assert q_45[2, 2] > 2.0 * q_all0[2, 2]


def test_transformed_qbar_known_angles():
    from wing_design.materials.unidir import transformed_Qbar
    Q = reduced_stiffness_Q(T700_EPOXY)
    scale = Q[0, 0]
    # 0 deg leaves Q unchanged; 90 deg swaps 11<->22 with no shear coupling.
    assert np.allclose(transformed_Qbar(Q, 0.0), Q, rtol=1e-12)
    q90 = transformed_Qbar(Q, 90.0)
    assert abs(q90[0, 0] - Q[1, 1]) < 1e-6 * scale
    assert abs(q90[1, 1] - Q[0, 0]) < 1e-6 * scale
    assert abs(q90[2, 2] - Q[2, 2]) < 1e-6 * scale
    assert abs(q90[0, 2]) < 1e-6 * scale and abs(q90[1, 2]) < 1e-6 * scale
    # +-45 are antisymmetric in the shear-coupling terms (so the balanced average is 0),
    # but each is individually strongly coupled.
    qp = transformed_Qbar(Q, 45.0)
    qm = transformed_Qbar(Q, -45.0)
    assert abs(qp[0, 2] + qm[0, 2]) < 1e-6 * scale
    assert abs(qp[1, 2] + qm[1, 2]) < 1e-6 * scale
    assert abs(qp[0, 2]) > 0.1 * scale
