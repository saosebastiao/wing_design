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
