import numpy as np

from wing_design.materials.unidir import T700_EPOXY
from wing_design.materials.failure import (
    tsai_wu_coefficients, tsai_wu_index, tsai_wu_strength_ratio,
    tsai_wu_strength_ratio_grad,
    ply_strength_ratio, laminate_min_strength_ratio,
)

PLY = T700_EPOXY


def test_index_unit_at_uniaxial_tension():
    assert np.isclose(tsai_wu_index([PLY.Xt_Pa, 0.0, 0.0], PLY), 1.0, rtol=1e-9)


def test_strength_ratio_one_at_each_strength_point():
    assert np.isclose(tsai_wu_strength_ratio([PLY.Xt_Pa, 0.0, 0.0], PLY), 1.0, rtol=1e-9)
    assert np.isclose(tsai_wu_strength_ratio([-PLY.Xc_Pa, 0.0, 0.0], PLY), 1.0, rtol=1e-9)
    assert np.isclose(tsai_wu_strength_ratio([0.0, PLY.Yt_Pa, 0.0], PLY), 1.0, rtol=1e-9)
    assert np.isclose(tsai_wu_strength_ratio([0.0, -PLY.Yc_Pa, 0.0], PLY), 1.0, rtol=1e-9)
    assert np.isclose(tsai_wu_strength_ratio([0.0, 0.0, PLY.S12_Pa], PLY), 1.0, rtol=1e-9)


def test_strength_ratio_scales_inverse_with_shear():
    r1 = tsai_wu_strength_ratio([0.0, 0.0, 1.0e7], PLY)
    r2 = tsai_wu_strength_ratio([0.0, 0.0, 5.0e6], PLY)
    assert np.isclose(r2, 2.0 * r1, rtol=1e-9)


def test_zero_stress_is_safe():
    assert tsai_wu_strength_ratio([0.0, 0.0, 0.0], PLY) >= 1.0e8


def test_f12_convention():
    F1, F2, F11, F22, F66, F12 = tsai_wu_coefficients(PLY)
    assert np.isclose(F12, -0.5 * np.sqrt(F11 * F22), rtol=1e-12)


def test_ply_strength_ratio_fibre_vs_transverse():
    eps = [1.0e-3, 0.0, 0.0]
    r0 = ply_strength_ratio(PLY, eps, 0.0)
    r90 = ply_strength_ratio(PLY, eps, 90.0)
    assert r0 > r90


def test_laminate_min_skips_absent_orientations():
    eps = [1.0e-3, 0.0, 0.0]
    r_all = laminate_min_strength_ratio(PLY, eps, f0=0.34, f45=0.33, f90=0.33, offset_deg=0.0)
    r_0only = laminate_min_strength_ratio(PLY, eps, f0=1.0, f45=0.0, f90=0.0, offset_deg=0.0)
    assert r_0only >= r_all
    assert np.isclose(r_0only, ply_strength_ratio(PLY, eps, 0.0), rtol=1e-9)


def test_failure_helpers_exported():
    import wing_design.materials as m
    for name in ("tsai_wu_index", "tsai_wu_strength_ratio", "tsai_wu_coefficients",
                 "ply_strength_ratio", "laminate_min_strength_ratio"):
        assert hasattr(m, name), name


def test_batch_matches_scalar():
    rng = np.random.default_rng(0)
    M = 20
    eps = rng.normal(scale=1e-3, size=(M, 3))
    offs = rng.uniform(-90.0, 90.0, size=M)
    from wing_design.materials.failure import laminate_min_strength_ratio_batch
    batch = laminate_min_strength_ratio_batch(PLY, eps, f0=0.34, f45=0.33, f90=0.33, offset_deg=offs)
    scalar = np.array([
        laminate_min_strength_ratio(PLY, eps[i], f0=0.34, f45=0.33, f90=0.33, offset_deg=float(offs[i]))
        for i in range(M)
    ])
    assert batch.shape == (M,)
    assert np.allclose(batch, scalar, rtol=1e-9, atol=0.0)


def test_batch_skips_absent_orientations():
    rng = np.random.default_rng(1)
    M = 12
    eps = rng.normal(scale=1e-3, size=(M, 3))
    offs = np.zeros(M)
    from wing_design.materials.failure import laminate_min_strength_ratio_batch
    batch = laminate_min_strength_ratio_batch(PLY, eps, f0=1.0, f45=0.0, f90=0.0, offset_deg=offs)
    scalar = np.array([
        laminate_min_strength_ratio(PLY, eps[i], f0=1.0, f45=0.0, f90=0.0, offset_deg=0.0)
        for i in range(M)
    ])
    assert np.allclose(batch, scalar, rtol=1e-9)


def test_batch_handles_zero_and_shear_rows():
    from wing_design.materials.failure import laminate_min_strength_ratio_batch
    eps = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1e-3]])
    batch = laminate_min_strength_ratio_batch(PLY, eps, f0=1.0, f45=0.0, f90=0.0, offset_deg=np.zeros(2))
    assert batch[0] >= 1.0e8
    assert np.isfinite(batch[1]) and batch[1] > 0


def test_strength_ratio_grad_matches_fd():
    rng = np.random.default_rng(3)
    # representative material-axis stress magnitudes (Pa) with all components active
    scales = np.array([6.0e8, 4.0e7, 5.0e7])
    for _ in range(6):
        sigma = rng.normal(size=3) * scales
        ana = tsai_wu_strength_ratio_grad(sigma, PLY)
        fd = np.zeros(3)
        for k in range(3):
            h = 1.0e-3 * max(abs(sigma[k]), 1.0e3)
            sp = sigma.copy(); sp[k] += h
            sm = sigma.copy(); sm[k] -= h
            fd[k] = (
                tsai_wu_strength_ratio(sp, PLY) - tsai_wu_strength_ratio(sm, PLY)
            ) / (2.0 * h)
        assert np.allclose(ana, fd, rtol=1e-6, atol=1e-6 * np.abs(fd).max())


def test_batch_array_fractions_match_per_element():
    rng = np.random.default_rng(7)
    M = 16
    eps = rng.normal(scale=1e-3, size=(M, 3))
    offs = rng.uniform(-90.0, 90.0, size=M)
    f0 = rng.uniform(0.1, 0.6, size=M)
    f45 = rng.uniform(0.1, 0.6, size=M) * (1.0 - f0)
    f90 = 1.0 - f0 - f45
    from wing_design.materials.failure import laminate_min_strength_ratio_batch, laminate_min_strength_ratio
    batch = laminate_min_strength_ratio_batch(PLY, eps, f0=f0, f45=f45, f90=f90, offset_deg=offs)
    per = np.array([
        laminate_min_strength_ratio(PLY, eps[i], f0=float(f0[i]), f45=float(f45[i]), f90=float(f90[i]),
                                    offset_deg=float(offs[i]))
        for i in range(M)
    ])
    assert batch.shape == (M,)
    assert np.allclose(batch, per, rtol=1e-9, atol=0.0)
