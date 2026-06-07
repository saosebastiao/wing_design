import numpy as np

from wing_design.beams.layout import stress_weighted_targets


def test_uniform_weights_recover_even_spacing():
    t = stress_weighted_targets(np.ones(8), 8)
    assert np.allclose(t, np.arange(8) / 8.0)


def test_targets_in_unit_interval_and_sorted():
    t = stress_weighted_targets(np.array([1.0, 2.0, 5.0, 1.0, 1.0, 3.0]), 10)
    assert t.shape == (10,)
    assert np.all((t >= 0.0) & (t < 1.0))
    assert np.all(np.diff(t) >= 0.0)


def test_peaked_weights_cluster_targets():
    w = np.ones(8)
    w[2] = 100.0
    t = stress_weighted_targets(w, 8)
    near_peak = np.sum((t >= 2.0 / 8.0) & (t < 3.0 / 8.0))
    assert near_peak >= 4
