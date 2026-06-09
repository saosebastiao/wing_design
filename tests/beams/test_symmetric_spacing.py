import numpy as np

from wing_design.scenario import small_scenario
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import beam_radius_groups
from wing_design.beams.layout import chord_symmetrize_weights, stress_weighted_targets


def test_symmetrize_is_mirror_symmetric():
    rng = np.random.default_rng(0)
    n = 8
    w = rng.uniform(1.0, 5.0, size=n)
    ws = chord_symmetrize_weights(w)
    assert ws.shape == (n,)
    for b in range(n):
        assert np.isclose(ws[b], ws[(n - 1 - b) % n])           # mirror-symmetric
        assert np.isclose(ws[b], max(w[b], w[(n - 1 - b) % n]))  # per-pair max
    assert np.allclose(chord_symmetrize_weights(ws), ws)         # idempotent


def test_symmetric_spacing_keeps_grouping():
    rng = np.random.default_rng(1)
    n_beams, n_levels = 8, 5
    w = rng.uniform(1.0, 5.0, size=n_beams)
    af = stress_weighted_targets(chord_symmetrize_weights(w), n_beams)
    m = build_beam_shell_model(small_scenario().geometry, n_beams=n_beams, n_levels=n_levels,
                               beam_radius=0.02, arc_fractions=af)
    _, ng = beam_radius_groups(m)
    assert ng == (n_beams // 2 + 1) * (n_levels - 1)   # symmetry detected (reduced), not the n fallback


def test_asymmetric_spacing_falls_back():
    n_beams, n_levels = 8, 5
    af = np.linspace(0.0, 1.0, n_beams, endpoint=False) ** 1.5   # deliberately asymmetric
    m = build_beam_shell_model(small_scenario().geometry, n_beams=n_beams, n_levels=n_levels,
                               beam_radius=0.02, arc_fractions=af)
    _, ng = beam_radius_groups(m)
    assert ng == n_beams * (n_levels - 1)              # fallback to independent radii


def test_even_spacing_is_symmetric():
    n_beams, n_levels = 8, 5
    af = np.arange(n_beams) / n_beams
    m = build_beam_shell_model(small_scenario().geometry, n_beams=n_beams, n_levels=n_levels,
                               beam_radius=0.02, arc_fractions=af)
    _, ng = beam_radius_groups(m)
    assert ng == (n_beams // 2 + 1) * (n_levels - 1)
