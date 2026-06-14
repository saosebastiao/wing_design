import numpy as np

from wing_design.geometry import small_wingsail, medium_wingsail
from wing_design.beams.fea_model import build_beam_frame, project_panels_to_beam_nodes, solve_beam_frame
from wing_design.aero.loads import PanelLoads
from wing_design.beams.shell_model import (
    skin_triangles,
    build_beam_shell_model,
    solve_beam_shell_model,
    braced_segment_mask,
)
from wing_design.beams.layout import stress_weighted_targets


def test_skin_triangulation_counts_and_validity():
    nb, nl = 8, 5
    tris = skin_triangles(nb, nl)
    assert tris.shape == (2 * nb * (nl - 1), 3)
    assert tris.min() >= 0 and tris.max() < nb * nl
    assert len(np.unique(tris)) == nb * nl


def test_model_builds_and_solves():
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, beam_radius=0.02)
    assert model.beam_elements.shape[0] == 8 * (5 - 1)
    assert model.shell_tris.shape[0] == 2 * 8 * (5 - 1)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 20.0
    res = solve_beam_shell_model(model, loads)
    assert np.isfinite(res.displacements).all()
    assert np.allclose(res.displacements[model.fixed_nodes], 0.0)


def test_skin_stiffer_than_ring_frame():
    spec = small_wingsail
    nb, nl, r = 16, 6, 0.02
    model = build_beam_shell_model(spec, n_beams=nb, n_levels=nl, beam_radius=r)
    frame = build_beam_frame(spec, n_beams=nb, n_levels=nl, beam_radius=r)

    panels = PanelLoads(
        centers_xyz=np.array([[0.0, 2.5, 0.1]]),
        areas=np.array([1.0]),
        forces_xyz=np.array([[0.0, 0.0, 1000.0]]),
        normals_xyz=np.array([[0.0, 0.0, 1.0]]),
        chords=np.array([0.8]),
        spanwise_widths=np.array([0.5]),
    )
    loads = project_panels_to_beam_nodes(frame, panels)

    d_ring = np.linalg.norm(solve_beam_frame(frame, loads).displacements[frame.tip_nodes, :3], axis=1).max()
    d_skin = np.linalg.norm(solve_beam_shell_model(model, loads).displacements[model.tip_nodes, :3], axis=1).max()
    assert d_skin < d_ring
    # The closed skin is a far better transverse/torsional load path than discrete
    # rings (~8.6x here at n_levels=6; the ratio shrinks as n_levels rises because
    # more ring levels add ring stiffness — ~2.4x at n_levels=12). Guard the gain.
    assert d_ring / d_skin > 5.0


def test_skin_datum_angles_shape_and_range():
    from wing_design.beams.shell_model import skin_datum_angles
    spec = small_wingsail
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    ang = skin_datum_angles(model, datum_dir=(0.0, 0.0, 1.0))
    assert ang.shape == (model.shell_tris.shape[0],)
    assert np.all(np.abs(ang) <= np.pi + 1e-9)


def test_build_model_accepts_arc_fractions():
    spec = small_wingsail
    even = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    af = np.arange(8) / 8.0
    same = build_beam_shell_model(spec, n_beams=8, n_levels=5, arc_fractions=af)
    assert np.allclose(even.nodes, same.nodes)   # even fractions == default


def test_nonuniform_model_solves():
    spec = small_wingsail
    af = stress_weighted_targets(np.array([1, 1, 5, 1, 1, 1, 3, 1.0]), 8)
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, arc_fractions=af)
    assert model.nodes.shape == (8 * 5, 3)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 20.0
    res = solve_beam_shell_model(model, loads)
    assert np.isfinite(res.displacements).all()
    assert np.allclose(res.displacements[model.fixed_nodes], 0.0)


def test_lateral_bracing_adds_expected_ring_elements():
    nb, nl = 8, 6
    base = build_beam_shell_model(medium_wingsail, n_beams=nb, n_levels=nl)
    braced = build_beam_shell_model(medium_wingsail, n_beams=nb, n_levels=nl,
                                    lateral_bracing=True)  # default stations="interior"
    n_rings = nl - 2
    assert braced.brace_elements is not None
    assert braced.brace_elements.shape[0] == n_rings * nb
    assert braced.beam_elements.shape[0] == base.beam_elements.shape[0] + n_rings * nb
    for e in braced.brace_elements:
        a, b = braced.beam_elements[e]
        assert np.isclose(braced.nodes[a, 2], braced.nodes[b, 2])
    assert base.brace_elements is None


def test_braced_segment_mask_interior():
    nb, nl = 8, 6
    m = build_beam_shell_model(medium_wingsail, n_beams=nb, n_levels=nl,
                               lateral_bracing=True)
    mask = braced_segment_mask(m)
    assert mask.shape == (m.n_form_elements,)
    per_beam = mask.reshape(nb, nl - 1)
    assert not per_beam[:, 0].any()
    assert not per_beam[:, -1].any()
    assert per_beam[:, 1:-1].all()


def test_lateral_bracing_explicit_stations_tuple():
    nb, nl = 8, 6
    m = build_beam_shell_model(medium_wingsail, n_beams=nb, n_levels=nl,
                               lateral_bracing=True, brace_stations=(1, 3))
    import numpy as np
    assert np.array_equal(m.brace_stations, np.array([1, 3]))
    assert m.brace_elements.shape[0] == 2 * nb          # two rings
    # every brace element connects two nodes at z-level 1 or 3
    zlev = m.nodes[:nl, 2]
    for e in m.brace_elements:
        a, b = m.beam_elements[e]
        assert np.isclose(m.nodes[a, 2], m.nodes[b, 2])
        assert np.isclose(m.nodes[a, 2], zlev[1]) or np.isclose(m.nodes[a, 2], zlev[3])
