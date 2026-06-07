import numpy as np

from wing_design.geometry import WingSpec
from wing_design.beams.fea_model import build_beam_frame, project_panels_to_beam_nodes, solve_beam_frame
from wing_design.aero.loads import PanelLoads
from wing_design.beams.shell_model import (
    skin_triangles,
    build_beam_shell_model,
    solve_beam_shell_model,
)


def test_skin_triangulation_counts_and_validity():
    nb, nl = 8, 5
    tris = skin_triangles(nb, nl)
    assert tris.shape == (2 * nb * (nl - 1), 3)
    assert tris.min() >= 0 and tris.max() < nb * nl
    assert len(np.unique(tris)) == nb * nl


def test_model_builds_and_solves():
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5, beam_radius=0.02)
    assert model.beam_elements.shape[0] == 8 * (5 - 1)
    assert model.shell_tris.shape[0] == 2 * 8 * (5 - 1)
    loads = np.zeros((model.nodes.shape[0], 6))
    loads[model.tip_nodes, 2] = 20.0
    res = solve_beam_shell_model(model, loads)
    assert np.isfinite(res.displacements).all()
    assert np.allclose(res.displacements[model.fixed_nodes], 0.0)


def test_skin_stiffer_than_ring_frame():
    spec = WingSpec()
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
    spec = WingSpec()
    model = build_beam_shell_model(spec, n_beams=8, n_levels=5)
    ang = skin_datum_angles(model, datum_dir=(0.0, 0.0, 1.0))
    assert ang.shape == (model.shell_tris.shape[0],)
    assert np.all(np.abs(ang) <= np.pi + 1e-9)
