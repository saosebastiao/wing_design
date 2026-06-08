import numpy as np

from wing_design.geometry import small_wingsail
from wing_design.aero.loads import PanelLoads
from wing_design.beams.fea_model import (
    build_beam_frame,
    project_panels_to_beam_nodes,
    solve_beam_frame,
    summarize_frame,
)
from wing_design.structural.projection import R_GEOM_FROM_AERO


def test_frame_topology():
    spec = small_wingsail
    nb, nl = 16, 10
    frame = build_beam_frame(spec, n_beams=nb, n_levels=nl)
    assert frame.nodes.shape == (nb * nl, 3)
    # longitudinal: nb*(nl-1); rings: nl*nb
    assert frame.elements.shape[0] == nb * (nl - 1) + nl * nb
    assert frame.fixed_nodes.shape[0] == nb
    # the clamped ring is the most-negative-z (keel-step) level
    z_fixed = frame.nodes[frame.fixed_nodes, 2]
    assert np.allclose(z_fixed, frame.nodes[:, 2].min())
    # the tip ring is the most-positive-z level
    assert frame.tip_nodes.shape[0] == nb
    z_tip = frame.nodes[frame.tip_nodes, 2]
    assert np.allclose(z_tip, frame.nodes[:, 2].max())
    assert frame.E > 0.0
    assert frame.G > 0.0


def test_projection_conserves_force():
    spec = small_wingsail
    frame = build_beam_frame(spec, n_beams=16, n_levels=10)
    # one fake panel in the aero frame (span along +Y), out on the wing
    panels = PanelLoads(
        centers_xyz=np.array([[0.0, 2.0, 0.1]]),
        areas=np.array([0.5]),
        forces_xyz=np.array([[10.0, 0.0, 500.0]]),
        normals_xyz=np.array([[0.0, 0.0, 1.0]]),
        chords=np.array([0.8]),
        spanwise_widths=np.array([0.3]),
    )
    loads = project_panels_to_beam_nodes(frame, panels, safety_factor=2.0)
    expected = R_GEOM_FROM_AERO @ (panels.forces_xyz[0] * 2.0)
    assert np.allclose(loads[:, :3].sum(axis=0), expected)
    assert np.allclose(loads[:, 3:], 0.0)  # forces only, no applied moments


def test_solve_runs_and_deflects():
    spec = small_wingsail
    frame = build_beam_frame(spec, n_beams=16, n_levels=12)
    loads = np.zeros((frame.nodes.shape[0], 6))
    tip = [b * frame.n_levels + 0 for b in range(frame.n_beams)]  # level 0 = tip
    loads[tip, 0] = 50.0  # push the whole tip ring chordwise
    res = solve_beam_frame(frame, loads)
    assert np.isfinite(res.displacements).all()
    assert res.max_translation_m > 0.0
    # clamped base ring stays put
    assert np.allclose(res.displacements[frame.fixed_nodes], 0.0)


def test_summarize_frame_metrics():
    spec = small_wingsail
    frame = build_beam_frame(spec, n_beams=16, n_levels=12)
    loads = np.zeros((frame.nodes.shape[0], 6))
    tip = [b * frame.n_levels + 0 for b in range(frame.n_beams)]
    loads[tip, 0] = 50.0
    res = solve_beam_frame(frame, loads)
    m = summarize_frame(frame, res, "gust", applied_force_N=800.0)
    assert m.case_name == "gust"
    assert m.applied_force_N == 800.0
    assert m.tip_translation_m > 0.0
    assert m.max_axial_stress_Pa >= 0.0
    assert m.max_bending_stress_Pa >= 0.0
    assert m.max_vm_stress_Pa >= 0.0
    assert np.isfinite(m.tip_twist_deg)
