import numpy as np

from wing_design.geometry import small_wingsail
from wing_design.beams.fea_model import build_beam_frame
from wing_design.beams.sizing import element_lengths, n_longitudinal, frame_mass, SizingConfig, size_beams


def test_n_longitudinal():
    spec = small_wingsail
    frame = build_beam_frame(spec, n_beams=8, n_levels=5)
    assert n_longitudinal(frame) == 8 * (5 - 1)


def test_element_lengths_match_geometry():
    spec = small_wingsail
    frame = build_beam_frame(spec, n_beams=8, n_levels=5)
    L = element_lengths(frame)
    assert L.shape[0] == frame.elements.shape[0]
    i, j = frame.elements[0]
    expected = np.linalg.norm(frame.nodes[j] - frame.nodes[i])
    assert abs(L[0] - expected) < 1e-12


def test_frame_mass_uniform_radius():
    spec = small_wingsail
    frame = build_beam_frame(spec, n_beams=8, n_levels=5)
    nl = n_longitudinal(frame)
    L = element_lengths(frame)
    r_long, r_ring, rho = 0.02, 0.01, 1550.0
    radii = np.full(nl, r_long)
    m = frame_mass(frame, radii, ring_radius=r_ring, rho=rho)
    expected = rho * (
        np.pi * r_long**2 * L[:nl].sum() + np.pi * r_ring**2 * L[nl:].sum()
    )
    assert abs(m - expected) / expected < 1e-12


def test_size_beams_reduces_mass_and_respects_stress():
    # Tiny frame + a small synthetic tip load. With generous deflection/twist
    # limits and a high stress allowable, the optimum drives every longitudinal
    # radius to r_min, so sized mass must be well below the r_max starting mass.
    spec = small_wingsail
    frame = build_beam_frame(spec, n_beams=4, n_levels=3)
    nl = n_longitudinal(frame)

    loads = np.zeros((frame.nodes.shape[0], 6))
    loads[frame.tip_nodes, 2] = 10.0  # gentle push on the tip ring

    cfg = SizingConfig(
        sigma_allow_Pa=1.0e8,
        tip_defl_max_m=1.0,         # generous → not binding
        tip_twist_max_deg=90.0,     # generous → not binding
        r_min=0.004,
        r_max=0.03,
        ring_radius=0.008,
    )
    rho = 1550.0
    start_mass = frame_mass(frame, np.full(nl, cfg.r_max), ring_radius=cfg.ring_radius, rho=rho)

    result = size_beams(frame, [loads], cfg, rho=rho, maxiter=30)

    assert result.radii.shape == (nl,)
    assert result.mass_kg < start_mass                       # mass was reduced
    assert result.max_vm_stress_Pa <= cfg.sigma_allow_Pa * 1.05  # stress feasible
    assert np.all(result.radii >= cfg.r_min - 1e-9)          # bounds respected
    assert np.all(result.radii <= cfg.r_max + 1e-9)
