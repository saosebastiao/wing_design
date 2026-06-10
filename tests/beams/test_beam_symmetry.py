import pytest

pytestmark = pytest.mark.sizing  # full SLSQP sizing runs (measured 15-99 s each)

import numpy as np

from wing_design.scenario import small_scenario
from wing_design.beams.shell_model import build_beam_shell_model
from wing_design.beams.shell_sizing import beam_radius_groups


def _model(n_beams=8, n_levels=5, arc_fractions=None):
    P = small_scenario()
    return build_beam_shell_model(P.geometry, n_beams=n_beams, n_levels=n_levels,
                                  beam_radius=0.02, arc_fractions=arc_fractions)


def test_symmetric_default_groups_reduced():
    m = _model(n_beams=8, n_levels=5)
    g, ng = beam_radius_groups(m)
    n = m.beam_elements.shape[0]
    assert g.shape == (n,)
    # even n_beams: U = n_beams//2 + 1 unique beams
    assert ng == (8 // 2 + 1) * (5 - 1)
    # mirror beams b and 8-b share group ids per level
    seg = 5 - 1
    for b in range(1, 4):
        mb = (8 - b) % 8
        assert np.array_equal(g[b * seg:(b + 1) * seg], g[mb * seg:(mb + 1) * seg])
    # chord-line beams 0 and 4 are NOT shared with any other beam
    g0 = set(g[0:seg]); g4 = set(g[4 * seg:5 * seg])
    others = set(g) - g0 - g4
    assert g0.isdisjoint(others) and g4.isdisjoint(others)


def test_asymmetric_placement_falls_back():
    nb, nl = 8, 5
    # deliberately asymmetric arc placement (not mirror-symmetric)
    af = np.linspace(0.0, 1.0, nb, endpoint=False) ** 1.5
    m = _model(n_beams=nb, n_levels=nl, arc_fractions=af)
    g, ng = beam_radius_groups(m)
    n = m.beam_elements.shape[0]
    assert ng == n
    assert np.array_equal(g, np.arange(n))


from wing_design.scenario import small_scenario as _ss
from wing_design.beams import (
    LaminateSizingConfig, build_beam_frame, build_beam_shell_model as _bm,
    project_panels_to_beam_nodes, size_beam_shell_laminate,
)
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.materials.unidir import T700_EPOXY


def _sized(n_beams=8, n_levels=5, **cfgkw):
    P = _ss()
    spec = P.geometry
    model = _bm(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    airplane = build_airplane(spec)
    env = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0, **cfgkw)
    r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=25)
    return model, r


def test_sizer_radii_are_mirror_symmetric():
    nb, nl = 8, 5
    model, r = _sized(nb, nl, buckling_safety_factor=1.5)
    assert r.radii.shape == (nb * (nl - 1),)
    R = r.radii.reshape(nb, nl - 1)
    for b in range(1, nb // 2):
        assert np.allclose(R[b], R[(nb - b) % nb], rtol=1e-9, atol=1e-12)


def test_sizer_radii_taper_monotonic_keel_to_tip():
    nb, nl = 8, 5
    model, r = _sized(nb, nl, buckling_safety_factor=1.5)
    R = r.radii.reshape(nb, nl - 1)
    # default_z_levels is tip->keel, so segment index increases toward the keel;
    # radius must be NON-DECREASING with segment index (>= tip ... <= keel).
    for b in range(nb):
        assert np.all(np.diff(R[b]) >= -1e-9)
