import numpy as np
import pytest
from wing_design.scenario import small_scenario
from wing_design.beams.shell_model import build_beam_shell_model, model_with_tip_gusset
from wing_design.beams.tip_coupling import tip_clique_elements


def _m(**kw):
    return build_beam_shell_model(small_scenario().geometry, n_beams=8, n_levels=5, beam_radius=0.02, **kw)


def test_default_no_gusset():
    m = _m()
    assert m.tip_gusset_elements is None and m.tip_gusset_radius is None


def test_build_with_gusset():
    m = _m(tip_gusset_radius=0.05)
    assert m.tip_gusset_radius == 0.05
    exp = tip_clique_elements(m.tip_nodes)
    assert m.tip_gusset_elements.shape == exp.shape
    assert np.array_equal(m.tip_gusset_elements, exp)


def test_model_with_tip_gusset_roundtrip():
    m = _m()
    g = model_with_tip_gusset(m, 0.06)
    assert g.tip_gusset_radius == 0.06 and g.tip_gusset_elements is not None
    assert m.tip_gusset_elements is None  # original unchanged


from wing_design.structural.frame import BeamSection
from wing_design.materials.unidir import T700_EPOXY, laminate_stiffness
from wing_design.structural.beam_shell import solve_beam_shell_laminate
from wing_design.beams.tip_coupling import tip_clique_elements


def _fixed_case(n_beams=8, n_levels=5):
    m = _m(n_beams=n_beams, n_levels=n_levels) if False else build_beam_shell_model(
        small_scenario().geometry, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02)
    secs = [BeamSection.circular(0.01)] * m.beam_elements.shape[0]
    A, D, _ = laminate_stiffness(T700_EPOXY, f0=0.34, f45=0.33, f90=0.33, thickness=0.0015)
    loads = np.zeros((m.nodes.shape[0], 6))
    # asymmetric chordwise tip load -> tip twist
    loads[m.tip_nodes[0], 0] = 500.0
    return m, secs, A, D, loads


def test_gusset_reduces_tip_twist_fixed_design():
    m, secs, A, D, loads = _fixed_case()
    base = solve_beam_shell_laminate(m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D, fixed_nodes=m.fixed_nodes, loads=loads)
    clique = tip_clique_elements(m.tip_nodes)
    gus = solve_beam_shell_laminate(m.nodes, m.beam_elements, secs, m.shell_tris,
        E_beam=m.E_beam, G_beam=m.G_beam, A_skin=A, D_skin=D, fixed_nodes=m.fixed_nodes, loads=loads,
        gusset_elements=clique, gusset_section=BeamSection.circular(0.05))
    base_tw = np.abs(base.displacements[m.tip_nodes, 5]).max()
    gus_tw = np.abs(gus.displacements[m.tip_nodes, 5]).max()
    assert gus_tw < 0.5 * base_tw            # large reduction
    assert base.axial_force.shape == gus.axial_force.shape == (m.beam_elements.shape[0],)  # recovery design-only


from wing_design.beams import (LaminateSizingConfig, build_beam_frame, project_panels_to_beam_nodes,
                               size_beam_shell_laminate, model_with_tip_gusset, beam_radius_groups)
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams.laminate_sizing import laminate_result_is_feasible


def _sizer_case(gusset=False, n_beams=8, n_levels=5):
    P = small_scenario(); spec = P.geometry
    m = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m,
        **({"tip_gusset_radius": 0.05} if gusset else {}))
    fr = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    env = sweep_envelope(build_airplane(spec), P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(fr, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02*spec.span,
        tip_twist_max_deg=5.0, buckling_safety_factor=1.5, n_skin_bands=2)
    return P, m, loads, cfg


@pytest.mark.sizing  # measured 192 s (2026-06-10)
def test_sizer_with_gusset_feasible_not_heavier():
    P, m, loads, cfg = _sizer_case(gusset=True)
    Pn, mn, loadsn, cfgn = _sizer_case(gusset=False)
    rn = size_beam_shell_laminate(mn, loadsn, cfgn, ply=Pn.material, rho=Pn.rho_kgm3, maxiter=60)
    # warm-start from no-gusset solution: the stiff gusset shifts the landscape; cold start from r_max
    # converges to a local minimum, but starting near the no-gusset optimum reaches the correct basin.
    group_of_element, G = beam_radius_groups(m)
    B = cfg.n_skin_bands
    L = 1  # not per_band_layup
    x0 = np.concatenate([
        np.array([float(rn.radii[group_of_element == g].mean()) for g in range(G)]),
        rn.t_bands,
        rn.f0_bands[:L],
        rn.f45_bands[:L],
    ])
    rg = size_beam_shell_laminate(m, loads, cfg, ply=P.material, rho=P.rho_kgm3, maxiter=200, x0=x0)
    assert laminate_result_is_feasible(rg, cfg)
    assert rg.mass_kg <= rn.mass_kg + 2.0     # twist relief shouldn't cost mass


@pytest.mark.sizing  # measured 34 s (2026-06-10)
def test_analytic_jacobian_composes_with_gusset():
    P, m, loads, cfg = _sizer_case(gusset=True)
    cfg_an = LaminateSizingConfig(sigma_allow_Pa=cfg.sigma_allow_Pa, tip_defl_max_m=cfg.tip_defl_max_m,
        tip_twist_max_deg=cfg.tip_twist_max_deg, buckling_safety_factor=1.5, n_skin_bands=2,
        use_analytic_jacobian=True)
    r = size_beam_shell_laminate(m, loads, cfg_an, ply=P.material, rho=P.rho_kgm3, maxiter=200)
    assert laminate_result_is_feasible(r, cfg_an)
