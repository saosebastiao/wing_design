import numpy as np
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
