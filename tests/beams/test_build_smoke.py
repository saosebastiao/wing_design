from wing_design.geometry import WingSpec
from wing_design.beams.build import build_form_beams, build_skin_wrap, build_assembly


def test_form_beams_build():
    spec = WingSpec()
    beams = build_form_beams(spec, n_beams=8, n_levels=6, beam_radius=0.02)
    assert len(beams) == 8
    for b in beams:
        assert b.volume > 0.0


def test_skin_wrap_builds():
    spec = WingSpec()
    skin = build_skin_wrap(spec, wall=0.003)
    assert skin.volume > 0.0


def test_assembly_spans_tip_to_keelstep():
    spec = WingSpec()
    asm = build_assembly(spec, n_beams=8, n_levels=6, beam_radius=0.02, wall=0.003)
    bb = asm.bounding_box()
    # z must run from the keel-step (negative) up to the tip (= span).
    assert bb.max.Z > spec.span - 0.05
    assert bb.min.Z < -(spec.transition_length + spec.spar_length) + 0.05
