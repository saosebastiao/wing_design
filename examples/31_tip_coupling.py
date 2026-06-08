"""Tip-coupling study on the medium wingsail: does a hard tip gusset redistribute load between beams?

A fixed uniform-radius beam-shell design is solved under the load envelope with NO tip
gusset (tip joined only through the skin) and with an increasingly stiff gusset (a
clique of stiff connector beams tying the tip nodes). Measures, per gusset stiffness:
the worst beam von Mises, the beam-stress spread (max/mean = how evenly load is shared),
and the tip deflection/twist. The question: how much does a hard tip coupling transfer
load between beams, given the load-bearing skin already couples the tip?
"""
from __future__ import annotations

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    solve_beam_shell_tip_coupled,
)
from wing_design.structural.frame import von_mises_per_element


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8

    model = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    load_arrays = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
                   for ar in envelope if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    secs = [model.section] * model.beam_elements.shape[0]

    print("uniform beam r=20mm, skin 3mm; tip joined by skin + optional stiff gusset")
    print(f"  {'gusset':>10} {'max beam σ':>11} {'σ spread':>9} {'u_tip':>8} {'twist':>8}")
    print(f"  {'r [mm]':>10} {'MPa':>11} {'max/mean':>9} {'mm':>8} {'deg':>8}")
    rows = {}
    for g in (None, 0.02, 0.04, 0.08, 0.15):
        vm_max = 0.0; spread = 0.0; d = 0.0; tw = 0.0
        for loads in load_arrays:
            res, n_beam = solve_beam_shell_tip_coupled(model, loads, gusset_radius=g)
            vm = von_mises_per_element(res, secs)        # already sliced to design beams
            vm_max = max(vm_max, float(vm.max()))
            spread = max(spread, float(vm.max() / max(vm.mean(), 1e-9)))
            tip = model.tip_nodes
            d = max(d, float(np.linalg.norm(res.displacements[tip, :3], axis=1).max()))
            tw = max(tw, float(np.degrees(np.abs(res.displacements[tip, 5]).max())))
        label = "none" if g is None else f"{g*1e3:.0f}"
        rows[label] = (vm_max, spread, d, tw)
        print(f"  {label:>10} {vm_max/1e6:>11.2f} {spread:>9.2f} {d*1e3:>8.2f} {tw:>8.3f}")

    n, r = rows["none"], rows["150"]   # skin-only vs near-rigid gusset
    print(f"\n  skin-only -> rigid gusset: peak beam σ {100*(r[0]/n[0]-1):+.0f}%, "
          f"σ-spread {n[1]:.2f}->{r[1]:.2f}, tip defl {100*(r[2]/n[2]-1):+.0f}%, "
          f"tip twist {n[3]:.3f}->{r[3]:.3f} deg ({n[3]/max(r[3],1e-9):.0f}x lower)")
    print("  -> the hard tip joint barely redistributes BEAM stress (skin already shares")
    print("     spanwise load); its real effect is TORSIONAL — it near-eliminates tip twist")
    print("     and modestly stiffens the tip, and it saturates at low gusset stiffness.")


if __name__ == "__main__":
    main()
