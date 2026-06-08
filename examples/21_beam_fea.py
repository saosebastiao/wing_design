"""Beam-frame FEA of the medium wingsail form-beam structure under the LL load envelope.

Builds the cantilevered space-frame (16 longitudinal form beams + transverse
rings), runs the LiftingLine aero envelope, projects each case's panel loads onto
the nearest beam node, solves, and prints per-case tip deflection, twist, and
member stress. Metrics for the *unsized* structure — the baseline Phase C sizes.
"""
from __future__ import annotations

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    build_beam_frame,
    project_panels_to_beam_nodes,
    solve_beam_frame,
    summarize_frame,
)


def main() -> None:
    P = medium_scenario()
    spec = P.geometry

    frame = build_beam_frame(
        spec,
        n_beams=16,
        n_levels=20,
        beam_radius=0.02,
        material=P.material,
        knockdown=P.material_iso.skin_E_knockdown,
        nu=P.nu_iso,
    )
    print(f"Frame: {frame.nodes.shape[0]} nodes, {frame.elements.shape[0]} elements")
    print(f"  E = {frame.E / 1e9:.1f} GPa, G = {frame.G / 1e9:.1f} GPa, beam r = 20 mm")
    print(f"  clamped keel-step ring: {frame.fixed_nodes.shape[0]} nodes")

    airplane = build_airplane(spec)
    envelope = sweep_envelope(
        airplane,
        P.load_cases,
        method="lifting_line",
        spanwise_resolution=P.aero.spanwise_resolution,
    )

    print(f"\n  {'case':<14} {'|F|':>9} {'u_tip':>9} {'twist':>8} {'σ_ax':>8} {'σ_bend':>8} {'σ_vm':>8}")
    print(f"  {'':14} {'N':>9} {'mm':>9} {'deg':>8} {'MPa':>8} {'MPa':>8} {'MPa':>8}")
    for ar in envelope:
        if ar.panels is None or abs(ar.factored_normal_force_N) < 1.0:
            continue
        loads = project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
        applied = float(np.linalg.norm(loads[:, :3].sum(axis=0)))
        result = solve_beam_frame(frame, loads)
        m = summarize_frame(frame, result, ar.case.name, applied)
        print(
            f"  {m.case_name:<14} {m.applied_force_N:>9.1f} {m.tip_translation_m * 1e3:>9.2f} "
            f"{m.tip_twist_deg:>8.3f} {m.max_axial_stress_Pa / 1e6:>8.2f} "
            f"{m.max_bending_stress_Pa / 1e6:>8.2f} {m.max_vm_stress_Pa / 1e6:>8.2f}"
        )


if __name__ == "__main__":
    main()
