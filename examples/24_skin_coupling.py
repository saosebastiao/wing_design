"""Quantify the load-bearing skin's stiffness contribution on the medium wingsail.

Builds the Phase-B ring frame and the Phase-E beam-shell model (skin replaces rings)
at the SAME beam radius, runs the LiftingLine load envelope, and compares tip
deflection per case. Tests the Phase-C/D finding that the skin — not thicker beams —
is the mass-efficient stiffness lever.
"""
from __future__ import annotations

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    solve_beam_frame,
    solve_beam_shell_model,
)


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels, r = 16, 12, 0.02

    frame = build_beam_frame(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=r,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
    )
    model = build_beam_shell_model(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=r,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m,
    )
    print(f"beam radius {r*1e3:.0f} mm; skin thickness {P.skin_sizing.t_baseline_m*1e3:.0f} mm")
    print(f"ring frame: {frame.elements.shape[0]} elements | "
          f"beam-shell: {model.beam_elements.shape[0]} beams + {model.shell_tris.shape[0]} skin tris")

    airplane = build_airplane(spec)
    envelope = sweep_envelope(
        airplane, P.load_cases, method="lifting_line",
        spanwise_resolution=P.aero.spanwise_resolution,
    )

    print(f"\n  {'case':<14} {'u_tip(rings)':>13} {'u_tip(skin)':>12} {'stiffer x':>10}")
    print(f"  {'':14} {'mm':>13} {'mm':>12} {'':>10}")
    for ar in envelope:
        if ar.panels is None or abs(ar.factored_normal_force_N) < 1.0:
            continue
        loads = project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
        d_ring = float(np.linalg.norm(solve_beam_frame(frame, loads).displacements[frame.tip_nodes, :3], axis=1).max())
        d_skin = float(np.linalg.norm(solve_beam_shell_model(model, loads).displacements[model.tip_nodes, :3], axis=1).max())
        ratio = d_ring / d_skin if d_skin > 0 else float("inf")
        print(f"  {ar.case.name:<14} {d_ring*1e3:>13.2f} {d_skin*1e3:>12.2f} {ratio:>9.1f}x")


if __name__ == "__main__":
    main()
