"""FEA-in-the-loop sizing of the medium wingsail form-beam frame to minimum mass.

Builds the Phase-B beam frame, projects the LiftingLine load envelope onto its
nodes, then sizes every longitudinal element's radius by minimizing beam mass
subject to stress + tip-deflection + tip-twist constraints (SLSQP, re-solving the
frame FEA each iteration). Prints the sized metrics and a per-beam root->tip radius
profile, then reports the Phase-3 tube-spar baseline mass for context.

NOTE: the tube spar is a single central bending member; this frame is the entire
load-bearing skin structure (16 beams + rings). The two masses are reported side
by side but are NOT directly comparable -- the spar would still need a skin.
"""
from __future__ import annotations

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    SizingConfig,
    build_beam_frame,
    n_longitudinal,
    project_panels_to_beam_nodes,
    size_beams,
)
from wing_design.structural import size_tube_spar


def main() -> None:
    P = medium_scenario()
    spec = P.geometry

    frame = build_beam_frame(
        spec, n_beams=16, n_levels=8, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
    )
    print(f"Frame: {frame.nodes.shape[0]} nodes, {frame.elements.shape[0]} elements, "
          f"{n_longitudinal(frame)} sizing variables")

    airplane = build_airplane(spec)
    envelope = sweep_envelope(
        airplane, P.load_cases, method="lifting_line",
        spanwise_resolution=P.aero.spanwise_resolution,
    )
    load_arrays = [
        project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
        for ar in envelope
        if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0
    ]
    print(f"Sizing against {len(load_arrays)} non-feathered load cases.")

    cfg = SizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span,
        tip_twist_max_deg=5.0,
    )
    print(f"\nConstraints: sigma_allow={cfg.sigma_allow_Pa/1e6:.0f} MPa, "
          f"tip_defl<={cfg.tip_defl_max_m*1e3:.0f} mm, tip_twist<={cfg.tip_twist_max_deg:.1f} deg")
    print(f"Radius bounds: [{cfg.r_min*1e3:.0f}, {cfg.r_max*1e3:.0f}] mm; "
          f"ring radius {cfg.ring_radius*1e3:.0f} mm\n")

    print("Sizing (SLSQP, FEA re-solve per step) -- this takes a minute...")
    result = size_beams(frame, load_arrays, cfg, rho=P.rho_kgm3, maxiter=60)

    # 0.1% slack: an active constraint sits exactly on its limit, so a strict
    # <= trips on floating-point round-off (e.g. deflection = 100.0000001 mm).
    tol = 1.0e-3
    feasible = (
        result.max_vm_stress_Pa <= cfg.sigma_allow_Pa * (1 + tol)
        and result.tip_defl_m <= cfg.tip_defl_max_m * (1 + tol)
        and result.tip_twist_deg <= cfg.tip_twist_max_deg * (1 + tol)
    )
    print(f"\n  SLSQP converged flag = {result.converged}  (iters={result.n_iter})")
    print(f"  feasible (stress+defl+twist) = {feasible}")
    print(f"  beam mass        = {result.mass_kg:8.3f} kg")
    print(f"  max long. sigma_vm = {result.max_vm_stress_Pa/1e6:8.1f} MPa "
          f"(allow {cfg.sigma_allow_Pa/1e6:.0f})")
    print(f"  max ring  sigma_vm = {result.max_ring_vm_stress_Pa/1e6:8.1f} MPa "
          f"(rings fixed at r={cfg.ring_radius*1e3:.0f} mm; not sized)")
    print(f"  tip deflection   = {result.tip_defl_m*1e3:8.1f} mm "
          f"(limit {cfg.tip_defl_max_m*1e3:.0f})")
    print(f"  tip twist        = {result.tip_twist_deg:8.3f} deg  "
          f"(limit {cfg.tip_twist_max_deg:.1f})")
    if result.max_ring_vm_stress_Pa > cfg.sigma_allow_Pa:
        print("  WARNING: ring elements exceed allowable -- increase ring_radius.")

    # Per-beam radius profile in tip->keel segment order. Longitudinal radii are
    # beam-major, level-minor: beam b occupies indices [b*(n_levels-1), (b+1)*(n_levels-1)).
    seg = frame.n_levels - 1
    radii_mm = result.radii.reshape(frame.n_beams, seg) * 1e3
    print("\n  per-beam radii [mm], tip->keel segment order:")
    for b in range(frame.n_beams):
        row = " ".join(f"{v:4.1f}" for v in radii_mm[b])
        print(f"    beam {b:2d}: {row}")

    baseline = size_tube_spar(spec, envelope, material=P.material)
    print(f"\nPhase-3 tube-spar baseline: mass={baseline.mass_total_kg:.3f} kg "
          f"(sigma_max={baseline.max_stress_Pa/1e6:.0f} MPa, governing={baseline.governing_case})")
    print("  NOTE: not directly comparable -- the spar is a single bending member;")
    print("  the form-beam frame is the whole load-bearing skin structure.")


if __name__ == "__main__":
    main()
