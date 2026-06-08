"""Co-size beam radii and skin thickness for the medium wingsail with the skin load-bearing.

Re-sizes the beam-shell structure (skin replaces rings) minimizing total beam+skin
mass under beam-stress, skin-stress, tip-deflection and tip-twist constraints, then
compares against Phase-C's beams-only sizing (deflection-bound) to show where the
mass payoff of the load-bearing skin lands.
"""
from __future__ import annotations

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    BeamShellSizingConfig,
    SizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    size_beam_shell,
    size_beams,
)


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8

    frame = build_beam_frame(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
    )
    model = build_beam_shell_model(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m,
    )

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

    defl_lim = 0.02 * spec.span
    print("Phase-C reference: beams-only sizing (skin absent, rings present)...")
    c_cfg = SizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=5.0)
    c = size_beams(frame, load_arrays, c_cfg, rho=P.rho_kgm3, maxiter=60)
    print(f"  beams-only mass = {c.mass_kg:6.2f} kg (tip defl {c.tip_defl_m*1e3:.0f} mm, "
          f"max sigma {c.max_vm_stress_Pa/1e6:.0f} MPa)")

    print("\nPhase-E.2: co-size beams + skin (skin load-bearing)...")
    e_cfg = BeamShellSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=5.0)
    e = size_beam_shell(model, load_arrays, e_cfg, rho=P.rho_kgm3, maxiter=150)
    print(f"  converged={e.converged}  iters={e.n_iter}")
    print(f"  total mass      = {e.mass_kg:6.2f} kg   (beams {e.beam_mass_kg:.2f} + skin {e.skin_mass_kg:.2f})")
    print(f"  skin thickness  = {e.t_skin*1e3:6.2f} mm   (bounds {e_cfg.t_min*1e3:.1f}-{e_cfg.t_max*1e3:.0f})")
    print(f"  beam radii      = {e.radii.min()*1e3:.1f}-{e.radii.max()*1e3:.1f} mm")
    print(f"  max beam sigma  = {e.max_beam_vm_Pa/1e6:6.0f} MPa   max skin sigma = {e.max_skin_vm_Pa/1e6:.0f} MPa "
          f"(allow {e_cfg.sigma_allow_Pa/1e6:.0f})")
    print(f"  tip deflection  = {e.tip_defl_m*1e3:6.1f} mm   tip twist = {e.tip_twist_deg:.2f} deg "
          f"(limits {defl_lim*1e3:.0f} mm / 5 deg)")

    if e.mass_kg < c.mass_kg:
        print(f"\n  -> load-bearing skin cuts structural mass {c.mass_kg:.1f} -> {e.mass_kg:.1f} kg "
              f"({100*(1-e.mass_kg/c.mass_kg):.0f}% lighter).")
    else:
        print(f"\n  -> co-sized mass {e.mass_kg:.1f} kg vs beams-only {c.mass_kg:.1f} kg.")


if __name__ == "__main__":
    main()
