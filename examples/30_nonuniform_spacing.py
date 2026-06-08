"""Stress-weighted vs even beam spacing for the medium wingsail to test whether load-driven clustering reduces mass.

A baseline even-spacing solve gives the skin membrane-stress distribution around the
cross-section; the N beams are then re-placed at equal-cumulative-stress arc positions
(clustering where the skin is most loaded). Both layouts are re-sized under the same
load envelope + constraints (with buckling) and their masses compared -- testing
whether stress-driven layout beats even spacing.
"""
from __future__ import annotations

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    BeamShellSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    cross_section_stress_weights,
    project_panels_to_beam_nodes,
    size_beam_shell,
    stress_weighted_targets,
)


def _report(tag, r, defl_lim, twist_lim):
    print(f"  {tag:<14} mass {r.mass_kg:6.2f} kg (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f}) | "
          f"σbeam {r.max_beam_vm_Pa/1e6:.0f} σskin {r.max_skin_vm_Pa/1e6:.0f} MPa | "
          f"defl {r.tip_defl_m*1e3:.0f}/{defl_lim*1e3:.0f} twist {r.tip_twist_deg:.2f}/{twist_lim:.0f} | "
          f"buck {r.max_beam_buckling_util:.2f}/{r.max_panel_buckling_util:.2f}")


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8

    even = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    load_arrays = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
                   for ar in envelope if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    weights = cross_section_stress_weights(even, load_arrays)
    af = stress_weighted_targets(weights, n_beams)
    nonu = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m, arc_fractions=af)

    print("per-segment skin stress weights [MPa] (0=TE around to TE):")
    print("  " + " ".join(f"{w/1e6:.0f}" for w in weights))
    print(f"  concentration: max/mean = {weights.max()/weights.mean():.1f}x")

    defl_lim, twist_lim = 0.02 * spec.span, 5.0
    cfg = BeamShellSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim,
                                tip_twist_max_deg=twist_lim, buckling_safety_factor=1.5)
    print("\nSizing both layouts (with buckling, ~2 sizings)...")
    r_even = size_beam_shell(even, load_arrays, cfg, rho=P.rho_kgm3, maxiter=150)
    r_nonu = size_beam_shell(nonu, load_arrays, cfg, rho=P.rho_kgm3, maxiter=150)
    _report("even spacing", r_even, defl_lim, twist_lim)
    _report("stress-weighted", r_nonu, defl_lim, twist_lim)

    dm = 100 * (r_nonu.mass_kg / r_even.mass_kg - 1)
    if r_nonu.mass_kg < r_even.mass_kg:
        print(f"\n  -> stress-weighted spacing is {-dm:.0f}% LIGHTER ({r_even.mass_kg:.1f} -> {r_nonu.mass_kg:.1f} kg).")
    else:
        print(f"\n  -> stress-weighted spacing is {dm:+.0f}% (NOT lighter: {r_even.mass_kg:.1f} -> {r_nonu.mass_kg:.1f} kg) "
              f"-- one-shot clustering didn't beat even spacing here; needs iteration or the 2nd beam family (F.2).")


if __name__ == "__main__":
    main()
