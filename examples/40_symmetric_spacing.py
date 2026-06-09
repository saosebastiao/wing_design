"""Mirror-symmetric stress-weighted beam spacing vs even spacing (medium wingsail).

Builds an even-spaced model, solves the load envelope to get per-perimeter-segment skin
stress, symmetrizes it across the chord (max-of-mirror) so the placement stays groupable,
re-places the beams at the resulting stress-weighted arc positions, and sizes both layouts
under identical constraints (span datum + buckling + 4 bands, analytic Jacobian). Prints
the stress-concentration ratio and the mass comparison. FEA-only.
"""
from __future__ import annotations

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig, beam_radius_groups, build_beam_frame, build_beam_shell_model,
    chord_symmetrize_weights, cross_section_stress_weights, project_panels_to_beam_nodes,
    size_beam_shell_laminate, stress_weighted_targets,
)
from wing_design.materials.unidir import T700_EPOXY


def main() -> None:
    P = medium_scenario(); spec = P.geometry
    n_beams, n_levels = 16, 8
    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)

    def model(af=None):
        return build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
            material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
            skin_thickness=P.skin_sizing.t_baseline_m, arc_fractions=af)

    airplane = build_airplane(spec)
    env = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                         spanwise_resolution=P.aero.spanwise_resolution)
    even = model()
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in env if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    w = cross_section_stress_weights(even, loads)
    w_sym = chord_symmetrize_weights(w)
    af = stress_weighted_targets(w_sym, n_beams)
    sym = model(af)
    print(f"stress concentration (max/mean segment): {w.max()/w.mean():.2f}; "
          f"symmetric-spaced groups={beam_radius_groups(sym)[1]} (even={beam_radius_groups(even)[1]})")

    defl_lim, twist_lim = 0.02 * spec.span, 5.0
    cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim,
        tip_twist_max_deg=twist_lim, ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=1.5,
        n_skin_bands=4, use_analytic_jacobian=True)

    print("sizing even spacing...")
    re = size_beam_shell_laminate(even, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=300)
    print("sizing symmetric stress-weighted spacing...")
    rs = size_beam_shell_laminate(sym, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=300)

    for tag, r in (("even", re), ("symmetric-weighted", rs)):
        print(f"  [{tag}] mass={r.mass_kg:.2f} kg (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f}) "
              f"conv={r.converged} twist={r.tip_twist_deg:.2f} buck={r.max_beam_buckling_util:.2f}/{r.max_panel_buckling_util:.2f}")
    dm = rs.mass_kg - re.mass_kg
    print(f"\n  Δmass {dm:+.2f} kg ({100*dm/re.mass_kg:+.1f}%)")


if __name__ == "__main__":
    main()
