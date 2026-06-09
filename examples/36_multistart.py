"""Validate the multi-start sizer machinery on the medium wingsail (small/cheap settings).

Runs size_beam_shell_laminate_multistart at a deliberately SMALL mesh (n_levels=5,
n_beams=8) with n_starts=4 and low maxiter -- this is a machinery check, NOT a converged
headline design. Prints each start's mass + feasibility and the chosen best (multi-start
is never worse than the default single start). Intended to run in minutes.
"""
from __future__ import annotations

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    size_beam_shell_laminate_multistart,
)
from wing_design.materials.unidir import T700_EPOXY


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 8, 5  # small / cheap -- machinery validation only

    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    model = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    load_arrays = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
                   for ar in envelope if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=1.5, n_skin_bands=4)

    res = size_beam_shell_laminate_multistart(
        model, load_arrays, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
        n_starts=4, seed=0, maxiter=120)

    print(f"n_starts={res.n_starts}  feasible={res.n_feasible}  best_start={res.best_start_index}")
    for i, (m, ok) in enumerate(zip(res.start_masses, res.start_feasible)):
        tag = "default" if i == 0 else f"random#{i}"
        print(f"  start {i} ({tag}): mass {m:.2f} kg  feasible={ok}")
    b = res.best
    print(f"\n  BEST: {b.mass_kg:.2f} kg (start {res.best_start_index}); "
          f"converged={b.converged} twist {b.tip_twist_deg:.2f} "
          f"buck {b.max_beam_buckling_util:.2f}/{b.max_panel_buckling_util:.2f}")
    print("  (multi-start is never worse than the default single start, by construction.)")


if __name__ == "__main__":
    main()
