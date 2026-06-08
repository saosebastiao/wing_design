"""Re-size the medium wingsail CLT design with vs without buckling constraints to quantify their impact.

The E.4 design (25.6 kg) was twist-governed with very thin members (0.7 mm skin,
4-14 mm beams) and stresses far under allowable -- textbook buckling territory.
This re-runs the CLT co-sizing with closed-form beam-Euler + skin-panel buckling
constraints (safety factor 1.5) and compares mass + binding behaviour to the
no-buckling baseline. If buckling binds, the prior mass numbers were optimistic.
"""
from __future__ import annotations

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    size_beam_shell_laminate,
)
from wing_design.materials.unidir import T700_EPOXY


def _report(tag, r, cfg, defl_lim, twist_lim):
    print(f"\n{tag}: converged={r.converged} iters={r.n_iter}")
    print(f"  total mass    = {r.mass_kg:6.2f} kg (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f})")
    print(f"  t_skin {r.t_skin*1e3:.2f} mm | beam radii {r.radii.min()*1e3:.1f}-{r.radii.max()*1e3:.1f} mm "
          f"| layup 0/{r.f0:.2f} ±45/{r.f45:.2f} 90/{r.f90:.2f}")
    print(f"  max beam σ {r.max_beam_vm_Pa/1e6:.0f} MPa | max skin σ {r.max_skin_vm_Pa/1e6:.0f} MPa")
    print(f"  tip defl {r.tip_defl_m*1e3:.1f}/{defl_lim*1e3:.0f} mm | tip twist {r.tip_twist_deg:.2f}/{twist_lim:.0f} deg")
    print(f"  buckling util: beam {r.max_beam_buckling_util:.2f} | panel {r.max_panel_buckling_util:.2f} "
          f"(constraint <= 1.0 when on)")


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
    defl_lim, twist_lim = 0.02 * spec.span, 5.0

    print("Re-sizing the CLT design with vs without buckling (this takes a few minutes)...")
    nob = size_beam_shell_laminate(
        model, load_arrays,
        LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim),
        ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=150)
    _report("WITHOUT buckling (E.4 baseline)", nob, None, defl_lim, twist_lim)

    buck = size_beam_shell_laminate(
        model, load_arrays,
        LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim,
                             buckling_safety_factor=1.5),
        ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=200)
    _report("WITH buckling (SF=1.5)", buck, None, defl_lim, twist_lim)

    binds = buck.max_beam_buckling_util > 0.95 or buck.max_panel_buckling_util > 0.95
    print(f"\n  -> buckling {'BINDS' if binds else 'is slack'}; "
          f"mass {nob.mass_kg:.1f} -> {buck.mass_kg:.1f} kg "
          f"({100*(buck.mass_kg/nob.mass_kg-1):+.0f}%).")


if __name__ == "__main__":
    main()
