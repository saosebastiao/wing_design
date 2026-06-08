"""Establish a consistent span-axis ply-angle datum for the medium wingsail CLT skin.

E.4 found the optimal layup was not a manufacturable prescription because ply angles
were measured against each skin triangle's arbitrary local frame (~50/50 spanwise/
chordwise). Here the CLT co-sizing is run with `ply_angle_datum=(0,0,1)` (span axis),
so "0deg" means spanwise consistently across the whole skin and the optimized
`(f0, f45, f90)` is a coherent global layup. Compares to the per-triangle-local
(E.4) result under the same loads and limits.
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


def _report(tag, r, defl_lim, twist_lim):
    print(f"\n{tag}: converged={r.converged} iters={r.n_iter}")
    print(f"  total mass = {r.mass_kg:6.2f} kg (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f}), "
          f"t_skin {r.t_skin*1e3:.2f} mm")
    print(f"  layup: 0deg {r.f0:.2f} | ±45 {r.f45:.2f} | 90deg {r.f90:.2f}")
    print(f"  max beam σ {r.max_beam_vm_Pa/1e6:.0f} MPa | max skin σ {r.max_skin_vm_Pa/1e6:.0f} MPa | "
          f"tip defl {r.tip_defl_m*1e3:.1f}/{defl_lim*1e3:.0f} mm | twist {r.tip_twist_deg:.2f}/{twist_lim:.0f} deg")


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

    print("Co-sizing CLT skin: per-triangle-local (E.4) vs span datum (E.4b)...")
    local = size_beam_shell_laminate(
        model, load_arrays,
        LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim),
        ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=150)
    _report("per-triangle-local (E.4, layup NOT a global prescription)", local, defl_lim, twist_lim)

    datum = size_beam_shell_laminate(
        model, load_arrays,
        LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim,
                             ply_angle_datum=(0.0, 0.0, 1.0)),
        ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=150)
    _report("span datum (E.4b, layup IS a global prescription: 0deg=spanwise)", datum, defl_lim, twist_lim)

    print(f"\n  -> with the span datum the optimal layup is manufacturable: "
          f"{datum.f0*100:.0f}% spanwise / {datum.f45*100:.0f}% ±45 / {datum.f90*100:.0f}% chordwise "
          f"at {datum.mass_kg:.1f} kg (vs {local.mass_kg:.1f} kg per-tri-local).")


if __name__ == "__main__":
    main()
