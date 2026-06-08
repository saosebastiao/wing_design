"""Final medium wingsail design: CLT co-sizing with span ply-angle datum AND buckling constraints.

Earlier increments measured these separately: E.4b found 19.0 kg with a coherent
span-datum layup but WITHOUT buckling; the buckling study added ~+2% on the isotropic
skin. This combines both -- a manufacturable, span-datum CLT layup co-sized under
beam-stress, skin-stress, tip-deflection, tip-twist AND closed-form buckling
constraints -- to produce the real, defensible headline mass for the shell-beam wingsail.
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


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8

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

    defl_lim, twist_lim, sf_buck = 0.02 * spec.span, 5.0, 1.5
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim,
        ply_angle_datum=(0.0, 0.0, 1.0),   # span-datum: 0deg = spanwise (E.4b)
        buckling_safety_factor=sf_buck,    # closed-form Euler + panel buckling
    )
    print("Final co-sizing: span-datum CLT skin + buckling (this takes a few minutes)...")
    r = size_beam_shell_laminate(model, load_arrays, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=200)

    print(f"\n  converged = {r.converged}  iters = {r.n_iter}")
    print(f"  TOTAL MASS    = {r.mass_kg:6.2f} kg  (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f})")
    print(f"  skin thickness = {r.t_skin*1e3:.2f} mm")
    print(f"  layup (span datum) = {r.f0*100:.0f}% spanwise / {r.f45*100:.0f}% ±45 / {r.f90*100:.0f}% chordwise")
    print(f"  beam radii    = {r.radii.min()*1e3:.1f}-{r.radii.max()*1e3:.1f} mm")
    print(f"  max beam σ {r.max_beam_vm_Pa/1e6:.0f} | skin σ {r.max_skin_vm_Pa/1e6:.0f} MPa (allow {P.sigma_allow_Pa/1e6:.0f})")
    print(f"  tip defl {r.tip_defl_m*1e3:.1f}/{defl_lim*1e3:.0f} mm | twist {r.tip_twist_deg:.2f}/{twist_lim:.0f} deg")
    print(f"  buckling util: beam {r.max_beam_buckling_util:.2f} | panel {r.max_panel_buckling_util:.2f} (SF {sf_buck})")

    print("\n  reference points: E.4b datum-only (no buckling) 19.0 kg; isotropic+buckling ~26.1 kg.")


if __name__ == "__main__":
    main()
