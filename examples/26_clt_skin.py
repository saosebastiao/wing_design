"""CLT anisotropic skin co-sizing for the medium wingsail with ply-fraction as a design variable.

The E.2 result was twist-governed with an isotropic-equivalent skin. Here the skin
is a CLT laminate whose 0/±45/90 ply-area fractions are design variables co-sized
with beam radii and skin thickness, so the optimizer can orient the plies for
whatever constraint balance is binding (axial/bending stiffness, shear/torsion, or
mass). Compares against the E.2 isotropic co-sizing under the same loads and limits.
"""
from __future__ import annotations

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    BeamShellSizingConfig,
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    size_beam_shell,
    size_beam_shell_laminate,
)
from wing_design.materials.unidir import T700_EPOXY


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
    twist_lim = 5.0

    print("E.2 reference: isotropic-equivalent skin co-sizing...")
    iso_cfg = BeamShellSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim)
    iso = size_beam_shell(model, load_arrays, iso_cfg, rho=P.rho_kgm3, maxiter=150)
    print(f"  isotropic: total {iso.mass_kg:6.2f} kg (beams {iso.beam_mass_kg:.2f} + skin {iso.skin_mass_kg:.2f}), "
          f"t_skin {iso.t_skin*1e3:.2f} mm, twist {iso.tip_twist_deg:.2f} deg")

    print("\nE.4: CLT anisotropic skin, co-sizing the layup...")
    clt_cfg = LaminateSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim)
    clt = size_beam_shell_laminate(model, load_arrays, clt_cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=150)
    print(f"  converged={clt.converged}  iters={clt.n_iter}")
    print(f"  CLT: total {clt.mass_kg:6.2f} kg (beams {clt.beam_mass_kg:.2f} + skin {clt.skin_mass_kg:.2f}), "
          f"t_skin {clt.t_skin*1e3:.2f} mm")
    print(f"  layup fractions: 0deg {clt.f0:.2f} | +-45 {clt.f45:.2f} | 90deg {clt.f90:.2f}")
    print(f"  beam radii {clt.radii.min()*1e3:.1f}-{clt.radii.max()*1e3:.1f} mm")
    print(f"  max beam sigma {clt.max_beam_vm_Pa/1e6:.0f} MPa | max skin sigma {clt.max_skin_vm_Pa/1e6:.0f} MPa "
          f"(allow {clt_cfg.sigma_allow_Pa/1e6:.0f})")
    print(f"  tip defl {clt.tip_defl_m*1e3:.1f} mm | tip twist {clt.tip_twist_deg:.2f} deg "
          f"(limits {defl_lim*1e3:.0f} mm / {twist_lim:.0f} deg)")

    layup = f"0deg {clt.f0:.2f} / ±45 {clt.f45:.2f} / 90deg {clt.f90:.2f}"
    if clt.mass_kg < iso.mass_kg:
        print(f"\n  -> CLT-optimised skin ({layup}) cuts mass {iso.mass_kg:.1f} -> {clt.mass_kg:.1f} kg "
              f"({100*(1-clt.mass_kg/iso.mass_kg):.0f}% lighter than the isotropic skin).")
    else:
        print(f"\n  -> CLT mass {clt.mass_kg:.1f} kg vs isotropic {iso.mass_kg:.1f} kg (layup {layup}).")


if __name__ == "__main__":
    main()
