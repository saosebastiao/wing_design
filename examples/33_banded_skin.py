"""Per-spanwise-band skin thickness on the medium wingsail: does tapering beat a uniform skin?

Sizes the span-datum CLT design (as in examples/32) twice under identical constraints:
uniform skin (n_skin_bands=1) vs 4 spanwise thickness bands. Prints the per-band
thicknesses and a mass comparison. FEA-only (no CAD).
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
    skin_band_map,
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

    def cfg(nb):
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim,
            ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=sf_buck, n_skin_bands=nb)

    # which band is the tip end vs keel end? (level 0 = tip per default_z_levels)
    bm4 = skin_band_map(model, 4)
    tri_z = model.nodes[model.shell_tris].mean(axis=1)[:, 2]
    band_mean_z = [float(tri_z[bm4 == k].mean()) for k in range(4)]

    print("sizing uniform skin (1 band)...")
    uni = size_beam_shell_laminate(model, load_arrays, cfg(1), ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=200)
    # Banded has 4x the thickness DVs and a tighter (twist+buckling) active set, so it
    # needs more SLSQP iterations to converge to a feasible optimum.
    print("sizing banded skin (4 bands)...")
    ban = size_beam_shell_laminate(model, load_arrays, cfg(4), ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=500)

    def show(tag, r):
        print(f"\n  [{tag}] converged={r.converged} iters={r.n_iter}")
        print(f"    TOTAL MASS = {r.mass_kg:6.2f} kg  (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f})")
        if r.t_bands.shape[0] == 1:
            print(f"    skin thickness (uniform) = {r.t_skin*1e3:.2f} mm")
        else:
            bands = "  ".join(f"{t*1e3:.2f}" for t in r.t_bands)
            print(f"    skin t by band [{bands}] mm (mean {r.t_skin*1e3:.2f}); band z-mean {['%.1f'%z for z in band_mean_z]}")
        print(f"    tip defl {r.tip_defl_m*1e3:.1f}/{defl_lim*1e3:.0f} mm | twist {r.tip_twist_deg:.2f}/{twist_lim:.0f} deg")
        print(f"    buckling util: beam {r.max_beam_buckling_util:.2f} | panel {r.max_panel_buckling_util:.2f}")

    show("uniform", uni)
    show("banded", ban)

    dmass = ban.mass_kg - uni.mass_kg
    print(f"\n  Δmass {dmass:+.2f} kg ({100*dmass/uni.mass_kg:+.1f}%)")
    verdict = ("lighter" if dmass < -0.05 else "heavier" if dmass > 0.05 else "≈ equal")
    print(f"  Verdict: banded skin is {verdict}. (band z-mean low = keel/root, high = tip)")


if __name__ == "__main__":
    main()
