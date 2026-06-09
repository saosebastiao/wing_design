"""Per-band skin layup vs a single global layup on the medium (22 m) wingsail.

Both runs use 4 spanwise thickness bands, the span ply-angle datum, and buckling
(SF 1.5). Per-band layup gives each band its own (f0,f45,f90); since mass is
fraction-independent, any mass win is INDIRECT -- a band tuned to its local load can
meet the buckling/twist/stress constraints at a thinner thickness. Prints the per-band
layup + thickness taper and the mass delta. FEA-only.
"""
from __future__ import annotations

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

    def cfg(per_band):
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim,
            ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=sf_buck,
            n_skin_bands=4, per_band_layup=per_band)

    print("sizing global layup (banded thickness)...")
    g = size_beam_shell_laminate(model, load_arrays, cfg(False), ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=300)
    print("sizing per-band layup (banded thickness)...")
    pb = size_beam_shell_laminate(model, load_arrays, cfg(True), ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=600)

    def show(tag, r):
        print(f"\n  [{tag}] converged={r.converged} iters={r.n_iter}")
        print(f"    TOTAL MASS = {r.mass_kg:6.2f} kg  (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f})")
        tb = "  ".join(f"{t*1e3:.2f}" for t in r.t_bands)
        print(f"    t by band [{tb}] mm (mean {r.t_skin*1e3:.2f})")
        for b in range(r.f0_bands.shape[0]):
            print(f"    band {b}: layup {r.f0_bands[b]*100:.0f}/{r.f45_bands[b]*100:.0f}/{r.f90_bands[b]*100:.0f} span/±45/chord")
        print(f"    tip defl {r.tip_defl_m*1e3:.1f}/{defl_lim*1e3:.0f} mm | twist {r.tip_twist_deg:.2f}/{twist_lim:.0f} deg")
        print(f"    buckling util: beam {r.max_beam_buckling_util:.2f} | panel {r.max_panel_buckling_util:.2f}")

    show("global layup", g)
    show("per-band layup", pb)

    dmass = pb.mass_kg - g.mass_kg
    print(f"\n  Δmass {dmass:+.2f} kg ({100*dmass/g.mass_kg:+.1f}%)")
    verdict = ("lighter" if dmass < -0.05 else "heavier" if dmass > 0.05 else "≈ equal")
    print(f"  Verdict: per-band layup is {verdict} (indirect: per-band fibre mix -> thinner bands).")


if __name__ == "__main__":
    main()
