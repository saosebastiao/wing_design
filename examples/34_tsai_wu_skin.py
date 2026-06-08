"""Per-ply Tsai-Wu vs laminate-average von-Mises skin failure for the medium wingsail.

Sizes the span-datum CLT design (as in examples/32, uniform skin) twice under identical
constraints: the default von-Mises skin proxy vs a per-ply Tsai-Wu criterion (material
SF 2.0). Prints mass, layup, and the governing skin margin for each. FEA-only.
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
    sf_mat = P.material_iso.sigma_allow_safety_factor

    def cfg(mode):
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim,
            ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=sf_buck,
            skin_failure=mode, tsai_wu_safety_factor=sf_mat)

    print("sizing with von-Mises skin proxy...")
    vm = size_beam_shell_laminate(model, load_arrays, cfg("von_mises"), ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=200)
    print("sizing with per-ply Tsai-Wu...")
    tw = size_beam_shell_laminate(model, load_arrays, cfg("tsai_wu"), ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=300)

    def show(tag, r):
        print(f"\n  [{tag}] converged={r.converged} iters={r.n_iter}")
        print(f"    TOTAL MASS = {r.mass_kg:6.2f} kg  (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f})")
        print(f"    skin t {r.t_skin*1e3:.2f} mm | layup {r.f0*100:.0f}/{r.f45*100:.0f}/{r.f90*100:.0f} span/±45/chord")
        msr = "-" if r.min_skin_strength_ratio is None else f"{r.min_skin_strength_ratio:.2f}"
        print(f"    skin: vM sigma {r.max_skin_vm_Pa/1e6:.0f} MPa (allow {P.sigma_allow_Pa/1e6:.0f}) | Tsai-Wu min R {msr} (SF {sf_mat:.1f})")
        print(f"    tip defl {r.tip_defl_m*1e3:.1f}/{defl_lim*1e3:.0f} mm | twist {r.tip_twist_deg:.2f}/{twist_lim:.0f} deg")
        print(f"    buckling util: beam {r.max_beam_buckling_util:.2f} | panel {r.max_panel_buckling_util:.2f}")

    show("von_mises", vm)
    show("tsai_wu", tw)

    dmass = tw.mass_kg - vm.mass_kg
    print(f"\n  Δmass {dmass:+.2f} kg ({100*dmass/vm.mass_kg:+.1f}%)")
    print("  Verdict: per-ply Tsai-Wu " +
          ("lightens" if dmass < -0.05 else "adds" if dmass > 0.05 else "≈ no change to") +
          " the design; compare which skin mode governs (layup shift, R vs vM margin).")


if __name__ == "__main__":
    main()
