"""Tip gusset in the sizing loop: does rigid tip binding lighten the twist-governed design?

Sizes the medium (22 m) wingsail twice under identical constraints (span datum, buckling
SF 1.5, 4 thickness bands): once with a free tip (no gusset), once with a rigid massless
tip gusset (a stiff connector-beam clique among the tip nodes). The gusset-on run is
WARM-STARTED from the gusset-off optimum — because the off design is feasible with slack
once tip twist is relaxed, this guarantees a fair "never worse" comparison and lets the
optimizer trade the freed twist budget for thinner material. Prints mass, tip twist, and
the governing constraints for both. FEA-only.
"""
from __future__ import annotations

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig,
    beam_radius_groups,
    build_beam_frame,
    build_beam_shell_model,
    model_with_tip_gusset,
    project_panels_to_beam_nodes,
    size_beam_shell_laminate,
)
from wing_design.materials.unidir import T700_EPOXY


def _x0_from_result(model, r, n_skin_bands):
    """Reconstruct the sizer design vector x0 = [r_group(G), t_band(B), f0, f45] from a result
    (global layup, per_band_layup=False -> L=1)."""
    group_of_element, G = beam_radius_groups(model)
    seg = G  # r_group has one value per group; take each group's representative radius
    r_group = np.zeros(G)
    seen = set()
    for e, g in enumerate(group_of_element):
        if g not in seen:
            r_group[g] = r.radii[e]
            seen.add(g)
    return np.concatenate([r_group, np.asarray(r.t_bands, float),
                           [float(r.f0_bands[0])], [float(r.f45_bands[0])]])


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels, nb_bands = 16, 8, 4

    frame = build_beam_frame(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso)
    base = build_beam_shell_model(spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m)
    gus = model_with_tip_gusset(base, 0.10)  # rigid connector-beam radius (saturated)
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in envelope if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    defl_lim, twist_lim = 0.02 * spec.span, 5.0
    # Analytic Jacobian: exact gradients converge much better than noisy FD near the
    # active beam-buckling constraint (the gusset frees the twist budget, so the optimizer
    # pushes beams thin and buckling becomes the binding constraint).
    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim,
        ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=1.5, n_skin_bands=nb_bands,
        use_analytic_jacobian=True)

    print("sizing free tip (no gusset)...")
    off = size_beam_shell_laminate(base, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=300)
    print("sizing with rigid tip gusset (warm-started from no-gusset optimum)...")
    x0 = _x0_from_result(base, off, nb_bands)
    on = size_beam_shell_laminate(gus, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=300, x0=x0)

    def show(tag, r):
        print(f"\n  [{tag}] converged={r.converged} iters={r.n_iter}")
        print(f"    TOTAL MASS = {r.mass_kg:6.2f} kg  (beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f})")
        print(f"    tip defl {r.tip_defl_m*1e3:.1f}/{defl_lim*1e3:.0f} mm | twist {r.tip_twist_deg:.2f}/{twist_lim:.0f} deg")
        print(f"    buckling util: beam {r.max_beam_buckling_util:.2f} | panel {r.max_panel_buckling_util:.2f}")
        print(f"    beam radii {r.radii.min()*1e3:.1f}-{r.radii.max()*1e3:.1f} mm | skin t mean {r.t_skin*1e3:.2f} mm")

    show("free tip", off)
    show("tip gusset", on)
    dm = on.mass_kg - off.mass_kg
    print(f"\n  Δmass {dm:+.2f} kg ({100*dm/off.mass_kg:+.1f}%) | twist {off.tip_twist_deg:.2f}° -> {on.tip_twist_deg:.2f}°")
    print(f"  Verdict: tip gusset is {'lighter' if dm < -0.05 else 'heavier' if dm > 0.05 else '≈ equal'}; "
          f"governing: free-tip {'twist' if off.tip_twist_deg > twist_lim-0.1 else 'buckling'} -> "
          f"gusset {'twist' if on.tip_twist_deg > twist_lim-0.1 else 'buckling/stress'}.")


if __name__ == "__main__":
    main()
