"""V.1 — Shadow prices on the medium headline: which requirement costs the most kg?

Runs the medium 16x8 headline sizing (span-datum CLT + closed-form buckling, analytic
Jacobian — the example 32/39 configuration) and reports the SLSQP KKT multipliers
converted to physical shadow prices: kg saved per unit of relaxation for each limit
(tip twist 5 deg, tip deflection 2% span, sigma_allow) and kg cost per unit of
buckling safety factor. The 5 deg / 2% limits are admitted heuristics — this tells us
which *requirement* to renegotiate first (backlog P#8). Conversion FD-validated in
tests/beams/test_shadow_prices.py (0.5% at h=5%).
"""
from __future__ import annotations

import time

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    size_beam_shell_laminate,
)
from wing_design.beams.laminate_sizing import laminate_result_is_feasible
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
    loads = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in envelope if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    defl_lim, twist_lim, sf_buck = 0.02 * spec.span, 5.0, 1.5
    # 1 band = the example-32 fully-constrained config; 4 bands = the 2264.6 kg
    # headline config (example 39 free tip), where twist binds and the prices differ.
    for n_bands in (1, 4):
        cfg = LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim, tip_twist_max_deg=twist_lim,
            ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=sf_buck,
            n_skin_bands=n_bands, use_analytic_jacobian=True,
        )
        print(f"\n=== medium sizing, n_skin_bands={n_bands} (analytic Jacobian, maxiter=400) ===")
        t0 = time.perf_counter()
        r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=400)
        wall = time.perf_counter() - t0
        feas = laminate_result_is_feasible(r, cfg)
        print(f"\n  converged={r.converged} feasible={feas} iters={r.n_iter} wall={wall:.0f} s")
        print(f"  TOTAL MASS = {r.mass_kg:.1f} kg (beams {r.beam_mass_kg:.1f} + skin {r.skin_mass_kg:.1f})")
        print(f"  twist {r.tip_twist_deg:.2f}/{twist_lim} deg | defl {r.tip_defl_m*1e3:.0f}/{defl_lim*1e3:.0f} mm"
              f" | buck util beam {r.max_beam_buckling_util:.2f} / panel {r.max_panel_buckling_util:.2f}")

        sp = r.shadow_prices
        if sp is None:
            print("\n  shadow prices unavailable (multiplier layout mismatch)")
            continue
        print("\n  SHADOW PRICES (dm*/dparam at the optimum; valid for small changes):")
        print(f"    tip twist limit   ({twist_lim} deg):   {sp['tip_twist_max_deg']:+10.2f} kg/deg"
              f"   -> +1 deg allowance saves {-sp['tip_twist_max_deg']:.1f} kg")
        print(f"    tip defl limit    ({defl_lim:.2f} m): {sp['tip_defl_max_m']:+10.1f} kg/m"
              f"    -> +100 mm allowance saves {-0.1*sp['tip_defl_max_m']:.1f} kg")
        print(f"    sigma_allow       ({P.sigma_allow_Pa/1e6:.0f} MPa): {sp['sigma_allow_Pa']*1e6:+10.3f} kg/MPa")
        print(f"    buckling SF beam  ({sf_buck}):    {sp['buckling_sf_beam']:+10.1f} kg/SF-unit"
              f"  -> SF 1.5->1.4 saves {0.1*sp['buckling_sf_beam']:.1f} kg")
        print(f"    buckling SF panel ({sf_buck}):    {sp['buckling_sf_panel']:+10.1f} kg/SF-unit"
              f"  -> SF 1.5->1.4 saves {0.1*sp['buckling_sf_panel']:.1f} kg")
    print("\n  (limit-type prices <= 0: relaxing sheds kg; SF prices >= 0: raising SF costs kg)")


if __name__ == "__main__":
    main()
