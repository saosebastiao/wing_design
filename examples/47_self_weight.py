"""V.4 — Self-weight + inertial/heel load cases on the medium wingsail.

Sizes the medium 16x8 design (strip-width panels — the V.6 direction) three ways:
  A. aero only (the V.3b strip baseline);
  B. aero x {upright gravity, 30 deg heel gravity} — every aero case combined with
     each acceleration, the design's own mass lumped to nodes every evaluate;
  C. B + a 1 g lateral slam superposed on the heeled case (>1 g at the rig is
     routine; static-equivalent, backlog V#5/V#12).
The analytic adjoint carries the lam^T dF/dx design-dependent-load term
(FD-validated in tests/beams/test_body_loads.py). Wing frame: z = span (up),
x = chord, y = thickness/lateral; heel rotates gravity about the chord axis.
"""
from __future__ import annotations

import time

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
from wing_design.beams.laminate_sizing import laminate_result_is_feasible
from wing_design.materials.unidir import T700_EPOXY

G0 = 9.81
HEEL_DEG = 30.0


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

    th = np.radians(HEEL_DEG)
    upright = (0.0, 0.0, -G0)
    heeled = (0.0, G0 * np.sin(th), -G0 * np.cos(th))
    heeled_slam = (0.0, G0 * np.sin(th) + G0, -G0 * np.cos(th))
    cases = {
        "A aero only": (),
        "B + gravity (upright, 30deg heel)": (upright, heeled),
        "C + heel & 1g lateral slam": (upright, heeled, heeled_slam),
    }

    def cfg(accels):
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
            tip_twist_max_deg=5.0, ply_angle_datum=(0.0, 0.0, 1.0),
            buckling_safety_factor=1.5, use_analytic_jacobian=True,
            panel_width_mode="strip", accel_vectors=accels)

    base_mass = None
    for tag, accels in cases.items():
        t0 = time.perf_counter()
        r = size_beam_shell_laminate(model, loads, cfg(accels), ply=T700_EPOXY,
                                     rho=P.rho_kgm3, maxiter=400)
        wall = time.perf_counter() - t0
        feas = laminate_result_is_feasible(r, cfg(accels))
        govern = []
        if r.tip_twist_deg > 4.9: govern.append("twist")
        if r.tip_defl_m > 0.98 * 0.02 * spec.span: govern.append("defl")
        if r.max_beam_buckling_util > 0.98: govern.append("beam-buck")
        if r.max_panel_buckling_util > 0.98: govern.append("panel-buck")
        line = (f"[{tag}] mass={r.mass_kg:7.1f} kg conv={r.converged} feas={feas} "
                f"iters={r.n_iter} wall={wall:5.0f}s governing={'+'.join(govern) or 'none'}")
        if base_mass is None:
            base_mass = r.mass_kg
        else:
            line += f"  ({100*(r.mass_kg-base_mass)/base_mass:+.1f}% vs aero-only)"
        print(line, flush=True)


if __name__ == "__main__":
    main()
