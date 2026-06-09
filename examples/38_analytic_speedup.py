"""Analytic adjoint Jacobian vs finite-difference: same optimum, far fewer FEA solves.

Sizes a SMALL beam-shell problem twice under identical constraints (span datum + buckling
+ banded skin) — once with the default finite-difference SLSQP Jacobian, once with the
analytic adjoint Jacobian (`use_analytic_jacobian=True`) — and reports each run's mass,
iterations, feasibility, and wall-clock. The two should reach the same feasible optimum;
the analytic run avoids SLSQP's ~n_DV full FEA solves per iteration. FEA-only.
"""
from __future__ import annotations

import time

from wing_design import small_scenario
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
    P = small_scenario()
    spec = P.geometry
    n_beams, n_levels = 12, 6   # small, but non-trivial DV count

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

    def cfg(analytic):
        return LaminateSizingConfig(
            sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
            ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=1.5, n_skin_bands=4,
            use_analytic_jacobian=analytic)

    for tag, analytic in (("finite-difference", False), ("analytic adjoint", True)):
        t0 = time.perf_counter()
        r = size_beam_shell_laminate(model, loads, cfg(analytic), ply=T700_EPOXY,
                                     rho=P.rho_kgm3, maxiter=200)
        dt = time.perf_counter() - t0
        print(f"[{tag}] mass={r.mass_kg:.3f} kg  iters={r.n_iter}  converged={r.converged}  "
              f"twist={r.tip_twist_deg:.2f}  buck={r.max_beam_buckling_util:.2f}/{r.max_panel_buckling_util:.2f}  "
              f"wall={dt:.1f}s")


if __name__ == "__main__":
    main()
