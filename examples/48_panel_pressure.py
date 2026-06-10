"""V.5 — Distributed panel pressure + strip-bending in the skin failure check.

Medium 16x8, strip-width panels (V.6 direction), three configurations:
  A. nearest-node load projection, membrane-only skin check (the V.3b baseline);
  B. skin-distributed projection (`project_panels_to_skin`: each aero panel a
     third to its nearest triangle's nodes) — same total force, smoother path;
  C. B + lateral-pressure strip-bending in the skin failure metric
     (sigma_b = 0.75 q w^2/t^2 via `panel_pressure_per_tri`, backlog V#4).
Pressure-compression buckling interaction and Tsai-Wu bending remain recorded
caveats (spec 2026-06-10-panel-pressure-design.md).
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
from wing_design.beams.fea_model import panel_pressure_per_tri, project_panels_to_skin
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
    active = [ar for ar in envelope
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    loads_node = [project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
                  for ar in active]
    loads_skin = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
                  for ar in active]
    pressures = [panel_pressure_per_tri(model, ar.panels, safety_factor=ar.case.safety_factor)
                 for ar in active]
    qmax = max(float(q.max()) for q in pressures)
    print(f"{len(active)} load cases; peak panel pressure {qmax/1e3:.1f} kPa")

    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
        tip_twist_max_deg=5.0, ply_angle_datum=(0.0, 0.0, 1.0),
        buckling_safety_factor=1.5, use_analytic_jacobian=True,
        panel_width_mode="strip")

    runs = [
        ("A node-lumped, membrane-only", loads_node, None),
        ("B skin-distributed", loads_skin, None),
        ("C B + strip-bending check", loads_skin, pressures),
    ]
    base_mass = None
    for tag, loads, pq in runs:
        t0 = time.perf_counter()
        r = size_beam_shell_laminate(model, loads, cfg, ply=T700_EPOXY, rho=P.rho_kgm3,
                                     maxiter=400, panel_pressures=pq)
        wall = time.perf_counter() - t0
        feas = laminate_result_is_feasible(r, cfg)
        govern = []
        if r.tip_twist_deg > 4.9: govern.append("twist")
        if r.tip_defl_m > 0.98 * 0.02 * spec.span: govern.append("defl")
        if r.max_beam_buckling_util > 0.98: govern.append("beam-buck")
        if r.max_panel_buckling_util > 0.98: govern.append("panel-buck")
        if r.max_skin_vm_Pa > 0.95 * P.sigma_allow_Pa: govern.append("skin-vm")
        line = (f"[{tag}] mass={r.mass_kg:7.1f} kg conv={r.converged} feas={feas} "
                f"iters={r.n_iter} wall={wall:5.0f}s skin_vm={r.max_skin_vm_Pa/1e6:.0f}MPa "
                f"governing={'+'.join(govern) or 'none'}")
        if base_mass is None:
            base_mass = r.mass_kg
        else:
            line += f"  ({100*(r.mass_kg-base_mass)/base_mass:+.1f}% vs A)"
        print(line, flush=True)


if __name__ == "__main__":
    main()
