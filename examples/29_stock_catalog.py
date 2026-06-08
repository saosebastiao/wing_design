"""Discretize the medium wingsail beam radii to a stock catalog via CP-SAT, showing the mass-vs-part-count trade.

Sizes the beam-shell structure continuously (with buckling), then snaps the per-element
radii to a stock catalog via a CP-SAT min-mass assignment with a cap of K distinct
sizes (co-linear grouping). Each K's discrete design is re-solved by FEA to confirm
feasibility -- rounding up from the continuous optimum preserves it. Shows the
mass-vs-part-count trade.
"""
from __future__ import annotations

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    BeamShellSizingConfig,
    beam_lengths,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    select_stock_sizes,
    size_beam_shell,
    skin_areas,
)
from wing_design.structural import solve_beam_shell, recover_membrane_stress, membrane_von_mises
from wing_design.structural.frame import BeamSection, von_mises_per_element
from wing_design.structural.buckling import beam_euler_utilization, panel_buckling_utilization


def verify(model, radii, t_skin, load_arrays, sf_buck):
    """Worst-case metrics + the PER-ELEMENT beam buckling util (for the repair loop).

    Returns (vm, sk, d, tw, max_beam_buck, max_panel_buck, beam_buck_per_elem).
    """
    radii = np.asarray(radii, dtype=float)
    sections = [BeamSection.circular(float(r)) for r in radii]
    D11 = model.E_skin * t_skin**3 / (12.0 * (1.0 - model.nu_skin**2))
    Atri = skin_areas(model)
    Lb = beam_lengths(model)
    vm = sk = d = tw = pb = 0.0
    beam_util = np.zeros(radii.shape[0])
    for loads in load_arrays:
        res = solve_beam_shell(model.nodes, model.beam_elements, sections, model.shell_tris,
                               E_beam=model.E_beam, G_beam=model.G_beam,
                               E_skin=model.E_skin, nu_skin=model.nu_skin, t_skin=t_skin,
                               fixed_nodes=model.fixed_nodes, loads=loads)
        vm = max(vm, float(von_mises_per_element(res, sections).max()))
        skin_s = recover_membrane_stress(model.nodes, model.shell_tris, res.displacements,
                                         E=model.E_skin, nu=model.nu_skin)
        sk = max(sk, float(membrane_von_mises(skin_s).max()))
        tip = model.tip_nodes
        d = max(d, float(np.linalg.norm(res.displacements[tip, :3], axis=1).max()))
        tw = max(tw, float(np.degrees(np.abs(res.displacements[tip, 5]).max())))
        beam_util = np.maximum(beam_util, beam_euler_utilization(
            res.axial_force, radii, Lb, E=model.E_beam, K=1.0, safety_factor=sf_buck))
        pb = max(pb, float(panel_buckling_utilization(skin_s, Atri, D11=D11, t=t_skin,
                                                      kc=4.0, safety_factor=sf_buck).max()))
    return vm, sk, d, tw, float(beam_util.max()), pb, beam_util


def select_and_repair(cont_radii, catalog, Lb, rho, K, model, t_skin, load_arrays, sf_buck, max_iter=10):
    """CP-SAT select, then greedily bump beam-buckling-overloaded elements up the catalog.

    Naive round-up is NOT buckling-safe: non-uniform stiffening redistributes compression
    onto the pinned r_min beams. Raise their per-element floor to the next catalog size and
    re-select until beam buckling is feasible (or the catalog is exhausted).
    """
    floors = np.asarray(cont_radii, dtype=float).copy()
    sel = select_stock_sizes(floors, catalog, Lb, rho=rho, max_distinct_sizes=K)
    for it in range(max_iter):
        vm, sk, d, tw, bb, pb, util = verify(model, sel.assigned_radii, t_skin, load_arrays, sf_buck)
        if bb <= 1.001:
            return sel, (vm, sk, d, tw, bb, pb), it
        bumped = False
        for e in np.where(util > 1.001)[0]:
            nxt = catalog[catalog > sel.assigned_radii[e] + 1e-9]
            if nxt.size:
                floors[e] = max(floors[e], float(nxt[0]))
                bumped = True
        if not bumped:
            break  # stuck at catalog max -> needs a bigger catalog
        sel = select_stock_sizes(floors, catalog, Lb, rho=rho, max_distinct_sizes=K)
    vm, sk, d, tw, bb, pb, _ = verify(model, sel.assigned_radii, t_skin, load_arrays, sf_buck)
    return sel, (vm, sk, d, tw, bb, pb), max_iter


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8
    sf_buck = 1.5

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
    defl_lim, twist_lim = 0.02 * spec.span, 5.0

    print("Continuous sizing (with buckling SF=1.5)...")
    cfg = BeamShellSizingConfig(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=defl_lim,
                                tip_twist_max_deg=twist_lim, buckling_safety_factor=sf_buck)
    cont = size_beam_shell(model, load_arrays, cfg, rho=P.rho_kgm3, maxiter=150)
    print(f"  continuous beam mass = {cont.beam_mass_kg:.2f} kg, radii "
          f"{cont.radii.min()*1e3:.1f}-{cont.radii.max()*1e3:.1f} mm, t_skin {cont.t_skin*1e3:.2f} mm")

    catalog = np.arange(4, 22, 2) * 1e-3   # 4..20 mm in 2 mm steps
    Lb = beam_lengths(model)
    allow = P.sigma_allow_Pa / 1e6

    print(f"\n  catalog [mm]: {[int(c*1e3) for c in catalog]}")
    print(f"  {'K':>2} {'#sizes':>7} {'beam kg':>8} {'vs cont':>8} {'rep':>4} {'σbeam':>7} {'σskin':>7} "
          f"{'defl':>7} {'twist':>7} {'buck b/p':>10}  feasible")
    for K in (2, 4, 8):
        sel, (vm, sk, d, tw, bb, pb), nrep = select_and_repair(
            cont.radii, catalog, Lb, P.rho_kgm3, K, model, cont.t_skin, load_arrays, sf_buck)
        feasible = (vm <= P.sigma_allow_Pa and sk <= P.sigma_allow_Pa
                    and d <= defl_lim * 1.001 and tw <= twist_lim * 1.001
                    and bb <= 1.001 and pb <= 1.001)
        print(f"  {K:>2} {len(sel.distinct_sizes):>7} {sel.mass_kg:>8.2f} "
              f"{100*(sel.mass_kg/cont.beam_mass_kg-1):>+7.0f}% {nrep:>4} {vm/1e6:>7.0f} {sk/1e6:>7.0f} "
              f"{d*1e3:>7.1f} {tw:>7.2f} {bb:>4.2f}/{pb:<4.2f} {'YES' if feasible else 'NO'}")
    print(f"  (limits: σ≤{allow:.0f} MPa, defl≤{defl_lim*1e3:.0f} mm, twist≤{twist_lim:.0f} deg, buck util≤1.0)")
    print("\n  -> naive round-up is NOT buckling-safe (non-uniform stiffening redistributes")
    print("     compression onto pinned r_min beams); a greedy bump-and-reverify repair ('rep'")
    print("     iterations) restores feasibility. Fewer distinct sizes => heavier but manufacturable.")


if __name__ == "__main__":
    main()
