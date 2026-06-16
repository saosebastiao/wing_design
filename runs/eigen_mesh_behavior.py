"""Diagnose WHY the headline's linear-buckling lambda is NON-MONOTONIC in mesh.

Background: the operational headline runs/multistart_v2_best.npz (1021.6 kg,
unbraced, no-slam) has post-hoc worst buckling lambda that came out NON-MONOTONIC
across spanwise mesh refinement (mesh_converge_diag-style resampling of the FIXED
design onto finer models):

    n_levels=8  ->  lambda1 = 2.76
    n_levels=10 ->  lambda1 = 1.52
    n_levels=12 ->  lambda1 = 1.24
    n_levels=16 ->  lambda1 = 1.39      (no plateau)

This blocks a mesh-converged re-baseline. Candidate explanations:
  (a) genuine mesh-induced MODE-SWITCHING: the k=1 lowest mode at one mesh is a
      different physical mode than at another, and they cross. A coarse mesh that
      can't resolve a short-wavelength mode simply MISSES the true lowest mode.
  (b) under-resolution that PLATEAUS at finer mesh.
  (c) noise from the design->finer-mesh RESAMPLING.

To separate these we compute the k=5 LOWEST eigenvalues (not just k=1) for the
WORST (aero x accel) combo, at n_levels in {8,10,12,14,16,20,24}. If the n=8
lambda1 (~2.76) has a lambda2/lambda3 sitting near the n=12 lambda1 (~1.24), the
coarse mesh was MISSING that lower mode -> mode-switching/under-resolution, not
noise. If lambda1..2 settle by n=20/24, we have the converged worst-lambda and the
verification mesh for the re-baseline.

ANALYSIS ONLY. No sizer changes, no source-file edits. Reuses mesh_converge_diag's
Carrier / map_hollow_vector / resample_tube_segments and its model-build +
load-projection pattern; reuses chain_rebuild's ACCELS / build_sections_from_result
and replicates eigen_worst's solve + geometric-stiffness setup, but calls
linear_buckling(..., k=5) and keeps all 5 lambdas of the worst combo.

Run: PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python runs/eigen_mesh_behavior.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import build_beam_shell_model, resample_segment_radii
from wing_design.beams.body_loads import body_load_vector
from wing_design.beams.fea_model import project_panels_to_skin
from wing_design.beams.shell_model import skin_datum_angles
from wing_design.beams.shell_sizing import beam_radius_groups, skin_band_map
from wing_design.materials.unidir import (
    PVC_H80, T700_EPOXY, laminate_stiffness_offset, sandwich_D_factor,
)
from wing_design.structural.beam_shell import (
    solve_beam_shell_laminate, solve_beam_shell_laminate_factored,
)
from wing_design.structural.eigen_buckling import linear_buckling
from wing_design.structural.geometric_stiffness import assemble_geometric_stiffness
from wing_design.structural.shell import recover_membrane_stress_C

# reuse chain_rebuild's constants + section builder verbatim
import chain_rebuild as cr
# reuse the resampling helpers + Carrier from the existing diagnostic
import mesh_converge_diag as mc

RUNS = Path(__file__).parent
N_BEAMS = 16
N_FROM = 8                      # the headline's native spanwise mesh
N_LIST = [8, 10, 12, 14, 16, 20, 24, 28, 32]
K_MODES = 5


def eigen_modes_worst(model, loads_aero, r, rho):
    """Replicates chain_rebuild.eigen_worst's solve + Kg setup, but:
      1. scans all (aero x accel) combos by lambda1 (k=1) to find the WORST one,
      2. re-solves that worst combo with k=K_MODES and returns its full spectrum.

    Returns (lams5, worst_id) where lams5 is up to K_MODES ascending multipliers
    of the worst combo and worst_id labels which (case, accel) produced it.
    """
    band_of_tri = skin_band_map(model, 1)
    t_tri = r.t_bands[band_of_tri]
    th = skin_datum_angles(model, cr.DATUM)
    mats = [laminate_stiffness_offset(T700_EPOXY, f0=float(r.f0_bands[0]),
                                      f45=float(r.f45_bands[0]), f90=float(r.f90_bands[0]),
                                      thickness=float(t_tri[e]),
                                      offset_deg=float(np.degrees(th[e])))
            for e in range(t_tri.shape[0])]
    A_arg = np.stack([m[0] for m in mats]); D_arg = np.stack([m[1] for m in mats])
    C_arg = np.stack([m[2] for m in mats])
    extra = None
    if r.t_core is not None:
        c_tri = np.asarray(r.t_core, dtype=float)[band_of_tri]
        D_arg = D_arg * sandwich_D_factor(t_tri, c_tri)[:, None, None]
        extra = PVC_H80.rho * c_tri
    sections, gusset, gusset_sec = cr.build_sections_from_result(model, r)
    areas_e = np.array([s.A for s in sections])
    fac = solve_beam_shell_laminate_factored(
        model.nodes, model.beam_elements, sections, model.shell_tris,
        E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
        fixed_nodes=model.fixed_nodes, loads=loads_aero[0],
        gusset_elements=gusset, gusset_section=gusset_sec)

    def solve_combo(lc, acc, k):
        f = lc + body_load_vector(model, r.radii, t_tri, rho=rho, accel=acc,
                                  areas=areas_e, extra_skin_density=extra)
        res = solve_beam_shell_laminate(
            model.nodes, model.beam_elements, sections, model.shell_tris,
            E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
            fixed_nodes=model.fixed_nodes, loads=f,
            gusset_elements=gusset, gusset_section=gusset_sec)
        s = recover_membrane_stress_C(model.nodes, model.shell_tris,
                                      res.displacements, C=C_arg)
        Kg = assemble_geometric_stiffness(
            model.nodes, beam_elements=model.beam_elements,
            axial_forces=res.axial_force, shell_tris=model.shell_tris,
            t_tri=t_tri, tri_stress_local=np.asarray(s))
        lams, _ = linear_buckling(fac.K_ff, Kg[np.ix_(fac.free, fac.free)], k=k)
        return lams

    # 1) find worst combo by lambda1
    best = (np.inf, None)
    for i, lc in enumerate(loads_aero):
        for j, acc in enumerate(cr.ACCELS):
            lams = solve_combo(lc, acc, k=1)
            l1 = float(lams[0]) if lams.size else np.inf
            if l1 < best[0]:
                best = (l1, (i, j))
    # 2) re-solve the worst combo for the full spectrum
    i, j = best[1]
    lams5 = solve_combo(loads_aero[i], cr.ACCELS[j], k=K_MODES)
    return lams5, best[1]


def main() -> None:
    d = np.load(RUNS / "multistart_v2_best.npz", allow_pickle=True)
    radii0 = d["radii"]; t_bands = d["t_bands"]
    f0 = d["f0_bands"]; f45 = d["f45_bands"]; f90 = d["f90_bands"]
    r_tube0 = d["r_tube"]; t_wall0 = d["t_wall"]
    t_hollow0 = d["t_hollow"]; t_core = d["t_core"]
    print(f"loaded headline: radii{radii0.shape} t_hollow{t_hollow0.shape} "
          f"r_tube{r_tube0.shape}", flush=True)

    P = medium_scenario()
    spec = P.geometry
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    active = [ar for ar in envelope
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]

    base_mk = dict(n_beams=N_BEAMS, beam_radius=0.02, material=P.material,
                   knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
                   skin_thickness=P.skin_sizing.t_baseline_m)

    # coarse (native) model — the resampling source frame
    model_from = build_beam_shell_model(spec, core_tube=True, hollow_beams=True,
                                        n_levels=N_FROM, **base_mk)

    rows = []
    for N in N_LIST:
        t0 = time.perf_counter()
        model = build_beam_shell_model(spec, core_tube=True, hollow_beams=True,
                                       n_levels=N, **base_mk)
        if N == N_FROM:
            radii = np.asarray(radii0, dtype=float)
            t_hollow = np.asarray(t_hollow0, dtype=float)
            r_tube = np.asarray(r_tube0, dtype=float)
            t_wall = np.asarray(t_wall0, dtype=float)
        else:
            radii = resample_segment_radii(radii0, n_beams=N_BEAMS,
                                           n_from=N_FROM, n_to=N)
            t_hollow = mc.map_hollow_vector(t_hollow0, model_from, model, N_FROM, N)
            r_tube = mc.resample_tube_segments(r_tube0, N_FROM, N)
            t_wall = mc.resample_tube_segments(t_wall0, N_FROM, N)

        r = mc.Carrier(radii=radii, t_bands=t_bands, f0_bands=f0, f45_bands=f45,
                       f90_bands=f90, r_tube=r_tube, t_wall=t_wall,
                       t_hollow=t_hollow, t_core=t_core)

        loads = [project_panels_to_skin(model, ar.panels,
                                        safety_factor=ar.case.safety_factor)
                 for ar in active]

        lams5, worst_id = eigen_modes_worst(model, loads, r, P.rho_kgm3)
        wall = time.perf_counter() - t0
        n_free = int(model_free_dofs(model))
        rows.append((N, lams5, worst_id, wall, n_free))
        ls = "  ".join(f"{x:7.4f}" for x in lams5)
        print(f"[N={N:2d}] worst_combo={worst_id} dofs~{n_free:5d}  "
              f"lambda1..{K_MODES}: {ls}   wall={wall:6.1f}s", flush=True)

    # ---- table ----
    print("\n=== k=5 LOWEST BUCKLING EIGENVALUES vs SPANWISE MESH (fixed headline) ===")
    hdr = f"{'n_lev':>5} | " + " | ".join(f"lam{i+1:>5}" for i in range(K_MODES))
    hdr += " | " + " | ".join(f"d{i+1}{i+2}".rjust(7) for i in range(K_MODES - 1))
    print(hdr)
    print("-" * len(hdr))
    for N, lams5, wid, wall, nf in rows:
        cells = " | ".join(f"{x:6.3f}" for x in lams5)
        gaps = " | ".join(f"{lams5[i+1]-lams5[i]:7.3f}" for i in range(K_MODES - 1))
        print(f"{N:>5} | {cells} | {gaps}")

    # ---- diagnosis ----
    print("\n=== DIAGNOSIS ===")
    l1 = [r[1][0] for r in rows]
    Ns = [r[0] for r in rows]

    # (a) mode-switching: does a coarse-mesh higher mode descend to become the
    #     fine-mesh lambda1? Compare n=8 spectrum to the fine-mesh lambda1.
    sp8 = rows[0][1]
    fine_l1 = l1[-1]
    near8 = [(i + 1, v) for i, v in enumerate(sp8) if v <= fine_l1 * 1.6]
    print(f"n=8 spectrum: {['%.3f' % v for v in sp8]}")
    print(f"  -> fine-mesh (n={Ns[-1]}) lambda1 = {fine_l1:.3f}")
    if len(near8) > 1:
        print(f"  -> n=8 has {len(near8)} modes within 1.6x of the fine lambda1 "
              f"(modes {[i for i, _ in near8]}): a coarse mesh that under-resolves "
              f"the true lowest mode reports a higher lambda1 = MODE-SWITCHING / "
              f"under-resolution, NOT pure resampling noise.")
    else:
        print(f"  -> n=8 lambda1 sits alone; the drop is not explained by a nearby "
              f"n=8 higher mode -> leans toward resampling/physical, inspect smoothness.")

    # (b) plateau: relative change of lambda1 between the last meshes
    print("\nlambda1 trajectory:")
    for k in range(len(Ns)):
        dlt = "" if k == 0 else f"  ({100*(l1[k]-l1[k-1])/l1[k-1]:+5.1f}% vs n={Ns[k-1]})"
        print(f"  n={Ns[k]:2d}: lambda1={l1[k]:.4f}{dlt}")
    # plateau test on the finest pairs
    plateau_n = None
    for k in range(1, len(Ns)):
        if abs(l1[k] - l1[k - 1]) / l1[k - 1] < 0.05:
            plateau_n = Ns[k]
            break
    if plateau_n is not None:
        print(f"  -> lambda1 changes <5% first at n={plateau_n}; "
              f"apparent converged worst-lambda ~ {l1[Ns.index(plateau_n)]:.3f}")
    else:
        print("  -> lambda1 never settles within 5% across the swept meshes "
              "(no plateau even at n=24).")

    # (c) smoothness of lambda1 (sign changes of the first difference)
    diffs = np.diff(l1)
    sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
    print(f"\nlambda1 first-difference sign changes: {sign_changes} "
          f"({'monotone' if sign_changes == 0 else 'non-monotone'}). "
          f"diffs={['%.3f' % x for x in diffs]}")


def model_free_dofs(model):
    """Free-DOF count for context (fixed nodes removed, 6 dof/node)."""
    n_nodes = model.nodes.shape[0]
    n_fixed = len(model.fixed_nodes)
    return 6 * (n_nodes - n_fixed)


if __name__ == "__main__":
    main()
