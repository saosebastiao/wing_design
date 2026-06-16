"""Re-verify the 3g SLAM SURVIVAL design's buckling eigenvalue at a CONVERGED
spanwise mesh (n_levels >= 24).

Background: runs/slam_survival_3g_20mm.npz (1575.3 kg, BRACED, rings 20 mm) was
eigen-verified at lambda 3.77 -- but at n_levels=8. runs/eigen_mesh_behavior.py
established that the n=8 eigen is UNCONSERVATIVELY overstiff: for the operational
(unbraced, no-slam) headline lambda1 fell from 2.76 (n=8) to a converged cluster
floor ~1.5-1.6 at n>=24. The survival lambda 3.77 is likely similarly inflated.

This script confirms whether the survival design still clears the GATE=1.5 at a
converged mesh. ANALYSIS ONLY, eigen-only (no sizing), no source-file edits.

Method (reuses existing machinery):
  - Resampling helpers from runs/mesh_converge_diag.py: Carrier, map_hollow_vector,
    resample_tube_segments + wing_design.beams.resample_segment_radii. Skin bands
    are single-band => mesh-independent (reused unchanged). brace_radius is the
    single sized scalar (20 mm) -> reused unchanged on every mesh.
  - Braced eigen adapted from runs/slam_braced.py::eigen_worst_braced: appends
    BeamSection.circular(r.brace_radius) for every brace element so the sections
    list matches model.beam_elements. Here we report k=5 lowest modes for the WORST
    (aero x accel) combo (mirroring runs/eigen_mesh_behavior.py::eigen_modes_worst).
  - SLAM accels: 4-case envelope (slam_g=3, g=9.81):
        (0,0,-g), (0, g*sin30, -g*cos30), (0, +3g, -g), (0, -3g, -g)
  - For each n_levels in {8,12,24,28} (n_beams=16) build the BRACED finer model
    (lateral_bracing=True; finer mesh => more interior ring stations, all at the
    sized brace_radius=20 mm), resample the survival design onto it, project the
    slam aero loads, and compute the worst k=5 buckling lambda under the 4-case
    slam envelope.

Run: PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python runs/survival_mesh_eigen.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

RUNS = Path(__file__).parent
sys.path.insert(0, str(RUNS))

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import build_beam_shell_model, resample_segment_radii
from wing_design.beams.body_loads import body_load_vector
from wing_design.beams.fea_model import project_panels_to_skin
from wing_design.beams.shell_model import skin_datum_angles
from wing_design.beams.shell_sizing import skin_band_map
from wing_design.materials.unidir import (
    PVC_H80, T700_EPOXY, laminate_stiffness_offset, sandwich_D_factor,
)
from wing_design.structural.beam_shell import (
    solve_beam_shell_laminate, solve_beam_shell_laminate_factored,
)
from wing_design.structural.eigen_buckling import linear_buckling
from wing_design.structural.frame import BeamSection
from wing_design.structural.geometric_stiffness import assemble_geometric_stiffness
from wing_design.structural.shell import recover_membrane_stress_C

# reuse chain_rebuild's DATUM + section builder, and mesh_converge_diag's
# Carrier / hollow + tube resampling helpers (all import via runs/ on sys.path).
import chain_rebuild as cr
import mesh_converge_diag as mc

N_BEAMS = 16
N_FROM = 8                        # the survival design's native spanwise mesh
# Sweep capped at n=48: lambda1 declines monotonically (no recovery) and has
# already crossed the gate by n=36, so n=48 is decisive. (n=56 was run once and
# kept descending in the same trend but cost ~10 min for a confirmatory point;
# omitted to keep the verification eigen-only/modest.)
N_LIST = [8, 12, 24, 28, 32, 36, 40, 48]
K_MODES = 5
GATE = 1.5

G0 = 9.81
TH = np.radians(30.0)
SLAM_G = 3.0
# 4-case SLAM accel envelope (matches runs/slam_braced.py for slam_g=3)
ACCELS = (
    (0.0, 0.0, -G0),
    (0.0, G0 * np.sin(TH), -G0 * np.cos(TH)),
    (0.0, SLAM_G * G0, -G0),
    (0.0, -SLAM_G * G0, -G0),
)


def eigen_modes_worst_braced(model, loads_aero, r, rho, accels):
    """k=5 lowest buckling eigenvalues of the WORST (aero x accel) combo for a
    BRACED model.

    Adapts runs/slam_braced.py::eigen_worst_braced (appends a circular brace
    section for every brace element so sections match beam_elements) but mirrors
    runs/eigen_mesh_behavior.py::eigen_modes_worst: scan combos by lambda1 (k=1),
    then re-solve the worst combo with k=K_MODES and keep its full spectrum.

    Returns (lams5, worst_id) where worst_id = (aero_case_idx, accel_idx).
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

    # form + tube sections from helper, then append brace sections (braced model).
    sections, gusset, gusset_sec = cr.build_sections_from_result(model, r)
    if (getattr(model, "brace_elements", None) is not None
            and getattr(r, "brace_radius", None) is not None):
        n_brace = len(model.brace_elements)
        brace_r = float(np.atleast_1d(r.brace_radius)[0])
        sections += [BeamSection.circular(brace_r)] * n_brace
    assert len(sections) == model.beam_elements.shape[0], (
        f"Section count mismatch: {len(sections)} != {model.beam_elements.shape[0]}"
    )

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
        for j, acc in enumerate(accels):
            lams = solve_combo(lc, acc, k=1)
            l1 = float(lams[0]) if lams.size else np.inf
            if l1 < best[0]:
                best = (l1, (i, j))
    # 2) re-solve the worst combo for the full spectrum
    i, j = best[1]
    lams5 = solve_combo(loads_aero[i], accels[j], k=K_MODES)
    return lams5, best[1]


def main() -> None:
    d = np.load(RUNS / "slam_survival_3g_20mm.npz", allow_pickle=True)
    radii0 = np.asarray(d["radii"], dtype=float)
    t_bands = d["t_bands"]; f0 = d["f0_bands"]; f45 = d["f45_bands"]; f90 = d["f90_bands"]
    r_tube0 = np.asarray(d["r_tube"], dtype=float)
    t_wall0 = np.asarray(d["t_wall"], dtype=float)
    t_hollow0 = np.asarray(d["t_hollow"], dtype=float)
    t_core = d["t_core"]
    brace_radius = float(np.atleast_1d(d["brace_radius"])[0])
    print(f"loaded SURVIVAL design: radii{radii0.shape} t_hollow{t_hollow0.shape} "
          f"r_tube{r_tube0.shape} brace_r={brace_radius*1e3:.3f} mm "
          f"(meta eigen@n=8 = 3.7750, gate={GATE})", flush=True)

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

    # coarse (native) BRACED model -- the resampling source frame
    model_from = build_beam_shell_model(spec, core_tube=True, hollow_beams=True,
                                        n_levels=N_FROM, lateral_bracing=True,
                                        **base_mk)

    rows = []
    for N in N_LIST:
        t0 = time.perf_counter()
        model = build_beam_shell_model(spec, core_tube=True, hollow_beams=True,
                                       n_levels=N, lateral_bracing=True, **base_mk)
        if N == N_FROM:
            radii = radii0; t_hollow = t_hollow0
            r_tube = r_tube0; t_wall = t_wall0
        else:
            radii = resample_segment_radii(radii0, n_beams=N_BEAMS,
                                           n_from=N_FROM, n_to=N)
            t_hollow = mc.map_hollow_vector(t_hollow0, model_from, model, N_FROM, N)
            r_tube = mc.resample_tube_segments(r_tube0, N_FROM, N)
            t_wall = mc.resample_tube_segments(t_wall0, N_FROM, N)

        # bands single-band -> mesh-independent; brace_radius is the single sized
        # scalar reused unchanged on every mesh.
        r = mc.Carrier(radii=radii, t_bands=t_bands, f0_bands=f0, f45_bands=f45,
                       f90_bands=f90, r_tube=r_tube, t_wall=t_wall,
                       t_hollow=t_hollow, t_core=t_core, brace_radius=brace_radius)

        loads = [project_panels_to_skin(model, ar.panels,
                                        safety_factor=ar.case.safety_factor)
                 for ar in active]

        lams5, worst_id = eigen_modes_worst_braced(model, loads, r, P.rho_kgm3, ACCELS)
        wall = time.perf_counter() - t0
        n_brace = len(model.brace_elements) if getattr(model, "brace_elements", None) is not None else 0
        rows.append((N, lams5, worst_id, wall, n_brace))
        ls = "  ".join(f"{x:7.4f}" for x in lams5)
        print(f"[N={N:2d}] worst_combo={worst_id} n_brace={n_brace:3d}  "
              f"lambda1..{K_MODES}: {ls}   wall={wall:6.1f}s", flush=True)

    # ---- table ----
    print("\n=== SURVIVAL (1575.3 kg, BRACED, rings 20mm) k=5 LOWEST BUCKLING "
          "EIGENVALUES vs SPANWISE MESH (worst 4-case SLAM combo) ===")
    hdr = f"{'n_lev':>5} | " + " | ".join(f"lam{i+1:>5}" for i in range(K_MODES))
    print(hdr)
    print("-" * len(hdr))
    for N, lams5, wid, wall, nb in rows:
        cells = " | ".join(f"{x:6.3f}" for x in lams5)
        print(f"{N:>5} | {cells}")

    # ---- verdict ----
    l1 = [float(rw[1][0]) for rw in rows]
    Ns = [rw[0] for rw in rows]
    l1_n8 = l1[0]
    print("\n=== VERDICT ===")
    print(f"lambda1 @ n=8         = {l1_n8:.4f}  (meta recorded 3.7750)")
    print("lambda1 trajectory (with per-refine %% change):")
    for k in range(len(Ns)):
        dlt = "" if k == 0 else f"  ({100*(l1[k]-l1[k-1])/l1[k-1]:+5.1f}%)"
        print(f"  n={Ns[k]:2d}: lambda1={l1[k]:.4f}{dlt}")

    # plateau test: first mesh where lambda1 changes <5% vs the previous mesh
    plateau_n = None
    for k in range(1, len(Ns)):
        if abs(l1[k] - l1[k - 1]) / l1[k - 1] < 0.05:
            plateau_n = Ns[k]
            plateau_v = l1[k]
            break
    last_v = l1[-1]
    if plateau_n is not None:
        print(f"\nlambda1 plateaus (<5%/refine) first at n={plateau_n}: "
              f"converged worst-lambda ~ {plateau_v:.3f}")
        verdict_v = plateau_v
        converged = True
    else:
        print(f"\nlambda1 has NOT plateaued (>5%/refine through n={Ns[-1]}); "
              f"still declining. Last value n={Ns[-1]}: {last_v:.3f} is an "
              f"UPPER bound on the true converged worst-lambda.")
        verdict_v = last_v
        converged = False

    if converged and verdict_v >= GATE:
        print(f"PASS: converged worst-lambda {verdict_v:.3f} >= GATE {GATE}. "
              f"The 1575.3 kg survival rating HOLDS at converged mesh.")
    elif (not converged) and last_v >= GATE:
        print(f"INCONCLUSIVE (still declining) but last (finest) lambda1 "
              f"{last_v:.3f} >= GATE {GATE} -- margin shrinking; finer mesh "
              f"could breach 1.5. Survival rating NOT yet confirmed at "
              f"converged mesh.")
    else:
        print(f"FAIL: worst-lambda {verdict_v:.3f} < GATE {GATE}. The survival "
              f"design needs more buckling capacity (heavier rings or beams) "
              f"-- REAL FINDING.")


if __name__ == "__main__":
    main()
