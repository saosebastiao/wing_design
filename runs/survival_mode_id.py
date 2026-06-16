"""Identify the GOVERNING converged-mesh slam buckling MODE of the 3 g survival
design (runs/slam_survival_3g_20mm.npz, braced, rings 20 mm).

Context: the worst SLAM buckling eigenvalue of this design declines monotonically
with spanwise refinement (3.77 @ n8 -> 1.25 @ n48, no plateau, below the 1.5
gate; see runs/survival_mesh_eigen.py). Before we can fix/resize it we must know
WHAT is buckling. This is ANALYSIS ONLY, eigen-only, no source edits.

Method (reuses runs/survival_mesh_eigen.py machinery verbatim):
  - Build the BRACED finer-mesh model, resample the n=8 design onto it, append
    ring sections at the sized brace_radius, find the worst (aero x slam-accel)
    combo, and extract the k=1 buckling EIGENVECTOR of that worst combo.
  - Map the reduced free-DOF eigenvector back to a full (n_nodes, 6) field via
    fac.free (DOF i of node n is global index 6*n+i; translations 0:3, rot 3:6).
  - Partition nodes into: TUBE nodes (pivot-axis nodes appended after the
    n_beams*n_levels form grid) vs FORM-beam grid nodes. Quantify translation
    magnitude per region, per spanwise z-band, and per tube segment, and decide
    LOCAL (concentrated -- e.g. the thin r/t~=79 tube transition, seg 4) vs
    GLOBAL (spread sway).
  - Spanwise convergence of the same diagnostic at a couple of meshes, plus one
    or two finer points (n=56, 64) to confirm decline/plateau (eigen-only).

Run: PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python runs/survival_mode_id.py
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

import chain_rebuild as cr
import mesh_converge_diag as mc

N_BEAMS = 16
N_FROM = 8
# Mesh points for the mode-ID diagnostic + finer confirmation of the decline.
# MEASURED (this run): lambda1 2.421 (n24) -> 1.437 (n40) -> 1.117 (n56),
# monotonic ~-22%/refine, NO plateau; the mode shape is IDENTICAL at all three
# (100% form-beam, LOCAL, mid-span peak z~7.7-7.9 m). n=64 was queued but skipped
# once n=56 confirmed the decline is unbroken and the mode unchanged (dense eigh
# cost ~8 min @ n56, ~15+ min @ n64 for a redundant confirmatory point -- the
# answer is decisive on these three). Wall: n24 32 s, n40 175 s, n56 497 s.
N_LIST = [24, 40, 56]
GATE = 1.5

G0 = 9.81
TH = np.radians(30.0)
SLAM_G = 3.0
ACCELS = (
    (0.0, 0.0, -G0),
    (0.0, G0 * np.sin(TH), -G0 * np.cos(TH)),
    (0.0, SLAM_G * G0, -G0),
    (0.0, -SLAM_G * G0, -G0),
)


def _build_args(model, r):
    """Skin A/D/C laminate args + body-load extras + sections, exactly as the
    verified eigen path builds them (mirrors eigen_modes_worst_braced)."""
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
    if (getattr(model, "brace_elements", None) is not None
            and getattr(r, "brace_radius", None) is not None):
        n_brace = len(model.brace_elements)
        brace_r = float(np.atleast_1d(r.brace_radius)[0])
        sections += [BeamSection.circular(brace_r)] * n_brace
    assert len(sections) == model.beam_elements.shape[0]
    return t_tri, A_arg, D_arg, C_arg, extra, sections, gusset, gusset_sec


def worst_mode_eigvec(model, loads_aero, r, rho, accels):
    """Return (lam1, eigvec_full, worst_id) for the worst (aero x accel) combo.

    eigvec_full is the k=1 buckling mode mapped back to the FULL global DOF
    vector (length 6*n_nodes) via fac.free; fixed DOFs are zero.
    """
    t_tri, A_arg, D_arg, C_arg, extra, sections, gusset, gusset_sec = _build_args(model, r)
    areas_e = np.array([s.A for s in sections])
    fac = solve_beam_shell_laminate_factored(
        model.nodes, model.beam_elements, sections, model.shell_tris,
        E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
        fixed_nodes=model.fixed_nodes, loads=loads_aero[0],
        gusset_elements=gusset, gusset_section=gusset_sec)
    ndof = 6 * model.nodes.shape[0]

    def solve_combo(lc, acc):
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
        return linear_buckling(fac.K_ff, Kg[np.ix_(fac.free, fac.free)], k=1)

    best = (np.inf, None, None)
    for i, lc in enumerate(loads_aero):
        for j, acc in enumerate(accels):
            lams, vecs = solve_combo(lc, acc)
            if lams.size and float(lams[0]) < best[0]:
                full = np.zeros(ndof)
                full[fac.free] = np.asarray(vecs[:, 0]).ravel().real
                best = (float(lams[0]), full, (i, j))
    return best


def characterize(model, lam, eigvec_full, r):
    """Print + return the regional modal-translation breakdown for one mode."""
    nb, nl = model.n_beams, model.n_levels
    n_form = nb * nl
    n_nodes = model.nodes.shape[0]
    field = eigvec_full.reshape(n_nodes, 6)
    trans = field[:, 0:3]
    tmag = np.linalg.norm(trans, axis=1)        # per-node translation magnitude

    # normalize so the peak nodal translation is 1 (shape only; lam carries scale)
    peak = float(tmag.max())
    if peak > 0:
        tmag = tmag / peak

    # --- region split: form-beam grid nodes [0:n_form) vs tube nodes [n_form:) ---
    form_t = tmag[:n_form]
    tube_t = tmag[n_form:n_nodes]
    # energy proxy = sum of squared translation magnitude (modal "translation energy")
    e_form = float(np.sum(form_t ** 2))
    e_tube = float(np.sum(tube_t ** 2))
    e_tot = e_form + e_tube
    frac_form = e_form / e_tot if e_tot else 0.0
    frac_tube = e_tube / e_tot if e_tot else 0.0

    # peak node overall
    pk = int(np.argmax(tmag))
    pk_is_tube = pk >= n_form
    pk_xyz = model.nodes[pk]

    # --- spanwise band: which z-level carries the largest summed modal trans ---
    z = model.nodes[:, 2]
    # per form-level summed energy
    lvl_e = np.zeros(nl)
    for b in range(nb):
        for k in range(nl):
            lvl_e[k] += form_t[b * nl + k] ** 2
    # per tube-segment (node n_form+k) energy
    tube_node_e = tube_t ** 2  # length nl (one tube node per level)

    zmin, zmax = float(z.min()), float(z.max())
    span = zmax - zmin
    # root/mid/tip thirds by z over [0, zmax] (in-wing); below-root is part of root
    def band_of(zk):
        f = (zk - 0.0) / zmax if zmax > 0 else 0.0
        if f < 1.0 / 3:
            return 0
        if f < 2.0 / 3:
            return 1
        return 2
    band_e = np.zeros(3)
    z_form_levels = np.array([model.nodes[k][2] for k in range(nl)])  # beam0 level z
    for k in range(nl):
        band_e[band_of(z_form_levels[k])] += lvl_e[k]
    band_e_t = np.zeros(3)
    for k in range(nl):
        band_e_t[band_of(z_form_levels[k])] += tube_node_e[k] if k < tube_node_e.size else 0.0

    pk_lvl = int(np.argmax(lvl_e))
    pk_tube_node = int(np.argmax(tube_node_e)) if tube_node_e.size else -1

    # localization: participation ratio of nodal translation^2 (a 'how many nodes'
    # measure). PR = (sum w)^2 / sum(w^2); ~N for fully spread, ~1 for single node.
    w = tmag ** 2
    pr = float((w.sum() ** 2) / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0

    print(f"\n--- MODE @ n_levels={nl}: lambda1 = {lam:.4f} ---")
    print(f"  nodes: form={n_form}  tube={n_nodes-n_form}  total={n_nodes}")
    print(f"  TRANSLATION ENERGY split: form-beams {100*frac_form:5.1f}%   "
          f"tube {100*frac_tube:5.1f}%")
    print(f"  peak node: id={pk} {'TUBE' if pk_is_tube else 'FORM'} "
          f"xyz=({pk_xyz[0]:.2f},{pk_xyz[1]:.2f},{pk_xyz[2]:.2f}) z={pk_xyz[2]:.2f}m")
    print(f"  participation ratio (eff. #nodes carrying the mode) = {pr:.1f} "
          f"of {n_nodes}  ({'LOCAL' if pr < 0.06*n_nodes else 'spread/GLOBAL'})")
    print(f"  spanwise band energy (root/mid/tip, z in [0,{zmax:.1f}] m):")
    print(f"      FORM: root {100*band_e[0]/max(band_e.sum(),1e-30):5.1f}%  "
          f"mid {100*band_e[1]/max(band_e.sum(),1e-30):5.1f}%  "
          f"tip {100*band_e[2]/max(band_e.sum(),1e-30):5.1f}%")
    if band_e_t.sum() > 0:
        print(f"      TUBE: root {100*band_e_t[0]/band_e_t.sum():5.1f}%  "
              f"mid {100*band_e_t[1]/band_e_t.sum():5.1f}%  "
              f"tip {100*band_e_t[2]/band_e_t.sum():5.1f}%")
    print(f"  peak FORM z-level k={pk_lvl} (z={z_form_levels[pk_lvl]:.2f} m)")
    if pk_tube_node >= 0:
        zt = model.nodes[n_form + pk_tube_node][2]
        print(f"  peak TUBE node k={pk_tube_node} (z={zt:.2f} m)")

    # --- tube-segment locator: map peak tube-node z onto the n=8 design's tube
    #     segments to see if it sits at the thin r/t~79 transition (seg 4). ---
    rt = np.asarray(r.r_tube, float); tw = np.asarray(r.t_wall, float)
    print(f"  n=8 tube r/t per segment: {np.round(rt/tw,1)}  "
          f"(seg4 r/t={rt[4]/tw[4]:.0f} is the thin transition)")
    return dict(lam=lam, frac_form=frac_form, frac_tube=frac_tube, pr=pr,
                n_nodes=n_nodes, peak_z=float(pk_xyz[2]), peak_is_tube=pk_is_tube,
                band_form=band_e / max(band_e.sum(), 1e-30))


def main() -> None:
    d = np.load(RUNS / "slam_survival_3g_20mm.npz", allow_pickle=True)
    radii0 = np.asarray(d["radii"], dtype=float)
    t_bands = d["t_bands"]; f0 = d["f0_bands"]; f45 = d["f45_bands"]; f90 = d["f90_bands"]
    r_tube0 = np.asarray(d["r_tube"], dtype=float)
    t_wall0 = np.asarray(d["t_wall"], dtype=float)
    t_hollow0 = np.asarray(d["t_hollow"], dtype=float)
    t_core = d["t_core"]
    brace_radius = float(np.atleast_1d(d["brace_radius"])[0])
    print(f"loaded SURVIVAL design: braced rings {brace_radius*1e3:.1f} mm; "
          f"tube r/t = {np.round(r_tube0/t_wall0,1)} (7 seg, seg4 is thin r/t~79)",
          flush=True)

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
    model_from = build_beam_shell_model(spec, core_tube=True, hollow_beams=True,
                                        n_levels=N_FROM, lateral_bracing=True,
                                        **base_mk)

    summary = []
    for N in N_LIST:
        t0 = time.perf_counter()
        model = build_beam_shell_model(spec, core_tube=True, hollow_beams=True,
                                       n_levels=N, lateral_bracing=True, **base_mk)
        radii = resample_segment_radii(radii0, n_beams=N_BEAMS, n_from=N_FROM, n_to=N)
        t_hollow = mc.map_hollow_vector(t_hollow0, model_from, model, N_FROM, N)
        r_tube = mc.resample_tube_segments(r_tube0, N_FROM, N)
        t_wall = mc.resample_tube_segments(t_wall0, N_FROM, N)
        r = mc.Carrier(radii=radii, t_bands=t_bands, f0_bands=f0, f45_bands=f45,
                       f90_bands=f90, r_tube=r_tube, t_wall=t_wall,
                       t_hollow=t_hollow, t_core=t_core, brace_radius=brace_radius)
        loads = [project_panels_to_skin(model, ar.panels,
                                        safety_factor=ar.case.safety_factor)
                 for ar in active]
        lam, vec, wid = worst_mode_eigvec(model, loads, r, P.rho_kgm3, ACCELS)
        wall = time.perf_counter() - t0
        # use the n=8 r/t in the locator (resampled r/t is ~equal anyway)
        r_for_char = mc.Carrier(radii=radii, t_bands=t_bands, f0_bands=f0,
                                f45_bands=f45, f90_bands=f90, r_tube=r_tube0,
                                t_wall=t_wall0, t_hollow=t_hollow, t_core=t_core,
                                brace_radius=brace_radius)
        info = characterize(model, lam, vec, r_for_char)
        info["N"] = N; info["wall"] = wall; info["worst_id"] = wid
        summary.append(info)
        print(f"  [N={N}] worst_combo={wid}  wall={wall:.1f}s", flush=True)

    print("\n=== SUMMARY: worst SLAM mode vs spanwise mesh ===")
    print(f"{'n':>4} | {'lam1':>7} | {'tube%':>6} | {'form%':>6} | "
          f"{'peak_z':>7} | {'peakReg':>7} | {'PR':>6} | {'wall_s':>7}")
    for s in summary:
        print(f"{s['N']:>4} | {s['lam']:>7.4f} | {100*s['frac_tube']:>6.1f} | "
              f"{100*s['frac_form']:>6.1f} | {s['peak_z']:>7.2f} | "
              f"{'TUBE' if s['peak_is_tube'] else 'FORM':>7} | {s['pr']:>6.1f} | "
              f"{s['wall']:>7.1f}")

    # plateau check on this run's points
    lams = [s["lam"] for s in summary]
    Ns = [s["N"] for s in summary]
    print("\nlambda1 trajectory (this run):")
    for k in range(len(Ns)):
        dlt = "" if k == 0 else f"  ({100*(lams[k]-lams[k-1])/lams[k-1]:+5.1f}%)"
        print(f"  n={Ns[k]:2d}: {lams[k]:.4f}{dlt}")
    if len(Ns) >= 2 and abs(lams[-1] - lams[-2]) / lams[-2] < 0.05:
        print(f"PLATEAU: lambda1 settled near {lams[-1]:.3f} "
              f"(<5% over last refine).")
    else:
        print(f"STILL DECLINING through n={Ns[-1]} (lambda1={lams[-1]:.3f}); "
              f"this is an UPPER bound on the converged worst-lambda.")


if __name__ == "__main__":
    main()
