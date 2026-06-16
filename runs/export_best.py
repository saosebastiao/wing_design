"""Export milestone CAD + FEA artifacts for the 1021.6 kg medium-wingsail headline.

The design lives in runs/multistart_v2_best.npz (multistart_v2.py best, IPOPT+KS,
foundation, datum_ortho, sandwich + hollow beams + core tube). CLAUDE.md requires
that a new headline mass produce exported (regenerable) CAD geometry and supporting
FEA fields. This script:

  1. reloads the saved design into a minimal _R carrier (chain_rebuild reload pattern);
  2. rebuilds the model + loads + pressures EXACTLY as multistart_v2.py does;
  3. independently re-verifies the worst linear-buckling eigenvalue AND keeps its mode
     shape (eigen_worst_mode below, a (lam, mode)-returning adaptation of
     chain_rebuild.eigen_worst), asserting agreement with the saved meta eigen_lam
     (2.7637) within 1% — this assertion IS the verification;
  4. writes the worst-mode VTU (mode_translation point field);
  5. lofts the sized beams to STL (lens, circular fallback) and the skin solid to STL.

Regenerate with:
  PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python runs/export_best.py
exports/ is gitignored; this script is the record of how the artifacts are produced.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # so `import chain_rebuild` works

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    build_beam_shell_model,
    build_sized_circular_beams,
    build_sized_lens_beams,
    build_sized_ring_braces,
    resample_segment_radii,
)
from wing_design.beams.body_loads import body_load_vector
from wing_design.beams.fea_model import panel_pressure_per_tri, project_panels_to_skin
from wing_design.beams.shell_model import skin_datum_angles
from wing_design.beams.shell_sizing import skin_band_map
from wing_design.geometry import build_wing_solid, medium_wingsail
from wing_design.materials.unidir import (
    PVC_H80, T700_EPOXY, laminate_stiffness_offset, sandwich_D_factor,
)
from wing_design.structural.beam_shell import (
    solve_beam_shell_laminate, solve_beam_shell_laminate_factored,
)
from wing_design.structural.eigen_buckling import linear_buckling
from wing_design.structural.geometric_stiffness import assemble_geometric_stiffness
from wing_design.structural.shell import recover_membrane_stress_C
from wing_design.viz.vtu import write_vtu_triangles

from chain_rebuild import ACCELS, DATUM, SF, build_sections_from_result
from wing_design.structural.frame import BeamSection

RUNS = Path(__file__).parent


def eigen_worst_mode(model, loads_aero, r, rho, accels=ACCELS):
    """Worst linear-buckling eigenvalue over (aero cases x accels) at the optimum,
    sandwich-aware (D x phi, core density in body loads, hollow/tube sections), AND
    the eigenvector of that worst combination.

    Adapted from chain_rebuild.eigen_worst (which returns only lmin) to also keep the
    mode shape, mirroring examples/49_rebaseline.eigen_worst's (lam, mode) return.

    BRACED-AWARE: if the model has brace_elements and r carries a brace_radius, append
    BeamSection.circular(brace_radius) for each brace element so the sections list
    length matches model.beam_elements.shape[0] (form → tube → brace), exactly as
    slam_braced.eigen_worst_braced does — otherwise the solve crashes on a braced model.

    Returns (lmin, (vec, free)) — vec is the eigenvector over the free DOFs.
    """
    band_of_tri = skin_band_map(model, 1)
    t_tri = r.t_bands[band_of_tri]
    th = skin_datum_angles(model, DATUM)
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
    sections, gusset, gusset_sec = build_sections_from_result(model, r)

    # Braced-aware: append a circular brace section per brace element so that
    # len(sections) == model.beam_elements.shape[0] (form → tube → brace).
    if (getattr(model, "brace_elements", None) is not None
            and getattr(r, "brace_radius", None) is not None):
        n_brace = len(model.brace_elements)
        brace_r = float(np.asarray(r.brace_radius).reshape(-1)[0])
        sections += [BeamSection.circular(brace_r)] * n_brace
        print(f"  [eigen] appended {n_brace} brace sections r={brace_r*1e3:.2f} mm "
              f"(total sections={len(sections)}, "
              f"beam_elements={model.beam_elements.shape[0]})", flush=True)

    assert len(sections) == model.beam_elements.shape[0], (
        f"Section count mismatch: {len(sections)} != {model.beam_elements.shape[0]}")

    areas_e = np.array([s.A for s in sections])
    fac = solve_beam_shell_laminate_factored(
        model.nodes, model.beam_elements, sections, model.shell_tris,
        E_beam=model.E_beam, G_beam=model.G_beam, A_skin=A_arg, D_skin=D_arg,
        fixed_nodes=model.fixed_nodes, loads=loads_aero[0],
        gusset_elements=gusset, gusset_section=gusset_sec)
    lmin, mode_min = np.inf, None
    for lc in loads_aero:
        for acc in accels:
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
            lams, modes = linear_buckling(fac.K_ff, Kg[np.ix_(fac.free, fac.free)], k=1)
            if lams.size and lams[0] < lmin:
                lmin, mode_min = float(lams[0]), (modes[:, 0], fac.free)
    return lmin, mode_min


def _brace_radius_from(d, meta, r):
    """Resolve the sized brace radius (m) from the npz: top-level `brace_radius`
    array key (stored as a (1,) array), else meta['brace_radius_mm'] / 1e3, else the
    carrier r.brace_radius. Returns None for an unbraced design."""
    if "brace_radius" in d.files:
        return float(np.asarray(d["brace_radius"]).reshape(-1)[0])
    if getattr(r, "brace_radius", None) is not None:
        return float(np.asarray(r.brace_radius).reshape(-1)[0])
    meta_br_mm = meta.get("brace_radius_mm")
    if meta_br_mm is not None and np.isfinite(float(meta_br_mm)):
        return float(meta_br_mm) / 1e3
    return None


def main(target: str = "multistart_v2_best.npz") -> None:
    t_start = time.perf_counter()
    out = Path("exports"); out.mkdir(exist_ok=True)

    # 1. reload the saved design into a minimal carrier (chain_rebuild _R pattern)
    f = RUNS / target
    if not f.exists():
        raise SystemExit(f"missing {f}")
    stem = f.stem  # output basenames derive from this so braced/unbraced don't collide
    d = np.load(f, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    print(f"loaded {f.name}: mass={meta['mass_kg']:.1f} kg "
          f"eigen_lam(saved)={meta['eigen_lam']:.4f} converged={meta['converged']}",
          flush=True)

    class _R:
        pass
    r = _R()
    for k in ("radii", "t_bands", "f0_bands", "f45_bands", "f90_bands"):
        setattr(r, k, d[k])
    for k in ("r_tube", "t_wall", "t_hollow", "t_core", "brace_radius"):
        setattr(r, k, d[k] if k in d.files else None)

    # Detect braced design (brace_radius npz key, carrier, or meta['brace_radius_mm']).
    brace_radius_val = _brace_radius_from(d, meta, r)
    braced = brace_radius_val is not None

    # Accel set for eigen re-verification must MATCH the set the design was sized /
    # eigen-gated against, else the recomputed lambda won't reproduce the saved one.
    # Default (unbraced multistart headline): chain_rebuild.ACCELS (2 cases).
    # Slam-survival designs carry meta['slam_g'] and were sized against 4 cases
    # (upright, 30deg heel, +Gg lateral slam, -Gg lateral slam).
    accels = ACCELS
    slam_g = meta.get("slam_g")
    if slam_g is not None:
        G0 = 9.81
        TH = np.radians(30.0)
        accels = ((0.0, 0.0, -G0),
                  (0.0, G0 * np.sin(TH), -G0 * np.cos(TH)),
                  (0.0, float(slam_g) * G0, -G0),
                  (0.0, -float(slam_g) * G0, -G0))

    if braced:
        # ensure the carrier exposes a scalar-resolvable brace_radius for the eigen
        r.brace_radius = brace_radius_val
        print(f"BRACED design detected: brace_radius={brace_radius_val*1e3:.2f} mm "
              f"→ building model with lateral_bracing=True "
              f"(slam_g={slam_g}, {len(accels)} accel cases)", flush=True)

    # 2. rebuild model + loads + pressures EXACTLY as the sizer built them
    P = medium_scenario()
    spec = P.geometry
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    active = [ar for ar in envelope
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    model = build_beam_shell_model(spec, n_beams=16, n_levels=8, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
        skin_thickness=P.skin_sizing.t_baseline_m, core_tube=True, hollow_beams=True,
        lateral_bracing=braced)
    loads = [project_panels_to_skin(model, ar.panels, safety_factor=ar.case.safety_factor)
             for ar in active]
    # pressures are projected identically; not needed for eigen but kept for parity
    _press = [panel_pressure_per_tri(model, ar.panels, safety_factor=ar.case.safety_factor)
              for ar in active]

    # 3. independent eigen re-verification (worst lambda + its mode)
    t0 = time.perf_counter()
    lam, mode = eigen_worst_mode(model, loads, r, P.rho_kgm3, accels=accels)
    print(f"recomputed worst eigen lambda = {lam:.4f} "
          f"(saved {meta['eigen_lam']:.4f}) in {time.perf_counter()-t0:.0f}s",
          flush=True)
    saved_lam = float(meta["eigen_lam"])
    rel = abs(lam - saved_lam) / abs(saved_lam)
    assert rel <= 0.01, (
        f"EIGEN MISMATCH: recomputed {lam:.6f} vs saved {saved_lam:.6f} "
        f"(rel err {rel*100:.3f}% > 1%) — refusing to export an unreproducible design")
    print(f"  eigen OK: rel err {rel*100:.4f}% <= 1%", flush=True)

    # 4. worst-mode VTU
    vtu_path = out / f"{stem}_worst_mode.vtu"
    if mode is None:
        raise SystemExit("no mode shape captured — cannot write VTU")
    vec, free = mode
    full = np.zeros(6 * model.nodes.shape[0])
    full[free] = vec
    write_vtu_triangles(vtu_path, model.nodes, model.shell_tris,
                        point_data={"mode_translation": full.reshape(-1, 6)[:, :3]})
    print(f"wrote {vtu_path}", flush=True)

    # 5a. loft sized beams to STL (lens, circular fallback)
    from build123d import Compound, export_stl
    n_geom = 40
    dense = resample_segment_radii(r.radii, n_beams=16, n_from=8, n_to=n_geom)
    try:
        beams = build_sized_lens_beams(spec, dense, n_beams=16, n_levels=n_geom)
        kind = "lens"
    except Exception as exc:
        print(f"  lens loft failed ({exc}); circular fallback", flush=True)
        beams = build_sized_circular_beams(spec, dense, n_beams=16, n_levels=n_geom)
        kind = "circular"
    beams_path = out / f"{stem}_beams.stl"
    comp = Compound(label=f"{stem}_beams", children=list(beams))
    export_stl(comp, str(beams_path))
    print(f"wrote {beams_path} (beams, {kind})", flush=True)

    # 5b. ring-brace solids to STL (only for braced designs; skipped if absent)
    braces_path = out / f"{stem}_braces.stl"
    n_rings = 0
    if braced and getattr(model, "brace_elements", None) is not None:
        ring_solids = build_sized_ring_braces(model, brace_radius_val)
        n_rings = len(ring_solids)
        print(f"ring braces: {n_rings} solids, radius={brace_radius_val*1e3:.2f} mm",
              flush=True)
        if ring_solids:
            ring_comp = Compound(label=f"{stem}_braces", children=ring_solids)
            export_stl(ring_comp, str(braces_path))
            print(f"wrote {braces_path} (ring braces)", flush=True)
    else:
        print("ring braces: not present in this design (unbraced path)", flush=True)

    # 5d. skin solid to STL (skip on failure — beams STL + VTU are the required ones)
    skin_path = out / f"{stem}_skin.stl"
    skin_ok = False
    try:
        skin = build_wing_solid(medium_wingsail)
        export_stl(skin, str(skin_path))
        skin_ok = True
        print(f"wrote {skin_path} (skin solid)", flush=True)
    except Exception as exc:
        print(f"  skin solid export SKIPPED ({type(exc).__name__}: {exc})", flush=True)

    # 6. summary
    wall = time.perf_counter() - t_start
    print(f"\n=== EXPORT SUMMARY ({stem}, {meta['mass_kg']:.1f} kg) ===", flush=True)
    print(f"mass            : {meta['mass_kg']:.1f} kg", flush=True)
    print(f"eigen lambda    : recomputed {lam:.4f} | saved {saved_lam:.4f} "
          f"(rel {rel*100:.4f}%, SF target {SF})", flush=True)
    print(f"converged/feas  : converged={meta['converged']} "
          f"n_feasible={meta.get('n_feasible')}/{len(meta.get('start_masses', []))} "
          f"best_start={meta.get('best_start')} used_incumbent={meta.get('used_incumbent')}",
          flush=True)
    radii_mm = np.asarray(r.radii) * 1e3
    print(f"beam radii      : {radii_mm.min():.2f} .. {radii_mm.max():.2f} mm "
          f"({radii_mm.size} segments)", flush=True)
    print(f"t_bands         : {[round(t*1e3, 3) for t in np.asarray(r.t_bands)]} mm",
          flush=True)
    print(f"layup f0/f45/f90: {float(r.f0_bands[0]):.3f} / {float(r.f45_bands[0]):.3f} "
          f"/ {float(r.f90_bands[0]):.3f}", flush=True)
    tcore_mm = np.asarray(r.t_core) * 1e3 if r.t_core is not None else None
    twall_mm = np.asarray(r.t_wall) * 1e3 if r.t_wall is not None else None
    print(f"t_core          : {[round(c, 3) for c in tcore_mm] if tcore_mm is not None else None} mm",
          flush=True)
    print(f"t_wall (tube)   : {[round(w, 3) for w in twall_mm] if twall_mm is not None else None} mm",
          flush=True)
    print("files written:", flush=True)
    print(f"  {vtu_path}", flush=True)
    print(f"  {beams_path}  ({kind})", flush=True)
    if braced and n_rings:
        print(f"  {braces_path}  (ring braces x{n_rings}, "
              f"r={brace_radius_val*1e3:.2f} mm)", flush=True)
    print(f"  {skin_path}" + ("" if skin_ok else "  [SKIPPED]"), flush=True)
    print(f"total wall-clock: {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "multistart_v2_best.npz")
