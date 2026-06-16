"""Mesh-convergence diagnostic for the CURRENT operational headline design.

Question (settling the V.2 OLD-finding caveat): V.2 found the medium headlines
NOT mesh-converged and UNCONSERVATIVE (n_levels 6->10 dropped mass 9-11%/refine),
because the OLD closed-form used element-length Euler beam buckling + b=sqrt(area)
panels — both gain *fake* capacity as the mesh refines. Since then the regime
CHANGED: the current headline uses the beam-on-elastic-FOUNDATION buckling model
(V.3c, physical, not element-length) + STRIP panel widths (V.3b) + an eigenvalue
verification. V.3 separately noted the TRUE eigenvalue RISES on finer meshes of a
FIXED design (coarse-mesh lambda is a conservative lower bound).

Hypothesis: the current headline is mesh-CONSERVATIVE on buckling, not
unconservative. This script verifies the bias DIRECTION cheaply by re-evaluating
the SAME fixed design at finer spanwise meshes (n_levels in {8,10,12,16}, n_beams
fixed at 16) and reporting the worst linear-buckling lambda vs n_levels.

ANALYSIS ONLY — no sizer changes, no source-file edits. Reuses chain_rebuild's
eigen_worst / build_sections_from_result / ACCELS (no-slam 2-accel set) verbatim.

Design->finer-mesh mapping (stated, per the task's "do your best & STATE it"):
  * Beam radii        : resample_segment_radii (V.0.5 protocol, beam-major).
  * Skin bands         : t_bands, f0/f45/f90_bands, t_core are SINGLE-BAND (shape 1),
                         hence mesh-INDEPENDENT — reused unchanged (skin_band_map(m,1)
                         is all-zeros at every n_levels, confirmed).
  * t_hollow           : per-radius-group vector (chord-symmetry groups). Expanded to a
                         per-form-element spanwise field on the coarse mesh, resampled
                         per-beam (segment-centre np.interp, same kernel as radii),
                         then re-aggregated to the finer model's groups by group-mean.
                         The finer model's own tmap (in build_sections_from_result)
                         keys the hollow subset out of it.
  * r_tube / t_wall    : per tube-segment along z (n_levels-1 entries). Resampled by
                         normalized spanwise (segment-centre) position, same np.interp.

Run: PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python runs/mesh_converge_diag.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import build_beam_shell_model, resample_segment_radii
from wing_design.beams.fea_model import project_panels_to_skin
from wing_design.beams.shell_sizing import beam_radius_groups

# reuse the exact verification machinery (no-slam ACCELS, eigen_worst, sections)
import chain_rebuild as cr

RUNS = Path(__file__).parent
N_BEAMS = 16
N_FROM = 8                       # the headline's native spanwise mesh
N_LIST = [8, 10, 12, 16]


class Carrier:
    """Minimal duck-typed stand-in for a LaminateSizingResult, holding the
    mapped design arrays that eigen_worst / build_sections_from_result read."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _hollow_tmap(model):
    """Reproduce build_sections_from_result's hollow-group -> t_hollow-index map:
    t_hollow is indexed only over the *hollow* radius-groups (a subset)."""
    goe, _ = beam_radius_groups(model)
    hset = sorted(int(e) for e in model.hollow_elements)
    hgroups = sorted({int(goe[e]) for e in hset})
    tmap = {g: i for i, g in enumerate(hgroups)}
    return goe, hset, tmap


def hollow_vector_to_field(t_hollow, model_from):
    """Expand the per-hollow-group t_hollow vector to a per-FORM-element field
    (n_form,), filling hollow elements by tmap and leaving non-hollow elements
    as NaN (carried through interp via per-beam masking below)."""
    goe, hset, tmap = _hollow_tmap(model_from)
    n_form = model_from.n_form_elements
    field = np.full(n_form, np.nan, dtype=float)
    for e in hset:
        field[e] = float(t_hollow[tmap[int(goe[e])]])
    return field


def resample_form_field(field_from, n_from, n_to):
    """Resample a per-form-element spanwise field (beam-major, n_beams*(n-1))
    from n_from to n_to levels — same segment-centre np.interp kernel as radii.
    NaN entries (non-hollow segments) are dropped per beam before interp so the
    in-wing hollow region interpolates only from in-wing source samples."""
    seg_from, seg_to = n_from - 1, n_to - 1
    x_from = (np.arange(seg_from) + 0.5) / seg_from
    x_to = (np.arange(seg_to) + 0.5) / seg_to
    out = np.full(N_BEAMS * seg_to, np.nan)
    for b in range(N_BEAMS):
        rb = field_from[b * seg_from:(b + 1) * seg_from]
        m = np.isfinite(rb)
        if not m.any():
            continue
        out[b * seg_to:(b + 1) * seg_to] = np.interp(x_to, x_from[m], rb[m])
    return out


def field_to_hollow_vector(field, model_to):
    """Aggregate a per-form-element field to the finer model's t_hollow vector
    (one entry per hollow group, in tmap order) by group-mean."""
    goe, hset, tmap = _hollow_tmap(model_to)
    n_groups = len(tmap)
    sums = np.zeros(n_groups)
    cnts = np.zeros(n_groups)
    for e in hset:
        i = tmap[int(goe[e])]
        sums[i] += field[e]
        cnts[i] += 1.0
    cnts[cnts == 0] = 1.0
    return sums / cnts


def map_hollow_vector(t_hollow, model_from, model_to, n_from, n_to):
    """coarse t_hollow -> finer t_hollow via the per-form-field detour."""
    field = hollow_vector_to_field(t_hollow, model_from)
    field_to = resample_form_field(field, n_from, n_to)
    return field_to_hollow_vector(field_to, model_to)


def resample_tube_segments(vec, n_from, n_to):
    """Resample a per-tube-segment vector (n-1 entries along z) by segment-centre
    position. Tube is one chain along the pivot axis, so a single np.interp."""
    seg_from, seg_to = n_from - 1, n_to - 1
    x_from = (np.arange(seg_from) + 0.5) / seg_from
    x_to = (np.arange(seg_to) + 0.5) / seg_to
    return np.interp(x_to, x_from, np.asarray(vec, dtype=float))


def main() -> None:
    d = np.load(RUNS / "multistart_v2_best.npz", allow_pickle=True)
    radii0 = d["radii"]; t_bands = d["t_bands"]
    f0 = d["f0_bands"]; f45 = d["f45_bands"]; f90 = d["f90_bands"]
    r_tube0 = d["r_tube"]; t_wall0 = d["t_wall"]
    t_hollow0 = d["t_hollow"]; t_core = d["t_core"]
    print(f"loaded headline: radii{radii0.shape} t_hollow{t_hollow0.shape} "
          f"r_tube{r_tube0.shape} t_bands{t_bands.shape} t_core{t_core.shape}",
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

    # coarse (native) model — used as the resampling source frame
    model_from = build_beam_shell_model(spec, core_tube=True, hollow_beams=True,
                                        n_levels=N_FROM, **base_mk)

    results = []
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
            t_hollow = map_hollow_vector(t_hollow0, model_from, model, N_FROM, N)
            r_tube = resample_tube_segments(r_tube0, N_FROM, N)
            t_wall = resample_tube_segments(t_wall0, N_FROM, N)

        # bands are single-band -> mesh-independent, reused unchanged
        r = Carrier(radii=radii, t_bands=t_bands, f0_bands=f0, f45_bands=f45,
                    f90_bands=f90, r_tube=r_tube, t_wall=t_wall,
                    t_hollow=t_hollow, t_core=t_core)

        # rebuild/project aero loads for THIS model's node count (same as chain)
        loads = [project_panels_to_skin(model, ar.panels,
                                        safety_factor=ar.case.safety_factor)
                 for ar in active]

        lam = cr.eigen_worst(model, loads, r, P.rho_kgm3)
        wall = time.perf_counter() - t0

        # self-consistency probes: tube fits, hollow walls < radius, all finite
        goe, _ = beam_radius_groups(model)
        hset = set(int(e) for e in model.hollow_elements)
        hgroups = sorted({int(goe[e]) for e in hset})
        tmap = {g: i for i, g in enumerate(hgroups)}
        wall_ok = all(float(t_hollow[tmap[int(goe[e])]]) <= float(radii[e]) + 1e-12
                      for e in hset)
        finite_ok = bool(np.all(np.isfinite(radii)) and np.all(np.isfinite(t_hollow))
                         and np.all(np.isfinite(r_tube)) and np.all(np.isfinite(t_wall)))
        results.append((N, lam, wall, wall_ok, finite_ok,
                        len(t_hollow), len(r_tube)))
        print(f"[N={N:2d}] worst lambda_cr = {lam:.4f}   wall={wall:6.1f}s   "
              f"hollow_wall<=r:{wall_ok} finite:{finite_ok} "
              f"(t_hollow={len(t_hollow)}, r_tube={len(r_tube)})", flush=True)

    print("\n=== MESH-CONVERGENCE OF FIXED HEADLINE DESIGN (foundation+strip+eigen) ===")
    print(f"{'n_levels':>8} | {'worst_lambda_cr':>16} | {'self-consistent':>15}")
    print("-" * 48)
    lam8 = results[0][1]
    for N, lam, wall, wok, fok, nh, nt in results:
        sc = "yes" if (wok and fok) else "NO"
        print(f"{N:>8} | {lam:>16.4f} | {sc:>15}")
    lam_last = results[-1][1]
    direction = ("RISES (mesh-CONSERVATIVE: coarse lambda is a lower bound; "
                 "16x8 headline is SAFE)" if lam_last >= lam8 else
                 "FALLS (UNCONSERVATIVE: 16x8 mass optimistic; re-baseline needed)")
    print(f"\nlambda(N={N_LIST[0]})={lam8:.4f} -> lambda(N={N_LIST[-1]})={lam_last:.4f}  "
          f"(delta {100*(lam_last-lam8)/lam8:+.1f}%)")
    print(f"VERDICT: {direction}")


if __name__ == "__main__":
    main()
