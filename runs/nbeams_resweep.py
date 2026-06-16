"""P.4b — re-sweep n_beams on the CURRENT operational regime.

The old P.4 "16 optimal" finding was taken at a STALE regime (pre-foundation /
pre-strip) with a COARSE-mesh-optimistic eigenvalue. The regime has since moved
to the operational config (datum_ortho panel D, foundation beam buckling, strip
panel widths, 2-accel no-slam set, IPOPT+KS) and we now know the n=8 eigenvalue
is a CONSERVATIVE lower bound that RISES to a ~1.5-1.6 cluster floor on converged
spanwise meshes (mesh_converge_diag / V.3). So "16 optimal" must be re-tested:
size each n_beams at the operational config, then judge feasibility on the
CONVERGED eigen floor (n>=24), not the n=8 eigen.

For each n_beams in {12, 14, 16, 20}:
  1. Run the canonical 5-stage warm chain (A0 tube-only cold -> A tube+hollow ->
     B sandwich -> C foundation -> D datum_ortho), keyed off n_beams, at
     n_levels=8 (chain_rebuild's stages/configs/eigen reused verbatim — only the
     n_beams of the model is parameterized; chain_rebuild hardcodes n_beams=16).
     Per-n_beams cold tube-only is fine: it is the tube+hollow/sandwich COLD that
     is eigen-rejected (P#1b), and the chain warm-starts those from A0.
  2. VERIFY buckling at converged spanwise meshes (n_levels in {24,28,32}) by
     resampling the stage-D design onto the finer model and re-running eigen_worst
     (eigen-only, cheap). The converged worst-lambda FLOOR (min over the converged
     meshes of mode-1 lambda) is the feasibility gate: feasible iff floor >= 1.5.

The optimum is the LIGHTEST n_beams whose CONVERGED eigen is feasible (>= 1.5).

MEMORY BUDGET (hard rule, see multistart_v2): each IPOPT sizing peaks ~18 GiB on
a 32 GiB machine -> NEVER two at once. The sweep loop is SERIAL (one chain point
at a time, one stage at a time). Eigen verification is light and also serial.

Each stage saves runs/nbeams_<N>_<tag>.npz immediately (resumable: existing npz
= stage skipped+reloaded), NEVER clobbering chain_rebuild's chain_*.npz. Per-point
rows accumulate into runs/nbeams_resweep.npz incrementally (a kill mid-sweep keeps
completed points).

Run (smoke, validates the pipeline — meaningless eigen at maxiter=3):
  PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python runs/nbeams_resweep.py --smoke
Run (real sweep — controller launches/monitors this, multi-hour):
  PYTHONPYCACHEPREFIX=$PWD/.pycache .venv/bin/python runs/nbeams_resweep.py
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import build_beam_shell_model, resample_segment_radii
from wing_design.beams.fea_model import panel_pressure_per_tri, project_panels_to_skin
from wing_design.beams.laminate_sizing import (
    LaminateSizingConfig, design_vector_from_result, laminate_result_is_feasible,
    size_beam_shell_laminate,
)
from wing_design.beams.shell_sizing import beam_radius_groups
from wing_design.materials.unidir import PVC_H80, T700_EPOXY

# reuse the verification machinery + operational constants verbatim
import chain_rebuild as cr
from chain_rebuild import ACCELS, DATUM, SF, eigen_worst
# reuse the design->finer-mesh resampling helpers (Carrier, hollow detour, tube)
import mesh_converge_diag as mcd
from mesh_converge_diag import (
    Carrier, _hollow_tmap, field_to_hollow_vector, hollow_vector_to_field,
    resample_tube_segments,
)

RUNS = Path(__file__).parent
N_BEAMS_LIST = [12, 14, 16, 20]
N_LEVELS = 8                       # sizing spanwise mesh (matches operational)
CONV_LEVELS = [24, 28, 32]         # converged verification meshes
CONV_FLOOR = 1.5                   # feasibility gate on the converged eigen floor


# ---------------------------------------------------------------------------
# n_beams-parameterized form-field resampler (mesh_converge_diag.resample_form_field
# hardcodes its module-level N_BEAMS=16; everything else there is n_beams-agnostic).
# ---------------------------------------------------------------------------
def resample_form_field(field_from, n_beams, n_from, n_to):
    """Resample a per-form-element spanwise field (beam-major, n_beams*(n-1)) from
    n_from to n_to levels — same segment-centre np.interp kernel as the radii
    resampler. NaN entries (non-hollow segments) are dropped per beam before
    interp so the in-wing hollow region interpolates only from in-wing samples."""
    seg_from, seg_to = n_from - 1, n_to - 1
    x_from = (np.arange(seg_from) + 0.5) / seg_from
    x_to = (np.arange(seg_to) + 0.5) / seg_to
    out = np.full(n_beams * seg_to, np.nan)
    for b in range(n_beams):
        rb = field_from[b * seg_from:(b + 1) * seg_from]
        m = np.isfinite(rb)
        if not m.any():
            continue
        out[b * seg_to:(b + 1) * seg_to] = np.interp(x_to, x_from[m], rb[m])
    return out


def map_hollow_vector(t_hollow, model_from, model_to, n_beams, n_from, n_to):
    """coarse t_hollow -> finer t_hollow via the per-form-field detour
    (n_beams-parameterized counterpart of mesh_converge_diag.map_hollow_vector)."""
    field = hollow_vector_to_field(t_hollow, model_from)
    field_to = resample_form_field(field, n_beams, n_from, n_to)
    return field_to_hollow_vector(field_to, model_to)


# ---------------------------------------------------------------------------
# Per-n_beams problem setup (aero envelope is n_beams-independent; only the
# beam-shell model and the load projection onto its nodes change).
# ---------------------------------------------------------------------------
def _scenario():
    P = medium_scenario()
    spec = P.geometry
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, P.load_cases, method="lifting_line",
                              spanwise_resolution=P.aero.spanwise_resolution)
    active = [ar for ar in envelope
              if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0]
    return P, spec, active


def _mk(P, n_beams, n_levels):
    return dict(n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
                material=P.material, knockdown=P.material_iso.skin_E_knockdown,
                nu=P.nu_iso, skin_thickness=P.skin_sizing.t_baseline_m)


def _stage_configs(P, spec):
    """The operational 5-stage chain configs, identical to chain_rebuild.main()."""
    base = dict(sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span,
                tip_twist_max_deg=5.0, ply_angle_datum=DATUM,
                buckling_safety_factor=SF, use_analytic_jacobian=True,
                panel_width_mode="strip", accel_vectors=ACCELS)
    return [
        ("A0_tube_only", False, LaminateSizingConfig(**base), 500),
        ("A_tube_hollow", True, LaminateSizingConfig(**base), 500),
        ("B_sandwich", True, LaminateSizingConfig(**base, core=PVC_H80, ks_rho=50.0,
                                                  optimizer="ipopt"), 2000),
        ("C_foundation", True, LaminateSizingConfig(**base, core=PVC_H80, ks_rho=50.0,
                                                    optimizer="ipopt",
                                                    beam_buckling_model="foundation"),
         2000),
        ("D_datum_ortho", True, LaminateSizingConfig(**base, core=PVC_H80, ks_rho=50.0,
                                                     optimizer="ipopt",
                                                     beam_buckling_model="foundation",
                                                     panel_d_mode="datum_ortho"),
         2000),
    ]


def _load_carrier(npz_path):
    """Rehydrate a minimal warm-start / design carrier from a saved stage npz."""
    d = np.load(npz_path, allow_pickle=False)
    c = Carrier()
    for k in ("radii", "t_bands", "f0_bands", "f45_bands", "f90_bands"):
        setattr(c, k, d[k])
    for k in ("r_tube", "t_wall", "t_hollow", "t_core"):
        setattr(c, k, d[k] if k in d.files else None)
    meta = json.loads(str(d["meta"]))
    return c, meta


def _save_stage(npz_path, r, x, wall_s, feas, lam):
    out = {k: getattr(r, k) for k in
           ("radii", "t_bands", "f0_bands", "f45_bands", "f90_bands")}
    for k in ("r_tube", "t_wall", "t_hollow", "t_core"):
        v = getattr(r, k)
        if v is not None:
            out[k] = v
    meta = dict(mass_kg=r.mass_kg, beam_mass_kg=r.beam_mass_kg,
                skin_mass_kg=r.skin_mass_kg, tube_mass_kg=r.tube_mass_kg,
                core_mass_kg=r.core_mass_kg, converged=bool(r.converged),
                used_incumbent=bool(r.used_incumbent), n_iter=int(r.n_iter),
                wall_s=float(wall_s), feasible=bool(feas), eigen_lam=float(lam),
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30)
    np.savez(npz_path, x=x, meta=json.dumps(meta), **out)
    return meta


# ---------------------------------------------------------------------------
# chain_for: the 5-stage warm chain at a given n_beams, returning stage-D.
# ---------------------------------------------------------------------------
def chain_for(n_beams, P, spec, active, smoke=False):
    """Run the operational 5-stage warm chain at this n_beams (SERIAL). Returns
    (stage_D_carrier, stage_D_meta, model_at_n_levels, loads, n8_eigen)."""
    mk_t = _mk(P, n_beams, N_LEVELS)
    model_t = build_beam_shell_model(spec, core_tube=True, hollow_beams=False, **mk_t)
    model = build_beam_shell_model(spec, core_tube=True, hollow_beams=True, **mk_t)
    assert model_t.nodes.shape == model.nodes.shape  # identical meshes
    loads = [project_panels_to_skin(model, ar.panels,
                                    safety_factor=ar.case.safety_factor)
             for ar in active]
    press = [panel_pressure_per_tri(model, ar.panels,
                                    safety_factor=ar.case.safety_factor)
             for ar in active]

    prefix = f"nbeams_{n_beams}"
    if smoke:
        prefix = f"smoke_{prefix}"
    stages = _stage_configs(P, spec)

    prev = None
    final = None
    for tag, hollow, cfg, maxiter in stages:
        if smoke:
            maxiter = 3
        m_st = model if hollow else model_t
        npz_path = RUNS / f"{prefix}_{tag}.npz"
        if npz_path.exists():
            prev, meta = _load_carrier(npz_path)
            print(f"[nbeams={n_beams}|{tag}] exists — reusing "
                  f"({meta['mass_kg']:.1f} kg)", flush=True)
            final = (prev, meta, m_st)
            continue
        x0 = (design_vector_from_result(m_st, cfg, prev)
              if prev is not None else None)
        t0 = time.perf_counter()
        r = size_beam_shell_laminate(m_st, loads, cfg, ply=T700_EPOXY,
                                     rho=P.rho_kgm3, maxiter=maxiter,
                                     panel_pressures=press, x0=x0)
        wall = time.perf_counter() - t0
        feas = laminate_result_is_feasible(r, cfg)
        lam = eigen_worst(m_st, loads, r, P.rho_kgm3)
        x_saved = design_vector_from_result(m_st, cfg, r)
        meta = _save_stage(npz_path, r, x_saved, wall, feas, lam)
        print(f"[nbeams={n_beams}|{tag}] mass={r.mass_kg:8.1f} kg "
              f"conv={r.converged} feas={feas} inc={r.used_incumbent} "
              f"iters={r.n_iter} wall={wall:.0f}s n8_eigen={lam:.3f}", flush=True)
        prev = r
        final = (r, meta, m_st)

    r_d, meta_d, _ = final
    # stage-D's native (n=8) eigen is the chain's recorded eigen on the n=8 mesh
    n8_eigen = float(meta_d["eigen_lam"])
    return r_d, meta_d, model, loads, n8_eigen


# ---------------------------------------------------------------------------
# converged_eigen: resample stage-D design to converged meshes, eigen floor.
# ---------------------------------------------------------------------------
def converged_eigen(n_beams, result, P, spec, active, model_from,
                    conv_levels=None):
    """Resample the stage-D design (a Carrier or result) from n_levels=8 onto each
    converged mesh (n_levels in conv_levels) and return the converged worst-lambda
    FLOOR = min over those meshes of the mode-1 worst-lambda. Also returns the
    per-mesh lambdas for reporting. Eigen-only — cheap, serial."""
    if conv_levels is None:
        conv_levels = CONV_LEVELS
    radii0 = np.asarray(result.radii, dtype=float)
    t_hollow0 = np.asarray(result.t_hollow, dtype=float)
    r_tube0 = np.asarray(result.r_tube, dtype=float)
    t_wall0 = np.asarray(result.t_wall, dtype=float)
    # single-band skin fields are mesh-independent (shape 1) -> reused unchanged
    t_bands = np.asarray(result.t_bands, dtype=float)
    f0 = np.asarray(result.f0_bands, dtype=float)
    f45 = np.asarray(result.f45_bands, dtype=float)
    f90 = np.asarray(result.f90_bands, dtype=float)
    t_core = (np.asarray(result.t_core, dtype=float)
              if result.t_core is not None else None)

    per_mesh = {}
    for N in conv_levels:
        t0 = time.perf_counter()
        model = build_beam_shell_model(spec, core_tube=True, hollow_beams=True,
                                       **_mk(P, n_beams, N))
        radii = resample_segment_radii(radii0, n_beams=n_beams,
                                       n_from=N_LEVELS, n_to=N)
        t_hollow = map_hollow_vector(t_hollow0, model_from, model, n_beams,
                                     N_LEVELS, N)
        r_tube = resample_tube_segments(r_tube0, N_LEVELS, N)
        t_wall = resample_tube_segments(t_wall0, N_LEVELS, N)
        r = Carrier(radii=radii, t_bands=t_bands, f0_bands=f0, f45_bands=f45,
                    f90_bands=f90, r_tube=r_tube, t_wall=t_wall,
                    t_hollow=t_hollow, t_core=t_core)
        loads = [project_panels_to_skin(model, ar.panels,
                                        safety_factor=ar.case.safety_factor)
                 for ar in active]
        lam = eigen_worst(model, loads, r, P.rho_kgm3)
        wall = time.perf_counter() - t0
        per_mesh[N] = float(lam)
        print(f"  [nbeams={n_beams}|conv N={N:2d}] lambda={lam:.4f} "
              f"wall={wall:.1f}s", flush=True)
    floor = min(per_mesh.values())
    return floor, per_mesh


# ---------------------------------------------------------------------------
# Incremental row store for runs/nbeams_resweep.npz (kill-safe).
# ---------------------------------------------------------------------------
def _rows_path(smoke):
    # smoke writes a separate file so it can NEVER pollute the real sweep's resume.
    return RUNS / ("smoke_nbeams_resweep.npz" if smoke else "nbeams_resweep.npz")


def _save_rows(rows, smoke=False):
    """Persist accumulated per-n_beams rows. rows: list of dicts."""
    np.savez(_rows_path(smoke), rows=json.dumps(rows))


def _load_rows(smoke=False):
    p = _rows_path(smoke)
    if p.exists():
        d = np.load(p, allow_pickle=False)
        return json.loads(str(d["rows"]))
    return []


def main(smoke: bool = False) -> None:
    P, spec, active = _scenario()
    nbeams_list = [12] if smoke else N_BEAMS_LIST
    conv_levels = [24] if smoke else CONV_LEVELS

    rows = _load_rows(smoke)
    done = {r["n_beams"] for r in rows}

    for n_beams in nbeams_list:
        if n_beams in done:
            print(f"[nbeams={n_beams}] already recorded — skipping", flush=True)
            continue
        print(f"\n=== n_beams={n_beams} : chain (SERIAL) ===", flush=True)
        r_d, meta_d, model8, loads8, n8_eigen = chain_for(
            n_beams, P, spec, active, smoke=smoke)
        print(f"=== n_beams={n_beams} : converged eigen verification ===",
              flush=True)
        conv_floor, per_mesh = converged_eigen(
            n_beams, r_d, P, spec, active, model8, conv_levels=conv_levels)
        feasible = bool(conv_floor >= CONV_FLOOR)
        row = dict(n_beams=int(n_beams), mass_kg=float(meta_d["mass_kg"]),
                   n8_eigen=float(n8_eigen), conv_eigen=float(conv_floor),
                   conv_per_mesh={int(k): float(v) for k, v in per_mesh.items()},
                   feasible=feasible, smoke=bool(smoke))
        rows.append(row)
        _save_rows(rows, smoke)     # incremental, kill-safe
        print(f"[nbeams={n_beams}] mass={row['mass_kg']:.1f} "
              f"n8_eig={row['n8_eigen']:.3f} conv_eig={row['conv_eigen']:.3f} "
              f"feasible(conv>={CONV_FLOOR})={feasible}", flush=True)

    # ---- final table + optimum ----
    print("\n=== P.4b n_beams RE-SWEEP (operational regime, converged eigen) ===")
    print(f"{'n_beams':>7} | {'mass_kg':>9} | {'n8_eigen':>8} | "
          f"{'conv_eigen':>10} | {'feasible':>8}")
    print("-" * 56)
    for row in sorted(rows, key=lambda r: r["n_beams"]):
        print(f"{row['n_beams']:>7} | {row['mass_kg']:>9.1f} | "
              f"{row['n8_eigen']:>8.3f} | {row['conv_eigen']:>10.4f} | "
              f"{str(row['feasible']):>8}")
    feas = [r for r in rows if r["feasible"]]
    if feas:
        opt = min(feas, key=lambda r: r["mass_kg"])
        print(f"\nOPTIMUM (lightest with conv_eigen>={CONV_FLOOR}): "
              f"n_beams={opt['n_beams']}  mass={opt['mass_kg']:.1f} kg  "
              f"conv_eigen={opt['conv_eigen']:.4f}")
    else:
        print(f"\nNO feasible point (none reached conv_eigen>={CONV_FLOOR})"
              f"{' [expected in --smoke: maxiter=3]' if smoke else ''}")
    print("NBEAMS_RESWEEP DONE", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
