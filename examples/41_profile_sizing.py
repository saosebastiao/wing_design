"""V.0.1 — Profile the medium sizing run (the gate for the V.0 speed work).

cProfiles a few SLSQP iterations (maxiter below) of the medium 16x8 problem on the
analytic-Jacobian path — the production configuration of example 32 — and ranks where
the wall-clock goes: K assembly vs SensCache build vs adjoint back-substitutions vs
stress recovery vs laminate build. The ranking decides whether V.0.2 (vectorize
per-element loops) or V.0.3 (KS aggregation of beam constraints) leads.

Profile covers ONLY size_beam_shell_laminate (setup excluded). Raw stats are dumped to
exports/profile_medium_sizing.prof (inspect later via pstats/snakeviz). cProfile adds
overhead to Python-loop-heavy code (the thing being measured), so treat shares as the
signal, absolute seconds as inflated.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from pathlib import Path

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    project_panels_to_beam_nodes,
    size_beam_shell_laminate,
)
from wing_design.materials.unidir import T700_EPOXY

MAXITER = 10


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8

    t0 = time.perf_counter()
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
    print(f"setup (geometry + aero): {time.perf_counter() - t0:.1f} s, "
          f"{len(load_arrays)} load cases, {model.beam_elements.shape[0]} beam elements, "
          f"{model.shell_tris.shape[0]} shell tris")

    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=1.5,
        use_analytic_jacobian=True,
    )

    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    r = size_beam_shell_laminate(model, load_arrays, cfg, ply=T700_EPOXY,
                                 rho=P.rho_kgm3, maxiter=MAXITER)
    prof.disable()
    wall = time.perf_counter() - t0
    print(f"\nsizing maxiter={MAXITER}: wall={wall:.1f} s ({wall / max(r.n_iter, 1):.1f} s/iter, "
          f"{r.n_iter} iters, converged={r.converged} [expected False at maxiter])")

    out = Path("exports"); out.mkdir(exist_ok=True)
    prof_path = out / "profile_medium_sizing.prof"
    prof.dump_stats(prof_path)
    print(f"raw stats -> {prof_path}\n")

    st = pstats.Stats(prof)
    st.sort_stats("cumulative")
    print("=" * 90, "\nTop 25 by cumulative time:")
    st.print_stats(25)
    st.sort_stats("tottime")
    print("=" * 90, "\nTop 25 by self time:")
    st.print_stats(25)

    # Bucket self-time by module — the V.0.1 deliverable.
    buckets = {
        "sensitivity.py (SensCache + adjoint + grads)": "beams/sensitivity",
        "beam_shell.py (K assembly + solve)": "structural/beam_shell",
        "shell.py (membrane/DKT elements + recovery)": "structural/shell",
        "frame.py (beam elements + recovery)": "structural/frame",
        "unidir.py (laminate CLT build)": "materials/unidir",
        "failure.py (strength ratios)": "materials/failure",
        "buckling.py (closed-form checks)": "structural/buckling",
        "laminate_sizing.py (driver/closures)": "beams/laminate_sizing",
        "scipy splu/spsolve": "scipy/sparse/linalg",
        "scipy slsqp": "_slsqp",
    }
    tot_by_bucket = dict.fromkeys(buckets, 0.0)
    total = 0.0
    for (fname, _lineno, _func), (_cc, _nc, tt, _ct, _callers) in st.stats.items():
        total += tt
        for label, frag in buckets.items():
            if frag in fname.replace("\\", "/"):
                tot_by_bucket[label] += tt
                break
    print("=" * 90, "\nSelf-time by bucket (cProfile-inflated; shares are the signal):")
    for label, tt in sorted(tot_by_bucket.items(), key=lambda kv: -kv[1]):
        print(f"  {tt:8.1f} s  {100.0 * tt / total:5.1f}%  {label}")
    print(f"  {total:8.1f} s  100.0%  TOTAL (all self time)")


if __name__ == "__main__":
    main()
