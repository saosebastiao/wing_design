"""Size the medium (22 m) wingsail and export the resulting structure as STL meshes.

Sizes with the default mirror-symmetric + monotonic-taper beam constraints (span datum +
buckling), densifies the sized beam radii onto a finer geometry resolution, lofts the
sized form beams (lens sections, circular fallback) and the OML skin, and writes STL
meshes (skin / beams / assembly) to exports/. Prints the sized headline.
"""
from __future__ import annotations

from pathlib import Path

from build123d import Compound, export_stl

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.geometry import build_wing_solid, medium_wingsail
from wing_design.beams import (
    LaminateSizingConfig,
    build_beam_frame,
    build_beam_shell_model,
    build_sized_circular_beams,
    build_sized_lens_beams,
    project_panels_to_beam_nodes,
    resample_segment_radii,
    size_beam_shell_laminate,
)
from wing_design.materials.unidir import T700_EPOXY


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8
    n_geom = 40  # geometry densification levels

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

    cfg = LaminateSizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa, tip_defl_max_m=0.02 * spec.span, tip_twist_max_deg=5.0,
        ply_angle_datum=(0.0, 0.0, 1.0), buckling_safety_factor=1.5, n_skin_bands=4)

    print("sizing (symmetric + monotonic beams, span datum + buckling)...")
    r = size_beam_shell_laminate(model, load_arrays, cfg, ply=T700_EPOXY, rho=P.rho_kgm3, maxiter=300)
    print(f"  converged={r.converged} iters={r.n_iter} mass={r.mass_kg:.2f} kg "
          f"(beams {r.beam_mass_kg:.2f} + skin {r.skin_mass_kg:.2f})")
    print(f"  beam radii {r.radii.min()*1e3:.1f}-{r.radii.max()*1e3:.1f} mm | "
          f"t_bands {[round(t*1e3,2) for t in r.t_bands]} mm")
    print(f"  twist {r.tip_twist_deg:.2f} deg | buck {r.max_beam_buckling_util:.2f}/{r.max_panel_buckling_util:.2f}")

    dense = resample_segment_radii(r.radii, n_beams=n_beams, n_from=n_levels, n_to=n_geom)
    try:
        beams = build_sized_lens_beams(spec, dense, n_beams=n_beams, n_levels=n_geom)
        kind = "lens"
    except Exception as exc:  # build123d lofting can fail on the lens faces
        print(f"  lens loft failed ({exc}); falling back to circular sections")
        beams = build_sized_circular_beams(spec, dense, n_beams=n_beams, n_levels=n_geom)
        kind = "circular"
    skin = build_wing_solid(medium_wingsail)

    out = Path(__file__).resolve().parent.parent / "exports"
    out.mkdir(exist_ok=True)
    beam_compound = Compound(label="beams", children=list(beams))
    assembly = Compound(label="wingsail_sized", children=[skin, beam_compound])
    export_stl(skin, str(out / "wingsail_sized_skin.stl"))
    export_stl(beam_compound, str(out / "wingsail_sized_beams.stl"))
    export_stl(assembly, str(out / "wingsail_sized_assembly.stl"))
    print(f"  wrote STL ({kind} beams) to {out}/wingsail_sized_*.stl")


if __name__ == "__main__":
    main()
