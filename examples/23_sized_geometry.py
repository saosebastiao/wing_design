"""Manufacturable sized form-beam geometry for the medium wingsail, exported as STEP.

Derives the spar radius, sizes the beams via the Phase-C FEA-in-the-loop loop,
builds inward-arc 'lens' beams sized to those areas (circular fallback if build123d
can't loft the lens faces), assembles everything, and exports a STEP to exports/.
Also prints the spline on-surface fidelity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from build123d import Compound, export_step

from wing_design import medium_scenario
from wing_design.aero import build_airplane, sweep_envelope
from wing_design.beams import (
    SizingConfig,
    build_beam_frame,
    build_sized_circular_beams,
    build_sized_lens_beams,
    project_panels_to_beam_nodes,
    resample_segment_radii,
    size_beams,
    spline_surface_error,
)
from wing_design.geometry import build_wing_solid


def main() -> None:
    P = medium_scenario()
    spec = P.geometry
    n_beams, n_levels = 16, 8
    n_geo = 60   # build geometry at high resolution, decoupled from the sizing resolution

    print(f"Derived spar radius = {spec.spar_radius*1e3:.0f} mm "
          f"(diameter {spec.spar_diameter*1e3:.0f} mm)")
    err_full = spline_surface_error(spec, n_beams=n_beams, n_levels=n_levels)
    err_wing = spline_surface_error(spec, n_beams=n_beams, n_levels=n_levels, z_min=0.0)
    print(f"Spline fidelity at {n_levels} levels: full {err_full*1e3:.0f} mm, "
          f"aero-surface {err_wing*1e3:.0f} mm")

    frame = build_beam_frame(
        spec, n_beams=n_beams, n_levels=n_levels, beam_radius=0.02,
        material=P.material, knockdown=P.material_iso.skin_E_knockdown, nu=P.nu_iso,
    )
    airplane = build_airplane(spec)
    envelope = sweep_envelope(
        airplane, P.load_cases, method="lifting_line",
        spanwise_resolution=P.aero.spanwise_resolution,
    )
    load_arrays = [
        project_panels_to_beam_nodes(frame, ar.panels, safety_factor=ar.case.safety_factor)
        for ar in envelope
        if ar.panels is not None and abs(ar.factored_normal_force_N) >= 1.0
    ]
    cfg = SizingConfig(
        sigma_allow_Pa=P.sigma_allow_Pa,
        tip_defl_max_m=0.02 * spec.span,
        tip_twist_max_deg=5.0,
    )
    print("\nSizing beams (SLSQP, ~1-2 min)...")
    sized = size_beams(frame, load_arrays, cfg, rho=P.rho_kgm3, maxiter=60)
    print(f"  sized mass={sized.mass_kg:.2f} kg, "
          f"radii {sized.radii.min()*1e3:.1f}-{sized.radii.max()*1e3:.1f} mm")

    # Sizing ran at n_levels (SLSQP speed); build the geometry at the finer n_geo by
    # interpolating the sized radii — decouples on-surface fidelity from sizing cost.
    geo_radii = resample_segment_radii(sized.radii, n_beams=n_beams, n_from=n_levels, n_to=n_geo)
    err_geo = spline_surface_error(spec, n_beams=n_beams, n_levels=n_geo)
    print(f"  geometry built at {n_geo} levels; on-surface fidelity worst {err_geo*1e3:.0f} mm "
          f"(vs {err_full*1e3:.0f} mm at the {n_levels}-level sizing resolution)")
    try:
        beams = build_sized_lens_beams(spec, geo_radii, n_beams=n_beams, n_levels=n_geo)
        section_kind = "lens"
    except Exception as exc:
        print(f"  lens build failed ({type(exc).__name__}: {exc}); falling back to circular.")
        beams = build_sized_circular_beams(spec, geo_radii, n_beams=n_beams, n_levels=n_geo)
        section_kind = "circular"
    print(f"  built {len(beams)} {section_kind} beams")

    wing = build_wing_solid(spec)   # eased transition → manufacturable junctions, no fillet needed
    wing.label = "wing"
    for i, b in enumerate(beams):
        b.label = f"beam_{i:02d}"
    asm = Compound(label="wingsail_sized", children=[wing, *beams])

    out = Path(__file__).resolve().parent.parent / "exports"
    out.mkdir(exist_ok=True)
    export_step(asm, str(out / "sized_geometry_v0.step"))
    print(f"\nWrote {out}/sized_geometry_v0.step  ({section_kind} sections)")
    print(f"Assembly bbox: {asm.bounding_box()}")


if __name__ == "__main__":
    main()
