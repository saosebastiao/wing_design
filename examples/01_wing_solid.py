"""Loft the medium (22 m) wingsail OML solid and export it to STEP and STL.

Demonstrates geometry.build_wing_solid on the medium wingsail: airfoil
sections + faired transition + cylindrical rotation spar, lofted into one
solid, then written to exports/ and displayed in OCP CAD Viewer.
"""
from __future__ import annotations

from pathlib import Path

from build123d import export_step, export_stl

from wing_design.geometry import build_wing_solid, medium_wingsail
from wing_design.viz import show_in_viewer


def main() -> None:
    spec = medium_wingsail
    part = build_wing_solid(spec)
    out = Path(__file__).resolve().parent.parent / "exports"
    out.mkdir(exist_ok=True)
    export_step(part, str(out / "wingsail_v0.step"))
    export_stl(part, str(out / "wingsail_v0.stl"))
    print(f"Wrote {out}/wingsail_v0.{{step,stl}}")
    print(f"Solid volume (model units = m): {part.volume:.4g} m^3")
    print(f"Bounding box: {part.bounding_box()}")
    show_in_viewer(part, names=["wingsail"])


if __name__ == "__main__":
    main()
