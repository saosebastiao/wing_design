"""Phase-1 sanity check: build the wingsail outer solid, export STEP+STL, view in OCP CAD Viewer."""
from __future__ import annotations

from pathlib import Path

from build123d import export_step, export_stl

from wing_design.geometry import WingSpec, build_wing_solid
from wing_design.viz import show_in_viewer


def main() -> None:
    spec = WingSpec()
    part = build_wing_solid(spec)
    out = Path(__file__).parent / "_out"
    out.mkdir(exist_ok=True)
    export_step(part, str(out / "wingsail_v0.step"))
    export_stl(part, str(out / "wingsail_v0.stl"))
    print(f"Wrote {out}/wingsail_v0.{{step,stl}}")
    print(f"Solid volume (model units = m): {part.volume:.4g} m^3")
    print(f"Bounding box: {part.bounding_box()}")
    show_in_viewer(part, names=["wingsail"])


if __name__ == "__main__":
    main()
