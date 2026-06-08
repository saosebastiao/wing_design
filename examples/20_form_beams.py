"""Build and export the shell-following form-beam assembly for the medium (22 m) wingsail.

Constructs the crude (un-sized) form-beam assembly from the medium scenario
and exports it to STEP. Beam cross-sections are a fixed radius here;
FEA-in-the-loop sizing arrives in Phase C.
"""
from __future__ import annotations

from pathlib import Path

from build123d import export_step

from wing_design import medium_scenario, show_in_viewer
from wing_design.beams import build_assembly


def main() -> None:
    spec = medium_scenario().geometry
    asm = build_assembly(spec, n_beams=16, n_levels=20, beam_radius=0.02, wall=0.003)

    out = Path(__file__).resolve().parent.parent / "exports"
    out.mkdir(exist_ok=True)
    export_step(asm, str(out / "form_beams_v0.step"))
    print(f"Wrote {out}/form_beams_v0.step")
    print(f"Children: {[c.label for c in asm.children]}")
    print(f"Bounding box: {asm.bounding_box()}")

    show_in_viewer(asm, names=["form_beams"])


if __name__ == "__main__":
    main()
