"""Visualize examples/06_frame_field.py output: frame_phi.vtu.

Tet mesh with the Poisson-fit parametrization φ : Ω → ℝ³ and the
smoothed principal eigenvectors (e1/e2/e3_smooth) attached as nodal
vectors. Default view: color by phi_1 (the family-0 parameter), glyph
e1_smooth at every 200th tet node so the arrow field is legible.

Usage:
  pvpython paraview/frame_phi.py
  pvpython paraview/frame_phi.py --field phi_2
  pvpython paraview/frame_phi.py --eig e3 --field sigma_3_MPa
"""
from __future__ import annotations

from paraview.simple import Show, GetActiveView  # type: ignore[import-not-found]

from _common import default_argparser, open_vtu, color_by, glyph_vectors, background, finish


def main() -> None:
    parser = default_argparser(
        "frame_phi.vtu",
        doc="Frame-field parametrization φ + smoothed principal directions.",
    )
    parser.add_argument("--field", default="phi_1",
                        choices=["phi_1", "phi_2", "phi_3", "sigma_1_MPa", "sigma_3_MPa"])
    parser.add_argument("--eig", default="e1", choices=["e1", "e2", "e3", "none"],
                        help="Eigenvector to glyph (default: e1_smooth). 'none' = no glyphs.")
    parser.add_argument("--every", type=int, default=200)
    args = parser.parse_args()

    reader = open_vtu(args.vtu)
    display = Show(reader)
    display.Representation = "Surface With Edges"
    display.Opacity = 0.55
    color_by(display, "POINTS", args.field)

    if args.eig != "none":
        glyph_vectors(reader, f"{args.eig}_smooth",
                      scale_factor=0.15, every_nth=args.every, color="black")

    view = GetActiveView()
    background(view)
    finish(args, view)


if __name__ == "__main__":
    main()
