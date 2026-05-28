"""Visualize examples/14_shell_wing.py output: shell_fea.vtu.

DKT+CST shell FEA on the OML triangulation. Each triangle has the
membrane stress tensor + 2-D principal stresses + 3-D principal
direction unit vectors + bending fiber stress.

Default view colors by σ_VM membrane (MPa) and glyphs `principal_dir_max`
(family-0 unit vectors) at every 4th triangle so you can read the
spar-cap layout directly off the skin.

Useful flags:
  --color sigma_VM_membrane_MPa | sigma_zz_membrane_MPa | sigma_1_MPa |
          sigma_2_MPa | skin_bending_fiber_MPa
  --glyph principal_dir_max | principal_dir_min | none
"""
from __future__ import annotations

from paraview.simple import Show, GetActiveView  # type: ignore[import-not-found]

from _common import (
    default_argparser, open_vtu, color_by, glyph_vectors, background, finish,
)


CELL_SCALARS = (
    "sigma_VM_membrane_MPa", "sigma_zz_membrane_MPa",
    "sigma_1_MPa", "sigma_2_MPa", "skin_bending_fiber_MPa", "loaded_tri",
)


def main() -> None:
    parser = default_argparser(
        "shell_fea.vtu", doc="Shell FEA membrane σ + principal directions.",
    )
    parser.add_argument("--color", default="sigma_VM_membrane_MPa",
                        choices=list(CELL_SCALARS))
    parser.add_argument("--glyph", default="principal_dir_max",
                        choices=["principal_dir_max", "principal_dir_min", "none"])
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--scale", type=float, default=0.06)
    args = parser.parse_args()

    reader = open_vtu(args.vtu)
    display = Show(reader)
    display.Representation = "Surface"
    color_by(display, "CELLS", args.color)

    if args.glyph != "none":
        glyph_vectors(reader, args.glyph,
                      scale_factor=args.scale, every_nth=args.every, color="black")

    view = GetActiveView()
    background(view)
    finish(args, view)


if __name__ == "__main__":
    main()
