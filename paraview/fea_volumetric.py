"""Visualize examples/04_volumetric_fea.py output: one fea_<case>.vtu.

Volumetric tet FEA on the wing solid. Coloring:
  * Cell von_mises (MPa-equivalent here is in Pa — see scalar bar units)
  * Optional warp by displacement vector × factor.

Usage (default opens exports/fea_survival.vtu):

  pvpython paraview/fea_volumetric.py [exports/fea_<case>.vtu]
  pvpython paraview/fea_volumetric.py --screenshot survival.png
"""
from __future__ import annotations

from paraview.simple import (  # type: ignore[import-not-found]
    Show, GetActiveView, WarpByVector,
)

from _common import default_argparser, open_vtu, color_by, background, finish


def main() -> None:
    parser = default_argparser(
        "fea_survival.vtu",
        doc="Volumetric tet FEA σ_VM with displacement warp.",
    )
    parser.add_argument(
        "--warp", type=float, default=10.0,
        help="Displacement warp factor (default: 10x). Set 0 to disable.",
    )
    args = parser.parse_args()

    reader = open_vtu(args.vtu)
    src = reader
    if args.warp > 0:
        warp = WarpByVector(Input=reader)
        warp.Vectors = ["POINTS", "displacement"]
        warp.ScaleFactor = args.warp
        src = warp

    display = Show(src)
    display.Representation = "Surface"
    color_by(display, "POINTS", "nodal_von_mises")

    view = GetActiveView()
    background(view)
    finish(args, view)


if __name__ == "__main__":
    main()
