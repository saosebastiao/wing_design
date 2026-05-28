"""Visualize examples/05_stress_lines.py output: stress_lines.vtu.

Volumetric streamlines (line cells) traced through the σ field for
each (case × tack × family). Color by `sigma_along_MPa` to see where
each candidate beam carries load; filter by `family` (0=tension,
1=shear, 2=compression) and/or `case` (see the case-index legend
printed by the example).

Usage:
  pvpython paraview/stress_lines.py
  pvpython paraview/stress_lines.py --family 0          # tension only
  pvpython paraview/stress_lines.py --color family       # categorical
"""
from __future__ import annotations

from paraview.simple import Show, GetActiveView  # type: ignore[import-not-found]

from _common import (
    default_argparser, open_vtu, color_by, set_thick_lines,
    threshold_between, background, finish,
)


def main() -> None:
    parser = default_argparser(
        "stress_lines.vtu", doc="Volumetric stress-line streamlines.",
    )
    parser.add_argument("--color", default="sigma_along_MPa",
                        choices=["sigma_along_MPa", "family", "case", "mirror"])
    parser.add_argument("--family", type=int, default=None,
                        help="Show only this family (0=σ1 tension, 1=σ2 shear, 2=σ3 compression).")
    args = parser.parse_args()

    reader = open_vtu(args.vtu)
    src = reader
    if args.family is not None:
        src = threshold_between(reader, "CELLS", "family", args.family, args.family)

    display = Show(src)
    color_by(display, "CELLS", args.color)
    set_thick_lines(display, width=2.5)

    view = GetActiveView()
    background(view)
    finish(args, view)


if __name__ == "__main__":
    main()
