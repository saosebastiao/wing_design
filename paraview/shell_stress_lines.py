"""Visualize examples/15_shell_stress_lines.py output: shell_stress_lines.vtu.

Surface streamlines on the OML — the candidate spar-cap centerlines.
Family 0 (σ_1, tension) lights up on the bottom skin and runs roughly
spanwise; family 1 (σ_2, compression) does the same on the top skin.
Pair with `shell_fea.vtu` in the same ParaView session for context.
"""
from __future__ import annotations

from paraview.simple import Show, GetActiveView  # type: ignore[import-not-found]

from _common import (
    default_argparser, open_vtu, color_by, set_thick_lines,
    threshold_between, background, finish,
)


def main() -> None:
    parser = default_argparser(
        "shell_stress_lines.vtu", doc="Phase 5e surface stress lines on the OML.",
    )
    parser.add_argument("--color", default="sigma_along_MPa",
                        choices=["sigma_along_MPa", "family", "streamline_id"])
    parser.add_argument("--family", type=int, default=None, choices=[0, 1, None],
                        help="0=σ1 tension (bottom skin), 1=σ2 compression (top skin).")
    args = parser.parse_args()

    reader = open_vtu(args.vtu)
    src = reader
    if args.family is not None:
        src = threshold_between(reader, "CELLS", "family", args.family, args.family)

    display = Show(src)
    color_by(display, "CELLS", args.color)
    set_thick_lines(display, width=3.5)

    view = GetActiveView()
    background(view)
    finish(args, view)


if __name__ == "__main__":
    main()
