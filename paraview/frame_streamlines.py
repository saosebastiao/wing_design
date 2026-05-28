"""Visualize examples/06_frame_field.py output: frame_streamlines.vtu.

Streamlines retraced through the *smoothed* (BFS sign-aligned) frame
field — compare against `stress_lines.vtu` to see how much cleaner the
curves are after the sign-align pass.
"""
from __future__ import annotations

from paraview.simple import Show, GetActiveView  # type: ignore[import-not-found]

from _common import (
    default_argparser, open_vtu, color_by, set_thick_lines,
    threshold_between, background, finish,
)


def main() -> None:
    parser = default_argparser(
        "frame_streamlines.vtu",
        doc="Streamlines through the smoothed (Phase 5b) frame field.",
    )
    parser.add_argument("--color", default="sigma_along_MPa",
                        choices=["sigma_along_MPa", "family"])
    parser.add_argument("--family", type=int, default=None, choices=[0, 1, 2, None])
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
