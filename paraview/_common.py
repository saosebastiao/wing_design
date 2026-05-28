"""Shared helpers for the per-export ParaView 6.x visualization scripts.

These scripts are meant to be run as

    /Applications/ParaView-6.1.1.app/Contents/bin/pvpython paraview/<name>.py

(optionally with `--screenshot foo.png` for headless capture). They use
the `paraview.simple` API that ships inside the ParaView app bundle, so
*they must be run with pvpython / pvbatch, not the project's uv-managed
Python interpreter* — the project venv has no paraview module.

The helpers in this module deliberately stay close to vanilla
`paraview.simple` so the per-export scripts read as recipes you can
copy into the GUI's Python shell.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from paraview.simple import (  # type: ignore[import-not-found]
    GetActiveView,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    GetScalarBar,
    Interact,
    Render,
    ResetCamera,
    SaveScreenshot,
    Show,
    XMLUnstructuredGridReader,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = REPO_ROOT / "exports"


def default_argparser(default_filename: str, *, doc: str = "") -> argparse.ArgumentParser:
    """Boilerplate CLI: positional VTU path (defaults to exports/<filename>),
    --screenshot for headless PNG, --size for image resolution."""
    p = argparse.ArgumentParser(description=doc)
    p.add_argument(
        "vtu",
        nargs="?",
        default=str(EXPORTS_DIR / default_filename),
        help=f"VTU file to load (default: exports/{default_filename})",
    )
    p.add_argument(
        "--screenshot",
        metavar="PATH",
        help="Save a PNG to PATH instead of opening an interactive window.",
    )
    p.add_argument(
        "--size",
        default="1600x900",
        help="WxH for screenshot or initial window (default: 1600x900).",
    )
    return p


def parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def open_vtu(path: str):
    """Load an XML VTU file; return the source proxy."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path!r} not found. Run the corresponding example first "
            f"(`just example <NN_name>`)."
        )
    return XMLUnstructuredGridReader(FileName=[path])


def color_by(display, association: str, field: str, *,
             component: str | None = None,
             preset: str = "Cool to Warm",
             rescale: bool = True,
             log_scale: bool = False) -> None:
    """Color the display by a named field. `association` is 'POINTS' or 'CELLS'."""
    from paraview.simple import ColorBy  # local import to keep import order safe
    if component is None:
        ColorBy(display, (association, field))
    else:
        ColorBy(display, (association, field, component))
    if rescale:
        display.RescaleTransferFunctionToDataRange(True, False)
    lut = GetColorTransferFunction(field)
    lut.ApplyPreset(preset, True)
    if log_scale:
        lut.MapControlPointsToLogSpace()
        lut.UseLogScale = 1
    view = GetActiveView()
    display.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(lut, view)
    bar.Title = field
    bar.ComponentTitle = "" if component is None else component


def glyph_vectors(source, field_name: str, *, scale_factor: float = 0.05,
                  glyph_type: str = "Arrow", every_nth: int = 1, color: str | None = None):
    """Add a Glyph filter coloring/orienting by a vector field on the source.

    Returns (glyph_proxy, display_proxy).
    """
    from paraview.simple import Glyph
    g = Glyph(Input=source, GlyphType=glyph_type)
    g.OrientationArray = ["CELLS", field_name]
    g.ScaleArray = ["CELLS", field_name]
    g.ScaleFactor = scale_factor
    g.GlyphMode = "Every Nth Point" if every_nth > 1 else "All Points"
    if every_nth > 1:
        g.Stride = every_nth
    d = Show(g)
    if color is not None:
        d.AmbientColor = _named_color(color)
        d.DiffuseColor = _named_color(color)
    return g, d


def threshold_between(source, field_assoc: str, field_name: str,
                       lo: float, hi: float):
    """Apply a Threshold filter; ParaView 6.x API uses LowerThreshold/UpperThreshold."""
    from paraview.simple import Threshold
    t = Threshold(Input=source)
    t.Scalars = [field_assoc, field_name]
    t.LowerThreshold = lo
    t.UpperThreshold = hi
    t.ThresholdMethod = "Between"
    return t


def set_thick_lines(display, width: float = 2.5) -> None:
    display.LineWidth = width
    display.RenderLinesAsTubes = 1


def background(view, rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
    view.Background = list(rgb)


def _wing_side_view(view) -> None:
    """Set a camera that shows the 5 m wingsail along its span.

    Geometry-frame convention: chord +X, normal +Y, span +Z, root at z=0.
    We look from the chord +X side toward the wing, +Y up, with the
    camera offset enough to fit the ~5 m span.
    """
    view.CameraPosition = [6.0, 5.0, -3.0]
    view.CameraFocalPoint = [0.5, 0.0, 2.5]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = 0
    ResetCamera(view)


def finish(args, view=None) -> None:
    """Render + either save screenshot or open the interactive window."""
    if view is None:
        view = GetActiveView()
    _wing_side_view(view)
    Render(view)
    if args.screenshot:
        w, h = parse_size(args.size)
        SaveScreenshot(args.screenshot, view, ImageResolution=[w, h])
        print(f"Wrote {args.screenshot}")
    else:
        print("Opening interactive window. Close it to exit.")
        Interact()


# ---------------------------------------------------------------------------
# tiny named-color helper (avoid pulling in matplotlib in the pv environment)
# ---------------------------------------------------------------------------

_NAMED_COLORS = {
    "black":  (0.0, 0.0, 0.0),
    "white":  (1.0, 1.0, 1.0),
    "red":    (0.85, 0.1, 0.1),
    "green":  (0.1, 0.7, 0.1),
    "blue":   (0.1, 0.3, 0.85),
    "orange": (1.0, 0.55, 0.0),
    "gray":   (0.55, 0.55, 0.55),
    "purple": (0.55, 0.0, 0.7),
}


def _named_color(name: str) -> list[float]:
    if name not in _NAMED_COLORS:
        raise ValueError(f"unknown color {name!r}; pick one of {sorted(_NAMED_COLORS)}")
    return list(_NAMED_COLORS[name])
