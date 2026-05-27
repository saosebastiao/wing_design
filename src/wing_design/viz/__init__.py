"""Visualization helpers for the OCP CAD Viewer VS Code extension.

Requires the **OCP CAD Viewer** extension by bernhard-42 installed in VS Code,
with the viewer running on port 3939 (the default). Start it from the command
palette: "OCP CAD Viewer: Open Viewer".

This module is a thin wrapper around `ocp_vscode.show` so examples can call
`show_in_viewer(obj)` without worrying about whether the viewer is up.
"""
from __future__ import annotations

import socket
import warnings
from typing import Any


def _viewer_alive(port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def show_in_viewer(*objects: Any, names: list[str] | None = None, port: int = 3939, **kwargs: Any) -> bool:
    """Send build123d / cadquery objects to the OCP CAD Viewer.

    Returns True if the viewer received the call, False (with a hint printed) if
    the viewer wasn't reachable. Importing this never fails even when the
    extension isn't installed.
    """
    if not _viewer_alive(port):
        print(
            f"OCP CAD Viewer not reachable on port {port} — skipping show().\n"
            "  - Install the 'OCP CAD Viewer' VS Code extension by bernhard-42.\n"
            "  - In VS Code, run 'OCP CAD Viewer: Open Viewer' from the command palette."
        )
        return False

    try:
        from ocp_vscode import set_port, show
    except ImportError:
        print("ocp_vscode not installed; run `uv add ocp_vscode`.")
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        set_port(port)
        try:
            if names is not None:
                show(*objects, names=names, **kwargs)
            else:
                show(*objects, **kwargs)
            return True
        except Exception as exc:
            print(f"OCP CAD Viewer rejected the payload: {exc}")
            return False
