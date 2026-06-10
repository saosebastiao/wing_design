"""Minimal dependency-free VTU (VTK XML unstructured grid) writer.

Enough for the project's FEA-field export convention (exports/*.vtu, viewed via
`just view` / `just shot`): triangle meshes with per-point and per-cell arrays.
ASCII format — files are small at sizing-mesh resolution. meshio left the venv at
some point (example 14 still imports it and is broken); this writer removes that
dependency for new examples.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _fmt_array(name: str, data: np.ndarray) -> str:
    a = np.asarray(data, dtype=float)
    ncomp = 1 if a.ndim == 1 else a.shape[1]
    body = "\n".join(" ".join(f"{v:.9g}" for v in np.atleast_1d(row)) for row in a)
    return (f'<DataArray type="Float64" Name="{name}" '
            f'NumberOfComponents="{ncomp}" format="ascii">\n{body}\n</DataArray>')


def write_vtu_triangles(
    path: str | Path,
    nodes: np.ndarray,
    tris: np.ndarray,
    *,
    point_data: dict[str, np.ndarray] | None = None,
    cell_data: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write an ASCII .vtu of a triangle mesh with optional point/cell arrays."""
    path = Path(path)
    nodes = np.asarray(nodes, dtype=float)
    tris = np.asarray(tris, dtype=int)
    n_pts, n_cells = nodes.shape[0], tris.shape[0]

    pts = "\n".join(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}" for p in nodes)
    conn = "\n".join(f"{t[0]} {t[1]} {t[2]}" for t in tris)
    offsets = " ".join(str(3 * (i + 1)) for i in range(n_cells))
    types = " ".join("5" for _ in range(n_cells))   # VTK_TRIANGLE = 5

    pd = "\n".join(_fmt_array(k, v) for k, v in (point_data or {}).items())
    cd = "\n".join(_fmt_array(k, v) for k, v in (cell_data or {}).items())

    path.write_text(f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
<UnstructuredGrid>
<Piece NumberOfPoints="{n_pts}" NumberOfCells="{n_cells}">
<Points>
<DataArray type="Float64" NumberOfComponents="3" format="ascii">
{pts}
</DataArray>
</Points>
<Cells>
<DataArray type="Int64" Name="connectivity" format="ascii">
{conn}
</DataArray>
<DataArray type="Int64" Name="offsets" format="ascii">
{offsets}
</DataArray>
<DataArray type="UInt8" Name="types" format="ascii">
{types}
</DataArray>
</Cells>
<PointData>
{pd}
</PointData>
<CellData>
{cd}
</CellData>
</Piece>
</UnstructuredGrid>
</VTKFile>
""")
    return path
