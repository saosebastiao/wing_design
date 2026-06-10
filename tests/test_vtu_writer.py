import numpy as np
from xml.etree import ElementTree

from wing_design.viz.vtu import write_vtu_triangles


def test_vtu_roundtrip_structure(tmp_path):
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.5]])
    tris = np.array([[0, 1, 2], [1, 3, 2]])
    p = write_vtu_triangles(
        tmp_path / "t.vtu", nodes, tris,
        point_data={"mode": np.arange(12, dtype=float).reshape(4, 3)},
        cell_data={"util": np.array([0.5, 1.0])},
    )
    root = ElementTree.parse(p).getroot()
    piece = root.find(".//Piece")
    assert piece.get("NumberOfPoints") == "4"
    assert piece.get("NumberOfCells") == "2"
    conn = root.find(".//Cells/DataArray[@Name='connectivity']").text.split()
    assert [int(x) for x in conn] == [0, 1, 2, 1, 3, 2]
    mode = root.find(".//PointData/DataArray[@Name='mode']")
    assert mode.get("NumberOfComponents") == "3"
    util = root.find(".//CellData/DataArray[@Name='util']")
    assert [float(x) for x in util.text.split()] == [0.5, 1.0]
