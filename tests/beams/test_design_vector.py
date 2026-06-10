import numpy as np
import pytest

from wing_design.beams.design_vector import DesignVector


def _dv():
    return DesignVector(("r_group", 5), ("t_band", 2), ("f0", 1), ("f45", 1))


def test_layout_matches_legacy_hand_indexing():
    # The sizer's historical layout: [r_group(G) | t_band(B) | f0(L) | f45(L)]
    dv = _dv()
    G, B, L = 5, 2, 1
    assert dv.nx == G + B + 2 * L
    assert dv.slice("r_group") == slice(0, G)
    assert dv.slice("t_band") == slice(G, G + B)
    assert dv.slice("f0") == slice(G + B, G + B + L)
    assert dv.slice("f45") == slice(G + B + L, G + B + 2 * L)


def test_pack_unpack_roundtrip():
    dv = _dv()
    x = np.arange(dv.nx, dtype=float)
    parts = dv.unpack(x)
    assert np.array_equal(dv.pack(parts), x)
    assert np.array_equal(parts["t_band"], [5.0, 6.0])
    assert parts["f45"][0] == 8.0


def test_unpack_returns_views():
    dv = _dv()
    x = np.zeros(dv.nx)
    dv.unpack(x)["r_group"][2] = 7.0
    assert x[2] == 7.0


def test_scatter_builds_gradient_rows():
    dv = _dv()
    g = dv.scatter("t_band", [1.5, -2.0])
    assert g.shape == (dv.nx,)
    assert g[dv.slice("t_band")].tolist() == [1.5, -2.0]
    assert np.count_nonzero(g) == 2
    g2 = dv.scatter("f0", [3.0], out=g)
    assert g2[dv.slice("f0")][0] == 3.0


def test_errors():
    dv = _dv()
    with pytest.raises(ValueError, match="shape"):
        dv.unpack(np.zeros(3))
    with pytest.raises(ValueError, match="missing"):
        dv.pack({"r_group": np.zeros(5)})
    with pytest.raises(ValueError, match="duplicate"):
        DesignVector(("a", 1), ("a", 2))
    with pytest.raises(KeyError):
        dv.slice("nope")
    assert "t_band" in dv and "nope" not in dv
    assert dv.width("f0") == 1
