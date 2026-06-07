import numpy as np
import pytest

from wing_design.beams.stock_catalog import select_stock_sizes


CAT = [0.010, 0.015, 0.020, 0.025]


def test_k1_uses_one_size_covering_max():
    req = [0.010, 0.012, 0.020]
    sel = select_stock_sizes(req, CAT, [1.0, 1.0, 1.0], rho=1550.0, max_distinct_sizes=1)
    assert len(sel.distinct_sizes) == 1
    assert np.allclose(sel.assigned_radii, 0.020)
    assert np.all(sel.assigned_radii >= np.array(req) - 1e-12)


def test_k2_minimizes_mass():
    req = [0.010, 0.012, 0.020]
    L = [1.0, 1.0, 1.0]
    sel2 = select_stock_sizes(req, CAT, L, rho=1550.0, max_distinct_sizes=2)
    sel1 = select_stock_sizes(req, CAT, L, rho=1550.0, max_distinct_sizes=1)
    assert len(sel2.distinct_sizes) <= 2
    assert np.all(sel2.assigned_radii >= np.array(req) - 1e-12)
    assert sel2.mass_kg <= sel1.mass_kg
    assert np.allclose(sorted(sel2.assigned_radii), [0.015, 0.015, 0.020])


def test_catalog_too_small_raises():
    with pytest.raises(ValueError):
        select_stock_sizes([0.030], [0.010, 0.020], [1.0], rho=1550.0, max_distinct_sizes=1)
