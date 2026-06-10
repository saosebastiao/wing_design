"""Named-block design vector for the sizing NLP (P.0, Toolbox #1).

Replaces the hand-indexed ``x = [r_group(G) | t_band(B) | f0(L) | f45(L)]`` layout
with declared blocks: each block has a name and a width, and every consumer asks
for slices by name instead of recomputing offsets. P.1 (hollow members) appends
``("r_tube", K), ("t_wall", K)`` without touching any existing unpack site.
"""
from __future__ import annotations

import numpy as np


class DesignVector:
    """Ordered named blocks of a flat design vector.

    >>> dv = DesignVector(("r_group", 5), ("t_band", 2), ("f0", 1), ("f45", 1))
    >>> dv.nx
    9
    >>> dv.slice("t_band")
    slice(5, 7, None)
    >>> dv.unpack(np.arange(9.0))["f0"]
    array([7.])
    """

    def __init__(self, *blocks: tuple[str, int]):
        if not blocks:
            raise ValueError("DesignVector needs at least one block")
        names = [b[0] for b in blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate block names: {names}")
        self._blocks: list[tuple[str, int]] = []
        self._slices: dict[str, slice] = {}
        pos = 0
        for name, width in blocks:
            width = int(width)
            if width < 0:
                raise ValueError(f"block {name!r} has negative width {width}")
            self._blocks.append((name, width))
            self._slices[name] = slice(pos, pos + width)
            pos += width
        self.nx = pos

    @property
    def blocks(self) -> tuple[tuple[str, int], ...]:
        return tuple(self._blocks)

    def __contains__(self, name: str) -> bool:
        return name in self._slices

    def width(self, name: str) -> int:
        s = self._slices[name]
        return s.stop - s.start

    def slice(self, name: str) -> slice:
        return self._slices[name]

    def get(self, x: np.ndarray, name: str) -> np.ndarray:
        """View of block ``name`` in ``x`` (no copy)."""
        return np.asarray(x)[self._slices[name]]

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """All blocks as name -> view (no copies)."""
        x = np.asarray(x)
        if x.shape != (self.nx,):
            raise ValueError(f"x must have shape ({self.nx},), got {x.shape}")
        return {name: x[s] for name, s in self._slices.items()}

    def pack(self, parts: dict[str, np.ndarray]) -> np.ndarray:
        """Flat x from per-block arrays; every block must be present and sized."""
        x = np.empty(self.nx)
        missing = [n for n, _ in self._blocks if n not in parts]
        if missing:
            raise ValueError(f"missing blocks: {missing}")
        for name, s in self._slices.items():
            a = np.asarray(parts[name], dtype=float)
            if a.shape != (s.stop - s.start,):
                raise ValueError(
                    f"block {name!r} expects shape ({s.stop - s.start},), got {a.shape}")
            x[s] = a
        return x

    def scatter(self, name: str, values: np.ndarray, out: np.ndarray | None = None
                ) -> np.ndarray:
        """(nx,) vector with ``values`` in block ``name`` and zeros elsewhere
        (or added into ``out`` if given) — the gradient-assembly helper."""
        g = np.zeros(self.nx) if out is None else out
        g[self._slices[name]] += np.asarray(values, dtype=float)
        return g
