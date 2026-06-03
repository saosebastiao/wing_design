"""Lump a spanwise distributed force onto frame nodes for the gate.

The aero solver gives a spanwise normal-force density; the frame gate wants
nodal point loads. We assign each in-span node a tributary share of the line
load (half the distance to each neighbor by z), in a fixed global direction.
This is the milestone's approximate aero->structure coupling (spec §7.3).
"""
from __future__ import annotations

import numpy as np


def lump_spanwise_force_to_nodes(frame, density_fn, *, z_min, z_max, direction):
    """Return {node_index: (fx, fy, fz)} lumping a line load onto in-span nodes.

    density_fn(z) -> N/m. Each node within [z_min, z_max] gets density(z) times
    its tributary length (midpoints to its sorted z-neighbors), times `direction`
    (a unit-ish 3-vector). Nodes are grouped by z so the total integrates the
    density over the span.
    """
    direction = np.asarray(direction, dtype=float)
    z = frame.coords[:, 2]
    in_span = [i for i in range(len(z)) if z_min - 1e-9 <= z[i] <= z_max + 1e-9]
    in_span.sort(key=lambda i: z[i])
    # NOTE: returns {} if no node falls in [z_min, z_max]; callers that expect a
    # loaded structure should check for an empty result.

    loads = {}
    for rank, i in enumerate(in_span):
        zi = z[i]
        lo = z[in_span[rank - 1]] if rank > 0 else z_min
        hi = z[in_span[rank + 1]] if rank < len(in_span) - 1 else z_max
        tributary = 0.5 * (zi - lo) + 0.5 * (hi - zi)
        f = float(density_fn(zi)) * tributary
        vec = f * direction
        loads[i] = (float(vec[0]), float(vec[1]), float(vec[2]))
    return loads
