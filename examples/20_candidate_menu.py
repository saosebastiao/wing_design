"""Phase 1C: build the candidate menu from the wing and write a VTU of the beam
centerlines for ParaView inspection.

Run: just example 20_candidate_menu
"""
from pathlib import Path

import numpy as np
import meshio

from wing_design import default_scenario
from wing_design.generative import build_candidate_menu

EXPORT = Path("exports")


def main() -> None:
    params = default_scenario()
    menu = build_candidate_menu(params)

    print(f"beams:            {len(menu.beams)}")
    print(f"cross-sections:   {len(menu.cross_sections)}")
    print(f"coverage targets: {len(menu.coverage_targets)}")
    print(f"conflict tuples:  {len(menu.conflicts.forbidden)}")
    for b in menu.beams:
        print(f"  beam {b.id}: {b.start_kind.value}->{b.end_kind.value} "
              f"host={b.host_id} mirror={b.mirror_id} covers={b.covers}")

    # VTU: all beam centerlines as polylines.
    points = []
    lines = []
    for b in menu.beams:
        base = len(points)
        for p in b.control_points:
            points.append(p)
        for k in range(len(b.control_points) - 1):
            lines.append([base + k, base + k + 1])
    EXPORT.mkdir(exist_ok=True)
    out = EXPORT / "candidate_menu.vtu"
    meshio.write_points_cells(
        str(out), points=np.array(points, dtype=float), cells=[("line", np.array(lines))]
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
