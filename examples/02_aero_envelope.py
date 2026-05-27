"""Phase-2 sanity check: sweep the load case envelope and print totals + spanwise profile."""
from __future__ import annotations

import numpy as np

from wing_design.aero import DESIGN_CASES, build_airplane, sweep_envelope
from wing_design.geometry import WingSpec


def main() -> None:
    spec = WingSpec()
    airplane = build_airplane(spec)
    results = sweep_envelope(airplane, DESIGN_CASES)

    print(f"Wingsail: span={spec.span} m, root={spec.root_chord} m, tip={spec.tip_chord} m")
    print(f"           S_ref={airplane.s_ref:.3f} m^2,  AR={spec.span**2 / airplane.s_ref:.2f}\n")
    print(
        f"{'case':<14} {'AWS':>6} {'AoA':>5} {'q':>7} "
        f"{'CL':>6} {'CD':>7} {'L_factored':>11} {'D':>7} {'M_root':>9}"
    )
    print(f"{'':14} {'m/s':>6} {'deg':>5} {'Pa':>7} {'-':>6} {'-':>7} {'N':>11} {'N':>7} {'N·m':>9}")
    print("-" * 86)

    worst = None
    for r in results:
        # Root bending moment under elliptic load: M = L_eq * (4 b / 3 pi)  (centroid of half-ellipse)
        root_moment_Nm = r.factored_normal_force_N * (4.0 * r.span_m) / (3.0 * np.pi)
        if worst is None or root_moment_Nm > worst:
            worst = root_moment_Nm
            worst_name = r.case.name
        print(
            f"{r.case.name:<14} {r.case.airspeed_mps:>6.1f} {r.case.alpha_deg:>5.1f} "
            f"{r.dynamic_pressure_Pa:>7.1f} {r.CL:>6.3f} {r.CD:>7.4f} "
            f"{r.factored_normal_force_N:>11.1f} {r.drag_N:>7.1f} {root_moment_Nm:>9.1f}"
        )

    print(f"\nGoverning case for root bending: {worst_name}  (M_root = {worst:.1f} N·m)")

    # Spanwise sample of the governing case
    governing = next(r for r in results if r.case.name == worst_name)
    y = np.linspace(0.0, governing.span_m, 11)
    q = governing.distributed_normal_force(y)
    print(f"\nSpanwise normal force (elliptic) for {worst_name}:")
    print("  y [m]:   " + "  ".join(f"{v:5.2f}" for v in y))
    print("  q [N/m]: " + "  ".join(f"{v:5.1f}" for v in q))


if __name__ == "__main__":
    main()
