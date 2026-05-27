"""Phase-3 baseline: size the tapered CFRP tube spar against the full load envelope."""
from __future__ import annotations

from wing_design.aero import DESIGN_CASES, build_airplane, sweep_envelope
from wing_design.geometry import WingSpec
from wing_design.materials import T700_EPOXY
from wing_design.structural import size_tube_spar


def main() -> None:
    spec = WingSpec()
    airplane = build_airplane(spec)
    envelope = sweep_envelope(airplane, DESIGN_CASES)
    sizing = size_tube_spar(spec, envelope, material=T700_EPOXY)

    print(f"Material:           {sizing.material.name}")
    print(f"Root diameter:      {sizing.diameter_root_m * 1000:.1f} mm")
    print(f"Tip  diameter:      {sizing.diameter_tip_m * 1000:.1f} mm")
    print(f"Root wall thk:      {sizing.wall_root_m * 1000:.2f} mm")
    print(f"Tip  wall thk:      {sizing.wall_tip_m * 1000:.2f} mm")
    print(f"Max axial stress:   {sizing.max_stress_Pa / 1e6:.1f} MPa")
    print(f"Max tip deflection: {sizing.max_tip_deflection_m * 1000:.1f} mm")
    print(f"Governing case:     {sizing.governing_case}")
    print(f"Spar mass:          {sizing.mass_spar_kg:.2f} kg")
    print(f"Stub mass:          {sizing.mass_stub_kg:.2f} kg")
    print(f"Total:              {sizing.mass_total_kg:.2f} kg")


if __name__ == "__main__":
    main()
