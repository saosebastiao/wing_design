"""Phase 1C: end-to-end thin slice. Build the candidate menu, let CP-SAT select
minimum-mass designs, judge the best with the frame gate under the nominal load
case, and export the selected truss to STEP/STL.

Run: just example 21_generate_truss
"""
from pathlib import Path

from build123d import export_step, export_stl

from wing_design import default_scenario
from wing_design.aero.cases import DESIGN_CASES
from wing_design.aero.loads import run_case_lifting_line
from wing_design.aero.model import build_airplane
from wing_design.generative import (
    build_candidate_menu,
    build_frame,
    lump_spanwise_force_to_nodes,
    solve_designs,
    solve_frame,
    wing_candidate_to_part,
)
from wing_design.generative.gate import tip_node_indices

EXPORT = Path("exports")


def main() -> None:
    params = default_scenario()
    menu = build_candidate_menu(params)

    designs = solve_designs(menu, params.generative,
                            top_n=params.generative.top_n_designs)
    print(f"CP-SAT returned {len(designs)} candidate design(s)")
    if not designs:
        print("no feasible design — check menu constraints")
        return

    # Aero for the nominal case -> spanwise normal-force density.
    spec = params.geometry
    airplane = build_airplane(spec)
    case = DESIGN_CASES[0]
    aero = run_case_lifting_line(airplane, case,
                                 spanwise_resolution=params.aero.spanwise_resolution)

    chosen = None
    for d in designs:
        frame = build_frame(d, menu)
        loads = lump_spanwise_force_to_nodes(
            frame,
            lambda z: float(aero.distributed_normal_force(min(max(z, 0.0), spec.span))),
            z_min=0.0, z_max=spec.span, direction=(0.0, 1.0, 0.0),
        )
        result = solve_frame(frame, params, loads, governing_case=case.name)
        print(f"  design mass={d.mass_kg:.2f} kg  ratio={result.max_stress_ratio:.3f} "
              f"tip={result.tip_deflection_m*1000:.1f} mm  feasible={result.feasible}")
        if result.feasible:
            chosen = (d, result)
            break

    if chosen is None:
        print("no design passed the gate under the nominal case")
        return

    design, result = chosen
    part = wing_candidate_to_part(design, menu)
    EXPORT.mkdir(exist_ok=True)
    export_step(part, str(EXPORT / "generated_truss.step"))
    export_stl(part, str(EXPORT / "generated_truss.stl"))
    print(f"chosen design: mass={design.mass_kg:.2f} kg, "
          f"feasible under {result.governing_case}")
    print(f"wrote {EXPORT/'generated_truss.step'} and .stl")


if __name__ == "__main__":
    main()
