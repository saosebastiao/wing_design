"""M2-core: end-to-end deflection-driven generation. Build the candidate menu,
then enumerate CP-SAT designs by mass and gate each against the full load-case
envelope (worst case governs), exporting the lightest design that survives.

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
    generate_truss,
    wing_candidate_to_part,
)

EXPORT = Path("exports")


def _density_fn(aero, span):
    """Spanwise normal-force density for a case, clamped to the wing span."""
    return lambda z: float(aero.distributed_normal_force(min(max(z, 0.0), span)))


def main() -> None:
    params = default_scenario()
    spec = params.geometry
    menu = build_candidate_menu(params)

    # Aero per sizing case (skip 'feathered' — it carries ~no load).
    airplane = build_airplane(spec)
    case_load_fns = {}
    for case in DESIGN_CASES:
        if case.name == "feathered":
            continue
        aero = run_case_lifting_line(airplane, case,
                                     spanwise_resolution=params.aero.spanwise_resolution)
        case_load_fns[case.name] = _density_fn(aero, spec.span)

    result = generate_truss(menu, params, case_load_fns,
                            max_candidates=params.generative.top_n_designs * 8)

    print("gated frontier (ascending mass):")
    for g in result.frontier:
        v = g.verdict
        print(f"  mass={g.design.mass_kg:6.2f} kg  ratio={v.max_stress_ratio:.3f}  "
              f"tip={v.tip_deflection_m*1000:6.1f} mm  feasible={v.feasible}  "
              f"governing={v.governing_case}")

    if result.chosen is None:
        print("\nNo design in the menu survives the load envelope — enlarge the "
              "cross-section catalog or add beams.")
        return

    part = wing_candidate_to_part(result.chosen, menu)
    EXPORT.mkdir(exist_ok=True)
    export_step(part, str(EXPORT / "generated_truss.step"))
    export_stl(part, str(EXPORT / "generated_truss.stl"))
    print(f"\nchosen: mass={result.chosen.mass_kg:.2f} kg  "
          f"governed by {result.verdict.governing_case}  "
          f"(ratio={result.verdict.max_stress_ratio:.3f}, "
          f"tip={result.verdict.tip_deflection_m*1000:.1f} mm)")
    print(f"wrote {EXPORT/'generated_truss.step'} and .stl")


if __name__ == "__main__":
    main()
