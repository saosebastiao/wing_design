# Tip gusset in the sizing model: design

**Date:** 2026-06-09
**Status:** Approved (brainstorming → ready for implementation plan)

## Goal

Optionally bind all beam tips with a rigid gusset (a stiff connector-beam clique among
the tip-ring nodes) inside the CLT laminate sizing solve, so the twist-governed sized
designs can relax the binding tip-twist constraint and shed mass. Opt-in, massless
(stiffness-only), composes with the analytic Jacobian / symmetry / banding.

## Motivation

The sized designs are tip-twist-governed (twist pinned at the 5° limit; buckling util
1.0). The earlier investigation (`solve_beam_shell_tip_coupled`, isotropic, not in the
sizing loop) showed a rigid tip joint cuts tip twist ~50×. Wiring that coupling into the
laminate sizer lets the optimizer trade the freed twist budget for thinner material — a
likely net mass win, and a physically plausible structure (a titanium tip fitting all
beams seat into).

## Decisions (from brainstorming)

- **Opt-in**, default off (`tip_gusset_radius=None`) → byte-identical to today.
- **Massless, stiffness-only** — the gusset adds stiffness to `K`, no mass to the total
  (the real fitting's mass is a separate fixed item, deferred).
- **Fixed representative radius** — one connector-beam radius large enough to be
  effectively rigid (the study saturated early). Not a design variable.
- **Assembled into K, excluded from force recovery** — keeps `n_design` clean (no
  per-element slicing downstream).

## Design

### 1. Carry the gusset on the model — `beams/shell_model.py`

`build_beam_shell_model(..., tip_gusset_radius: float | None = None)`. When set:
`tip_gusset_elements = tip_clique_elements(tip_nodes)` (all unordered pairs among the
tip-ring nodes; from `beams.tip_coupling`), stored on the model along with
`tip_gusset_radius`. Default None ⇒ both fields None (off). Add `model_with_tip_gusset(
model, radius) -> BeamShellModel` (via `dataclasses.replace`) mirroring
`model_with_helix`. New `BeamShellModel` fields: `tip_gusset_elements: np.ndarray | None
= None`, `tip_gusset_radius: float | None = None` (additive, defaulted → backward
compatible).

### 2. Assemble into K, exclude from recovery — `structural/beam_shell.py`

Add optional `gusset_elements: np.ndarray | None = None` and `gusset_section:
BeamSection | None = None` to `solve_beam_shell_laminate` and
`solve_beam_shell_laminate_factored` (and the shared `_assemble_beam_shell_laminate`
helper). When given, the gusset elements are assembled into the global `K` exactly like
beam elements (using `E_beam/G_beam` and `gusset_section`), **but are NOT included in the
beam internal-force recovery** — so `FrameResult.axial_force/bending_moment/torsion`
stay length `n_design = len(beam_elements)`. The factored solve likewise keeps
`transforms/klocals` for the design beams only (the gusset contributes to `K_ff`/`u` but
carries no design sensitivity).

### 3. Sizer wiring — `beams/laminate_sizing.py`

When `model.tip_gusset_elements is not None`, the sizer passes `gusset_elements =
model.tip_gusset_elements` and `gusset_section = BeamSection.circular(
model.tip_gusset_radius)` to every laminate solve (both the FD `evaluate` path and the
analytic factored path). Nothing else changes:
- constraint values: `von_mises_per_element`, buckling, defl/twist are over the design
  beams / tip nodes exactly as now (forces are design-only; tip nodes unchanged).
- **analytic Jacobian:** the gusset is constant stiffness ⇒ `∂K/∂x` has no gusset term;
  the sensitivity loops only over the `n_design` beams; the factored `K` (and thus `u`,
  the adjoint solves) include the gusset. So the adjoint gradients compose unchanged.
- **mass:** unchanged — only design beams + skin (gusset massless).

### 4. Example — `examples/39_tip_gusset.py`

Re-size the medium wingsail twice — `tip_gusset_radius=None` vs a representative radius —
under identical constraints (span datum, buckling SF 1.5, banded skin). Print mass +
split, tip twist, governing constraint (buckling util vs twist), beam radii range, and
the Δmass / Δtwist. Run timed (orchestrator). Expectation: gusset-on tip twist drops well
below the limit (no longer binding) and total mass drops as the optimizer thins material.

### 5. Testing — `tests/beams/test_tip_gusset.py`

- Model carries the clique when `tip_gusset_radius` set; `model_with_tip_gusset`
  round-trip; off ⇒ fields None.
- **Fixed-design twist drop:** at one fixed design + load, `solve` with the gusset gives
  tip twist ≪ without (large reduction), confirming the coupling works in the laminate
  solve.
- **Sizer feasible + not heavier:** a small sizing with the gusset converges feasible and
  `mass ≤ no-gusset mass + tol` (twist relief should not cost mass).
- **Analytic Jacobian composes:** one FD-validation of a constraint gradient on a
  gusset-on model (`use_analytic_jacobian=True`) — analytic ≈ central FD.
- **Default byte-identical:** no-gusset path unchanged; full suite green.

## Components / file map

| File | Responsibility | Change |
|---|---|---|
| `beams/shell_model.py` | `tip_gusset_radius` param + `tip_gusset_elements` field + `model_with_tip_gusset` | modify |
| `structural/beam_shell.py` | optional gusset assembly (into K, excluded from recovery) in both laminate solves + shared helper | modify |
| `beams/laminate_sizing.py` | pass gusset to the solve (FD + factored paths) when present | modify |
| `beams/__init__.py` | export `model_with_tip_gusset` | modify |
| `examples/39_tip_gusset.py` | gusset-off-vs-on sized comparison | **new** |
| `tests/beams/test_tip_gusset.py` | tests | **new** |
| `docs/plan.md` | finding + Decisions-log row (at close, with run numbers) | modify |

## Success criteria

- Opt-in tip gusset (default off → byte-identical) as a rigid massless stiffness coupling
  in the laminate sizer; composes with the analytic Jacobian, symmetry, banding, datum,
  Tsai-Wu.
- At a fixed design, the gusset drastically reduces tip twist (sanity vs the ~50×
  investigation).
- A measured medium-wingsail gusset-off-vs-on comparison showing the twist constraint
  relaxing and a mass reduction (or honest report if not).
- Full suite green.

## Out of scope (deferred)

- Gusset mass / stress accounting (massless now).
- Gusset stiffness as a design variable, or a stiffness sweep in tests.
- The isotropic `solve_beam_shell` path and the standalone `solve_beam_shell_tip_coupled`
  investigation wrapper (left as-is).
- Re-running every historical headline with the gusset.
