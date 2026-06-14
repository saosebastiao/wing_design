# Lateral ring-frame bracing for slam survival (task #22): design

## Measured basis

The medium headline (1021.6 kg, converged, λ_cr 2.76) is **beam-buckling-governed**
at util 1.000, everything else slack (twist 0.26, defl 0.57, panel 0.71). The
re-introduced slam load case fails this topology:

- **3 g lateral** (`runs/slam.py 3`): 1567.8 kg (+53.5%), unconverged, closed-form
  "feasible" (beam-buckling util 1.001) but **eigen-REJECTED at λ_cr 0.81 < 1.0**.
- **2 g lateral** (`runs/slam.py 2`): never reached feasibility (IPOPT restoration
  stall), eigen 0.60 — inconclusive on physics, conclusive that the unbraced
  topology + closed-form recipe cannot solve the slam problem.

Two findings drive this design (findings.md, governing-physics point 9): (a) a
chord-normal inertial load has **no efficient load path** in a pure spanwise
cantilever — it needs transverse members; (b) the closed-form/foundation
beam-buckling model is **non-conservative under slam** (reports feasible at true
λ 0.81), because the governing slam mode is a longer-wavelength lateral mode the
per-element foundation model does not capture.

## Decisions locked (brainstorm 2026-06-14)

1. **Eigen strategy:** bracing first + proven closed-form in-loop buckling + the
   existing post-hoc eigen solve as the acceptance gate (λ_cr ≥ 1.5), with an
   automated SF-bump re-size loop if the gate fails. Build full analytic in-loop
   eigenvalue sensitivity (dλ/dx) ONLY if bracing + calibration cannot reach
   λ ≥ 1.5. (No eigenvalue-sensitivity code exists today; it would be a large
   standalone numerical project — deferred.)
2. **Bracing topology:** transverse **ring frames / cross-members** (slender beam
   members tying adjacent form beams at spanwise stations — length-cutting), not
   shear webs (heavier, new element type) or diagonal lattice (F.2: +83% under
   aero). Reuses the existing beam-element + radius-DV machinery.

## Model — the rings (`src/wing_design/beams/shell_model.py`)

New geometry generator, opt-in, in the core-tube pattern (the model already has a
clean hook-in at `shell_model.py:112–157`):

- Config flags on `build_beam_shell_model`: `lateral_bracing: bool = False` and
  `brace_stations: "interior" | "all" | sequence[int] = "interior"`.
- At each selected spanwise station k, emit a closed ring: elements
  `(b, k) → (b+1, k)` for b in 0..n_beams−1 (wrap last→first). One ring =
  `n_beams` elements; default stations = all interior levels (1..n_levels−2).
- Append to `beam_elements`; record indices as a new `BeamShellModel` field
  `brace_elements` (analogous to `tube_elements` / `hollow_elements`,
  `shell_model.py:41–75`). No new node coordinates (rings reuse existing
  form-beam grid nodes).
- Because braces are appended to `beam_elements`, they enter `K` and every
  `solve_beam_shell_laminate*` solve automatically (`beam_shell.py:308–364`) —
  no solver change.

## Design variable + mass (`src/wing_design/beams/laminate_sizing.py`)

- One new DV block `("brace_radius", 1)` — a single shared **solid circular-rod**
  radius (YAGNI; per-station split is a later increment). Bounds
  `[r_min, r_brace_max]`, built exactly like `r_tube`
  (`laminate_sizing.py:371–379`, bounds at `:184–199`); add `r_brace_max` to the
  config.
- Section: `BeamSection.circular(r_brace)` for every brace element.
- Mass: `Σ_brace length·π r_brace²·ρ`, added to the existing mass accumulation
  and its gradient (∂m/∂r_brace = 2π r ρ Σlength).
- Stiffness gradient: braces are beam elements, so ∂K/∂r_brace reuses
  `dkloc_dr` + the `SensCache` (`sensitivity.py`, `test_sensitivity.py:47`) — **no
  new ∂K code**.

## The crux — rings must enter the in-loop buckling constraint

If rings are only passive FEA elements, the in-loop buckling constraint won't
credit them and the optimizer will **zero `r_brace`** (as it first did the tube).
The governing slam mode is a lateral mode the per-element foundation model is
blind to, so SF calibration alone cannot pull rings in.

**Approach (chosen): augment the foundation stiffness.** The rings add discrete
lateral restraint that smears into additional elastic-foundation stiffness:

    k_eff(e) = k_skin(e) + k_ring(r_brace)

where `k_skin` is the current `kgeo · D22` term (`foundation_k` closure,
`laminate_sizing.py:426–465`) and `k_ring` is the ring's lateral restraint per
unit length contributed to the braced form-beam segments (ring bending stiffness
EI_ring ∝ r_brace⁴, divided by station spacing and tributary length — exact form
derived in the plan). The foundation buckling load
`Pcr = 2√(k_eff·EI)` (`buckling.py:114–129`) then rises with `r_brace`, giving the
optimizer an in-loop incentive to size the rings.

- The new term `∂k_ring/∂r_brace` is added to the per-element foundation chain
  (the `chains` list already threads `t_band`/`f0`/`f45`/`t_core` columns;
  `laminate_sizing.py:454–464`). It is the **one genuinely new analytic
  gradient** and gets a central-difference FD-validation test (rel-err ≤ 1e-6,
  tiny mesh) per `test_sensitivity.py:71–100` BEFORE it is trusted.
- The **post-hoc eigen solve is the honest backstop**: if the smeared-foundation
  credit under-models the global mode, the eigen gate (below) catches it.

**Fallback (only if the gate keeps failing):** derive a dedicated closed-form
*global* lateral-buckling load instead of the smeared term, or escalate to the
deferred analytic in-loop eigenvalue constraint.

## Constraints (`constraints.py` / `laminate_sizing.py:1231–1325`)

- **brace_vm**: von-Mises stress utilization on the brace elements — a new
  `ConstraintSpec` reusing the `beam_vm` pipeline (braces are beam elements);
  shadow `("limit", "sigma_allow_Pa", …)`. KS-aggregated when `ks_rho` is set.
- **brace_validity / bounds**: handled by the DV bounds (no separate spec needed
  for a solid rod).
- Brace **self-buckling** is deferred — added only if the post-hoc eigen mode
  shows brace buckling (rings are short; unlikely to govern).

## Eigen gate + SF-bump loop (`runs/` wrapper)

Post-hoc verification stays `eigen_worst` (`chain_rebuild.py:81–125`,
sandwich/section-aware). Wrap the slam re-probe in an automated loop:

1. Size with the closed-form constraints (incl. augmented `k_eff`) at the current
   `buckling_safety_factor`.
2. Run `eigen_worst` over the full slam envelope.
3. If λ_cr ≥ 1.5 → accept (converged + feasible + gate passed). Else raise the
   in-loop `buckling_safety_factor` by a step (e.g. ×1.15), warm-restart, repeat
   up to a small cap.

This is the V.3/V.3b/V.3c "closed-form, eigen-calibrated" pattern, automated.

## Run plan

- Re-probe **3 g first** (hard case) on the braced topology, warm-started from the
  3 g geometry (`runs/slam_3g.npz`, already in the lateral-loaded regime), single
  worker (~18 GiB peak, memory-budget rule).
- Then **2 g** to place the survivability boundary.
- Success criterion: **converged AND feasible AND λ_cr ≥ 1.5**, masses reported
  with measured wall-clock and peak RSS, per house rules.

## Milestone artifacts

Once a survivor exists (new headline under the slam envelope): a new ring-loft
CAD builder (`build.py`, analogous to `build_sized_circular_beams:113–141`) +
worst-mode VTU, exported and recorded per the milestone convention. Until then,
the run npz + utilizations are the record.

## Testing

- FD-validation test for `∂k_ring/∂r_brace` (central difference, rel-err ≤ 1e-6,
  tiny mesh) — the single new gradient.
- A model-construction test: `lateral_bracing=True` produces the expected ring
  element count (`n_rings · n_beams`) on the right nodes; default off is
  byte-identical to today (regression guard).
- A sizer smoke test: a tiny braced problem runs, sizes a nonzero `r_brace` when
  the foundation credit is active, and `design_vector_from_result` round-trips the
  new block.

## Non-goals (YAGNI)

- Per-station or hollow brace sections (single solid shared radius first).
- Analytic in-loop eigenvalue sensitivity (deferred unless the gate can't pass).
- Shear webs / diagonal lattice (different topologies, not this increment).
- Wrapped-joint allowable / assembly-order manufacturability checks (M-phase).
- Brace self-buckling constraint (added only if the eigen mode demands it).
