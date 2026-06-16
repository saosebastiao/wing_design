# M.2 Ply discretization (skin laminate → integer plies): design

## Measured basis

The validated operational headline (1021.6 kg, `runs/multistart_v2_best.npz`) has a
**continuous** skin laminate: single band, faces ≈ 1.15 mm (~4.6 plies at 0.25 mm),
layup f0/f45/f90 = **0 / 1.00 / 0 (pure ±45)**. This is **not buildable**: (a)
non-integer ply counts; (b) the manufacturing concept *mandates* the skin's first
layer be a hoop (chordwise / 90°) wind for the binding prestress, but the optimum
carries **zero 90°**. M.2 produces the integer-ply, hoop-floored **as-built**
laminate and reports the ideal→as-built mass delta — the first of the Phase-M
"as-built number" items (the plan calls M.1+M.2 the gate for any as-built claim).

## Decisions locked (brainstorm 2026-06-16)

1. **Round-and-repair**, NOT a constrained re-optimize: discretize the existing
   sized design, then FEA re-verify + repair upward to feasibility (the plan's
   "bump-and-reverify" pattern). A constrained MINLP re-optimize is a later
   refinement if the rounding penalty proves large.
2. **0.25 mm/ply.** (From the backlog; expose as a config constant.)
3. **Mandatory first-layer hoop floor: n90 ≥ 1 per band** (hoop = chordwise = 90°
   in the span datum). This will perturb the pure-±45 optimum — a real
   manufacturability effect, surfaced, not hidden.
4. **Re-verify buckling at n ≥ 24** (converged cluster floor, per the CLAUDE.md
   convention) plus all closed-form constraints.
5. **Scope: skin laminate FACES only.** Sandwich-core foam-sheet catalog and
   beam-wall plies are separate, deferred.

## Module — `discretize_laminate` (new, post-processing)

New file `src/wing_design/beams/ply_discretization.py` (keep it small and
self-contained; it consumes a `LaminateSizingResult` + `LaminateSizingConfig` and
returns a discretized result + a report — it does NOT touch the sizer).

### Ply model + rounding (pure function, easy to unit-test)
`round_laminate(t_band, f0, f45, f90, *, t_ply=0.25e-3, hoop_min=1) -> PlySchedule`:
- `n = max(1, round(t_band / t_ply))` total plies.
- Split `n` into integer `(n0, n45, n90)` from `(f0, f45, f90)` by **largest-
  remainder** rounding so `n0+n45+n90 == n` exactly.
- Enforce the hoop floor: if `n90 < hoop_min`, raise `n90` to `hoop_min` and
  remove that many plies from the largest of `{n0, n45}` (so the total `n` is
  preserved at this step — repair later adds plies if the thinner laminate becomes
  infeasible). If `n` is too small to host the hoop floor + a structural ply,
  bump `n` up by 1 (a band thinner than `hoop_min+1` plies can't be built anyway).
- Return `PlySchedule(n0, n45, n90, t=n*t_ply, f0=n0/n, f45=n45/n, f90=n90/n)`.

### Re-verify + repair
`discretize_laminate(model, config, result, loads, presses)`:
1. Round every band → integer `PlySchedule`s; build a modified design (new
   `t_bands`, `f0/f45/f90_bands` from the schedules; everything else — radii,
   tube, core, rings — unchanged).
2. Re-evaluate feasibility at the modified design WITHOUT re-optimizing: the
   `maxiter=0` evaluate pattern from `runs/inspect_best.py` gives all constraint
   utilizations + mass; the **n≥24 eigen** via the `runs/mesh_converge_diag.py`
   resampling + `chain_rebuild.eigen_worst` (converged cluster floor over the
   lowest 2–3 modes).
3. **Repair loop:** while infeasible, add ONE ply to the band/orientation that
   most relieves the worst-violated constraint (heuristic: thickness for
   deflection/panel-buckling, the structural orientation ±45 for shear/buckling,
   90° already floored), rebuild, re-check. Cap iterations; if it can't repair,
   report the residual violation rather than loop forever.
4. Return: the discretized + repaired design, its mass, the per-band ply
   schedules, the ideal vs as-built mass delta, and which constraint(s) drove any
   repair.

## Output / deliverable

A run script `runs/asbuilt_m2.py` that loads the headline, runs
`discretize_laminate`, and prints: ideal mass 1021.6 kg → as-built integer-ply
mass, the per-band ply schedule (e.g. `[1×90° + 4×±45]`), the binding constraint
after discretization, the n≥24 eigen, and the % penalty from (a) ply rounding and
(b) the mandatory hoop floor separately if cheap to attribute. Record as the first
Phase-M finding (ideal vs as-built).

## Testing

- **Rounding unit tests** (`tests/beams/test_ply_discretization.py`):
  ply-count sum preserved; hoop floor enforced (`n90 ≥ 1`); an exact-multiple
  input (t = k·0.25 mm, clean fractions) is unchanged except the hoop floor;
  a pure-±45 input gets ≥1 hoop ply with the rest ±45.
- **Repair/feasibility test:** a deliberately rounded-down design is repaired to
  a feasible one (constraints satisfied) and the repair terminates.
- **Integration test:** discretize the headline; assert the as-built mass ≥ ideal
  (rounding + hoop never lighten it), finite, and the result is feasible +
  eigen-verified at n≥24.
- No new analytic gradient (post-processing) → no FD-validation test needed.

## Non-goals (YAGNI)

- Constrained re-optimize / MINLP (later, only if rounding penalty is large).
- Sandwich-core foam-sheet catalog (separate M item).
- Beam-wall ply discretization (beams are wound/RTM at continuous radii — separate).
- Windable-angle constraints (M.3), bondline/as-built adders (M.4) — later Phase-M
  items that layer on this discretized design.
