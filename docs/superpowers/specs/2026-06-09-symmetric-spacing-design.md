# Non-uniform spacing, mirror-symmetric: design

**Date:** 2026-06-09
**Status:** Approved (brainstorming → ready for implementation plan)

## Goal

Place beams at stress-weighted arc positions (revisiting F.1) but **symmetrized across the
chord**, so beam `b` and its chord mirror `n−b` land at mirror positions. This keeps
`beam_radius_groups`'s symmetry detection active (so mirror-paired radii stay grouped and
the design stays symmetric). Measure even-spacing vs symmetric-stress-weighted on the
current model.

## Motivation

F.1 found stress-weighted (non-uniform) beam spacing gave only ~1% on this
skin/buckling-dominated design, and it was abandoned for even spacing. The user wants it
back, but it must not break the new mirror-symmetry of the radii: an asymmetric placement
makes `beam_radius_groups` fall back to independent radii (losing the symmetric sizing).
Symmetrizing the stress weights across the chord yields a mirror-symmetric placement that
stays groupable, while still clustering beams toward the higher-stressed perimeter
regions. A free-rotating wingsail flies both tacks, so a chord-symmetric layout is also
the physically right call.

## Decision (from brainstorming)

- **Max-of-mirror** symmetrization: `w_sym[i] = max(w[i], w[mirror(i)])` — place beams
  where *either* side is highly stressed (both-tack-conservative, robust to an asymmetric
  single-case baseline).

## Design

### 1. Symmetrization helper — `beams/layout.py`

```
chord_symmetrize_weights(weights: np.ndarray) -> np.ndarray
```
`weights` is the per-perimeter-segment stress intensity (length `n_beams`, segment `b`
between beams `b` and `b+1`, as produced by `cross_section_stress_weights`). Across the
chord, segment `b` mirrors segment `(n−1−b) % n` (the traversal runs TE→upper→LE→lower→TE,
so beam `b` ↔ `n−b` and segment `b` ↔ `n−1−b`). Returns
`w_sym[b] = max(w[b], w[(n−1−b) % n])` — symmetric by construction. Pure function,
unit-testable. Export it from `beams/__init__.py`.

The arc fractions then come from the existing
`stress_weighted_targets(w_sym, n_beams)` (cumulative-weight placement); with symmetric
weights the result is a mirror-symmetric arc placement.

### 2. Workflow — uses the existing `arc_fractions` path (no new plumbing)

`build_beam_shell_model(..., arc_fractions=...)` already exists. The pipeline (in the
example): even-spaced model → solve the load envelope → `w =
cross_section_stress_weights(model, load_arrays)` → `w_sym = chord_symmetrize_weights(w)`
→ `af = stress_weighted_targets(w_sym, n_beams)` → rebuild the model with
`arc_fractions=af` → size. Everything downstream — `beam_radius_groups` symmetry
detection, monotonic taper, banding, datum offsets, the analytic Jacobian, beam lengths,
triangle frames — is **position-driven off `model.nodes`**, so it composes automatically;
the only requirement is that `af` is mirror-symmetric, which §1 guarantees.

### 3. Example — `examples/40_symmetric_spacing.py`

Medium wingsail, identical constraints (span datum, buckling SF 1.5, `n_skin_bands=4`,
`use_analytic_jacobian=True` for speed/convergence). Size **even spacing** vs
**symmetric-stress-weighted spacing**; print mass + beam/skin split, the
stress-concentration ratio (max/mean segment weight), beam radii range, tip twist,
buckling utils, and Δmass. Run timed (orchestrator). Honest expectation: ~1% (F.1) on
this skin-dominated design; report the measured value.

### 4. Testing — `tests/beams/test_symmetric_spacing.py`

- `chord_symmetrize_weights`: output mirror-symmetric (`w_sym[b] == w_sym[(n−1−b)%n]`),
  equals the per-pair max, length `n_beams`; idempotent on already-symmetric input.
- **Composition (key test):** build `build_beam_shell_model(arc_fractions=af)` with a
  symmetric `af = stress_weighted_targets(chord_symmetrize_weights(rng_weights), n)` →
  `beam_radius_groups` returns the **reduced** group count (symmetry detected), NOT the
  `n` fallback. A deliberately asymmetric `af` → `n` fallback (contrast).
- Sanity: even spacing `af = arange(n)/n` is detected symmetric (it is — `b/n` ↔ `1−b/n`
  are chord mirrors).

## Components / file map

| File | Responsibility | Change |
|---|---|---|
| `beams/layout.py` | `chord_symmetrize_weights` (max-of-mirror) | modify |
| `beams/__init__.py` | export it | modify |
| `examples/40_symmetric_spacing.py` | even-vs-symmetric-spaced sized comparison | **new** |
| `tests/beams/test_symmetric_spacing.py` | symmetrize + grouping-preserved tests | **new** |
| `docs/plan.md` | finding + Decisions-log row (at close) | modify |

## Success criteria

- `chord_symmetrize_weights` produces mirror-symmetric weights (max-of-mirror);
  unit-tested.
- A symmetric stress-weighted placement keeps `beam_radius_groups` grouping (radii stay
  mirror-symmetric) — the composition test passes.
- A measured even-vs-symmetric-spaced comparison on the medium wingsail (honest, given
  F.1's ~1%).
- Full suite green; even spacing remains the default (opt-in via `arc_fractions`).

## Out of scope (deferred)

- Changing the default placement (even stays default).
- Per-spanwise-level spacing (one `arc_fractions` for all levels, as F.1).
- A second diagonal beam family (F.2, discarded).
- Iterating the placement (one-shot from a baseline solve, as F.1).
