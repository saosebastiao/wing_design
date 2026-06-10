---
name: sizing-run
description: Use when launching, monitoring, or reporting wing_design sizing/FEA runs — examples in examples/2x–4x, medium/large wingsail optimizations, sweeps, or any run expected to take more than a minute.
---

# Running sizing/FEA jobs

## Launch

- Run through just: `just example 39_tip_gusset` or `just py examples/39_tip_gusset.py`
  (inherits PYTHONPYCACHEPREFIX; venv-correct).
- **Always time it:** wrap with `time` or capture Bash duration. The measured
  wall-clock goes in the finding — estimates have run ~10× off.
- Expected scales (measured): small-problem sizing (n_beams=12, n_levels=6) ≈ 5–10 min;
  medium wingsail (16×8, maxiter 300, analytic Jacobian) ≈ 1–1.7 h; FD is ~2× slower.
  Anything ≥ 10 min: run in the background and check on completion — don't block.

## Before reporting ANY mass number

1. Converged? (optimizer success / iterations < maxiter)
2. Feasible? (`laminate_result_is_feasible`; buckling/twist/deflection utilizations)
3. Beats the noise floor? Same-config differences under ~2–3% are INSIDE the noise
   floor — do NOT claim them as wins or losses.
4. Wall-clock measured and noted.

If any of 1–2 fail, the number is not a result — it's a diagnostic. Say so.

## Sweeps and warm starts

Pass `x0=` from the nearest prior optimum (examples 39/40 pattern); fine meshes start
from coarse solutions resampled (`resample_segment_radii` pattern). Sweep points are
embarrassingly parallel — separate processes, not threads.

## Artifacts

- Everything generated goes to `exports/` (gitignored). Geometry: STL/STEP via the
  sized-export path (`examples/37_sized_export.py` pattern). FEA fields: `.vtu` +
  `just shot <viz> [out.png]` for screenshots.
- **Milestone runs** (new headline, validated lever, re-baseline) must export CAD +
  analysis artifacts and the finding must say how to regenerate them — see CLAUDE.md.

## After the run

Record via the `record-finding` skill: findings.md entry + decisions-log row, with
mass, governing constraint, iterations, wall-clock.
