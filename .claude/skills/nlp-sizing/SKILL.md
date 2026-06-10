---
name: nlp-sizing
description: Use when modifying or running the sizing optimization (laminate_sizing.py, sensitivity.py) — adding constraints or design variables, extracting Lagrange multipliers / shadow prices, diagnosing SLSQP non-convergence or stalls, aggregating constraints (KS), or deciding whether to switch to IPOPT.
---

# NLP sizing: SLSQP/IPOPT conventions for this repo

## Hard-won conventions (violating these has burned real runs)

- **Normalize everything to O(1).** Objective = mass/m_ref; every constraint =
  `1 − value/limit ≥ 0`. Raw Pa-vs-metre scales make SLSQP's QP ill-conditioned and it
  quits early at infeasible points (Phase-C lesson).
- **Inequality sign:** scipy wants `fun(x) ≥ 0` feasible. All constraints here follow
  `1 − value/limit` (so feasible = positive, binding = 0).
- **Noise floor ~2–3%.** ftol=1e-4 on normalized mass + local-basin scatter (FD vs
  analytic landed 2.2% apart) means same-config mass differences under ~2–3% are NOT
  signal.
- **Result = converged AND feasible.** `res.success` alone is insufficient; run the
  feasibility check (`laminate_result_is_feasible`) before reporting mass.

## Lagrange multipliers / shadow prices

scipy ≥ 1.15 exposes `res.multipliers` from SLSQP (project venv has 1.17.1) — do NOT
assume they're unavailable. They correspond to the *normalized* objective and
constraints, so convert before quoting kg-per-unit: for constraint `g̃ = 1 − v/L`,
`∂m/∂L ≈ m_ref · λ̃ · ∂g̃/∂L` at the optimum. **Verify the conversion by
finite-differencing one limit on a small problem before trusting the numbers** — the
sign/scale algebra is easy to fumble.

## Adding a constraint or DV block (until the P.0 refactor lands)

Touch all ~6 places or the analytic path silently breaks: (1) `evaluate()` metric,
(2) constraint closure, (3) constraint dict + idx bookkeeping (`laminate_sizing.py`
~322–568), (4) gradient fn in `sensitivity.py`, (5) Jacobian registration,
(6) result dataclass + feasibility check. Every new gradient gets a central-FD test,
rel-err ≤ ~1e-4, tiny mesh (`tests/beams/test_sensitivity.py` patterns).
Design-dependent *loads* (e.g. prestress `f_pre`) additionally need a `λᵀ·∂f/∂x` term
— the adjoint currently assumes loads are design-independent.

## KS aggregation (when collapsing vector constraints)

`KS(g) = (1/ρ)·ln Σ exp(ρ·gᵢ)` over-estimates the max → conservative. Start ρ ≈ 50 on
normalized constraints; re-validate the optimum against the hard-max formulation
(same pattern as `test_analytic_sizing.py` FD-vs-analytic equivalence).

## SLSQP troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Quits early, infeasible point | Lost O(1) normalization on a new term |
| Hits maxiter, slow progress | More DVs need more iters (4 bands ≈ 354); warm-start from a neighbor (`x0=`) |
| "Positive directional derivative" | Degenerate/conflicting constraints; check new constraint's sign + gradient vs FD |
| Different optima across runs | Local basins (~2% apart); multi-start wrapper exists |

## IPOPT escape hatch

Triggers: DVs > ~150, or SLSQP stalls persist after normalization/warm-start fixes.
casadi 3.7.2 (bundles IPOPT) is already in the venv via AeroSandbox; cyipopt is the
cleaner scipy-like API if adding a dep. Key options: `hessian_approximation =
limited-memory`, `mu_strategy = adaptive`, tol on the same normalized scale. IPOPT
returns multipliers natively and handles many constraints well; it wants gradients —
use the analytic Jacobian path, not FD.
