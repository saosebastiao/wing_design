---
name: record-finding
description: Use when a sizing run, investigation, or phase increment has produced a result (positive or negative) that needs to be recorded — after completing any measurement run, or when the user says to record/log a finding or decision.
---

# Recording a finding

Findings live in `docs/findings.md` (append-only archive). plan.md holds NO findings —
only update its phase checklist/status line if a phase item completed.

## 1. Append the finding block (findings.md → Phase history & findings)

House format (match existing entries exactly):

```markdown
**<Increment name> done (YYYY-MM-DD) — <verdict: e.g. "−8.1% (clear win)" /
"NEGATIVE result (heavier)" / "correct, modest speedup">.** <What was built/run, with
module.function names in backticks.> <The measurement: masses, governing constraints,
utilizations, before → after with %.> **Why:** <the mechanism, one or two sentences.>
<Caveats / what is NOT shown.> (<wall-clock>, <FD or analytic Jacobian>.)
```

Requirements:
- **Measured wall-clock** included — never an estimate.
- Converged + feasible stated for every mass quoted (or the number labeled diagnostic).
- Negative results recorded with the same rigor; say *why* it lost (see the tip-gusset
  and diagonal-lattice entries for the pattern).
- If it changes the cross-cutting picture, also update the "governing-physics picture"
  list and/or "Headline results" tables at the top of findings.md.

## 2. Append a decisions-log row (findings.md → Decisions log table)

`| <Increment name> | <Choice + key numbers + what stays default/opt-in> |`

## 3. Milestone? Export artifacts

If this is a milestone (new headline mass, lever validated, re-baseline): export CAD
(STL/STEP) + analysis artifacts per CLAUDE.md, and reference the regeneration command
in the finding text.

## 4. Commit

Style: `<Area>: record finding (<one-line verdict>)` — e.g.
`Symmetric spacing: record finding (negative for mass; +2.7%)`.
