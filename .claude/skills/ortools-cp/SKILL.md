---
name: ortools-cp
description:
  Guide for modeling and solving combinatorial optimization problems with
  Google OR-Tools' CP-SAT solver in Python. Use this when writing models
  that mix integer / boolean variables, linear constraints, scheduling
  (intervals + no-overlap), conditional logic, or when picking solver
  parameters for combinatorial search.
---

# OR-Tools CP-SAT (Python)

CP-SAT is the modern solver in `ortools.sat.python.cp_model`. It handles
mixed integer + boolean + linear models with a built-in scheduling
extension (intervals, no-overlap, cumulative). For most optimization
problems on this project, **start with CP-SAT** — don't reach for
`pywraplp` (linear/MIP) unless you have a pure LP/MIP with thousands of
continuous variables.

## When to use

CP-SAT fits when **any** of these are true:

- Variables are integers or booleans (CP-SAT cannot model continuous
  variables — use `pywraplp` for those).
- The problem involves scheduling, sequencing, or assignment with
  "this-or-that" structure (intervals on shared resources, alternative
  options, conditional precedence).
- The model has boolean reifications ("constraint X applies *only if*
  condition B is true").
- You want every feasible solution, not just one optimum
  (`enumerate_all_solutions`).
- The objective is integer or rational and the problem has heavy
  combinatorial structure (LP relaxations would be weak).

Reach for the other solvers instead when:

- The model is a pure LP (continuous variables, linear constraints,
  linear objective) — use `pywraplp` with the GLOP backend.
- The model is a MIP at a scale CP-SAT struggles with (10⁵+ variables,
  most continuous) — use `pywraplp` with SCIP or CBC.
- The problem is routing / vehicle assignment / TSP — use
  `pywrapcp` (routing solver, separate package).

## The five-step skeleton

Every CP-SAT script has the same five sections. Keep them visible:

```python
from ortools.sat.python import cp_model

def solve():
    # 1. Model
    model = cp_model.CpModel()

    # 2. Variables
    x = model.new_int_var(0, 10, "x")
    y = model.new_int_var(0, 10, "y")

    # 3. Constraints
    model.add(x + y == 7)
    model.add(x <= 2 * y)

    # 4. Objective (optional — drop for pure feasibility)
    model.maximize(3 * x + y)

    # 5. Solve
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return {"x": solver.value(x), "y": solver.value(y),
            "objective": solver.objective_value}
```

If your code drifts away from this shape (e.g. variables interleaved
with constraints, no clear `solve()` boundary), refactor before adding
features. Readability wins.

## API style — use snake_case

OR-Tools 9.x added snake_case aliases for every method. **Always use
snake_case** in new code:

| Old CamelCase             | Modern snake_case (use this)      |
|---------------------------|-----------------------------------|
| `NewIntVar`               | `new_int_var`                     |
| `NewBoolVar`              | `new_bool_var`                    |
| `NewIntervalVar`          | `new_interval_var`                |
| `Add`                     | `add`                             |
| `AddNoOverlap`            | `add_no_overlap`                  |
| `AddAllDifferent`         | `add_all_different`               |
| `Maximize` / `Minimize`   | `maximize` / `minimize`           |
| `Solve`                   | `solve`                           |
| `OnlyEnforceIf`           | `only_enforce_if`                 |
| `OnSolutionCallback`      | `on_solution_callback`            |

The CamelCase forms still work but mixing styles in one file is jarring.
The official Python examples now use snake_case throughout.

## Idiomatic problem-domain modeling

The pattern that scales best is: **describe the problem with
dataclasses, then build the model from them.** Keeps the constraints
phrased in domain language, not in raw indices.

```python
from dataclasses import dataclass
from ortools.sat.python import cp_model

@dataclass(frozen=True)
class Task:
    id: str
    duration: int           # in some time unit (minutes, etc.)
    machine: int            # which resource it runs on
    must_precede: tuple[str, ...] = ()   # IDs of downstream tasks

def schedule(tasks: list[Task], horizon: int) -> dict[str, int]:
    model = cp_model.CpModel()
    by_id = {t.id: t for t in tasks}

    # One interval per task. Keep the start vars in a parallel dict
    # so we can look them up by domain ID.
    start = {t.id: model.new_int_var(0, horizon, f"start_{t.id}") for t in tasks}
    interval = {
        t.id: model.new_interval_var(start[t.id], t.duration,
                                     start[t.id] + t.duration,
                                     f"iv_{t.id}")
        for t in tasks
    }

    # No two tasks on the same machine overlap.
    by_machine: dict[int, list] = {}
    for t in tasks:
        by_machine.setdefault(t.machine, []).append(interval[t.id])
    for ivs in by_machine.values():
        model.add_no_overlap(ivs)

    # Precedence: every task ends before its successors start.
    for t in tasks:
        for succ_id in t.must_precede:
            model.add(start[succ_id] >= start[t.id] + t.duration)

    # Makespan objective.
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, [start[t.id] + t.duration for t in tasks])
    model.minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    if solver.solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("infeasible")
    return {t.id: solver.value(start[t.id]) for t in tasks}
```

Why this style works:

- Constraints read in the same vocabulary as the problem statement
  (`start[succ_id] >= start[t.id] + t.duration`, not
  `model.add(s[7] >= s[2] + 3)`).
- Looking up a variable by domain key (`start[t.id]`) is way easier to
  debug than parallel arrays indexed by integers.
- Adding a new field to the dataclass (e.g. `priority: int`) is a
  local change — the rest of the code keeps reading naturally.

## Variables

```python
# Integer variable on a closed range [lb, ub].
x = model.new_int_var(0, 100, "x")

# Boolean (0/1) — use a dedicated method, NOT new_int_var(0, 1, …).
b = model.new_bool_var("b")

# Variable on a non-contiguous domain (e.g. {1, 3, 5, 7}).
v = model.new_int_var_from_domain(
    cp_model.Domain.from_values([1, 3, 5, 7]), "v")

# Constant — wrap as an int. CP-SAT lifts Python ints automatically
# in linear expressions; you rarely need new_constant().
five = 5            # this is fine
five_var = model.new_constant(5)   # only if you need it as a variable
```

Boolean negation: `~b` (or `b.Not()`) — produces the literal "b is
False", **not** an integer. Use it inside `only_enforce_if`.

## Linear constraints

```python
# Simple linear constraints — Python operators just work.
model.add(2 * x + 3 * y <= 10)
model.add(x + y >= 1)
model.add(x == y)
model.add(x != y)

# Sum over a collection — sum() is fine but LinearExpr.sum() is faster
# for very large sums (it skips Python's pairwise __add__).
model.add(sum(xs[i] for i in range(n)) == k)
model.add(cp_model.LinearExpr.sum(xs) == k)

# Weighted sum.
model.add(cp_model.LinearExpr.weighted_sum(xs, weights) <= cap)
```

## Boolean constraints and reification

This is what makes CP-SAT shine over pure LP/MIP — first-class boolean
logic without big-M hacks.

```python
# At-most-one / exactly-one over a set of booleans.
model.add_at_most_one([b1, b2, b3])
model.add_exactly_one([b1, b2, b3])

# Implications and clauses.
model.add_implication(b1, b2)          # b1 → b2
model.add_bool_or([b1, b2, b3])        # b1 ∨ b2 ∨ b3
model.add_bool_and([b1, b2, b3])       # b1 ∧ b2 ∧ b3

# Reify a constraint on a literal: "constraint holds IFF b is true".
# Express the if-direction with only_enforce_if(b);
# express the only-if direction with only_enforce_if(~b).
model.add(x >= 10).only_enforce_if(b)        # b → (x >= 10)
model.add(x <  10).only_enforce_if(~b)       # ¬b → (x < 10)
```

The `only_enforce_if` form is the right tool for "this constraint
applies only when this condition holds." Don't try to encode it with
big-M — CP-SAT's lazy clause learning will exploit the reified form.

## Global constraints

CP-SAT has built-in "global" constraints that are much stronger than
their hand-rolled equivalents:

```python
model.add_all_different([x1, x2, x3, x4])    # all values distinct
model.add_max_equality(z, [x1, x2, x3])      # z == max(x1, x2, x3)
model.add_min_equality(z, [x1, x2, x3])      # z == min(x1, x2, x3)
model.add_abs_equality(z, x)                 # z == |x|
model.add_modulo_equality(z, x, k)           # z == x mod k
model.add_division_equality(z, x, k)         # z == x // k
model.add_multiplication_equality(z, [x, y]) # z == x * y (nonlinear!)
model.add_element(index_var, [a, b, c, d], value_var)
# Look up the index_var-th element of the array; value_var equals it.

# Force a sequence of variables to form a valid TSP-style circuit.
arcs = [(i, j, lit_ij) for i, j in edges]
model.add_circuit(arcs)
```

Prefer these over manual encodings — they're propagated specially.

## Scheduling (intervals + no-overlap + cumulative)

```python
# Fixed-duration task on a single machine.
start = model.new_int_var(0, horizon, "start")
interval = model.new_interval_var(start, duration, start + duration, "iv")

# Optional task — only "exists" if presence literal is true.
present = model.new_bool_var("present")
optional = model.new_optional_interval_var(
    start, duration, start + duration, present, "iv_opt")

# All tasks on one machine — no two can overlap in time.
model.add_no_overlap([iv_a, iv_b, iv_c])

# Resource constrained — at any time, sum of active task demands
# can't exceed capacity.
model.add_cumulative([iv_a, iv_b, iv_c], demands=[2, 1, 3], capacity=4)
```

Intervals are the right primitive whenever you'd say "this thing
takes some time." Don't roll your own `start <= other.start - my_dur`
disjunctions — `add_no_overlap` is dramatically more efficient.

## Objective

```python
model.minimize(makespan)
model.maximize(2 * x + 3 * y)
# Multi-term sums — same as constraints, can use LinearExpr.sum().
model.minimize(sum(cost[i] * picked[i] for i in items))
```

Only one objective per model. For multi-objective work, weight + scalarize
or run sequential single-objective solves.

## Solving

```python
solver = cp_model.CpSolver()

# Useful parameters (set before solve):
solver.parameters.max_time_in_seconds = 60.0    # wall-clock cap
solver.parameters.num_workers = 8               # parallel search
solver.parameters.log_search_progress = True    # human-readable log
solver.parameters.random_seed = 42              # reproducibility
solver.parameters.linearization_level = 1       # 0..2 (default 1)

status = solver.solve(model)

# Status interpretation:
#   OPTIMAL    — proven optimum
#   FEASIBLE   — a solution but optimality not proven (e.g. hit time limit)
#   INFEASIBLE — proven no solution
#   UNKNOWN    — search aborted before any answer
#   MODEL_INVALID — coding error in the model
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    val = solver.value(x)
    obj = solver.objective_value
    print(f"x={val} obj={obj} status={solver.status_name(status)}")
    print(f"branches={solver.num_branches}  wall_time={solver.wall_time:.2f}s")
```

Three parameters cover 90 % of tuning:

- `max_time_in_seconds` — always set one; default is unbounded.
- `num_workers` — bump to your core count for the parallel portfolio;
  meaningful speedups on hard instances.
- `log_search_progress = True` — for any non-trivial problem, the log
  tells you whether the solver is making progress and what's blocking.

## Enumerating multiple solutions

For "show me every feasible answer" or "show me the first N":

```python
class SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables, limit=None):
        super().__init__()
        self._vars = variables
        self._limit = limit
        self.solutions: list[dict] = []

    def on_solution_callback(self):
        self.solutions.append({v.name: self.value(v) for v in self._vars})
        if self._limit and len(self.solutions) >= self._limit:
            self.stop_search()

solver = cp_model.CpSolver()
solver.parameters.enumerate_all_solutions = True
collector = SolutionCollector([x, y, z], limit=100)
solver.solve(model, collector)
print(f"Found {len(collector.solutions)} solutions")
```

`enumerate_all_solutions = True` works only when the model has *no*
objective (CP-SAT can't enumerate optima of a soft-objective model
directly — it would never finish). If you want every feasible
solution within K of the optimum, solve once with the objective, then
add a constraint pinning the objective to that range and re-solve
with enumeration on.

## Adding hints (warm starts)

If you have a known good (but maybe suboptimal) solution, hint it:

```python
model.add_hint(x, 7)
model.add_hint(y, 3)
```

CP-SAT will try the hinted values first. Doesn't restrict the search,
just biases it. Useful when re-solving after a small change to the
problem.

## Common pitfalls

- **Forgetting to wrap booleans in `only_enforce_if`.** Writing
  `model.add((x >= 10) == b)` looks tempting but the `==` isn't a
  reification — it builds a useless integer equality. Use
  `model.add(x >= 10).only_enforce_if(b)` and the reverse for `~b`.
- **Mixing CamelCase and snake_case.** They're aliases that resolve to
  the same backend, but a code review will flag it. Pick one (use
  snake_case) and stay there.
- **Treating booleans as integers.** `model.new_int_var(0, 1, ...)` is
  *not* a boolean variable for reification purposes — use
  `new_bool_var`. The literal `~b` and the `only_enforce_if` API only
  work on real boolean variables.
- **Floating-point coefficients.** CP-SAT works on integers. Float
  coefficients get rounded — multiply through by a fixed factor (e.g.
  scale costs to integer cents) and document the scale factor.
- **Big horizons.** For scheduling, set `horizon` to the smallest valid
  value (e.g. sum of all durations, or a known upper bound from a
  greedy heuristic). A loose horizon balloons the search tree without
  any modeling benefit.
- **Implicit `sum()` over generators of size > ~10 000.** Use
  `cp_model.LinearExpr.sum([...])` instead — it's an order of magnitude
  faster to build because it skips pairwise `__add__`.
- **Calling `solver.value(x)` before `solver.solve(model)`** — the
  value doesn't exist yet. Always check `status` first.
- **Forgetting `num_workers`.** Default is 1. On any modern laptop,
  bumping to 8 is a free 4-8× speedup on hard instances.

## Quick sanity check

If you want to confirm CP-SAT itself is working before debugging your
model, this minimal program should print `x=2 y=0` and `obj=4`:

```python
from ortools.sat.python import cp_model
m = cp_model.CpModel()
x = m.new_int_var(0, 10, "x")
y = m.new_int_var(0, 10, "y")
m.add(x + y <= 2)
m.maximize(2 * x + y)
s = cp_model.CpSolver()
s.solve(m)
print(f"x={s.value(x)} y={s.value(y)} obj={s.objective_value}")
```

## Documentation

- Official Python reference:
  <https://developers.google.com/optimization/reference/python/sat/python/cp_model>
- Tutorials (cover assignment, packing, scheduling, routing, with
  Python examples):
  <https://developers.google.com/optimization>
- CP-SAT primer (the canonical "how to think in CP-SAT" guide):
  <https://github.com/d-krupke/cpsat-primer>
- The version installed in this project is pinned in `pyproject.toml`
  (`ortools >= 9.15.6755`). The snake_case API used throughout this
  guide is available from 9.x onward.
