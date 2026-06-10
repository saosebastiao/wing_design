---
name: justfile-tasks
description: Use when running project tasks (examples, scripts, ParaView visualization, dependency management), adding or modifying just recipes, or automating a repeated command sequence in this repo.
---

# Project task automation via just

All routine commands go through the justfile — it exports
`PYTHONPYCACHEPREFIX=.pycache` so bytecode stays out of the source tree. Don't run
bare `python script.py`. `just --list` shows everything.

## Recipe quick reference

| Task | Command |
|---|---|
| Run an example | `just example 39_tip_gusset` |
| Run any script | `just py path/to/script.py` |
| Run a wing_design module | `just exec <module>` |
| IPython shell in venv | `just shell` |
| Deps: sync / add / add dev / upgrade | `just sync` / `just add <pkg>` / `just add-dev <pkg>` / `just upgrade` |
| ParaView interactive | `just view shell_fea [--color region_id]` |
| ParaView screenshot | `just shot shell_fea [out.png]` |
| Regenerate all doc figures | `just shots` |
| Open ParaView GUI on a file | `just pv exports/shell_fea.vtu` |
| Clean caches | `just clean` |

Tests have no recipe yet: `uv run pytest` (or `uv run pytest tests/beams -q`).
ParaView path is a variable — override with `just PV=/path/to/bin view ...`.

## Adding a recipe

Match house style: a `# comment` line above each recipe (it becomes the `just --list`
description), kebab-case names, `uv run` for anything Python.

```just
# One-line description (usage: just bench 39_tip_gusset)
bench name:
    time uv run python examples/{{name}}.py
```

- Parameters: `name:` required, `out="":` default, `*args:` variadic passthrough.
- Multi-line shell logic: start the body with `#!/usr/bin/env bash` + `set -e`
  (see the `shots` recipe) — otherwise each line runs in a separate shell.
- Variables: `NAME := "value"` at top level; reference as `{{NAME}}`.
- Recipe dependencies: `release: test build` runs `test` then `build`.

## When to add one

A command sequence typed ≥ 3 times, or anything with non-obvious flags/paths
(ParaView batch mode, screenshot plumbing) belongs in the justfile, not in chat
history. Timed sizing runs are a good candidate (`bench` above) per the
measure-wall-clock convention in CLAUDE.md.
