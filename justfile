default:
    @just --list

# Sync the venv with pyproject.toml / uv.lock
sync:
    uv sync

# Add a runtime dependency (usage: just add build123d)
add pkg:
    uv add {{pkg}}

# Add a dev-only dependency
add-dev pkg:
    uv add --dev {{pkg}}

# Upgrade locked dependencies
upgrade:
    uv lock --upgrade
    uv sync

# Run the wing-design entrypoint
run *args:
    uv run wing-design {{args}}

# Run an arbitrary python script under the project venv
py *args:
    uv run python {{args}}

# Open an IPython shell with the project venv
shell:
    uv run ipython

# Execute a module in src/wing_design (usage: just exec airfoil)
exec module:
    uv run python -m wing_design.{{module}}

# Run a numbered example script (usage: just example 01_wing_solid)
example name:
    uv run python examples/{{name}}.py

# Run all examples in order
examples:
    for f in examples/*.py; do echo "--- $f ---"; uv run python "$f" || exit 1; done

# ---------------------------------------------------------------------------
# ParaView 6.x visualization
# ---------------------------------------------------------------------------
# Override on the command line if ParaView is installed elsewhere:
#   just PV=/path/to/bin view shell_fea
PV := "/Applications/ParaView-6.1.1.app/Contents/bin"
PARAVIEW_APP := "/Applications/ParaView-6.1.1.app"

# Open a ParaView visualization interactively (usage: just view shell_fea [--color region_id])
view name *args:
    {{PV}}/pvpython paraview/{{name}}.py {{args}}

# Save a screenshot of one visualization (usage: just shot shell_fea [out.png])
shot name out="":
    #!/usr/bin/env bash
    set -e
    out="{{out}}"
    if [ -z "$out" ]; then out="docs/figures/{{name}}.png"; fi
    mkdir -p "$(dirname "$out")"
    {{PV}}/pvbatch paraview/{{name}}.py --screenshot "$out"

# Regenerate every screenshot under docs/figures/ (Phase 4-5 exports only)
shots:
    #!/usr/bin/env bash
    set -e
    mkdir -p docs/figures
    pairs=(
      "fea_volumetric            exports/fea_survival.vtu"
      "stress_lines              exports/stress_lines.vtu"
      "frame_phi                 exports/frame_phi.vtu"
      "frame_streamlines         exports/frame_streamlines.vtu"
      "shell_fea                 exports/shell_fea.vtu"
      "shell_stress_lines        exports/shell_stress_lines.vtu"
    )
    for entry in "${pairs[@]}"; do
      read script vtu <<< "$entry"
      label=$(basename "$vtu" .vtu)
      out="docs/figures/${label}.png"
      printf "%-40s → %s\n" "$label" "$out"
      {{PV}}/pvbatch "paraview/${script}.py" "$vtu" --screenshot "$out" >/dev/null 2>&1 \
        || { echo "  FAIL ($script on $vtu)"; exit 1; }
    done
    echo "Wrote $(ls docs/figures/*.png | wc -l) PNGs to docs/figures/"

# Open the ParaView GUI on an export file (usage: just pv exports/shell_fea.vtu)
pv file:
    open -a "{{PARAVIEW_APP}}" {{file}}

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

# Remove build artifacts and caches
clean:
    rm -rf build dist *.egg-info .pytest_cache
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
