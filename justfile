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

# Remove build artifacts and caches
clean:
    rm -rf build dist *.egg-info .pytest_cache
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
