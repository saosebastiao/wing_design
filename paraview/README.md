# ParaView scripts — `paraview/`

Per-export visualization recipes for the VTU files produced by
`examples/`. **Targets ParaView 6.1.1** (any 6.x should work — the
`paraview.simple` API + VTU XML format + preset names used here are
stable across the 6.x series).

The scripts are intentionally short. Each one loads a single VTU,
sets a sensible color map + camera, and either opens an interactive
window or saves a PNG. They're meant to be readable recipes you can
copy into the ParaView GUI's *Python Shell* (View → Python Shell) and
adapt — not a black-box framework.

## Running

ParaView ships its own Python interpreter. Don't use the project's
`uv run python` — the project venv has no `paraview` module.

The `justfile` wraps the common invocations:

```sh
just view shell_per_region_skin                    # open interactive window
just view sized_truss --color max_abs_sigma_MPa    # pass extra flags through
just shot shell_per_region_skin                    # screenshot → docs/figures/shell_per_region_skin.png
just shot shell_fea /tmp/foo.png                   # custom output path
just shots                                         # regenerate every screenshot
just pv exports/shell_fea.vtu                      # launch the ParaView GUI
```

If ParaView is installed somewhere other than
`/Applications/ParaView-6.1.1.app/Contents/bin`, override the path on
the command line: `just PV=/usr/local/bin view shell_fea`.

The raw invocations work too:

```sh
PV=/Applications/ParaView-6.1.1.app/Contents/bin
$PV/pvpython paraview/shell_per_region_skin.py
$PV/pvbatch paraview/sized_truss.py exports/shell_frame_sized.vtu \
    --color max_abs_sigma_MPa --screenshot truss.png
```

Every script accepts a positional VTU path; if omitted it loads the
matching file from `exports/`. Run with `--help` for per-script flags.

## What each script visualizes

| Script | Default VTU | Phase | Visualization |
|--------|-------------|-------|---------------|
| `fea_volumetric.py` | `fea_<case>.vtu` (default: survival) | 4a | Tet σ_VM with displacement-warped solid |
| `frame_phi.py` | `frame_phi.vtu` | 5b | φ scalar + smoothed eigenvector glyphs on the tet volume |
| `frame_streamlines.py` | `frame_streamlines.vtu` | 5b | Retraced streamlines through the smoothed frame field |
| `stress_lines.py` | `stress_lines.vtu` | 5a | Raw volumetric streamlines per (case × tack × family) |
| `isocurves.py` | `isocurves.vtu` | 5c | Integer-isocurve segments per tet |
| `sized_truss.py` | `lp_sized_truss.vtu` (overridable) | 6a-c, 6e, 6g | Any Phase-6 sized truss line network |
| `shell_fea.py` | `shell_fea.vtu` | 4b | Shell σ_VM + principal-direction glyphs on the OML |
| `shell_stress_lines.py` | `shell_stress_lines.vtu` | 5e | Surface streamlines on the OML (spar caps) |
| `interior_candidates.py` | `interior_candidates.vtu` | 5f | Rib + spanwise + shear-web candidate network |
| `coupled_shell_frame.py` | `coupled_shell_frame.vtu` | 6f | Shell + frame coupled FEA with displacement warp |
| `shell_per_region_skin.py` | `shell_per_region_skin.vtu` | 6h | Per-region skin thickness |

`sized_truss.py` is the catch-all for any line-cell VTU with
`area_mm2` cell data: `truss_sized.vtu`, `lattice_truss_sized.vtu`,
`lp_sized_truss.vtu`, `ab_iteration_truss.vtu`, `shell_track_sized.vtu`,
`shell_frame_sized.vtu` all open with it.

See [`../docs/visualizations.md`](../docs/visualizations.md) for the
**physical interpretation** of each view — what the Michell pattern
should look like, where to expect the spar caps, what σ levels mean
"safe" vs "saturated", etc.

## What lives in `_common.py`

Helpers shared by every script. Read this if you're writing a new
visualization:

- `default_argparser(filename, doc)` — `[vtu] [--screenshot] [--size]`.
- `open_vtu(path)` — `XMLUnstructuredGridReader` + missing-file check.
- `color_by(display, association, field, preset='Cool to Warm')` —
  `ColorBy` + rescale + named preset + scalar bar.
- `glyph_vectors(source, field, scale_factor, every_nth, color)` —
  `Glyph` filter for vector fields (arrows / lines).
- `threshold_between(source, assoc, field, lo, hi)` — `Threshold`
  filter using `LowerThreshold` / `UpperThreshold` (ParaView 6.x API).
- `set_thick_lines(display, width)` — `RenderLinesAsTubes` + width.
- `finish(args, view)` — installs a 3/4 side camera for the 5 m
  wingsail, renders, and either saves a screenshot or opens an
  interactive window.

## State files (`.pvsm`)

ParaView's session-state format. Save one via *File → Save State*
after you've tuned a view in the GUI; reload via *File → Load State*.
We don't ship `.pvsm` files because (a) they embed the absolute VTU
path and break if the repo moves and (b) the Python scripts already
encode the recipe in a more portable way. If you want
session-specific tweaks (lights, custom annotations, multi-view
layouts), save your own `.pvsm` once you're happy and commit it next
to the script.

## Limitations

- Glyph density and camera defaults are tuned for the 5 m wingsail
  geometry. If you change `default_scenario()` to a wildly different
  span, edit `_common._wing_side_view`.
- The scripts assume the VTU file already exists. Run the
  corresponding `examples/NN_<phase>.py` first.
- macOS ParaView prints `[openvkl] INITIALIZATION ERROR: …` on
  startup. That's a missing volume-rendering plugin in the upstream
  build, not our problem; rendering still works.
