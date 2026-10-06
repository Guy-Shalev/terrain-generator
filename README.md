# Terrain Generator

Procedural worlds from simplified plate tectonics. Plates raise the mountains,
rain and rivers carve them, climate picks the biomes, and trees grow last.
NumPy + SciPy, with a Pygame viewer. The map wraps east to west and closes with
ocean at the poles.

## Install

Python 3.11, then the three libraries it is made of:

```bash
pip install numpy scipy pygame
```

`tkinter`, for the import dialog, ships with Python on Windows.

## Run

Run `start.bat`, or:

```bash
python viewer.py --seed 7 --size 640x448
```

```bash
python preview.py 7          # every layer to out/, export to out/export/seed7/
```

```bash
python test_terrain.py       # invariant checks
```

The viewer also takes `--windowed WxH` for the window size and `--plates N`.

## Viewer

Every action is in the menu bar, with its key beside it.

| Key | Does |
|---|---|
| drag, WASD, arrows | pan |
| wheel, `+`, Num `-` | zoom |
| `Z` | fit to window |
| `1`-`0`, `-` | the first eleven layers |
| `[` `]` | previous / next layer |
| `R` | new world, new seed |
| `T` | rebuild, same seed |
| `G` | plate motion arrows |
| `P` | save the current layer as a picture |
| `E` | export for an engine |
| `I` | import a world from an export's `config.json` |
| `Esc`, `Q` | close the menu, then quit |

The Edit menu holds seven sliders: plates (4-48), map size (192x144 to
3840x2160), land/sea, shelf coast roughness, rivers, lakes and temperature
(-20 to +20 C). The world rebuilds when you let go of one.

The bar along the bottom reads out what is under the mouse: height, plate and
crust, distance to the plate boundary, rain, temperature, flow, lake or river,
biome and canopy.

Layers: main view, elevation (no water), tectonic relief, pre-erosion, plates,
boundaries, stress, rainfall, temperature, biomes, trees, biome legend, main
view (biomes), drainage, erosion delta, slope, land / coast, river width,
canopy.

`P` and `E` write to one folder per world, `out/export/seed<N>/`. A new world
gets a new folder (`seed7-2`), so nothing is written over. Import rebuilds the
world from its seed and settings, not from the pictures; if the export was made
by other code it says so, since the rebuild may then differ.

## Export

| File | What it is |
|---|---|
| `height16.png` | land and sea floor height, 16-bit grey |
| `lakes16.png` | lake depth, 16-bit, 0 where dry |
| `biome.png` | biome index per cell, 8-bit |
| `canopy.png` | tree cover, 0 to 1, 8-bit |
| `rivers.json` | river centrelines, with the width at each point |
| `config.json` | the seed and every setting; rebuilds the world |
| `main_view.png` | a colour picture of the map, from the viewer only |

Each PNG has a `.json` beside it that says what its values mean. For the
heightmaps, `lo + code * units_per_code` is the generator's height,
`metres_per_unit` turns that into metres, and `sea_code` is sea level. For
`biome.png`, `classes` names each index. River points are `[x, y]` in cells, and
x can run past the right edge where a river crosses the seam: take it modulo the
width.

From Python, `generate(export.load_config(path))` rebuilds an exported world bit
for bit, as long as the code has not changed since.

## Tuning

Every setting lives in `Config` in `terrain/world.py`, grouped by stage, and can
be overridden per call:

```python
from terrain import generate
world = generate(seed=12, n_plates=9, land_fraction=0.4, collision_h=1.8)
```

Settings measured in pixels are tuned at `ref_width` and scaled to the map's own
width, so a world keeps its shape at any size.

`world.height` is the final elevation, with sea level at 0. Beside it are
`surface`, `flow`, `runoff`, `rivers`, `lakes`, `temp`, `biome`, `tect` (plate
fields), `water` (lake and river fields) and `trees` (the `cover` field, plus
`pos`, `kind` and `size` for every tree).

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| plates | `terrain/tectonics.py` | plates, their motion, crust type, and where they push, pull or slide |
| relief | `terrain/elevation.py` | ranges, trenches, arcs, rifts and ridges along the boundaries |
| hotspots | `terrain/elevation.py` | island chains along each plate's motion |
| texture | `terrain/elevation.py` | fine ridged detail, strongest where the land is active |
| coastline | `terrain/elevation.py` | sea level, coastal plains, frayed coasts and inlets |
| pre-rain | `terrain/climate.py` | a first rain pass, so erosion carves the wet side harder |
| erosion | `terrain/hydrology.py` | rivers cut valleys, slopes slump |
| temperature | `terrain/climate.py` | latitude, height and distance from the sea |
| climate | `terrain/climate.py` | wind carries rain inland and drops it on mountains; rain shadows behind |
| rivers | `terrain/rivers.py` | lakes filled to what their rain can keep, river paths, meanders and widths |
| biomes | `terrain/climate.py` | biome from temperature against rain |
| trees | `terrain/trees.py` | tree cover from rain and the tree line, then individual trees |

The reasons behind each stage are in its module's docstrings.

## Speed

About 3 s at 640x448, 12-14 s at 1024x768, 27-43 s at 1920x1080 and a little
over two minutes at 3840x2160. Times vary from run to run.

## Not done yet

- Nothing human-made: no towns, roads or borders.
- Climate is a yearly mean plus a seasonal swing. That is enough for Whittaker
  biomes, not for Köppen, which needs to know when the rain falls.
- Ocean currents warm or cool the sea, not the land beside it.
- Lake outflow is carved, not simulated, so a river cannot change course later.
- Meanders are noise, not migration.
- Trees do not slow erosion.
