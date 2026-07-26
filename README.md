# Terrain Generator

Procedural world terrain from simplified plate tectonics, in NumPy + SciPy, with
a Pygame viewer. The map is a cylinder: it wraps in x, clamps in y. Because y
does not wrap, crust is tapered towards the top and bottom (`polar_start`,
`polar_ocean`) so the world closes with polar ocean instead of slicing
continents off at the edge - the land-area threshold is a quantile, so this
moves land rather than removing it. The latitude that taper keys on wanders
(`polar_wobble`) and the taper starts early and bites late: at uniform
strength it crosses the land threshold at one latitude everywhere, and the
continents come out trimmed against two invisible horizontal rules.

```bash
python viewer.py --seed 7 --size 640x448
```

```bash
python preview.py 7          # write every layer to out/ as PNG
```

```bash
python test_terrain.py       # invariant checks
```

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| plates | `terrain/tectonics.py` | blue-noise seeds, noise-warped *additively weighted* Voronoi partition (plates vary in size), per-plate velocity, crust type and buoyancy |
| strain | `terrain/tectonics.py` | divergence and curl of the piecewise-constant velocity field; boundary values propagated inland by a nearest-boundary distance transform |
| crust | `terrain/tectonics.py` | continuous continentality: plate type biases it, fBm decides the actual outline |
| relief | `terrain/elevation.py` | isostatic base plus collision ranges, trenches, volcanic arcs, cordilleras, rifts, mid-ocean ridges, transform scars, seafloor ageing |
| hotspots | `terrain/elevation.py` | island chains smeared along each plate's own motion |
| texture | `terrain/elevation.py` | warped ridged fBm, amplitude weighted by local relief and tectonic activity |
| coastline | `terrain/elevation.py` | sea level by quantile, a bounded rise per cell away from the shore, then noise frays the shoreline |
| erosion | `terrain/hydrology.py` | thermal creep, depression filling, D8 routing, flow accumulation, stream-power incision, repeated |
| lakes and rivers | `terrain/rivers.py` | lake surface levelling, outlet carving, polyline extraction, meandering, discharge-based width, channel incision |

Boundary type is never assigned by hand. The velocity field is constant inside
each plate, so its derivatives are non-zero only where plates meet: negative
divergence is convergence, positive is rifting, and a large curl with little
divergence is a transform fault. Everything downstream reads those three fields.

Each of the three passes through a soft threshold (`conv_gate`, `rift_gate`,
`shear_gate`) first. Any two plates differ in velocity, so every boundary
carries *some* strain, and drawing it literally paints a landform along every
Voronoi edge - the plate partition then shows through the finished map as a
web of creases. Gating means only boundaries doing real work leave a mark, and
the rest of the ocean floor and the continental interiors stay unbroken.

## Viewer

Left-drag or WASD/arrows to pan, wheel to zoom, `Z` to fit. Number keys pick a
layer, `[` `]` cycle. `R` regenerates with a new seed, `T` with the same one,
`G` overlays plate motion arrows, `P` saves the current layer, `F1` toggles help.
The status bar reads out elevation, plate, crust type, distance to the nearest
plate boundary, and drainage area under the cursor.

Two sliders top right set plate count (4-48) and world size (192-1024 wide, 4:3).
They apply on release, not while dragging, because a rebuild takes a moment -
roughly 0.4 s at 256x192, 2.4 s at 512x384, 7 s at 768x576 and 18 s at 1024x768.
`generate(..., on_stage=fn)` reports each stage as it starts, which is what the
viewer paints on the banner while it works.

Layers: relief, elevation without water, tectonic relief (pre-texture),
pre-erosion, plates, boundary classes, stress, drainage, erosion delta, slope,
land/coast, river width.

Three separate fields would otherwise trace the plate partition and give it
away even after the strain gating: the plate boundary itself, the crust step
between a continental and an oceanic plate, and the seafloor-age contours
around spreading centres. Each is domain-warped by noise before use
(`plate_warp` + `plate_warp_fine`, `crust_warp`, `age_warp`), at a scale of
tens of cells, so none of them reads as a clean Voronoi arc. Blurring alone
does not help - it turns a hard edge into a smooth curve of the same shape.

A range whose crest sits near the coast would otherwise drop straight into the
water, because boundary uplift peaks where the plate boundary runs and crust
thins towards the margin - at an ocean-continent boundary those coincide.
`coast_slope` caps how fast land may climb away from the shore, within
`coast_plain_zone` of it, which puts a plain and foothills in front of such a
range. The cap is self-gating: a coast that already rises gently is under it
and untouched, and the allowed slope is noise-modulated along the shore so
some coasts still come down steeply. Building land out into the sea instead
does not work - any offset keyed on distance from the shore raises a bench of
near-constant width, which reads as a bright ribbon traced round every
landmass with a shore-parallel river down the middle of it.

Colour is deliberately restrained below sea level. Shelf water and open ocean
are the same water, so the sea ramp spans a narrow band of blues and the
shading is flattened under water; seafloor structure reads as relief, not as a
bright halo drawn around every coast. `crust_sharpness` and `crust_blur` set
how gradually continental crust thins out, and so how wide the shelf is.

## Tuning

Every knob lives in `Config` in `terrain/world.py`, grouped by stage, and can be
overridden per call:

```python
from terrain import generate
world = generate(seed=12, n_plates=9, land_fraction=0.4, collision_h=1.8)
```

`world.height` is the final elevation with sea level at 0.0; `world.flow` is
drainage area in cells; `world.rivers` and `world.lakes` are masks;
`world.tect` holds every plate field and `world.water` every water field.

## Lakes and rivers

Every closed basin gets one flat surface at its spill elevation, with a depth
field beneath it, so lakes read as sheets of water rather than as dimples in the
terrain. Each lake's pour point is then notched and a channel is carved
downstream, which gives it a real outflow and a valley to drain through; filling
and routing are redone afterwards because the surface changed. Some shallow
basins drain away entirely at that point, which is the intended outcome.

Rivers are traced out of the D8 network as head-to-mouth paths, smoothed to
shed the eight-direction staircase, then displaced sideways by 1-D noise along
their own arc length - amplitude scaling with channel width, tapered to zero at
both ends so tributaries stay attached to their trunk. Width comes from
discharge (`w` proportional to the square root of drainage area), and the bed is
incised under the finished channel.

`world.water` holds it all: `lake_id`, `lake_level`, `lake_depth`, `width`,
`polylines`, plus `flow` and `receivers`. `world.surface` returns the water
surface where there is water and the bed everywhere else, which is what the
relief layer hillshades so lakes come out flat.

## Not done yet

No climate, biomes, or anything human-made. Lake outflow is carved, not
simulated - lake levels do not respond to a water balance, and a river cannot
change course after the fact.

Cost is roughly 0.4 s at 256x192, 2.4 s at 512x384 and 18 s at 1024x768, and
varies by 30-40% run to run on the same machine. Erosion is about half of it
and the river stage most of the rest.

The large sizes used to be far worse (59 s at 1024x768). Five changes, none of
which alter the output - the fill and the accumulation are verified
bit-identical to the straightforward versions in `test_terrain.py`:

- depression filling gets a **multigrid warm start**. Planchon-Darboux only
  ever lowers its estimate, so any upper bound converges to the same surface;
  solving first on block-maxima with block-AND outlets is a valid bound and
  cuts the fine grid from ~165 iterations to ~84.
- its convergence test was `np.allclose`, which built several full-size
  temporaries per iteration and cost as much as the work it was checking. The
  iteration is monotone, so "did anything drop" is the whole test.
- the 3x3 minimum is separable by hand into six slice-wise `np.minimum` calls
  with reused buffers: ~9x faster than `ndimage.minimum_filter` at map sizes,
  bit-identical.
- thermal creep summed its eight transfers into one `delta` instead of holding
  eight excess fields (48 MB live at 1024x768), and folds the rate and the land
  mask into a single factor.
- flow accumulation peels the drainage tree a level at a time instead of
  walking cell by cell in elevation order, and each Perlin octave evaluates its
  four lattice corners once rather than six times.
