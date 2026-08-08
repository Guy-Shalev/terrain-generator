# Terrain Generator

Procedural world terrain from simplified plate tectonics, in NumPy + SciPy, with
a Pygame viewer. The map is a cylinder: it wraps in x, clamps in y. Crust is
tapered towards the poles (`polar_start`, `polar_ocean`) so the world closes
with ocean instead of slicing continents off at the edge.

```bash
python viewer.py --seed 7 --size 640x448
```

```bash
python preview.py 7          # layers to out/, engine export to out/export/seed7/
```

```bash
python test_terrain.py       # invariant checks
```

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| plates | `terrain/tectonics.py` | blue-noise seeds, noise-warped *additively weighted* Voronoi partition, per-plate velocity, crust type and buoyancy |
| strain | `terrain/tectonics.py` | divergence and curl of the piecewise-constant velocity field; boundary values propagated inland by a nearest-boundary distance transform |
| crust | `terrain/tectonics.py` | continuous continentality: plate type biases it, fBm decides the outline |
| relief | `terrain/elevation.py` | isostatic base plus collision ranges, trenches, arcs, cordilleras, rifts, mid-ocean ridges, transform scars, seafloor ageing |
| hotspots | `terrain/elevation.py` | island chains smeared along each plate's own motion |
| texture | `terrain/elevation.py` | warped ridged fBm, amplitude weighted by local relief and tectonic activity |
| coastline | `terrain/elevation.py` | sea level by quantile, bounded rise per cell away from the shore, slope-compensated fray and drowned inlets on continent-ocean margins |
| pre-rain | `terrain/climate.py` | a first rain pass on the un-eroded surface, to weight the carving |
| erosion | `terrain/hydrology.py` | thermal creep, depression filling, D8 routing, rain-weighted flow accumulation, stream-power incision, repeated |
| temperature | `terrain/climate.py` | latitude curve, moist-dependent lapse rate, continentality damped by sea ice |
| climate | `terrain/climate.py` | moisture capacity from sea surface temperature, gyre currents, zonal rain belts, moisture marched downwind for orographic shadows, runoff and evaporation |
| lakes and rivers | `terrain/rivers.py` | lake water balance, outlet carving, polyline extraction, meandering, discharge-based width, channel incision |
| biomes | `terrain/climate.py` | biomes from temperature against rain relative to evapotranspiration |
| trees | `terrain/trees.py` | canopy cover from moisture, timberline, gallery forest and grove noise; scattered into individual trees |

Boundary type is never assigned by hand. Velocity is constant inside each plate,
so its derivatives are non-zero only where plates meet: negative divergence is
convergence, positive is rifting, large curl with little divergence is a
transform fault. Each passes a soft threshold (`conv_gate`, `rift_gate`,
`shear_gate`) so only boundaries doing real work leave a mark — otherwise the
Voronoi partition shows through the finished map as a web of creases.

For the same reason the three fields that would trace the partition — the
boundary itself, the crust step, the seafloor-age contours — are domain-warped
before use (`plate_warp`, `crust_warp`, `age_warp`).

## Viewer

Left-drag or WASD/arrows to pan, wheel to zoom, `Z` to fit. Number keys pick a
layer, `[` `]` cycle. `R` regenerates with a new seed, `T` with the same one,
`G` overlays plate motion arrows, `P` saves the current layer, `E` exports the
world for an engine, `F1` toggles help. The status bar reads out elevation,
plate, crust type, distance to the nearest boundary, drainage area, temperature
and biome under the cursor.

Layers: main view, elevation without water, tectonic relief (pre-texture),
pre-erosion, plates, boundary classes, stress, rainfall, temperature, biomes,
trees, biome legend, drainage, erosion delta, slope, land/coast, river width,
canopy.

`biome legend` is the same classification drawn to be read rather than to sit
under a relief: one distinct colour per biome from `LEGEND_COLORS`, flat and
unsoftened so a cell matches its swatch exactly, internal borders where two
land biomes meet, and a key naming every biome the world actually has. Borders
skip the coast - the biome changes at every coastal cell and the sea ramp
already draws that line better - and skip rivers and lakes, where a border
reads as something built there. The key goes in whichever corner has the least
land under it.

Seven sliders top right. World: plate count (4-48), size (192-1024 wide, 4:3),
`land_fraction` as a percent, `margin_h` (0.00-0.40). Water: river count, lake
count. Climate: `temp_offset`, -20 to +20 degrees.

Both counts are cutoffs in the config and run backwards (`river_threshold`,
`lake_min_depth`), so the sliders carry a divisor instead and count upwards,
with 6 reproducing the config defaults. Sliders apply on release, not while
dragging, because a rebuild takes a moment. `generate(..., on_stage=fn)` reports
each stage as it starts, which is what the viewer paints on the banner.

## Tuning

Every knob lives in `Config` in `terrain/world.py`, grouped by stage, and can be
overridden per call:

```python
from terrain import generate
world = generate(seed=12, n_plates=9, land_fraction=0.4, collision_h=1.8)
```

`world.height` is the final elevation with sea level at 0.0; `world.surface` is
the water surface where there is water and the bed elsewhere; `world.flow` is
drainage area weighted by runoff; `world.runoff` is that weight; `world.rivers`
and `world.lakes` are masks; `world.tect` holds every plate field, `world.water`
every water field (`lake_id`, `lake_level`, `lake_depth`, `width`, `polylines`,
`widths`, `flow`, `receivers`), and `world.trees` the canopy `cover` field plus
the `pos`, `kind` and `size` of every tree.

### Map size

Noise `periods` counts are relative to the map and scale for free; warp
amplitudes, blur sigmas and landform widths are in pixels and do not. So
`generate` scales every pixel-denominated knob by `width / ref_width` before
anything runs (`_scale_to_size` in `terrain/world.py`, `_PX_FIELDS`). Rises per
cell (`coast_slope`, `talus`, `outlet_carve_slope`) scale inversely, and
`lake_min_area` by the square. It returns a copy, so the viewer's own `Config`
is not rescaled underneath it each rebuild. Set `ref_width == width` to disable.

`erosion_k` is deliberately left out — stream power couples area to slope in
cell units and follows no single factor. Octave counts are fixed, so a big map
carries proportionally less fine detail.

## Climate

Moisture capacity comes from the temperature under each cell at
Clausius-Clapeyron's 7% per degree (`rain_capacity`), which is why the equator
is wet without a band being drawn there, and why temperature is computed
*before* rain. Zonal belts (`cos(3 pi lat)`, wobbled) add what capacity cannot:
the dry subtropics, from air descending at 30 degrees.

Gyre currents tell the two sides of an ocean apart — cold and equatorward down
the eastern side (Benguela, Humboldt, California), warm and poleward up the
western (Gulf Stream, Kuroshio). `_shore_weight` marches direction by direction
rather than using a distance transform, since "how near is any land" is exactly
what this must not know. The field is zero-meaned over sea, so currents move
heat rather than add it.

Moisture is then marched downwind, one column at a time — easterly in the
tropics and poles, westerly between. A parcel drops `rain_base` per land cell
and `rain_orog` per unit of upwind climb, takes moisture back over sea, and gets
`rain_recycle` of each fall handed back. The first lap is thrown away (a
cylinder has no dry upwind edge). Recycling is load-bearing: without it, loss
inland is a pure exponential and half of every landmass comes out rainless. The
field is normalised to average one over land, so it drops into
`hydrology.accumulate` as weights without moving what `river_threshold` means.

Erosion takes the same weights off the `pre-rain` pass — erosion sharpens a
range without moving it, so the shadow falls in the same place either way.
`erosion_k` was raised from 0.0018 to 0.009 for this to reach the surface.

`lake_evap` is aridity in units of what a land cell sheds: `(E/P - 1) / 0.3`,
default 3.0 for a semi-arid world, 0 reproducing fill-to-the-rim. Local
evaporation is `lake_evap / runoff` floored at `rain_evap_cap`, so the same
basin is a full lake in a wet belt and a pan in a shadow.

## Temperature and biomes

Latitude sets the baseline (`temp_equator` to `temp_pole`, as its square),
height takes `temp_lapse` off it, and distance from the sea sets `swing`, the
seasonal half-range, saturating within `temp_cont_reach`. `temperature` returns
a pair because mean annual cannot tell taiga from tundra — what decides is the
warmest month.

`temp_lapse` is 38 C per unit of height, about half a true 6.5 C/km against this
terrain's scale, because the generator carries far more high ground than Earth
does. It varies with the *rank* of local runoff within this world's land
(`temp_lapse_moist`), since moist air cools at ~5 C/km against dry air's 9.8.
The rank is the midpoint of the two insertion points, or tied ground biases the
whole map's lapse rate. That makes stage order a loop on paper, cut by running
temperature twice — flat rate to feed the rain, runoff-aware to feed the biomes.

Biomes come from temperature against **rain relative to heat**:
`precip_mean_mm * runoff` over potential evapotranspiration, read off
biotemperature sampled twelve times around `swing`. That division is what
separates a desert from tundra. Permanent ice (<0) and the tree line (<6)
are checked first, on the warmest month; then five moisture bands against six
temperature bands:

|  | polar | subpolar | cool | temperate | subtropical | tropical |
|---|---|---|---|---|---|---|
| **arid** | desert | desert | desert | desert | desert | desert |
| **semiarid** | tundra | tundra | steppe | steppe | shrubland | savanna |
| **subhumid** | tundra | taiga | taiga | temp. forest | trop. seasonal | trop. seasonal |
| **humid** | tundra | taiga | temp. forest | temp. forest | temp. forest | trop. rainforest |
| **perhumid** | tundra | taiga | temp. rainforest | temp. rainforest | trop. rainforest | trop. rainforest |

The arid row is one biome across all six columns. Desert is a rainfall class -
that row is picked by moisture index alone - and the hot/cold split this used to
carry drew its line at the mode of the driest ground's own temperature
distribution, so cells half a degree apart came out as two biomes with two names
and two colours. Nothing downstream ever read the difference.

Both sets of cuts are shifted off textbook values because this generator's
climate is narrower than Earth's; the resulting biome shares land within a few
points of Earth's, except ice, which is low because the polar taper leaves
little polar land.

`biome_blur` smooths the climate fields *before* banding (so single hills do not
speckle the map) and changes `world.biome`. `biome_soften` blurs the *colours*
after, masked to land and divided by the blurred mask so coasts stay sharp;
nothing but the renderer reads it.

`temp_offset` slides the whole latitude curve, with three couplings that are
exactly nothing at slider 0: `precip_mean_mm` scales by `rain_per_degree`;
`temp_polar_amp` weights the shift towards the poles (a warmer world is a
flatter one); and continentality measures distance to *open* water, so a frozen
sea stops moderating its coast. Without those, desert ran 3% of land at -10 and
52% at +10; it now stays inside Earth's own 25-30% across the range, while ice,
rainforest and coastal continentality move instead.

## Trees

The one stage that reads the finished world and changes nothing in it — it runs
last, takes only finished fields, and draws from its own `default_rng([seed, 3])`
so it cannot move the river meanders on existing seeds.

Two outputs: a continuous `cover` field (canopy fraction per cell, what an
engine scatters from) and a point set (what makes a forest look like trees).
Four continuous terms, none derivable from the twelve-way biome cut:

- **Moisture** sets it, off `climate.moisture_index`, smoothstepped between
  `tree_mi_open` and `tree_mi_closed`, straddling the semiarid/subhumid cut.
- **The timberline** fades over `tree_line_fade` degrees of warmest month.
- **Gallery forest** puts trees along rivers in country too dry for them, scaled
  by `(1 - cover)` so it does nothing in a rainforest. Gated by the timberline.
- **Groves** clump it, applied to the moisture index *before* the curve — the
  curve saturates, so noise on the output leaves a closed forest with no
  clearings.

`scatter` places trees on a jittered grid — one candidate per `tree_spacing`
square, kept with probability `cover`. `tree_spacing` is deliberately **not**
scaled with map size, the only pixel knob that is not: what must stay constant
is how a forest looks, not how much ground a tree owns. Spacing must clear the
stamp (2.0 against a ~3-cell crown) or a closed canopy draws as a flat mass.

Four kinds, because that is how many silhouettes read apart at map scale.
`KIND_BY_BIOME` is a lookup, blurred by `kind_mix` so each tree draws its kind
from the local mixture — otherwise a taiga cap draws as a disc with a hard edge.
Masked to land and divided by the blurred mask, or a coastal forest loses trees
to the water offshore.

The main view draws them at `tree_relief` rather than full opacity — there they
are grain over the hypsometric ramp, not the subject. Ground is pulled
`GROUND_DESAT` towards its own luminance over land, or green trees vanish into
green ground. `paint_trees` composites kind by kind (four vectorised passes, not
thirty thousand slice assignments) and clips alpha rather than normalising, so a
closed canopy saturates into a mass while a savanna stays separate trees.

## Lakes and rivers

A lake holds the water surface its own catchment can keep, not the one its rim
allows: it loses `lake_evap` per cell of surface and gains its inflow, so it
settles at inflow / `lake_evap` — and with the basin's bed heights sorted, that
area *is* the level. Three outcomes from one rule: fill to the rim and spill,
sit part full and spill nothing (terminal), or be a pond in a wide hollow. Only
the connected piece holding the basin's deepest cell is kept, since a level
contour over a textured floor is not a connected set.

The accumulation is *gated*, not patched: each basin's cells point at its exit
so the whole inflow is finalised at one cell, where the running total is
replaced by the outflow the balance allows. Water a lake keeps leaves the
network, so channels below a terminal lake come out dry. D8 does not converge a
basin on its exit by itself — the fill's epsilon tilt drains it through the rim
strip outside the label.

Each spilling lake has its pour point notched and a channel carved downstream;
filling and routing are redone after. A gorge is only cut where outflow clears
`river_threshold`, or a high threshold leaves dry trenches winding across the
map.

That rewiring is right for accounting and wrong for geometry — it is a shortcut
across the lake bed, and channels cut at the basin rim stop several cells short
of the pond. So geometry gets its own receivers: a breadth-first tree grown out
from that basin's own lake (not the nearest water, or a path climbs the rim at
the sea), which cannot contain the two-cell loops a restricted gradient does.
Widths take a running maximum along each path, restoring the "discharge only
grows downstream" property the gate broke. `_connect_strays` is the backstop for
junctions the meander pulled apart; `test_rivers_reach_water` holds it to zero
strays at three sizes.

`lake_min_area` is 30 cells, not 6: the balance leaves more puddles, and a
puddle on a river's course reads as a break in it rather than a pond. Being an
area, it scales with the square of map size.

Rivers are traced out of the D8 network head-to-mouth, smoothed off the
eight-direction staircase, then displaced by 1-D noise along their arc length,
tapered to zero at both ends so tributaries stay attached. Width is
`river_width` times drainage area to `river_width_exp` (0.45, which holds the
trunks in without touching the headwaters), measured against `river_width_ref`
and never against `river_threshold` — dividing by the threshold made asking for
more rivers widen every river. Rivers are clipped to land after incision, so the
bed is still cut to the shore.

## Getting a world out

`terrain/export.py` writes six files, into a new folder per export under
`out/export/` (`seed7`, then `seed7-2` — an export is a thing you keep):

| file | what it is |
|---|---|
| `_height16.png` | the bed, 16-bit greyscale |
| `_lakes16.png` | lake depth, 16-bit, zero everywhere dry |
| `_biome.png` | `climate` biome index per cell, 8-bit, names in the sidecar |
| `_canopy.png` | canopy fraction, 8-bit, 0 to 1 |
| `_rivers.json` | channel centrelines and per-node width |
| `_config.json` | the seed, every knob, and a stamp of the code that read them |

Sixteen bits for terrain because 256 codes across a couple of height units
terraces every plain; eight for biome indices (they are labels, greyscale not
palettised) and for canopy (a density read to 1/255 is finer than any placement
rule asks). PNGs are written by hand — no interlacing, filter 0, three chunks
and a zlib stream, smaller than the dependency would be.

Each heightmap gets a JSON sidecar: `lo + code * units_per_code` puts it back in
generator units, `metres_per_unit` puts it on the ground (chosen, not measured —
it is the reading `temp_lapse` was calibrated against), `sea_code` says where
the coastline landed, `wrap: x` says the map is a cylinder. Pass `lo`/`hi` when
two exports must line up:

```python
m = export.heightmap(world.height, "height16.png")
export.heightmap(world.surface, "surface16.png", m["lo"], m["hi"])
```

`_config.json` is what rebuilds the world — `generate(export.load_config(path))`
returns it bit for bit. It overwrites `ref_width` with the map's own width, or
`generate` would scale the already-scaled config a second time and silently
build a different world at any size but the reference. `generator` carries a
hash of the package source and the two library versions, conservative in one
direction only: equal means the same map, different means *unverified*.
`load_config` warns on a mismatch, loads anyway, and drops unrecognised knobs.

## Not done yet

Nothing human-made. Trees are one-directional by design — vegetation does not
resist erosion, which would need a crude pre-canopy off the pre-rain pass and
would move every coastline on every seed. Climate is annual mean plus a seasonal
half-range: enough for Whittaker, not for Köppen-Geiger, which keys on *when*
rain falls. Ocean currents set the sea's temperature but not the land's.
Mediterranean shrubland is placed as subtropical semiarid, which is the right
corner of the diagram but not the real test (a dry summer). Nothing goes fully
dry — a basin catches at least its own footprint, so playas need evaporation to
beat rain locally. Lake outflow is carved, not simulated: a river cannot change
course after the fact. Meanders are noise, not migration — curvature-driven
migration with neck cutoffs was built and reverted, since at any visible rate
the channels wandered off their D8 course and sat oddly in their valleys.

## Cost

Roughly 0.5 s at 256x192, 2.3 s at 512x384 and 16 s at 1024x768, varying 30-40%
run to run. Erosion is about half, the river stage most of the rest; climate and
trees are about 2% each.

Down from 59 s at 1024x768, by seven changes that alter no output (the fill, the
accumulation, the halo and the handed-in routing are verified bit-identical in
`test_terrain.py`):

- depression filling gets a **multigrid warm start** — Planchon-Darboux only
  lowers its estimate, so any upper bound converges to the same surface. ~165
  iterations down to ~84.
- its convergence test was `np.allclose`, building full-size temporaries per
  iteration. The iteration is monotone, so "did anything drop" is the whole test.
- the 3x3 minimum is separable by hand into six slice-wise `np.minimum` calls
  with reused buffers: ~9x faster than `ndimage.minimum_filter`, bit-identical.
- thermal creep sums its eight transfers into one `delta` instead of holding
  eight excess fields (48 MB live at 1024x768).
- flow accumulation peels the drainage tree a level at a time instead of walking
  cell by cell in elevation order; each Perlin octave evaluates its four lattice
  corners once rather than six times.
- neighbour reads go through **`grid.Halo`**, a one-cell padded copy carrying
  the same wrap-x and clamp-y edges, so all eight offsets are slices of one
  buffer. `shift` was a fancy-index gather and the single most expensive thing
  the generator did — a thermal iteration's neighbour reads went from 19.4 ms to
  0.11 ms.
- the river stage is **handed the erosion stage's final fill and routing**
  instead of recomputing them. Only valid before the first notch, so it is used
  once and dropped.
