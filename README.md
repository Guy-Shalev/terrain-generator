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
| coastline | `terrain/elevation.py` | sea level by quantile, a bounded rise per cell away from the shore, slope-compensated fray and drowned inlets on continent-ocean margins, then noise frays the shoreline |
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

Six sliders top right, in two groups. The world: plate count (4-48), size
(192-1024 wide, 4:3), `land_fraction` as a whole percent (shown as land/sea),
and `margin_h` (0.00-0.40), the fray on continent-ocean margins. Then the
water: river count and lake count.

Both counts are cutoffs in the config and run backwards - a river needs
`river_threshold` of drainage area, a lake `lake_min_depth` of depth, so
raising either leaves fewer. The sliders carry a divisor instead, counting
upwards the way a reader expects, with 6 reproducing the config defaults.
Measured at 320x240: rivers 28 / 83 / 462 at slider 2 / 6 / 30 and lakes
0 / 10 / 16 at 1 / 6 / 30, each moving its own target and leaving the other
alone. Channel width (`river_width`, and `river_width_max` which caps it) and
meander amplitude (`meander_amp`) stay config-only.

Sea level is a quantile of the elevation field, so the whole land range works
and the result lands within a point or so of the setting; 0 and 100 are
degenerate but do not fail - an all-ocean world simply has no rivers. They
apply on release, not while dragging, because a rebuild takes a moment -
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

Where a continental plate meets an oceanic one the coastline used to come out
visibly smoother than the rest - long clean sweeps where an open coast has
inlets, lobes and offshore islands. The crust step across a plate-type
boundary is several times steeper than a noise-drawn coast, and `fray_coast`
perturbs *elevation*, so on a margin the same noise moves the shoreline about
half as far. `erode_margins` works on those margins only, gated by
`margin_zone`: `cont_self` and `cont_other` are nearest-boundary pulls and so
are defined everywhere, and without that gate a continental interior far from
any ocean still reads as a full type mismatch.

It does two things. The fray is scaled by the local slope against the median
coastal slope of that map, which restores the travel and stays
resolution-independent where a fixed slope constant would not. Then drowned
inlets are cut into the coast, from *ridged* noise rather than plain fBm
because its crests run in lines and a line cut into a coast is a ria, where
thresholded plain fBm digs round pits that read as craters. The cut is clipped
non-negative and subtracted, so it only ever removes land.

Both terms are deliberately **coarse**, which matters more than their depth.
The wanted effect is a coast that wanders, not one that is speckled, and fine
noise near the zero contour does not bend a coastline - it perforates it.
Cutting inland has the same failure in mirror: a channel centred behind the
shore strands the basin it cuts off instead of opening into the sea. Enclosed
bodies of water with no route to the ocean, counted against a stage-off
baseline, across four seeds:

| seed | stage off | fine noise, cuts inland | coarse, cuts on the shore |
|---|---|---|---|
| 7 | 10 | 33 | 7 |
| 3 | 0 | 43 | 4 |
| 21 | 1 | 27 | 4 |
| 5 | 1 | 21 | 5 |

The coarse settings also move the shoreline *further* (2.43 to 3.22 cells),
so there was nothing to trade off: the fine detail was buying holes, not
irregularity.

Everything this stage adds is clipped, and that is not decoration. Twice a
term here raised mountains along the shoreline instead of moving it:

- A fringing archipelago offshore, since removed. The idea is sound - islands
  make new coastline, where nudging a contour only ever redraws one line - but
  a gaussian window centred 6 cells offshore with a width of 8 still carries
  0.75 of its weight *at* the shoreline and 0.32 of it six cells inland. A
  retry needs a window narrow relative to its offset, or one masked to
  strictly negative `sd`.
- The fray itself, which was the one term left uncapped. fBm reaches about 3.5
  standard deviations and `margin_gain_max` multiplies it by up to three, so
  it could add over a unit of elevation - and it did so precisely where the
  coast is steepest. Uncapped it raised 648 cells by more than 0.4, all but a
  handful within twelve cells of the shore, taking the map's summit from 0.49
  to 0.86. Clipped to one standard deviation that falls to 130 cells, the
  summit returns to 0.49 and shoreline travel is unchanged at 1.8 cells - the
  clip costs nothing it was supposed to be doing.

Two things about this were not obvious and cost real time:

- **Mask the corridor by distance, not elevation.** A band of fixed elevation
  is only `band/slope` cells wide on the ground - about five on a margin - so
  the shoreline cannot travel past the point where its own mask has faded. It
  caps the stage hardest at the very coasts it exists to fix. Measured, ten
  times the amplitude bought 0.8 of a cell. A distance corridor is the same
  width whatever the slope.
- **fBm here has a standard deviation near 0.18, not 1.** An amplitude knob
  used raw is about five times weaker than it reads, which is why the island
  term sat at 0.055 against a shelf 0.236 deep and never surfaced. Both noise
  fields are normalised by their own spread, so `margin_h` is elevation per
  standard deviation and the knobs mean what they say. Reasoning about travel
  as amplitude/slope overstates it by the same factor; the empirical
  before/after shoreline distance is the measure to trust.

Displacing the field sideways instead is the obvious alternative and does not
work. A domain warp moves the contour by the displacement itself whatever the
slope, which is the appealing part, but that displacement is coherent over its
own wavelength: long waves slide the whole margin across as a smooth arc, and
short ones at a useful amplitude fold the field. Across four seeds and a sweep
of frequency and amplitude it never shifted the shoreline's roughness by more
than a couple of percent at any scale.

Colour is deliberately restrained below sea level. Shelf water and open ocean
are the same water, so the sea ramp spans a narrow band of blues and the
shading is flattened under water; seafloor structure reads as relief, not as a
bright halo drawn around every coast. `crust_sharpness` and `crust_blur` set
how gradually continental crust thins out, and so how wide the shelf is.

## Map size

`Config` mixes two unit systems, and it has to. Noise `periods` counts are
relative to the map, so they scale with it for free; everything else - warp
amplitudes, blur sigmas, the falloff widths of every landform - is in pixels
and does not. Left alone, growing the map shrinks every pixel quantity
relative to the world: boundary relief keeps a fixed pixel width and narrows
into a hard crease, the warps that hide the Voronoi partition become too small
to hide it, and straight plate edges surface in the open ocean. Measured on
seed 7 as boundary slope over the map's median slope:

| width | unscaled | scaled |
|---|---|---|
| 384 | 1.22 | 1.37 |
| 640 | 1.50 | 1.32 |
| 1024 | 1.99 | 1.31 |

So `generate` scales every pixel-denominated knob by `width / ref_width`
before anything runs (`_scale_to_size` in `terrain/world.py`, listed in
`_PX_FIELDS`). Rises per cell - `coast_slope`, `talus`, `outlet_carve_slope` -
scale the other way, and `lake_min_area` by the square, being an area. It
returns a copy, because the viewer keeps one `Config` and regenerates from it
and must not have it rescaled underneath it each time. Set `ref_width` equal
to `width` to turn the whole thing off.

`erosion_k` is deliberately left out: stream power couples drainage area to
slope in cell units and does not follow any single factor. Land area lands
within half a point across 256-1024 as it is.

Resolution is not fully free even so - octave counts are fixed, so the finest
noise sits a fixed number of octaves below the map rather than at a fixed
pixel size, and a big map carries proportionally less fine detail.

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

A gorge is only cut where the outflow clears `river_threshold` - the same bar a
channel has to clear to be drawn as a river. Carving used to ignore it, and the
two then disagreed: the gorge is real terrain, but with no river drawn in it a
high threshold left dry trenches winding across the map. At 384x288 that was 1%
of the dug cells at the default threshold and 30% at six times it, which is
where it becomes obvious. Discharge at the pour point is the right test because
it is exactly what would flow down the gorge, and it is measured on the
*filled* surface, where a lake's whole catchment already routes through its
spill point. The erosion stage has already accumulated that field and its
caller was discarding it, so the test costs nothing.

The trade is that basins whose outflow is under the bar keep their water
instead of being drained: on seed 7 at 384x288, 9 lakes become 45, and lake
area goes from 0.3% of the map to 1.0%. Land area and river count are
unchanged. They read as small tarns rather than as speckle, and it is the
consistent answer - a basin too small to feed a river is too small to cut a
gorge either.

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

Cost is roughly 0.35 s at 256x192, 1.8 s at 512x384 and 14 s at 1024x768, and
varies by 30-40% run to run on the same machine. Erosion is about half of it
and the river stage most of the rest.

The large sizes used to be far worse (59 s at 1024x768). Seven changes, none of
which alter the output - the fill, the accumulation, the halo and the handed-in
routing are verified bit-identical to the straightforward versions in
`test_terrain.py`:

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
- neighbour reads go through **`grid.Halo`**, a one-cell padded copy carrying
  the same wrap-x and clamp-y edges, so all eight offsets are slices of one
  buffer. `shift` is a broadcast fancy-index gather - it reads the whole grid
  out of order into a fresh array - and it was the most expensive single thing
  the generator did: 928 calls and over a second of a six second build at
  768x576, three quarters of them from the thermal loop. One contiguous copy
  replaces sixteen gathers a sweep, taking a thermal iteration's neighbour
  reads from 19.4 ms to 0.11 ms. Routing also caches the flat index grid a
  neighbour maps to, which depends only on shape and offset.
- the river stage is **handed the erosion stage's final fill and routing**
  instead of recomputing them. Erosion signs off by filling and routing its
  finished surface, `carve_outlets` opens by needing exactly that, and the
  result was being thrown away in between - a third of a second at 1024x768.
  It is only valid before the first notch, so it is used once and dropped.

Together those two took erosion 1.5-1.7x faster and the river stage up to
2.5x, and the whole build from 17.2 s to 14.1 s at 1024x768.
