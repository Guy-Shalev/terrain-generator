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
python preview.py 7          # layers to out/, engine export to out/export/seed7/
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
| pre-rain | `terrain/climate.py` | a first rain pass on the un-eroded surface, to weight the carving |
| erosion | `terrain/hydrology.py` | thermal creep, depression filling, D8 routing, rain-weighted flow accumulation, stream-power incision, repeated |
| temperature | `terrain/climate.py` | latitude curve, moist-dependent lapse rate, continentality damped by sea ice |
| climate | `terrain/climate.py` | moisture capacity from sea surface temperature, gyre currents warming and cooling it by basin side, zonal rain belts, moisture marched downwind for orographic shadows, runoff and evaporation fields |
| lakes and rivers | `terrain/rivers.py` | lake water balance, outlet carving, polyline extraction, meandering, discharge-based width, channel incision |
| biomes | `terrain/climate.py` | temperature from latitude, lapse rate and continentality; biomes from temperature against rain relative to evapotranspiration |
| trees | `terrain/trees.py` | canopy cover from moisture, timberline, gallery forest and grove noise; scattered into individual trees |

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
`G` overlays plate motion arrows, `P` saves the current layer, `E` exports the
world for an engine, `F1` toggles help.
The status bar reads out elevation, plate, crust type, distance to the nearest
plate boundary, drainage area, temperature and biome under the cursor.

Seven sliders top right, in three groups. The world: plate count (4-48), size
(192-1024 wide, 4:3), `land_fraction` as a whole percent (shown as land/sea),
and `margin_h` (0.00-0.40), the fray on continent-ocean margins. Then the
water: river count and lake count. Then the climate: `temp_offset`, in whole
degrees from -20 to +20, which warms the poles harder than the equator and takes
the rain and the sea ice with it.

Both counts are cutoffs in the config and run backwards - a river needs
`river_threshold` of drainage area, a lake `lake_min_depth` of depth, so
raising either leaves fewer. The sliders carry a divisor instead, counting
upwards the way a reader expects, with 6 reproducing the config defaults.
Measured at 320x240: rivers 28 / 83 / 462 at slider 2 / 6 / 30 and lakes
0 / 10 / 16 at 1 / 6 / 30, each moving its own target and leaving the other
alone. Channel width (`river_width`, and `river_width_max` which caps it),
meander amplitude (`meander_amp`) and the rest of the climate group stay
config-only.

Sea level is a quantile of the elevation field, so the whole land range works
and the result lands within a point or so of the setting; 0 and 100 are
degenerate but do not fail - an all-ocean world simply has no rivers. They
apply on release, not while dragging, because a rebuild takes a moment -
roughly 0.4 s at 256x192, 2.4 s at 512x384, 7 s at 768x576 and 18 s at 1024x768.
`generate(..., on_stage=fn)` reports each stage as it starts, which is what the
viewer paints on the banner while it works.

Layers: main view, elevation without water, tectonic relief (pre-texture),
pre-erosion, plates, boundary classes, stress, rainfall, temperature, biomes,
trees, drainage, erosion delta, slope, land/coast, river width, canopy. The
rainfall layer is ranked within the land distribution rather than scaled by it -
rain is skewed enough that a linear ramp paints every interior the same tan.
`trees` takes the last number key ahead of `drainage`, which moves to `[` `]`:
drainage is a layer you open to find out why a river went where it did, and
trees is the world.

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
drainage area weighted by runoff, so still in cells on average; `world.runoff`
is that weight; `world.rivers` and `world.lakes` are masks; `world.tect` holds
every plate field and `world.water` every water field, including the per-node
`widths` of each polyline. `world.trees` holds the canopy `cover` field and the
`pos`, `kind` and `size` of every individual tree.

## Climate

Rain sets two things: how much water each cell contributes to the network, and
how hard a lake has to work to stay wet. Both come out of one small model.

How much the air can carry comes from the temperature under it, at
Clausius-Clapeyron's own 7% per degree (`rain_capacity`): a tropical ocean hands
its air several times what a polar one does. That is why the equator is wet
without a band being drawn there, and it is the reason temperature is computed
*before* the rain now rather than after it.

Zonal belts keep the job only they can do. Wet on the equator, dry in the horse
latitudes, wet again at the storm tracks, as `cos(3 pi lat)`, with the latitude
they key on wobbling so they do not read as three ruler-straight stripes - but
the part that matters is the dry subtropics, which come from air *descending* at
30 degrees and not from how cold it is there. `rain_belts` went from 0.75 to
0.9 when capacity arrived, because the two were both drawing the equatorial
maximum and only one of them can also dig the trough at 30: measured over two
seeds at 384x288, desert ran 18% of land at 0.75, 23% at 0.85 and 29% at 0.95.
The belt scales the whole rain rate rather than its flat part: applied only to
`rain_base` it is invisible wherever there is relief, because the orographic
term is several times the base on any real slope.

The latitudinal profile that falls out, as mean runoff over land: **1.75** in
the tropics, **0.41** at 30 degrees, **0.75** at 60. That shape used to come
entirely from one hand-drawn cosine.

### Ocean currents

Latitude alone gives every ocean cell in a row the same temperature, so the only
thing that could tell two coasts apart was which way the wind blew over them.
A gyre turns at the edges of its basin and the two edges are not alike. Down the
*eastern* side of an ocean - the west coast of a continent - it carries water
towards the equator and pulls cold water up behind it: Benguela, Humboldt,
Canary, California. Up the *western* side it runs poleward and warm: the Gulf
Stream, the Kuroshio.

So the sign comes from which way the nearest shore lies. `_shore_weight` marches
that in one direction at a time, by the same recurrence the moisture uses and
for the same reason - a distance transform is isotropic, and "how near is any
land" is the one thing this must not know, since the shore behind a parcel of
water and the shore ahead of it are different facts.

The two latitude profiles differ because the two mechanisms do. Upwelling is a
subtropical band (`current_lat`) and fades either side of it. The warm limb
scales with latitude instead: it is an anomaly against the local mean, and at
the equator there is no meridional gradient left for it to carry up. The field
is averaged to zero over the sea, so currents move heat around rather than add
it - the temperature slider is calibrated, and a current field with a mean would
quietly bias every world against the number it was tuned to.

Measured against the same seeds with `current_cold` and `current_warm` at zero,
at 384x288. Sea surface temperature spread *within one latitude row* was 1.3-2.0
degrees, which is the wobble noise and nothing else; with currents it is
**4.1-4.4**. Downstream, runoff on land moves by 4-6% at the median and **17-18%
at the 90th percentile**, and **7-8% of land changes biome**. Desert goes up
about two points. Cost is 0.05 s of a 2.1 s build.

Sea ice barely notices, and that was the prediction that missed: the plan was
for warm water to keep a polar coast open and turn its tundra to taiga, but polar
sea here sits some 25 degrees below freezing and a 4 degree anomaly cannot cross
that. Only the narrow ring already near the ice edge flips, and the extent moves
by about 1%. What currents actually buy is rain on the mid-latitude coasts,
where the ocean is upwind.

Which is also why there is no Atacama. In the trades the wind blows east to west,
so a west coast's upwind is *continent* and a cold current sitting off it never
touches the marching air. The coastal deserts that do exist there are rain
shadows, and were already there.

Then moisture is marched downwind, one column at a time. Winds are zonal -
easterly in the tropics and at the poles, westerly between - so the march runs
along rows and the two groups are done in opposite directions. A parcel drops
`rain_base` of what it carries every land cell and `rain_orog` per unit of
upwind climb, takes moisture back up over the sea, and gets `rain_recycle` of
each fall handed straight back over land. The world is a cylinder, so there is
no upwind edge to start dry at: the first lap is thrown away and only the
moisture it leaves behind is kept.

That recycling term is not decoration. Without it the loss inland is a pure
exponential, and at rates that give a decent shadow half of every landmass came
out under a seventh of the mean with its rivers gone. With it, at 384x288 over
four seeds, land runoff runs 0.25 at the 5th percentile, 0.74 at the median and
2.7 at the 95th, and the windward side of a range gets **2.5 to 3.6 times** what
its lee does.

The field is normalised to average one over land, which is what lets it drop
into `hydrology.accumulate` as weights without moving what `river_threshold`
means - drainage area was already that field with every weight at one. Rivers
are drawn from the weighted flow, so they thin and vanish in a rain shadow.

Erosion takes the same weights, off a `pre-rain` pass run on the surface as it
stands before erosion. Chicken and egg: the rain that ought to weight the
carving is the rain the finished ranges cast their shadows with, and those
ranges do not exist yet - but erosion lowers a range and sharpens it without
moving where it is, so the shadow falls in the same place either way. Both rain
passes take the same derived stream, so the carving is weighted by the bands
the finished map is banded on rather than by ones offset from them by up to
`rain_wobble` of latitude.

**`erosion_k` had to move for this to be worth anything.** At the 0.0018 it
sat at, the fluvial term is small next to thermal creep, and weighting it by
rainfall took the cut on wet high ground from 4.2x the cut on dry ground to
only 4.7x - eleven percent, with the surface itself moving 0.0001 at the median
in a range spanning 1.8. Measured over three seeds at 384x288:

| `erosion_k` | wet/dry cut, flat weights | with rain | land |
|---|---|---|---|
| 0.0018 | 4.20 | 4.67 (+11%) | 28.0% |
| **0.009 (shipped)** | **3.07** | **4.35 (+42%)** | **27.9%** |
| 0.036 | 2.01 | 3.87 (+92%) | 27.5% |
| 0.108 | 1.57 | 3.54 (+126%) | 26.6% |

Note which column the weighting improves. Raising `k` on flat weights makes a
range *less* lopsided, not more - more incision everywhere evens the two flanks
out - and it is only with the rain in the accumulation that the extra carving
lands on the side that earns it. At 0.009 that is 3.07 against 4.35.

Land fraction holds all the way up the column, and at the shipped value the
mean cut is 0.0031 against 0.0019, so the extra carving arrives as texture on
the high ground rather than as a drowned world. Per seed the spread is wide -
7.92 against 5.94 on seed 7, 1.52 against 0.94 on seed 3 - because how much
rain shadow a world has at all depends on where its ranges stand relative to
the wind. The weighting costs 0.16 s of a 2.4 s build, all of it the extra rain
pass.

Evaporation is `lake_evap` divided by the local runoff, floored at
`rain_evap_cap`, so the same basin is a full lake in a wet belt and a pan in a
shadow. Over six seeds at 384x288, of 54 basins:

| | basins | mean flooded fraction | terminal |
|---|---|---|---|
| dry ground (runoff < 0.6) | 34 | 0.44 | 88% |
| wet ground (runoff >= 0.6) | 20 | 0.82 | 55% |

## Temperature and biomes

Rain on its own cannot say what grows anywhere. A dry cell at 60 degrees and a
dry cell at 25 get the same tan on the rainfall layer, and one is steppe while
the other is the Sahara. Temperature is the missing axis.

Three terms set it. Latitude gives the baseline, falling from `temp_equator` to
`temp_pole` as the square of it, on a latitude wobbled the same way the rain
belts are. Height takes `temp_lapse` off that, on land only. And distance from
the sea sets `swing`, the seasonal half-range: water holds its heat, so a coast
barely moves between seasons while an interior at the same latitude bakes and
then freezes. The continentality term saturates rather than growing linearly -
an ocean's moderating reach is spent within `temp_cont_reach` of it, and past
that one more cell inland changes nothing.

`swing` is why `temperature` returns a pair. Mean annual temperature cannot
tell taiga from tundra: a continental interior averages below freezing and
still grows forest, because its *summer* clears the tree line even though its
winter is far worse than any coast's. What decides is the warmest month.

`temp_lapse` is 38 C per unit of height, about half a true 6.5 C/km read
against this terrain's scale. That is deliberate. The generator carries far
more high ground than Earth does - the top tenth of its land sits above what
would be 2500 m - and at a true lapse rate that tenth freezes and the map comes
out a third taiga.

Biomes come from temperature against **rain relative to heat**, not rain. The
moisture axis is `precip_mean_mm * runoff` divided by potential
evapotranspiration, which Holdridge reads linearly off biotemperature - the
year averaged with every month below freezing counted as zero, sampled twelve
times around `swing`. That single division is the whole difference from a plain
Whittaker lookup, and it is what separates a cold desert from tundra: both are
dry in millimetres, but only one is dry relative to its own thirst. Get it
wrong in the obvious way - clipping the annual mean instead of the seasonal
cycle - and every freezing cell's evapotranspiration goes to zero, its moisture
index to infinity, and a quarter of the map to taiga.

Then five moisture bands against six temperature bands, with the two cases no
matrix handles checked first, because both key on the warmest month rather than
the mean: permanent ice below 0, and the tree line below 6.

|  | polar | subpolar | cool | temperate | subtropical | tropical |
|---|---|---|---|---|---|---|
| **arid** | cold desert | cold desert | cold desert | cold desert | hot desert | hot desert |
| **semiarid** | tundra | tundra | steppe | steppe | shrubland | savanna |
| **subhumid** | tundra | taiga | taiga | temp. forest | trop. seasonal | trop. seasonal |
| **humid** | tundra | taiga | temp. forest | temp. forest | temp. forest | trop. rainforest |
| **perhumid** | tundra | taiga | temp. rainforest | temp. rainforest | trop. rainforest | trop. rainforest |

Both sets of cuts are shifted off the textbook values, and for the same reason:
this generator's climate is narrower than Earth's. Its wettest land gets 2.4x
the mean where a real rainforest gets 3 to 5x, and its equator is only 5 C
above the tropical cut before the lapse rate takes the rest. Left on
Holdridge's own numbers the map bunched into two middle bands - deserts at 5%
of land against Earth's 20%, tropical rainforest at 1.5% against 6%, and a
third of everything temperate forest. The bands are the same provinces read off
a flatter distribution. Over five seeds at 384x288:

| biome | here | Earth |
|---|---|---|
| hot desert | 15% | 14% |
| taiga | 15% | 13% |
| steppe | 14% | 10% |
| temperate forest | 11% | 10% |
| cold desert | 10% | 6% |
| shrubland | 8% | 3% |
| tundra | 7% | 8% |
| tropical seasonal forest | 6% | 6% |
| savanna | 6% | 10% |
| temperate rainforest | 5% | 2% |
| tropical rainforest | 3% | 6% |
| ice | 1% | 5% |

Ice is low because the polar crust taper (`polar_start`, `polar_ocean`)
deliberately leaves little land at the poles - that is terrain, not climate.

Two blurs, doing different jobs. `biome_blur` smooths the climate fields
*before* they are banded: straight off the raw ones every hill that crosses a
cut drops a lone cell of another biome into the middle of a region and the map
reads as speckle. It only removes features narrower than itself, and a range
wide enough to have its own climate is much wider than that, so altitudinal
zonation survives it. It moves where a boundary falls, and so it changes
`world.biome`.

`biome_soften` blurs the *colours*, after. Classification is a hard cut - a
cell is one biome or another and `world.biome` says which - but nothing on the
ground changes over a single cell, and drawn literally every band boundary is a
stencil edge. The blur is masked to land and divided by the blurred mask, so a
transition inland is a gradient while the coast stays exactly as sharp as the
sea ramp draws it. Nothing but the renderer reads it.

Two layers draw it - `temperature`, absolute so that freezing sits at a fixed
place on the ramp, and `biomes` - and the main view mixes the biome
colour into its hypsometric tint at `biome_tint`, so height and vegetation read
off the same map. The trees go on over that at `tree_relief`; see below.

## Trees

The one stage that reads the finished world and changes nothing in it. Trees are
the end product - what the plates, the rain and the rivers add up to on the
ground - so the stage runs last, takes only finished fields, and feeds nothing
back. It draws from its own generator, `default_rng([seed, 3])`, for the reason
the climate stage does: drawing from the shared stream would move the river
meanders on every seed the generator has ever made. Verified - `height`,
`height_pre`, `height_eroded`, `biome`, `runoff` and `temp` all hash identically
to the commit before this one on seeds 7, 3 and 21, with the same river and lake
counts. Cost is 0.04 s of a 2.4 s build at 384x288.

Two things come out. A `cover` field, canopy fraction per cell, which is what an
engine scatters from; and a **point set**, which is what makes a forest look like
trees rather than like a green wash. The field is the intermediate and the points
are the product - the same move `rivers` makes by keeping polylines, because a
scatter *is* positions and rasterising it is the lossy step.

None of it is derivable from `biome.png`, and that is the point of the field
being continuous. The biome grid is a thirteen-way cut - a cell is temperate
forest or it is steppe - and every clearing, timberline and gallery strip in the
world lives in the ground between those two answers. Four terms, all continuous:

- **Moisture** sets it, off the same `climate.moisture_index` the biome bands are
  cut from, smoothstepped between `tree_mi_open` and `tree_mi_closed`. Those
  straddle the semiarid/subhumid cut at 0.7 on purpose: the classifier puts the
  forest boundary there, and this says the ground either side of it is thinning
  stands rather than a step.
- **The timberline** is a fade over `tree_line_fade` degrees of warmest month
  rather than the hard `TREE_C` cut a classifier needs. What a real mountain has
  is a band where the forest thins and then stops.
- **Gallery forest** puts trees along a river running through country whose own
  rainfall could not keep them - the Nile and the Okavango are green lines drawn
  on tan. Scaled by `(1 - cover)`, so it does nothing in a rainforest and
  everything in a savanna: it stands in for water the rain did not supply, and
  where the rain already did there is nothing left to add. Gated by the
  timberline too, or a river carries forest over the tree line with it.
- **Groves** clump the result, and the noise goes into the moisture index
  *before* the curve rather than onto the cover after it. The curve saturates at
  both ends, so noise applied to the output is flattened everywhere except the
  middle band and a closed forest comes out with no clearings in it at all.

The ordering that falls out, as mean cover over four seeds at 384x288, from one
moisture curve and no per-biome table anywhere:

| biome | cover | | biome | cover |
|---|---|---|---|---|
| temperate rainforest | 0.84 | | savanna | 0.36 |
| tropical rainforest | 0.83 | | shrubland | 0.29 |
| temperate forest | 0.66 | | steppe | 0.27 |
| tropical seasonal | 0.57 | | hot desert | 0.14 |
| taiga | 0.54 | | tundra | 0.11 |

Land averages 0.34 cover with 24% of it closed (>0.7), and in dry country a
riverbank carries **3.6 to 4.5 times** the canopy of ground eight cells away.
Desert is not zero and should not be: the only trees a desert has stand where the
water is, which is why `KIND_BY_BIOME` gives it palms.

`scatter` places them on a jittered grid - one candidate per `tree_spacing`
square, offset at random inside its own square, kept with probability `cover`.
That is a Poisson disc without the rejection loop, which at these densities buys
nothing a three-pixel stamp would show.

**`tree_spacing` is deliberately not scaled with map size**, and it is the only
pixel knob that is not. Every other one follows the map so a landform covers the
same fraction of the world at any resolution - but a tree is drawn at a fixed
stamp size, and holding its ground area fixed instead shrinks the canopy's
texture as the map grows. What has to stay constant here is how a forest looks,
not how many hectares a tree owns.

Two things about the drawing were wrong first and are worth keeping written down,
because both made the layer read as a slightly darker biome map:

- **Spacing has to clear the stamp.** A crown is about three cells across, so at
  the 1.3 it started on, a closed canopy piled four trees on every one that
  showed and the layer drew as a flat green mass at every zoom. At 2.0 the crowns
  touch and overlap slightly - which is what a closed canopy is - and a savanna at
  a third of the cover comes out as separate trees with ground between them.
- **The ground has to stop being green.** `LAND_RAMP` is green at every elevation
  a forest grows at, and so is half of `BIOME_COLORS`. Trees painted onto that
  are green on green and the canopy disappears into the ground it stands on. The
  base keeps its biome tint, so bare land still looks like the country it belongs
  to, but is pulled `GROUND_DESAT` of the way to its own luminance and warmed
  back - over land only, since draining the sea of colour too just looks broken.

Four kinds, because that is how many silhouettes read apart at map scale: a
spire, a dome, a flat crown on a bare stem, and a dot. `KIND_BY_BIOME` is a
look-up rather than a model - the climate that would decide leaf habit has
already been run and banded, and re-deriving it would be the same cut drawn
twice. Temperate rainforest is coniferous on purpose; the real ones are, from
Sitka to Valdivia.

That look-up on its own is a stencil, and it showed: a taiga cap on a mountain
came out as a solid disc of conifer with a hard line round it, because every
tree inside a thirteen-way cut gets the same answer and every tree one cell
outside gets a different one. So `kind_mix` blurs the **composition** at
`tree_mix`, and each tree draws its kind from the local mixture. This is the
mirror of `biome_soften`, which blurs how a boundary is drawn and changes
nothing about what is there; this changes what is there, and every individual
tree still keeps one kind and one silhouette. A cell in a transition does not
grow a half-conifer - it grows both, in the proportion its neighbourhood does.
Masked to land and divided by the blurred mask, like `biome_rgb`, or a coastal
forest loses a third of its trees to scrub because of the water offshore.

At 6 px, 15% of land is still a pure stand and the rest carries some mix, which
is the belt doing its job rather than mixing everything into an even scatter -
`test_tree_kinds_blend_across_a_boundary` pins both ends, and checks that
`tree_mix = 0` puts the hard lookup back. The kind is drawn *last* in `scatter`,
after positions and sizes, so adding it left both exactly where they were on
every existing seed.

The main view draws them too, at `tree_relief` rather than full opacity.
There they are texture and not the subject: the biome tint has already said
where forest is, and what the stamps add is grain over it - a canopy that looks
like canopy, an edge where it thins, and the gallery strips picking out rivers
the hypsometric ramp draws as one blue line through uniform green. At 1.0 they
bury the ramp and the layer stops being a relief map; at 0 they are off there,
which is what the `trees` layer is for. Same `paint_trees`, one alpha
multiplier, no second renderer.

`paint_trees` composites kind by kind, not tree by tree: thirty thousand small
slice assignments is a Python loop at map scale, four alpha grids is four
vectorised passes, and the only thing lost is painter's order *within* one kind,
which at three pixels a crown nobody can see. Alpha is clipped rather than
normalised - where a canopy closes the stamps overlap and the sum runs past one,
and that saturation is what makes a rainforest a solid mass while a savanna at a
third of the cover stays a field of separate trees.

## Height, and how fast it cools

`temp_lapse` is what a cell loses per unit of height, and it is not one number
any more. Rising air cools at about 9.8 C/km while it stays unsaturated and near
5 once it is condensing, because the latent heat it gives up pays back part of
the expansion - so a wet windward slope loses height far more gently than a
desert range at the same latitude, and its tree line rides up with it.
`temp_lapse_moist` is the spread either side: 0.25 gives a dry-to-wet ratio of
1.67 against the 1.96 the two adiabats really differ by.

It keys on the *rank* of the local runoff within this world's land, not on its
value, so the middle cell keeps exactly `temp_lapse` and the calibration that
number was chosen for survives being made local. The rank is the midpoint of the
two insertion points rather than one side of them - they agree on a continuous
field, but ground at exactly the same runoff has every tied cell taking the top
of its own run, which drags the mean rank above a half and biases the whole
map's lapse rate.

That makes the stage order a loop on paper: rain needs temperature for its
capacity, and temperature needs rain for its lapse rate. It is cut by running
temperature twice - once at the flat rate to feed the rain, once with the runoff
to feed the biomes - which costs 0.04 s at 384x288. Both passes take their own
generator off the seed rather than the shared stream: drawing here would shift
every later draw and move the river meanders on every seed the generator has
ever made. Verified - `height_pre` and `height_eroded` hash identically to the
commit before this one, while the rivers move, which is what a change to the
rain is supposed to do.

## What the temperature slider moves

`temp_offset` is a shift, not a setting: it slides the whole latitude curve and
never touches its ends. On its own that is the one thing warming does not do,
and it showed. Potential evapotranspiration climbed with the slider while
rainfall in millimetres could not move - `runoff` is normalised to average one
over land and cannot know the world got hotter - so the warm end turned the map
into sand: desert ran 3% of land at -10 and **52% at +10**. Three couplings, all
of them exactly nothing at slider 0, so the calibrated middle is untouched:

- **Rain follows temperature.** `precip_mean_mm` scales by `rain_per_degree` per
  degree of offset. Clausius-Clapeyron gives 7% per degree of what the air can
  *hold*; global rainfall is limited by the energy available to evaporate it and
  responds nearer 2-3%. The default 0.05 sits between them because it carries
  both jobs - it also stands in for how much moisture reaches an interior, which
  follows capacity more than the global mean. At 0.025 desert runs 11/27/40% at
  -10/0/+10; at 0.07, 44/27/23%; at 0.05, 25/27/30%.
- **The offset lands hardest on the poles.** A warmer world is a *flatter* one:
  the Eocene ran an equator-to-pole gradient near 30 C against today's 45.
  `temp_polar_amp` weights the shift by latitude, normalised over **land** - the
  weight averages to one across a uniform globe, but the polar taper keeps this
  world's land in the middle latitudes where the weight is below one, and a
  slider set to +10 delivered +6.9 C to the ground.
- **A frozen sea stops moderating its coast.** Continentality is distance to
  *open* water, not to any water: sea ice has a lid on it and behaves like land,
  which is why Siberia's coast is continental and Norway's is not. Computed in
  one extra pass - the first swing decides where the sea freezes, the second
  uses it - and the fixed point is not worth chasing, since the cells that would
  flip on a third pass are the ones sitting on the ice edge.

Measured over two seeds at 384x288:

| `temp_offset` | -10 | -5 | 0 | +5 | +10 |
|---|---|---|---|---|---|
| desert | 25.4% | 25.6% | 27.2% | 29.0% | 29.8% |
| ice | 16.1% | 6.6% | 1.5% | 0.2% | 0.0% |
| rainforest | 1.0% | 2.3% | 6.4% | 12.5% | 15.4% |
| equator-to-pole, C | 62.1 | 52.4 | 42.6 | 32.9 | 23.2 |
| land mean, C | 3.2 | 8.2 | 13.2 | 18.2 | 23.2 |
| sea frozen | 47.8% | 31.8% | 9.9% | 0.0% | 0.0% |

Desert now stays inside Earth's own 25-30% at every setting where it used to run
from 3 to 52, the gradient narrows as the world warms, and the land mean tracks
the slider one for one. What moves instead is what should: ice, rainforest, and
how continental the high-latitude coasts are - polar land swings 7.5 C either
side of its mean at slider 0 and 10.4 C at -10, because half the sea around it
has frozen over.

Climate stays one-directional. It reads the finished surface and the rain, and
feeds nothing back: rivers, lakes, the water balance and erosion are all
exactly what they were. The stage runs last for a second reason too - it draws
from `rng`, and running it earlier would move the river meanders on every seed
the generator has ever made.

## Lakes and rivers

A lake holds the water surface its own catchment can keep, not the one its rim
allows. In steady state a lake loses `lake_evap` per cell of water surface and
gains its inflow, so it settles at an area of inflow / `lake_evap` - and with
the basin's bed heights sorted, that area *is* the level: wet `k` cells and the
surface stands at the k-th lowest bed in the basin. Three outcomes fall out of
one rule. A basin whose inflow covers its whole spill area fills to the rim and
spills the surplus. One that cannot sits part full and **spills nothing at
all** - a terminal lake, with no river below it and no gorge cut for one. One
whose inflow only just clears its floor is a pond in a wide hollow.

`lake_evap` is aridity, in units of what a land cell sheds as runoff: open water
loses `E/P` of the local rainfall while land yields only about a third of it, so
the knob is `(E/P - 1) / 0.3` and the default 3.0 is a semi-arid world. 0
reproduces the old fill-to-the-rim behaviour exactly. It is a ratio of areas, so
it needs no scaling with map size. Measured over six seeds at 384x288, out of
27-79 basins:

| `lake_evap` | terminal | part full | lake cells/map |
|---|---|---|---|
| 0 | 0 | 0 | 510 |
| 1.5 | 0 | 0 | 522 |
| 3 | 7 | 7 | 694 |
| 6 | 30 | 29 | 760 |
| 12 | 60 | 60 | 533 |

Below about 2 nothing changes on this terrain, because basin inflow per basin
cell bottoms out around 2.6 - the balance can only bite where a basin is wide
against the catchment feeding it. Note that the water *rises* between 0 and 6:
a lake that stops spilling is also a lake nothing carves an outlet out of, so it
keeps a basin the old code would have drained. Nothing goes completely dry at
any setting, because a basin catches at least its own footprint and so never
falls below `cells / lake_evap` of area. Playas need evaporation to outrun the
rain locally, which is a spatial field rather than this scalar; the hook for one
is the `weights` argument of `hydrology.accumulate`, which takes runoff per cell
and is currently a uniform 1.

A lake is also one *sheet* of water. The level is a contour, and the cells
below it in a basin with any texture in its floor are not a connected set:
a part-full lake drew as a spider of tendrils and specks along the whole basin
rather than as a pond. Only the piece holding the basin's deepest cell is kept,
which is what water standing at that level would actually cover.

Each lake that still spills has its pour point notched and a channel carved
downstream, which gives it a real outflow and a valley to drain through; filling
and routing are redone afterwards because the surface changed. Some shallow
basins drain away entirely at that point, which is the intended outcome.

A gorge is only cut where the lake's outflow clears `river_threshold` - the same
bar a channel has to clear to be drawn as a river. Carving used to ignore it,
and the two then disagreed: the gorge is real terrain, but with no river drawn in
it a high threshold left dry trenches winding across the map. At 384x288 that
was 1% of the dug cells at the default threshold and 30% at six times it, which
is where it becomes obvious. The outflow is the right test because it is exactly
what would flow down the gorge - and for a terminal lake it is zero, so the
whole question answers itself.

The accumulation is *gated*, not run and then patched: each basin's cells are
pointed at its exit so the entire inflow is finalised at one cell, and there the
running total is replaced by the outflow the balance allows. Kahn's front is
exactly the set of cells with nothing left upstream, so the gate fires once and
sees everything. Water that a lake keeps therefore leaves the network, and the
channels below a terminal lake come out dry rather than merely narrow.

D8 does not already converge a basin on its exit - the fill's epsilon tilt
drains a basin through the shallow rim strip that `lake_min_depth` leaves outside
the label, so a labelled basin has 3 to 24 cells whose receiver is not in it, and
gating any one of them would see a fraction of the water.

## Rivers have to end somewhere

That rewiring is right for accounting and wrong for geometry, because it is a
shortcut straight across the lake bed. Channels used to be cut at the rim of the
basin instead, which stopped a river several cells short of the pond it was
running into - and once the climate stage made evaporation local, dry-region
lakes shrank and the gap got wide enough to see. A quarter of the river cells on
seed 1 at 512x384 belonged to a body of water touching neither lake nor sea,
a median of 4 to 7 cells short of one:

| stray river cells | 512x384 | 1024x768 |
|---|---|---|
| seed 7 | 0.6% | 1.0% |
| seed 3 | 3.1% | 6.8% |
| seed 1 | 25.0% | 9.9% |

So the geometry gets its own receivers. Inside a basin they are a breadth-first
tree grown out from that basin's own lake, each ring pointing back at the one
before it, and the tracer is allowed to walk a channel across any cell the fill
had to raise. Three things were learned the hard way there:

- **Aim at the basin's own lake, not at the nearest water.** Measured against
  water of any kind, a basin whose rim runs near the coast has its gradient
  pointing over that rim at the sea, so the path climbs out, misses the pond and
  stops on the far slope five to eight cells from anything.
- **A gradient is not enough; it needs to be a tree.** Restricting the gradient
  to the basin leaves cells with no improving neighbour, which fall back to
  their plain receiver - and a plain receiver pointing back at a rerouted cell
  is a two-cell loop that swallows a river silently. Seed 1 had one at (239, 60)
  and (239, 61). A BFS tree cannot contain one.
- **Discharge has to be carried across.** The accounting graph has already sent
  the basin's water to its exit, so the floor a river crosses reads about one
  cell's worth. Widths take a running maximum along each path, which also
  restores the "discharge only grows downstream" property the balance's gate
  broke.

That leaves the junctions. A tributary's mouth is pinned to the trunk cell it
joined, then the trunk meanders a cell or two away and the two no longer touch.
`_connect_strays` is the backstop: every body of water is a component of
`river | lake | sea`, and one touching neither lake nor sea is walked downstream
from its widest cell and painted until it reaches water, or erased if it never
does. It is a mop, not a mechanism - it bridges a couple of cells at a junction
and nothing more - but `test_rivers_reach_water` holds the whole thing to zero
strays at three sizes.

The trade is that basins whose outflow is under the bar keep their water
instead of being drained - which the balance makes commoner, since it is what
takes a lake's outflow to zero - and that needed `lake_min_area` raised from 6
cells to 30 to stay tidy. The extra survivors are puddles, and a puddle sitting on a
river's course cuts the channel in two: lakes and rivers have their own
colours, so three cells of lake blue in the middle of a river reads as a break
in it rather than as a pond. At 384x288 that was 18-19 interrupted channels a
map across seeds; at 30 cells it is none, on every seed and size tried, and
the largest lake is untouched. Being an area, the cutoff scales with the
square of map size, so it means the same lake at any resolution.

Rivers are also clipped to land. The raster overruns the shoreline both ways -
a mouth is a disc of channel width centred on the last path point, half of it
in the sea - and once sea, lake and river each had a colour, the overrun drew
a stripe of the wrong blue across the water. The clip happens after the
incision, so the bed is still cut right up to the shore.

Rivers are traced out of the D8 network as head-to-mouth paths, smoothed to
shed the eight-direction staircase, then displaced sideways by 1-D noise along
their own arc length - amplitude scaling with channel width, tapered to zero at
both ends so tributaries stay attached to their trunk. Width comes from
discharge, as `river_width` times drainage area to `river_width_exp`, and the
bed is incised under the finished channel.

Width is measured against `river_width_ref`, a fixed fraction of the map, and
never against `river_threshold`. The threshold is the "how many rivers" slider,
and dividing by it coupled the two: asking for more rivers made *every* river
wider, so at the top of the slider the whole network saturated at
`river_width_max` and drew as a mat of equally fat channels with no trunk in it.
At 512x384 with the slider at 30, the widest channel goes from 4.0 - the cap,
flat across the top of the network - to 2.6, and the median from 1.6 to 0.8.

That exponent is 0.45 rather than the classic half, which holds the trunks in
without touching the headwaters - at 512x384 the widest channel goes from 3.89
cells to 3.34 while the median stays on the 0.7 floor. It is the right lever
for the job: `river_width_max` never binds (nothing reaches it at any size, so
lowering it enough to matter flattens the whole top percentile to one width),
and lowering `river_width` narrows the mid-sized channels along with the big
ones.

`world.water` holds it all: `lake_id`, `lake_level`, `lake_depth`, `width`,
`polylines`, `widths`, plus `flow` and `receivers`. `world.surface` returns the water
surface where there is water and the bed everywhere else, which is what the
main view hillshades so lakes come out flat.

## Getting a world out

`terrain/export.py` writes six files. A heightmap alone imports as grey rock:
the bed says what shape the land is, the lake depths say where water sits on it,
the biome indices say what covers it, the canopy says how much of it, and the
river paths are vectors because that is what they were before they were
rasterised.

| file | what it is |
|---|---|
| `_height16.png` | the bed, 16-bit greyscale |
| `_lakes16.png` | lake depth, 16-bit, zero everywhere dry |
| `_biome.png` | `climate` biome index per cell, 8-bit, names in the sidecar |
| `_canopy.png` | canopy fraction, 8-bit, 0 to 1 |
| `_rivers.json` | channel centrelines and per-node width |
| `_config.json` | the seed, every knob, and a stamp of the code that read them |

The canopy goes out as the density, not as the trees. An engine scattering
vegetation has its own LOD budget and its own idea of what a tree is, and a
raster it can sample at any density beats several hundred thousand positions it
would have to thin out - the generator's own point set is a rendering of that
field rather than the other way round. It gets `field8` and not `heightmap`,
because `heightmap` normalises to the data's own range and writes a sidecar full
of terrain (`metres_per_unit`, `sea_code`) which on a coverage fraction is not
merely unused but wrong. Eight bits, too: the terrain needs sixteen because 256
codes across a couple of height units terraces every plain, while a scatter
density read to one part in 255 is already finer than any placement rule asks.

The rasters are the result, and a lossy one: 16 bits is a third of a metre at
`metres_per_unit`, and nothing in them carries the plates, the stress, the
temperature or the flow. `_config.json` is what rebuilds the world -
`generate(export.load_config(path))` returns it bit for bit.

That file overwrites `ref_width` with the map's own width, and that is the whole
subtlety. `generate` scales every pixel-denominated knob from `ref_width` to the
width being built and keeps the *scaled* config on the world, so handing that
copy back scales it a second time and builds a different world - silently, and
only at sizes other than the reference, which is why the round-trip test runs at
256 and asserts the naive version still fails.

The knobs alone are still a false promise, because every one of them can be
unchanged while an edit to `elevation.py` moves every coastline. So `generator`
carries a hash of the package's own source and the versions of the two libraries
whose arithmetic the output is made of. It is conservative in one direction only:
equal means the same code and so the same map, different means *unverified*
rather than different, since reformatting a comment moves the hash and nothing
else. `load_config` warns on a mismatch and loads anyway - it also drops knobs it
does not recognise, so a config from a build that has since gained or lost one
still rebuilds what it can.

`E` in the viewer writes the set for the world on screen, `preview.py` writes it
while it writes the layers. Both land in a folder of their own under
`out/export/`, named for the seed, and a **new one every time** - `seed7`, then
`seed7-2`. An export is a thing you keep, and the reason to press the key twice
is usually that the second world is worth comparing to the first; overwriting is
the one behaviour that cannot be undone from outside. The name is claimed by
creating the directory and catching the failure rather than by asking whether it
exists first, which is two steps with room for another export in between. The viewer exports the whole world, not the current view:
pan and zoom are for looking at it.

Eight bits is not enough for the terrain. The map spans a couple of height units,
and 256 codes across that terraces every plain and shelf, which is exactly where
a heightmap gets looked at flat. Biome indices *are* 8-bit, because they are
labels - there are thirteen of them and nothing in between - and greyscale
rather than palettised: what an engine wants is the index, to pick a texture or
a scatter rule with, and a palette buries that under colours it would have to
match back.

Written by hand, not through an imaging library. A greyscale PNG with no
interlacing and filter type 0 on every row is a signature, three chunks and a
zlib stream - smaller than the dependency would be.

A heightmap on its own is unitless, so a JSON sidecar goes with each:
`lo + code * units_per_code` puts the surface back in the generator's units and
`metres_per_unit` puts it on the ground. That last one is chosen, not measured -
`temp_lapse` is calibrated by reading the ~0.5 that land runs to as a 6 km range,
which puts a unit at 12 km, and every biome on the map is downstream of that
reading. An importer has to scale by something, and the number the climate was
tuned against beats a guess. `sea_code` says where the coastline landed, and
`wrap: x` says the map is a cylinder - which is also why river points can run
past the map's width rather than jumping at the seam.

The lake depths go out on their own scale rather than the bed's: the bed spans
the abyss to the summit, and a lake is metres deep in that, so shared it would
be a handful of codes wide. Pass `lo` and `hi` when two exports *do* have to
line up - a series of seeds, adjacent tiles, or the flat water surface against
the bed it sits in:

```python
m = export.heightmap(world.height, "height16.png")
export.heightmap(world.surface, "surface16.png", m["lo"], m["hi"])
```

That pair was written by default for a while, until it was measured: the bed and
the surface differ on lake cells and nowhere else, **106 cells of 49152** at
256x192, so the second file was 375 KB restating the first. The lake depths say
the same thing in 1.7 KB.

## Not done yet

Nothing human-made. Trees are one-directional by design and not by omission -
they are the end product, so nothing reads `world.trees` but the renderer and the
export, and vegetation does not resist erosion. That coupling is real (bare
ground gullies where forested slopes stay soil-mantled) and would need a crude
pre-canopy off the pre-rain pass to weight `erosion_k`, since erosion runs long
before the climate does. It would also move every coastline on every seed, which
is the reason it is not in. Climate is annual-mean plus a seasonal half-range, which is
enough for Whittaker but not for Koppen-Geiger - that keys on *when* the rain
falls, and the moisture march has no seasons to put it in. Ocean currents set
the sea's temperature but not the land's: a warm current makes the coast beside
it rainier and not milder, because advecting that heat inland is a term of its
own and this has none. Mediterranean shrubland is placed as subtropical semiarid, which is the
right corner of the diagram but not the real test, since the real test is a dry
summer. Climate feeds back only as far as the rain: erosion is weighted by it now, and
`erosion_k` was raised so that weighting could reach the surface, but the biome
map still has no say in where a river goes. Nothing goes fully dry either - a
basin catches at least its own footprint, so playas need evaporation to beat the
rain locally rather than a lake-surface rate that is merely high. Lake outflow is
carved rather than simulated: the level responds to a water balance now, the
channel it spills through does not, and a river cannot change course after the
fact. Meanders are noise, not migration: curvature-driven migration with neck
cutoffs and oxbow lakes was built and reverted, because at any rate that made
the bends visible the map read as wrigglier than the noise does and the migrated
channels wandered far enough off their D8 course to sit oddly in their valleys.
Worth another attempt only with a topographic term holding a channel to its own
valley floor, in place of the blunt cap on lateral drift that version used.

Cost is roughly 0.5 s at 256x192, 2.3 s at 512x384 and 16 s at 1024x768, and
varies by 30-40% run to run on the same machine. Erosion is about half of it and
the river stage most of the rest; climate is about 2% of it and trees another 2%.

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
