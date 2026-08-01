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
| climate | `terrain/climate.py` | zonal rain belts, moisture marched downwind for orographic shadows, runoff and evaporation fields |
| lakes and rivers | `terrain/rivers.py` | lake water balance, outlet carving, polyline extraction, meandering, discharge-based width, channel incision |
| biomes | `terrain/climate.py` | temperature from latitude, lapse rate and continentality; biomes from temperature against rain relative to evapotranspiration |

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

Layers: relief, elevation without water, tectonic relief (pre-texture),
pre-erosion, plates, boundary classes, stress, rainfall, drainage, erosion
delta, slope, land/coast, river width. The rainfall layer is ranked within the
land distribution rather than scaled by it - rain is skewed enough that a linear
ramp paints every interior the same tan.

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
`widths` of each polyline.

## Climate

Rain sets two things: how much water each cell contributes to the network, and
how hard a lake has to work to stay wet. Both come out of one small model.

Zonal belts give the latitudes - wet on the equator, dry in the horse
latitudes, wet again at the storm tracks, dry at the poles, as
`cos(3 pi lat)` - and the latitude they key on wobbles, the same trick the
polar taper uses, so they do not read as three ruler-straight stripes.
The belt scales the whole rain rate rather than its flat part: applied only to
`rain_base` it is invisible wherever there is relief, because the orographic
term is several times the base on any real slope.

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
Erosion still runs on unweighted flow: wet slopes ought to carve faster, but
that is a change to every tuned number in the erosion stage rather than a
change to this one.

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
place on the ramp, and `biomes` - and the main relief layer mixes the biome
colour into its hypsometric tint at `biome_tint`, so height and vegetation read
off the same map.

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
relief layer hillshades so lakes come out flat.

## Not done yet

Nothing human-made. Climate is annual-mean plus a seasonal half-range, which is
enough for Whittaker but not for Koppen-Geiger - that keys on *when* the rain
falls, and the moisture march has no seasons to put it in. No ocean currents
either, so there is no warm west coast and no Atacama: the only thing that
makes one coast differ from another at the same latitude is which way its wind
blows. Mediterranean shrubland is placed as subtropical semiarid, which is the
right corner of the diagram but not the real test, since the real test is a dry
summer. Climate feeds nothing back: erosion still runs on unweighted flow, so a
soaked windward slope carves no faster than the desert behind it, and the biome
map has no say in where a river goes. Nothing goes fully dry either - a
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
the river stage most of the rest; climate is about 2% of it.

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
