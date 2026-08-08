"""Pipeline: config, stage sequencing, and the finished World object."""
import time
from dataclasses import dataclass, field, asdict, replace

import numpy as np

from . import climate, elevation, hydrology, rivers, tectonics, trees


@dataclass
class Config:
    # --- grid ---
    width: int = 512
    height: int = 384
    seed: int = 0
    ref_width: int = 512    # size these knobs are tuned at; see `_scale_to_size`

    # --- plates ---
    n_plates: int = 21
    plate_warp: float = 34.0        # px of noise displacement on plate borders
    plate_warp_fine: float = 15.0   # second, shorter-scale displacement
    plate_warp_periods: int = 3
    plate_size_var: float = 0.42    # 0 = all plates the same size
    cont_plate_fraction: float = 0.42
    stress_blur: float = 2.0        # smoothing of the raw strain derivatives
    stress_spread: float = 5.0      # softens medial-axis rays after propagation
    # (lo, hi) soft thresholds: below lo a boundary leaves no landform at all,
    # above hi it acts at full strength. Keeps quiet boundaries invisible.
    conv_gate: tuple = (0.18, 0.48)
    rift_gate: tuple = (0.38, 0.78)
    shear_gate: tuple = (0.55, 0.92)

    # --- crust ---
    cont_base: float = 0.58
    ocean_base: float = -0.72
    cont_fraction: float = 0.38     # area of continental crust (shelves included)
    crust_periods: float = 2.0      # lower -> fewer, larger continents
    crust_plate_weight: float = 1.15  # how strongly plates dictate the outline
    crust_blur: float = 9.0
    crust_warp: float = 18.0        # detaches coastlines from the plate outline
    crust_warp_periods: float = 5.0
    crust_sharpness: float = 0.85    # higher -> narrower shelves, steeper margins
    upland_h: float = 0.30
    upland_periods: float = 3.0
    polar_start: float = 0.56       # |latitude| where crust starts thinning
    polar_ocean: float = 5.5        # taper strength, in std devs of the field
    polar_wobble: float = 0.13      # how far the ice line wanders, in latitude
    polar_wobble_periods: float = 2.5
    land_fraction: float = 0.29

    # --- boundary relief (amplitude / falloff width in px) ---
    collision_h: float = 1.30
    collision_w: float = 26.0
    trench_h: float = 0.70
    trench_w: float = 5.0
    arc_h: float = 0.85
    arc_offset: float = 16.0
    arc_w: float = 7.0
    arc_break_periods: float = 16.0   # along-strike spacing of arc volcanoes
    arc_break_bias: float = 0.2       # lower -> more gaps in the chain
    cordillera_h: float = 0.55
    cordillera_w: float = 30.0
    rift_depth: float = 0.45
    rift_w: float = 9.0
    rift_shoulder: float = 0.40
    ridge_h: float = 0.28
    ridge_w: float = 18.0
    transform_h: float = 0.13
    transform_w: float = 4.0
    ridge_age_thresh: float = 0.15
    age_depth: float = 0.35
    age_scale: float = 90.0
    age_warp: float = 26.0          # breaks up the seafloor-age contours
    age_warp_periods: float = 4.0

    # --- hotspots ---
    n_hotspots: int = 5
    hotspot_max_crust: float = 0.35   # only seed hotspots in oceanic crust
    hotspot_h: float = 1.15
    hotspot_sigma: float = 3.2
    hotspot_spacing: float = 11.0
    hotspot_chain: tuple = (3, 9)

    # --- texture / coast ---
    texture_h: float = 0.36
    texture_periods: int = 6
    texture_warp: float = 12.0
    abyss_h: float = 0.10
    coast_h: float = 0.15           # bays and headlands
    coast_h_fine: float = 0.055     # fretted edge
    coast_band: float = 0.13        # elevation range the fraying reaches
    coast_periods: int = 10
    coast_slope: float = 0.024      # max rise per cell as land leaves the shore
    coast_slope_periods: float = 5.0  # how that cap varies along the coast
    coast_plain_zone: float = 30.0    # how far inland the cap reaches, in px
    margin_h: float = 0.16          # extra fray on a continent-ocean margin
    margin_h_fine: float = 0.07
    margin_periods: float = 8.0     # coarse: finer noise breaks the coast up
    margin_zone: float = 45.0       # falloff from the plate boundary, in px
    margin_reach: float = 22.0      # width of the coastal corridor, in px
    margin_gain_max: float = 3.0    # cap on the slope compensation
    margin_cap: float = 1.0         # clip on the fray noise, in std devs
    margin_slope_blur: float = 2.0
    margin_cut_h: float = 0.30      # drowned inlets cut into a margin coast
    margin_cut_thresh: float = 1.0  # in std devs; higher -> fewer inlets
    margin_cut_cap: float = 1.5
    margin_cut_periods: float = 10.0
    margin_cut_offset: float = 1.0  # centre of the cut window, px inland
    margin_cut_w: float = 7.0

    # --- erosion ---
    erosion_passes: int = 4
    # Five times what it was, and the reason is the rain weighting rather than
    # the erosion. At 0.0018 the fluvial term is small next to thermal creep,
    # so weighting it by rainfall moved the cut on wet high ground from 4.2x
    # the cut on dry ground to only 4.7x; at this value it is 4.4x against a
    # flat-weighted 3.1x, which is a range with a dissected windward side and a
    # lee that keeps its bulk. Measured over three seeds at 384x288, land came
    # out at 27.9% against 28.0% and the mean cut at 0.0031 against 0.0019, so
    # the extra carving lands as texture on the high ground rather than as a
    # drowned world.
    erosion_k: float = 0.009
    erosion_m: float = 0.5
    erosion_n: float = 1.0
    thermal_iters: int = 12
    talus: float = 0.045

    # --- climate ---
    rain_base: float = 0.006        # moisture a land cell takes out of the air
    rain_orog: float = 1.0          # extra rain per unit of upwind climb
    rain_ocean_gain: float = 0.08   # moisture an ocean cell puts back
    # Depth of the zonal wet/dry bands, 0 = flat. Raised from 0.75 when the
    # capacity term arrived: temperature now supplies the wet equator and the
    # dry pole, so what is left for the belts is the job only they can do - the
    # subtropical dry band, which comes from air descending at 30 degrees and
    # not from how cold it is there. Measured over two seeds at 384x288, desert
    # as a share of land ran 18% at 0.75, 23% at 0.85 and 29% at 0.95.
    rain_belts: float = 0.9
    rain_wobble: float = 0.10       # how far the bands wander, in latitude
    rain_wobble_periods: float = 2.0
    rain_recycle: float = 0.55      # share of rain a land cell puts back up
    rain_blur: float = 2.5          # px; weather is not one cell wide
    rain_evap_cap: float = 0.2      # driest runoff `lake_evap` is divided by
    # Fractional rise in what the air can carry, per degree of the temperature
    # under it. Clausius-Clapeyron's own number, and the reason the equator is
    # wet without anyone drawing a band there.
    rain_capacity: float = 0.07
    temp_equator: float = 27.0      # mean annual C at the equator, at sea level
    temp_pole: float = -25.0        # ditto at the poles
    # Global shift; the viewer's temperature slider. Rain follows it through
    # `rain_per_degree` and it lands hardest on the poles through
    # `temp_polar_amp`, so warming flattens the world rather than merely raising
    # evapotranspiration everywhere.
    temp_offset: float = 0.0
    # How much of the shift the poles get over the equator. 0 is a uniform
    # offset; at 0.5 the poles move twice the slider and the equator half of it,
    # which is about the amplification a hothouse actually ran.
    temp_polar_amp: float = 0.5
    # C lost per unit of height. Land here runs to about 0.5, and reading that
    # as a 6 km range puts one height unit at 12 km, so Earth's 6.5 C/km lands
    # near 78. It is set at half that on purpose: this terrain carries far more
    # high ground than Earth does - the top tenth of it sits above what would be
    # 2500 m - and at a true lapse rate that tenth freezes and the map comes out
    # a third taiga. Lower it further and the tropics eat everything.
    temp_lapse: float = 38.0
    # Spread either side of that with local humidity: dry air cools at about
    # 9.8 C/km on the way up, saturated air at 5, because condensation pays back
    # part of the expansion. 0.25 gives a dry-to-wet ratio of 1.67 against the
    # 1.96 the two adiabats really differ by. 0 is the flat rate.
    temp_lapse_moist: float = 0.25
    # --- ocean currents ---
    # Degrees the gyres move the sea surface off its latitude. Earth's eastern
    # boundary upwelling runs 5-8 C below the zonal mean and the western
    # boundary currents 3-5 C above, and the asymmetry is real: pulling cold
    # water up from below is a stronger lever than carrying warm water along.
    # The field averages to zero over the sea, so these set how far the two
    # coasts of a continent part company, not how warm the world is.
    current_cold: float = 6.0
    current_warm: float = 4.0
    current_reach: float = 40.0     # px offshore the anomaly reaches
    current_lat: float = 0.35       # |latitude| the upwelling band sits on
    current_blur: float = 8.0       # px; a gyre is not a coastline
    temp_wobble: float = 0.06       # how far the isotherms wander, in latitude
    temp_wobble_periods: float = 2.0
    temp_swing: float = 22.0        # seasonal half-range at the pole, deep inland
    temp_maritime: float = 0.35     # share of that swing a coast still gets
    # px inland over which continentality saturates. Measured on seeds at
    # 512x384, land runs 10 px from the sea at the median and 39 at the 99th,
    # so a reach much above 15 leaves even the deepest interior only half
    # continental and the seasonal swing never arrives.
    temp_cont_reach: float = 15.0
    precip_mean_mm: float = 750.0   # what a runoff of 1.0 means, in mm/yr
    # Fractional change in that per degree of `temp_offset`. Clausius-Clapeyron
    # gives 7% per degree of what the air can *hold*, while global rainfall is
    # limited by the energy available to evaporate it and responds nearer 2 to
    # 3%. This number has to carry both, because it also stands in for how much
    # moisture reaches a continental interior, which follows the capacity more
    # than the global mean. Measured over two seeds at 384x288, desert as a
    # share of land at slider -10 / 0 / +10 runs 11 / 27 / 40% at 0.025 and
    # 44 / 27 / 23% at 0.07; at 0.05 it is 25 / 27 / 30%, which keeps both ends
    # of the slider inside Earth's own range instead of turning one into sand.
    rain_per_degree: float = 0.05
    # Two different jobs, easily confused. `biome_blur` smooths the climate
    # fields *before* they are banded, so it moves where a boundary falls and
    # changes `world.biome`. `biome_soften` blurs the colours *after*, so it
    # only changes how a boundary is drawn and nothing reads it but the render.
    biome_blur: float = 2.0         # px
    biome_soften: float = 2.5       # px
    biome_tint: float = 0.35        # biome colour mixed into the relief layer

    # --- lakes and rivers ---
    lake_min_depth: float = 4e-3    # shallower closed basins are just wet ground
    lake_min_area: int = 30         # cells; below this a basin is wet ground
    # Runoff a cell of lake surface loses, in units of what a land cell yields;
    # 0 fills every basin to its rim, as the generator used to. A lake settles
    # at an area of inflow / this, so it is a ratio and needs no scaling with
    # the map. 3 is about right physically - open water evaporates roughly the
    # local rainfall while land sheds only a third of it as runoff - and lands
    # where it should on this terrain: measured over three seeds at 384x288,
    # basin inflow per basin cell runs 2-28 with a median near 6-13, so most
    # basins still fill and spill while the widest, driest-fed ones do not.
    lake_evap: float = 3.0
    outlet_carve_passes: int = 2
    outlet_carve_depth: float = 0.012   # notch cut into a lake's pour point
    outlet_carve_slope: float = 6e-4    # gradient of the carved outflow channel
    outlet_carve_len: int = 220         # max cells carved downstream of a lake
    river_threshold: float = 0.0006     # drainage area needed to become a river
    river_width: float = 0.85           # channel width in cells at that threshold
    river_width_max: float = 4.0
    river_width_cap: float = 6.0        # widest channel in cells, at any map size
    river_width_exp: float = 0.45       # width goes as discharge to this power
    # Discharge at which a channel is `river_width` wide, as a fraction of the
    # map. Deliberately *not* `river_threshold`: that one is the "how many
    # rivers" slider, and measuring width against it made asking for more rivers
    # widen every river already there.
    river_width_ref: float = 0.0006
    river_incision: float = 0.02        # how deep the channel sits in the bed
    meander_amp: float = 1.5            # lateral swing in cells, times sqrt(width)
    meander_period: float = 30.0        # along-path wavelength, in cells
    meander_taper: float = 8.0          # vertices over which the swing fades out

    # --- trees ---
    # Where a canopy opens and where it closes, on `climate.moisture_index`.
    # Straddles MI_BANDS' semiarid/subhumid cuts on purpose: the biome grid puts
    # the forest boundary at 0.7, and these say that the ground either side of
    # it is thinning stands rather than a step from steppe to forest.
    tree_mi_open: float = 0.45
    tree_mi_closed: float = 1.5
    tree_line_fade: float = 5.0     # C of warmest month over which forest thins out
    tree_gallery: float = 0.75      # canopy a bank gets that its own rain would not
    tree_gallery_reach: float = 3.0  # px from water that reaches
    tree_grove: float = 0.55        # clumping, in std devs of the noise on `mi`
    tree_grove_periods: float = 22.0
    # Spacing has to clear the stamp, or the trees stop being trees. A crown is
    # about three cells across, so at 1.3 a closed canopy piles four of them on
    # every one that shows and the layer draws as a flat green mass at any zoom.
    # At 2.0 the crowns touch and overlap slightly, which is what a closed
    # canopy is, and a savanna at a third of the cover comes out as separate
    # trees with ground between them.
    tree_spacing: float = 2.0       # px between trees under a closed canopy
    tree_size_var: float = 0.35     # lognormal sigma on trunk size
    # px over which one kind of tree gives way to another. Blurs the *mix*, not
    # the colours: at 0 a biome boundary is a hard line between two species, and
    # what a real one has is a belt of mixed stand several times wider than a
    # cell. Wider than `biome_soften` because this is ecology rather than
    # anti-aliasing - the belt is a real place, not a soft edge.
    tree_mix: float = 6.0
    # Opacity of the trees on the *main view*, where they are grain over a
    # hypsometric ramp that is still doing the talking. The `trees` layer draws
    # them at full strength on ground desaturated to make room; this one cannot
    # do that without ceasing to be a relief map. 0 turns them off there.
    # Named for the relief rendering (`render.relief`) rather than for the
    # layer, which is why it kept the name when the layer changed its own.
    tree_relief: float = 0.55


# Knobs measured in pixels. `Config` is tuned at `ref_width`, and these have to
# follow the map or the world changes shape as it is resized: a landform of
# fixed pixel width is a smaller and smaller fraction of a growing map, so
# boundary relief narrows into a hard crease while everything keyed on
# `periods` - which is already relative to the map - stays put around it.
# Measured on seed 7, boundary slope over median slope ran 1.22 / 1.50 / 1.99
# at 384 / 640 / 1024 wide; scaled, it holds at 1.31-1.37.
_PX_FIELDS = (
    "plate_warp", "plate_warp_fine", "stress_blur", "stress_spread",
    "crust_blur", "crust_warp", "collision_w", "trench_w", "arc_offset",
    "arc_w", "cordillera_w", "rift_w", "ridge_w", "transform_w", "age_scale",
    "age_warp", "hotspot_sigma", "hotspot_spacing", "texture_warp",
    "coast_plain_zone", "rain_blur", "temp_cont_reach", "biome_blur",
    "biome_soften", "current_reach", "current_blur",
    "margin_zone", "margin_reach", "margin_cut_offset",
    "margin_cut_w", "margin_slope_blur", "meander_period", "meander_taper",
    "meander_amp", "river_width", "river_width_max",
    # `tree_spacing` is deliberately absent: see `trees.scatter`. It is the one
    # pixel quantity here whose job is how the map looks rather than how much
    # ground something covers.
    "tree_gallery_reach", "tree_mix",
)
# Rises per cell: the same climb spread over more cells is a gentler one. The
# two rain rates go here for the same reason - moisture must cross a continent
# in the same number of *continents*, not the same number of cells, or a bigger
# map dries its interiors out further. `rain_orog` needs no scaling: it is rain
# per unit climbed, and a range's total climb does not change with resolution.
_PER_CELL_FIELDS = ("coast_slope", "talus", "outlet_carve_slope",
                    "rain_base", "rain_ocean_gain")


def _scale_to_size(cfg):
    """Put every pixel-denominated knob back on the map's own scale.

    Returns a copy: callers that keep a `Config` around and regenerate from it
    - the viewer does - must not have it rescaled underneath them each time.

    `erosion_k` is deliberately left alone. Stream power couples drainage area
    to slope in cell units and does not scale by any single factor; land area
    comes out within a tenth of a point across sizes as it is.
    """
    s = cfg.width / max(1, cfg.ref_width)
    if s == 1.0:
        return cfg
    cfg = replace(cfg)
    for k in _PX_FIELDS:
        setattr(cfg, k, getattr(cfg, k) * s)
    for k in _PER_CELL_FIELDS:
        setattr(cfg, k, getattr(cfg, k) / s)
    cfg.outlet_carve_len = max(1, round(cfg.outlet_carve_len * s))
    cfg.lake_min_area = max(1, round(cfg.lake_min_area * s * s))   # an area
    # Channel width is a pixel quantity like the rest, so at HD and up a trunk
    # river is drawn a dozen cells across and reads as an estuary. Cap the
    # widest channel, and pull the whole width scale down by the same factor:
    # capping `river_width_max` alone would leave the tributaries scaled up
    # against a ceiling they all reach, and the network draws as a mat of
    # equal channels rather than a trunk with branches hanging off it.
    if cfg.river_width_max > cfg.river_width_cap:
        cfg.river_width *= cfg.river_width_cap / cfg.river_width_max
        cfg.river_width_max = cfg.river_width_cap
    return cfg


@dataclass
class World:
    cfg: Config
    tect: tectonics.Tectonics
    height_raw: np.ndarray      # tectonics only, before texture and erosion
    height_pre: np.ndarray      # after texture, before erosion
    height_eroded: np.ndarray   # after erosion, before rivers were cut in
    height: np.ndarray          # final
    water: rivers.Water
    runoff: np.ndarray          # water a cell contributes, 1 on average on land
    temp: np.ndarray            # mean annual temperature, C
    swing: np.ndarray           # seasonal half-range, C; warmest month is temp+swing
    biome: np.ndarray           # climate.BIOME_NAMES index per cell
    trees: trees.Trees          # canopy cover, and the individual trees on it
    timings: dict = field(default_factory=dict)

    @property
    def land(self):
        return self.height > 0.0

    @property
    def flow(self):
        return self.water.flow

    @property
    def rivers(self):
        return self.water.river_mask

    @property
    def lakes(self):
        return self.water.lake_mask

    @property
    def surface(self):
        """Height of the top of the water where there is water, of the bed elsewhere."""
        return self.water.surface(self.height)

    def info_at(self, y, x):
        p = int(self.tect.plate[y, x])
        w = self.water
        return {
            "plate": p,
            "crust": "continental" if self.tect.plate_cont[p] else "oceanic",
            "elev": float(self.height[y, x]),
            "flow": float(w.flow[y, x]),
            "rain": float(self.runoff[y, x]),
            "temp": float(self.temp[y, x]),
            "biome": climate.BIOME_NAMES[int(self.biome[y, x])],
            "canopy": float(self.trees.cover[y, x]) if self.trees else 0.0,
            "dist_to_boundary": float(self.tect.dist[y, x]),
            "river_w": float(w.width[y, x]),
            "lake_depth": float(w.lake_depth[y, x]),
        }


def generate(cfg=None, verbose=True, on_stage=None, **overrides):
    """`on_stage(name)` is called before each stage; a big map takes a minute
    and a caller with a window needs something to put on it."""
    cfg = cfg or Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    # After the overrides, so `generate(width=1024)` scales to 1024 and not to
    # whatever width the config happened to carry when it was built.
    cfg = _scale_to_size(cfg)
    rng = np.random.default_rng(cfg.seed)
    timings = {}

    def stage(name, fn):
        if on_stage:
            on_stage(name)
        t0 = time.perf_counter()
        out = fn()
        timings[name] = time.perf_counter() - t0
        if verbose:
            print(f"  {name:<12} {timings[name]:6.2f}s")
        return out

    if verbose:
        print(f"generating {cfg.width}x{cfg.height} seed={cfg.seed}")
    tect = stage("plates", lambda: tectonics.build(cfg, rng))
    raw = stage("relief", lambda: elevation.tectonic_relief(tect, cfg, rng))
    raw = stage("hotspots", lambda: elevation.hotspot_chains(raw, tect, cfg, rng))
    pre = stage("texture", lambda: elevation.texture(raw.copy(), tect, cfg, rng))

    # One sea level for every version of the surface, so the layers line up.
    level = float(np.quantile(pre, 1.0 - cfg.land_fraction))
    raw -= level
    pre -= level
    def coastline():
        # Cap the rise, shove the plate margins sideways, and only then fray, so
        # every reshaped shoreline gets fretted like any other rather than
        # arriving as a smooth arc.
        h = elevation.coastal_plain(pre, cfg, rng)
        h = elevation.erode_margins(h, tect, cfg, rng)
        return elevation.fray_coast(h, cfg, rng)

    pre = stage("coastline", coastline)

    # The rain the carving is weighted by, read off the surface as it stands
    # before erosion. Chicken and egg: the rain that ought to weight it is the
    # rain the finished ranges cast their shadows with, and those ranges do not
    # exist yet. The un-eroded relief is the stand-in, and a fair one - erosion
    # lowers a range and sharpens it but does not move where it is, so the
    # shadow falls in the same place. The alternative is what this did for a
    # long time, which is to carve a desert as hard as a rainforest.
    #
    # Both rain passes take the same derived stream, so they see the same belt
    # wobble and the erosion is weighted by the bands the final map is banded
    # on. Off the shared stream, as this used to be, the two disagreed by up to
    # `rain_wobble` of latitude and the carving landed beside its own rain.
    def erosion_rain():
        t = climate.temperature(pre, cfg, np.random.default_rng([cfg.seed, 1]))[0]
        return climate.runoff(pre, cfg, np.random.default_rng([cfg.seed, 2]),
                              temp=t)[0]

    wet = stage("pre-rain", erosion_rain)
    eroded, er_filled, er_flow, er_rec = stage("erosion", lambda: hydrology.stream_power(
        pre.copy(), 0.0, cfg.erosion_passes, cfg.erosion_k, cfg.erosion_m,
        cfg.erosion_n, cfg.thermal_iters, cfg.talus, weights=wet))
    # Rain is read off the eroded surface, which is the one the rivers will run
    # on: the ranges that cast the shadows are the ones erosion left standing.
    # Temperature comes first now, because the rain reads it: the air over a
    # warm sea carries several times what polar air does. It takes its own
    # stream off the seed rather than the shared one - drawing its noise here
    # would shift every later draw and move the river meanders on every seed the
    # generator has ever made - and the two passes take a fresh one each so they
    # see the same wobble. The first pass runs at the flat lapse rate, since the
    # humidity that bends it is what this rain is about to produce.
    sst = stage("temperature", lambda: climate.temperature(
        eroded, cfg, np.random.default_rng([cfg.seed, 1]))[0])
    runoff, evap = stage("climate", lambda: climate.runoff(
        eroded, cfg, np.random.default_rng([cfg.seed, 2]), temp=sst))
    # Erosion signs off by filling, routing and accumulating its finished
    # surface, and the river stage opens by needing exactly that. Hand it over
    # instead of letting it be recomputed.
    h, water = stage("rivers", lambda: rivers.build(
        eroded.copy(), cfg, rng, routed=(er_filled, er_rec, er_flow),
        runoff=runoff, evap=evap))

    def biomes():
        # Still last, because it wants the finished surface: rivers incise, a
        # coastal cell can cross sea level doing it, and the biome grid's idea
        # of the coastline has to be `world.land`'s. This pass gets the runoff,
        # so its lapse rate is the moist one - a wet windward slope loses height
        # far more gently than a desert range at the same latitude.
        t, sw = climate.temperature(h, cfg, np.random.default_rng([cfg.seed, 1]),
                                    humid=runoff)
        return t, sw, climate.biomes(h, t, sw, runoff, cfg)

    temp, swing, biome = stage("biomes", biomes)

    world = World(cfg=cfg, tect=tect, height_raw=raw, height_pre=pre,
                  height_eroded=eroded, height=h, water=water, runoff=runoff,
                  temp=temp, swing=swing, biome=biome, trees=None, timings=timings)
    # Last, and off its own generator. Trees read the finished world and change
    # nothing in it, so this stage cannot move anything upstream of it - but it
    # would if it drew from `rng`, which is why it does not: every seed the
    # generator has ever made still builds the same terrain, rivers and biomes
    # with this stage present as it did without it.
    world.trees = stage("trees", lambda: trees.build(
        world, np.random.default_rng([cfg.seed, 3])))
    if verbose:
        print(f"  {'total':<12} {sum(timings.values()):6.2f}s  "
              f"land={world.land.mean():.0%}  max={h.max():.2f}  min={h.min():.2f}  "
              f"lakes={water.lake_id.max()}  rivers={len(water.polylines)}  "
              f"trees={len(world.trees)}")
    return world
