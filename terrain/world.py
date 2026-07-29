"""Pipeline: config, stage sequencing, and the finished World object."""
import time
from dataclasses import dataclass, field, asdict, replace

import numpy as np

from . import elevation, hydrology, rivers, tectonics


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
    erosion_k: float = 0.0018
    erosion_m: float = 0.5
    erosion_n: float = 1.0
    thermal_iters: int = 12
    talus: float = 0.045

    # --- lakes and rivers ---
    lake_min_depth: float = 4e-3    # shallower closed basins are just wet ground
    lake_min_area: int = 6          # cells
    outlet_carve_passes: int = 2
    outlet_carve_depth: float = 0.012   # notch cut into a lake's pour point
    outlet_carve_slope: float = 6e-4    # gradient of the carved outflow channel
    outlet_carve_len: int = 220         # max cells carved downstream of a lake
    river_threshold: float = 0.0006     # drainage area needed to become a river
    river_width: float = 0.85           # channel width in cells at that threshold
    river_width_max: float = 4.0
    river_incision: float = 0.02        # how deep the channel sits in the bed
    meander_amp: float = 1.5            # lateral swing in cells, times sqrt(width)
    meander_period: float = 30.0        # along-path wavelength, in cells
    meander_taper: float = 8.0          # vertices over which the swing fades out


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
    "coast_plain_zone", "margin_zone", "margin_reach", "margin_cut_offset",
    "margin_cut_w", "margin_slope_blur", "meander_period", "meander_taper",
    "meander_amp", "river_width", "river_width_max",
)
# Rises per cell: the same climb spread over more cells is a gentler one.
_PER_CELL_FIELDS = ("coast_slope", "talus", "outlet_carve_slope")


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

    eroded, er_filled, er_flow, er_rec = stage("erosion", lambda: hydrology.stream_power(
        pre.copy(), 0.0, cfg.erosion_passes, cfg.erosion_k, cfg.erosion_m,
        cfg.erosion_n, cfg.thermal_iters, cfg.talus))
    # Erosion signs off by filling, routing and accumulating its finished
    # surface, and the river stage opens by needing exactly that. Hand it over
    # instead of letting it be recomputed.
    h, water = stage("rivers", lambda: rivers.build(
        eroded.copy(), cfg, rng, routed=(er_filled, er_rec, er_flow)))

    world = World(cfg=cfg, tect=tect, height_raw=raw, height_pre=pre,
                  height_eroded=eroded, height=h, water=water, timings=timings)
    if verbose:
        print(f"  {'total':<12} {sum(timings.values()):6.2f}s  "
              f"land={world.land.mean():.0%}  max={h.max():.2f}  min={h.min():.2f}  "
              f"lakes={water.lake_id.max()}  rivers={len(water.polylines)}")
    return world
