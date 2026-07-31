"""Layer rendering: numpy height/plate fields -> uint8 RGB images."""
import numpy as np

from . import grid

SEA_RAMP = [
    (-1.00, (11, 30, 64)), (-0.70, (13, 36, 75)), (-0.45, (16, 44, 88)),
    (-0.25, (20, 54, 100)), (-0.12, (25, 64, 112)), (-0.05, (30, 74, 123)),
    (0.0, (37, 86, 134)),
]
LAND_RAMP = [
    (0.00, (68, 108, 72)), (0.14, (88, 128, 76)), (0.34, (112, 142, 82)),
    (0.52, (142, 148, 94)), (0.68, (152, 132, 100)), (0.80, (136, 118, 106)),
    (0.90, (176, 174, 174)), (1.00, (252, 252, 255)),
]
PLATE_CLASS = {1: (220, 60, 50), 2: (60, 140, 235), 3: (250, 200, 60)}


def _ramp(vals, ramp):
    stops = np.array([s for s, _ in ramp])
    cols = np.array([c for _, c in ramp], dtype=float)
    out = np.empty(vals.shape + (3,))
    for c in range(3):
        out[..., c] = np.interp(vals, stops, cols[:, c])
    return out


def _rank(vals, pool, n=20000):
    """Where each value sits in `pool`'s distribution, 0..1."""
    if pool.size == 0:
        return np.zeros_like(vals)
    s = np.sort(pool if pool.size <= n else np.random.default_rng(0).choice(pool, n))
    return np.searchsorted(s, vals, side="right") / len(s)


def hypsometric(h, land_mix=0.35, sea_mix=0.22):
    """Blue below sea level, green-to-white above, each side scaled separately.

    Mostly absolute height, part rank within its own side, so the ramp is not
    wasted on the extremes. The sea ramp deliberately spans a narrow band of
    blues: open ocean and coastal shelf are the same water, and a wide ramp
    paints a bright halo around every coast. Seafloor structure comes through
    as hillshading instead.
    """
    land_v, sea_v = h[h > 0], h[h < 0]
    sea_scale = max(1e-6, -float(np.percentile(sea_v, 2)) if sea_v.size else 1.0)
    land_scale = max(1e-6, float(np.percentile(land_v, 98)) if land_v.size else 1.0)
    sea = _ramp((1 - sea_mix) * np.clip(h / sea_scale, -1, 0) -
                sea_mix * (1 - _rank(h, sea_v)), SEA_RAMP)
    land = _ramp((1 - land_mix) * np.clip(h / land_scale, 0, 1) +
                 land_mix * _rank(h, land_v), LAND_RAMP)
    return np.where((h > 0)[..., None], land, sea)


def hillshade(h, azimuth=315.0, altitude=45.0, z=28.0):
    dy, dx = grid.gradient(h * z)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.deg2rad(360.0 - azimuth + 90.0)
    alt = np.deg2rad(altitude)
    shade = (np.sin(alt) * np.cos(slope) +
             np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(shade, 0, 1)


def relief(h, shade=True, surface=None):
    """Hypsometric tint of the bed, hillshaded off `surface` if water sits on it."""
    rgb = hypsometric(h)
    if shade:
        s = hillshade(h if surface is None else surface)[..., None]
        # Flatter shading under water: at full land contrast the abyssal noise
        # mottles the whole ocean and undoes the point of a narrow sea ramp.
        lit = np.where((h > 0)[..., None], 0.55 + 0.75 * s, 0.82 + 0.28 * s)
        rgb = np.clip(rgb * lit, 0, 255)
    return rgb


# Narrower than the sea ramp on purpose. A lake drowns whatever valley was
# already cut through its basin, so the old channel is its deepest part by some
# margin - three times the median depth is ordinary - and a wide ramp draws that
# channel back in as a dark line running under the water, which reads as a river
# showing through rather than as a lake.
LAKE_RAMP = [(0.0, (104, 158, 188)), (0.35, (84, 138, 178)), (1.0, (60, 108, 158))]
LAKE_DEPTH_BLUR = 2.0   # softens the drowned channel out of the depth shading
RIVER_RAMP = [(0.0, (110, 165, 195)), (0.4, (78, 132, 176)), (1.0, (48, 96, 150))]


def water_map(w, shade=True):
    """The main map: relief with lakes at their own flat level and rivers by width.

    Lakes are shaded from the water surface, not the bed, so they read as flat
    sheets; the depth ramp underneath still shows how deep the basin is.
    """
    wat = w.water
    rgb = relief(w.height, shade, surface=w.surface)
    if wat.lake_mask.any():
        # Shaded off a blurred depth: the ramp is meant to say how deep the
        # basin is, not to trace the metre-wide channel at the bottom of it.
        # `wat.lake_depth` itself is left exact for anyone reading the field.
        soft = grid.blur(wat.lake_depth, LAKE_DEPTH_BLUR)
        d = np.clip(soft / max(1e-6, float(np.percentile(
            soft[wat.lake_mask], 92))), 0, 1)
        rgb = np.where(wat.lake_mask[..., None], _ramp(d, LAKE_RAMP), rgb)
    if wat.river_mask.any():
        v = np.clip(wat.width / max(1e-6, w.cfg.river_width_max), 0, 1)
        rgb = np.where(wat.river_mask[..., None], _ramp(v, RIVER_RAMP), rgb)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def plate_colors(n, rng=None):
    rng = rng or np.random.default_rng(7)
    hs = rng.permutation(n) / max(1, n)
    # cheap HSV->RGB at fixed s/v
    i = (hs * 6).astype(int) % 6
    f = hs * 6 - np.floor(hs * 6)
    v, s = 0.88, 0.55
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.stack([r, g, b], axis=-1) * 255).astype(float)


def plates(t, cls=None, shade_height=None):
    cols = plate_colors(len(t.seeds))
    rgb = cols[t.plate]
    rgb = np.where(t.is_cont[..., None], rgb, rgb * 0.55 + 30)
    if shade_height is not None:
        rgb = np.clip(rgb * (0.65 + 0.5 * hillshade(shade_height)[..., None]), 0, 255)
    if cls is not None:
        for k, c in PLATE_CLASS.items():
            rgb = np.where((cls == k)[..., None], np.array(c, dtype=float), rgb)
    return rgb.astype(np.uint8)


def scalar(a, lo=None, hi=None, ramp=None):
    """Generic diverging/sequential view of any field."""
    lo = float(a.min()) if lo is None else lo
    hi = float(a.max()) if hi is None else hi
    v = np.clip((a - lo) / max(1e-9, hi - lo), 0, 1)
    ramp = ramp or [(0.0, (12, 12, 30)), (0.35, (40, 70, 130)),
                    (0.6, (200, 120, 60)), (0.85, (240, 200, 90)), (1.0, (255, 255, 240))]
    return _ramp(v, ramp).astype(np.uint8)


def stress(t):
    """Red convergence, blue extension, green shear, all on one image."""
    rgb = np.zeros(t.plate.shape + (3,))
    rgb[..., 0] = t.conv
    rgb[..., 2] = t.rift
    rgb[..., 1] = t.shear * 0.8
    fade = np.exp(-t.dist / 25.0)[..., None]
    return np.clip(rgb * fade * 320 + 12, 0, 255).astype(np.uint8)


def flow(acc, h, threshold):
    """Log drainage area over a dim relief, so the whole network is visible."""
    base = relief(h, shade=True) * 0.55
    la = np.log10(np.maximum(acc, 1.0))
    lo = np.log10(max(2.0, threshold * 0.04))     # faintest tributary shown
    hi = np.log10(max(10.0, float(acc.max())))
    v = np.clip((la - lo) / (hi - lo), 0, 1)
    water = np.stack([60 + 60 * v, 130 + 90 * v, 190 + 60 * v], axis=-1)
    strength = np.where(h > 0, v ** 0.7, 0)[..., None]
    return np.clip(base * (1 - strength) + water * strength, 0, 255).astype(np.uint8)


RAIN_RAMP = [(0.0, (196, 172, 120)), (0.35, (176, 186, 120)),
             (0.7, (96, 158, 108)), (1.0, (36, 96, 132))]


def rain_map(w):
    """Runoff per cell over a dim relief: tan is desert, blue-green is soaked."""
    base = relief(w.height, shade=True) * 0.5
    # Ranked within the land distribution, not scaled by it: rain is skewed
    # enough - a wet coast is several times the median - that a linear ramp
    # paints every interior the same tan and shows none of the structure in it.
    v = _rank(w.runoff, w.runoff[w.land])
    rain = _ramp(v, RAIN_RAMP)
    return np.clip(np.where(w.land[..., None], rain * 0.72 + base * 0.5, base),
                   0, 255).astype(np.uint8)


def land_mask(h, lakes=None):
    rgb = np.where((h > 0)[..., None], np.array([232, 226, 208.0]),
                   np.array([28, 52, 84.0]))
    coast = (h > 0) != (grid.blur((h > 0).astype(float), 1.2) > 0.5)
    rgb = np.where(coast[..., None], np.array([20, 20, 20.0]), rgb)
    if lakes is not None:
        rgb = np.where(lakes[..., None], np.array([70, 120, 170.0]), rgb)
    return rgb.astype(np.uint8)


def slope_map(h):
    dy, dx = grid.gradient(h)
    return scalar(np.hypot(dy, dx), 0, float(np.percentile(np.hypot(dy, dx), 99.5)))


def erosion_diff(pre, post):
    d = post - pre
    lim = max(1e-6, float(np.percentile(np.abs(d), 99.5)))
    v = np.clip(d / lim, -1, 1)
    return _ramp(v, [(-1.0, (200, 60, 40)), (0.0, (25, 25, 28)),
                     (1.0, (60, 150, 220))]).astype(np.uint8)


LAYERS = [
    ("relief", water_map),
    ("elevation (no water)", lambda w: relief(w.height).astype(np.uint8)),
    ("tectonic relief", lambda w: relief(w.height_raw).astype(np.uint8)),
    ("pre-erosion", lambda w: relief(w.height_pre).astype(np.uint8)),
    ("plates", lambda w: plates(w.tect, shade_height=w.height)),
    ("boundaries", lambda w: plates(w.tect, cls=_cls(w), shade_height=w.height)),
    ("stress", lambda w: stress(w.tect)),
    ("rainfall", rain_map),
    ("drainage", lambda w: flow(w.flow, w.height, w.cfg.river_threshold * w.height.size)),
    ("erosion delta", lambda w: erosion_diff(w.height_pre, w.height_eroded)),
    ("slope", lambda w: slope_map(w.height)),
    ("land / coast", lambda w: land_mask(w.height, w.lakes)),
    ("river width", lambda w: river_width_map(w)),
]


def river_width_map(w):
    """Channel width and lake extent on a plain background, no shading."""
    rgb = np.where((w.height > 0)[..., None], np.array([238, 232, 216.0]),
                   np.array([32, 56, 88.0]))
    rgb = np.where(w.lakes[..., None], np.array([120, 160, 190.0]), rgb)
    v = np.clip(w.water.width / max(1e-6, w.cfg.river_width_max), 0, 1)
    return np.where(w.rivers[..., None],
                    _ramp(v, [(0.0, (150, 190, 214)), (0.5, (60, 118, 168)),
                              (1.0, (18, 48, 96))]), rgb).astype(np.uint8)


def _cls(w):
    from . import tectonics
    return tectonics.boundary_class(w.tect)


def layer(world, i):
    name, fn = LAYERS[i % len(LAYERS)]
    return name, fn(world)
