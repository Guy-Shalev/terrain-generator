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


def relief(h, shade=True, surface=None, tint=None, tint_mix=0.0):
    """Hypsometric tint of the bed, hillshaded off `surface` if water sits on it.

    `tint` is an optional per-cell colour - a biome map - blended into the land
    at `tint_mix` before shading. The sea is left alone.
    """
    rgb = hypsometric(h)
    if tint is not None and tint_mix > 0:
        rgb = np.where((h > 0)[..., None], rgb * (1 - tint_mix) + tint * tint_mix, rgb)
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


def overpaint_water(rgb, w):
    """Lay lakes and rivers over whatever the land was painted with."""
    wat = w.water
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
    return rgb


def water_map(w, shade=True):
    """The main view: relief with lakes at their own flat level and rivers by width.

    Lakes are shaded from the water surface, not the bed, so they read as flat
    sheets; the depth ramp underneath still shows how deep the basin is.

    The land is tinted towards its biome colour at `cfg.biome_tint`, so the
    hypsometric ramp still says how high somewhere is while its hue says what
    grows there. Mixed in before shading, and only over land: the sea has its
    own narrow ramp and there is no biome under it to say anything.

    Then the trees go on at `cfg.tree_relief`, well under full opacity. They are
    texture here, not the subject: the biome tint has already said where forest
    is, and what the stamps add is grain over it - a canopy that looks like
    canopy, an edge where it thins, the gallery strips picking out rivers that
    the ramp draws as one blue line through uniform green. At 1.0 they bury the
    hypsometric ramp and the layer stops being a relief map; at 0 they are off,
    which is what the `trees` layer is for.
    """
    rgb = relief(w.height, shade, surface=w.surface, tint=biome_rgb(w),
                 tint_mix=w.cfg.biome_tint)
    rgb = paint_trees(rgb, w.trees, w.height.shape, w.cfg.tree_relief)
    return np.clip(overpaint_water(rgb, w), 0, 255).astype(np.uint8)


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


# Indexed by `climate` biome id. Kept muted and in the same family as
# `LAND_RAMP`, because these get mixed into the main view rather than
# replacing it: a saturated palette here turns that layer into a political
# map. Ocean is never drawn from this - the sea ramp handles it.
BIOME_COLORS = np.array([
    (37, 86, 134),      # ocean, only ever a placeholder
    (240, 246, 252),    # ice
    (154, 158, 146),    # tundra
    (58, 94, 78),       # taiga
    (168, 164, 148),    # cold desert          grey, against the hot desert's tan
    (176, 182, 108),    # steppe
    (92, 134, 74),      # temperate forest
    (48, 102, 70),      # temperate rainforest
    (150, 138, 86),     # shrubland            olive-brown, against steppe's green
    (224, 190, 128),    # hot desert
    (202, 176, 92),     # savanna
    (122, 152, 64),     # tropical seasonal forest
    (34, 96, 48),       # tropical rainforest
], dtype=float)


def biome_rgb(w, soften=None):
    """Per-cell biome colour, sea left as the hypsometric ramp will paint it.

    Classification is a hard cut - a cell is one biome or another, and
    `world.biome` says which - but nothing on the ground changes over one cell,
    and drawn literally every band boundary is a stencil edge. So the colours
    are softened here rather than the ids anywhere: blurred, and blurred
    *masked*, so the coast stays as sharp as the sea ramp draws it instead of
    bleeding green into the water. Dividing by the blurred mask is what keeps a
    headland the full strength of its own colour rather than a fraction of it
    mixed with the ocean's nothing.
    """
    rgb = BIOME_COLORS[w.biome]
    s = w.cfg.biome_soften if soften is None else soften
    if s <= 0:
        return rgb
    m = w.land.astype(float)
    num = np.stack([grid.blur(rgb[..., c] * m, s) for c in range(3)], axis=-1)
    return num / np.maximum(grid.blur(m, s), 1e-6)[..., None]


def biome_map(w):
    """Flat biome colour, hillshaded, with the water drawn back over it."""
    rgb = biome_rgb(w)
    # Shallower than the main view's shading. The point here is the biome
    # boundaries, and at full contrast a mountain's own shadow reads as one.
    rgb = np.where(w.land[..., None], rgb * (0.78 + 0.38 * hillshade(w.height)[..., None]),
                   hypsometric(w.height))
    return np.clip(overpaint_water(rgb, w), 0, 255).astype(np.uint8)


TEMP_RAMP = [(-30.0, (78, 96, 168)), (-10.0, (120, 168, 208)),
             (0.0, (226, 234, 240)), (12.0, (232, 206, 132)),
             (24.0, (216, 130, 66)), (34.0, (166, 48, 44))]


def temp_map(w):
    """Mean annual temperature, absolute, over a dim relief.

    Absolute and not ranked, unlike the rainfall layer: degrees mean something
    on their own, and the freezing point sitting at a fixed place on the ramp
    is most of what the layer is for.
    """
    base = relief(w.height, shade=True) * 0.45
    return np.clip(_ramp(w.temp, TEMP_RAMP) * 0.75 + base * 0.5,
                   0, 255).astype(np.uint8)


# A tree, at map scale, is a silhouette three or four pixels across, and the
# whole job of these is that the four kinds read apart at that size. Each is a
# list of (dy, dx, weight) offsets from the trunk: a conifer is a spire, a
# broadleaf a dome, a palm a flat crown carried wide of a thin stem, and scrub a
# dot. Drawn as a colour ramp instead, all four are the same green.
TREE_STAMPS = [
    [(-2, 0, .40), (-1, 0, .85), (0, 0, 1.0), (0, -1, .45), (0, 1, .45), (1, 0, .65)],
    [(-1, 0, .75), (-1, -1, .35), (-1, 1, .35), (0, 0, 1.0), (0, -1, .85),
     (0, 1, .85), (1, 0, .70)],
    [(-1, 0, .45), (0, 0, .90), (0, -1, .80), (0, 1, .80), (0, -2, .50), (0, 2, .50)],
    [(0, 0, .75), (0, 1, .35)],
]
# Muted, and in the same family as BIOME_COLORS: these sit on top of the relief
# ramp, and a saturated green turns the layer into a golf course.
TREE_COLORS = np.array([
    (26, 58, 44),       # conifer     - blue-green, and the darkest of the four
    (52, 96, 50),       # broadleaf
    (96, 130, 58),      # palm        - yellower, which is what dry country does
    (114, 124, 72),     # scrub
], dtype=float)
TREE_SHADOW = np.array([16, 24, 20.0])
TREE_SHADOW_A = 0.38    # alpha of the offset shadow pass


def _splat(shape, ys, xs, offsets, weights):
    """Accumulate one stamp over every tree at once: one pass per offset, not
    per tree. Wraps in x and clamps in y, like every other neighbour read."""
    h, w = shape
    acc = np.zeros(shape)
    for dy, dx, a in offsets:
        np.add.at(acc, (np.clip(ys + dy, 0, h - 1), (xs + dx) % w), a * weights)
    return acc


def paint_trees(rgb, t, shape, strength=1.0):
    """Stamp individual trees over whatever the ground was painted with.

    `strength` scales every alpha, crowns and shadow alike, for the main view:
    there the trees are texture over a hypsometric ramp that is still doing the
    talking, and at full opacity they bury it.

    Composited kind by kind rather than tree by tree. Thirty thousand small
    slice assignments is a Python loop at map scale; four alpha grids is four
    vectorised passes, and the only thing lost is the painter's order *within* a
    kind, which at three pixels a crown nobody can see.

    Alpha is clipped, not normalised. Where a canopy closes, stamps overlap and
    the sum runs well past one - that saturation *is* closed canopy, and it is
    what makes a rainforest read as a solid mass while a savanna at a third of
    the cover stays a field of separate trees.
    """
    if len(t) == 0 or strength <= 0:
        return rgb
    ys = np.clip(t.pos[:, 0].astype(int), 0, shape[0] - 1)
    xs = t.pos[:, 1].astype(int) % shape[1]
    # One shadow pass for every kind together, offset down-right to agree with
    # the hillshade's north-west sun. Under it the canopy gains depth; without
    # it the trees look printed on.
    shade = np.clip(_splat(shape, ys + 1, xs + 1, [(0, 0, 1.0)], t.size), 0, 1)
    sa = (TREE_SHADOW_A * strength * shade)[..., None]
    rgb = rgb * (1 - sa) + TREE_SHADOW * sa
    for k, offsets in enumerate(TREE_STAMPS):
        m = t.kind == k
        if not m.any():
            continue
        a = strength * np.clip(
            _splat(shape, ys[m], xs[m], offsets, t.size[m]), 0, 1)[..., None]
        rgb = rgb * (1 - a) + TREE_COLORS[k] * a
    return rgb


GROUND_DESAT = 0.62     # how far the base is pulled towards its own luminance
GROUND_WARM = np.array([1.06, 1.00, 0.88])   # and then back towards earth


def tree_map(w):
    """The trees, on ground the colour of ground.

    The base keeps its biome tint - what is interesting here is the boundary
    between what has trees on it and what does not, and that only reads if the
    bare land beside a forest looks like the country it belongs to rather than
    like blank paper - but it is desaturated first, hard.

    That is the whole difference between this layer working and not. `LAND_RAMP`
    is green at every elevation a forest grows at and so is half of
    `BIOME_COLORS`, so trees painted straight onto it are green on green: the
    canopy disappears into the ground it stands on and the layer reads as a
    slightly darker biome map. Pulled towards luminance and warmed back, the
    ground keeps its relief and its regions while green becomes something only
    vegetation has.
    """
    base = relief(w.height, surface=w.surface, tint=biome_rgb(w), tint_mix=0.5)
    lum = base @ np.array([0.30, 0.59, 0.11])
    # Land only. The sea has no vegetation to distinguish itself from, and
    # draining it of colour along with the ground turns the ocean slate grey and
    # the map into something that looks broken rather than deliberate.
    ground = (base * (1 - GROUND_DESAT) + lum[..., None] * GROUND_DESAT) * GROUND_WARM
    base = np.where(w.land[..., None], ground * 0.94, base)
    return np.clip(overpaint_water(paint_trees(base, w.trees, w.height.shape), w),
                   0, 255).astype(np.uint8)


def canopy_map(w):
    """The density field the trees were scattered from, on its own."""
    base = relief(w.height, shade=True) * 0.45
    cov = _ramp(w.trees.cover, [(0.0, (198, 180, 144)), (0.35, (150, 164, 96)),
                                (0.7, (74, 122, 62)), (1.0, (26, 66, 42))])
    return np.clip(overpaint_water(
        np.where(w.land[..., None], cov * 0.78 + base * 0.5, base), w),
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
    ("main view", water_map),
    ("elevation (no water)", lambda w: relief(w.height).astype(np.uint8)),
    ("tectonic relief", lambda w: relief(w.height_raw).astype(np.uint8)),
    ("pre-erosion", lambda w: relief(w.height_pre).astype(np.uint8)),
    ("plates", lambda w: plates(w.tect, shade_height=w.height)),
    ("boundaries", lambda w: plates(w.tect, cls=_cls(w), shade_height=w.height)),
    ("stress", lambda w: stress(w.tect)),
    ("rainfall", rain_map),
    # Slotted in here rather than appended: the viewer binds number keys to the
    # first eleven layers only, and these two are worth reaching for.
    ("temperature", temp_map),
    ("biomes", biome_map),
    # Trees take the last number key, ahead of drainage. Both were worth
    # reaching for and only one can have it: drainage is a layer you open to
    # find out why a river went where it did, and this one is the world.
    ("trees", tree_map),
    ("drainage", lambda w: flow(w.flow, w.height, w.cfg.river_threshold * w.height.size)),
    ("erosion delta", lambda w: erosion_diff(w.height_pre, w.height_eroded)),
    ("slope", lambda w: slope_map(w.height)),
    ("land / coast", lambda w: land_mask(w.height, w.lakes)),
    ("river width", lambda w: river_width_map(w)),
    ("canopy", canopy_map),
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
