"""Stage 5: where the rain falls, and so how much water each cell contributes.

Two things set it. Zonal belts - wet on the equator, dry in the horse
latitudes, wet again at the mid-latitude storm tracks, dry at the poles - and
orographic transport: a parcel of air carries moisture inland from the sea,
drops some of it every cell, drops a lot of it climbing, and arrives at the far
side of a range with nothing left. The second is what makes a rain shadow, and
a rain shadow is what makes a desert that is not a stripe.

The output is runoff per cell, normalised to average one over land, so it drops
straight into `hydrology.accumulate` as weights without moving what
`river_threshold` means. Drainage area was already that field with every weight
at one.
"""
import numpy as np

from . import grid, noise


def _belts(shape, cfg, rng):
    """Zonal rain factor: three wet bands and three dry ones, wandering."""
    h, w = shape
    lat = grid.latitude(shape) + np.zeros((1, w))
    # The bands are latitude circles, and drawn as such they read as three
    # ruler-straight stripes across the map. Same trick the polar taper uses:
    # let the latitude they key on wander.
    lat = lat + cfg.rain_wobble * noise.fbm(h, w, rng, cfg.rain_wobble_periods, 3)
    band = np.cos(3.0 * np.pi * np.clip(lat, -1, 1))
    return 1.0 - cfg.rain_belts * (1.0 - band) * 0.5


def _march(h, land, belts, cfg):
    """Blow moisture from x=0 to x=w, raining as it goes. One row per latitude.

    The lap before the one that counts is there because the world is a cylinder:
    there is no upwind edge to start dry at, so the first pass is thrown away
    and only the state it leaves behind is kept.
    """
    hh, w = h.shape
    m = np.ones(hh)
    rain = np.zeros((hh, w))
    for lap in range(2):
        prev = h[:, -1]
        for x in range(w):
            hx = h[:, x]
            climb = np.maximum(hx - prev, 0.0)   # upwind gradient, this cell
            prev = hx
            wet = land[:, x]
            # The belt scales the whole rate, not just the flat part. Applied to
            # `rain_base` alone it is invisible wherever there is relief, which
            # is most of the land: the orographic term is several times the base
            # on any real slope, so the wet and dry latitudes come out the same.
            r = np.minimum(m * belts[:, x] * (cfg.rain_base +
                                              cfg.rain_orog * climb), m)
            r = np.where(wet, r, 0.0)
            # Over water the parcel takes moisture back up. Over land it gets a
            # share of what just fell handed back - evapotranspiration, and the
            # reason a continental interior is dry rather than empty. Without it
            # the loss is exponential in the distance from the coast: at the
            # rates that give a decent rain shadow, half of every landmass came
            # out at under a seventh of the mean and the rivers there vanished.
            m = m - r + np.where(wet, r * cfg.rain_recycle,
                                 (1.0 - m) * cfg.rain_ocean_gain)
            if lap:
                rain[:, x] = r
    return rain


def rainfall(h, cfg, rng, sea_level=0.0):
    """Rain per cell. Winds are zonal: easterly in the tropics and at the poles,
    westerly in between, which is why the marching is done in two groups."""
    land = h > sea_level
    belts = _belts(h.shape, cfg, rng)
    lat = np.abs(grid.latitude(h.shape))[:, 0]
    westerly = (lat > 1 / 3) & (lat < 2 / 3)

    rain = np.zeros_like(h)
    for sel, flip in ((westerly, False), (~westerly, True)):
        if not sel.any():
            continue
        sl = slice(None, None, -1) if flip else slice(None)
        out = _march(h[sel][:, sl], land[sel][:, sl], belts[sel][:, sl], cfg)
        rain[sel] = out[:, sl]
    # Weather is not a cell wide. The orographic term keys on the climb between
    # two neighbours, so undamped it dumps a year of rain on one row of cells
    # and leaves the next dry; blurring puts the total back over the range that
    # earned it. It also softens the join between the two wind bands.
    return grid.blur(rain, cfg.rain_blur)


def runoff(h, cfg, rng, sea_level=0.0):
    """Rainfall as accumulation weights: mean one over land, zero at sea.

    Returns `(runoff, evaporation)`. Evaporation is what a cell of lake surface
    loses in those same units, and it is `lake_evap` scaled by how dry the place
    is - a lake in a wet belt fills to its rim, the same basin in a rain shadow
    stands well below it or goes to salt. Scaling it by the reciprocal of the
    local runoff is the cheap reading of aridity: the two move together, since
    what the sky does not deliver the land cannot shed.
    """
    land = h > sea_level
    rain = rainfall(h, cfg, rng, sea_level)
    if not land.any():
        return np.zeros_like(h), np.full_like(h, cfg.lake_evap)
    mean = max(1e-9, float(rain[land].mean()))
    out = np.where(land, rain / mean, 0.0)
    evap = cfg.lake_evap / np.clip(np.where(land, out, 1.0), cfg.rain_evap_cap, None)
    return out, evap


def temperature(h, cfg, rng, sea_level=0.0):
    """Mean annual temperature and the seasonal half-range, both in degrees C.

    Three terms. Latitude sets the baseline, falling from the equator to the
    poles as the square of it. Height takes a lapse rate off that, on land
    only - the sea surface stands at whatever its latitude says. And distance
    from the sea sets how far the year swings either side of the mean: water
    holds its heat and the coast that sits on it barely moves between seasons,
    while an interior at the same latitude bakes and then freezes.

    That third term is why the pair is returned rather than the mean alone.
    Mean annual temperature cannot tell taiga from tundra - a Siberian interior
    averages below freezing and still grows forest, because its summer clears
    the tree line even though its winter is far worse than any coast's. What
    decides is the warmest month, and that is `temp + swing`.
    """
    lat = grid.latitude(h.shape)
    # Straight off the row index the isotherms are ruler-straight bands, the
    # same failure the rain belts have; wobble the latitude they key on.
    lat = np.clip(lat + cfg.temp_wobble * noise.fbm(
        h.shape[0], h.shape[1], rng, cfg.temp_wobble_periods, 3), -1.0, 1.0)
    base = cfg.temp_equator + (cfg.temp_pole - cfg.temp_equator) * lat ** 2
    # The offset lands hardest on the poles. A warmer world is a *flatter* one -
    # the Eocene ran an equator-to-pole gradient near 30 C against today's 45 -
    # so shifting every latitude by the same amount is the one thing warming
    # demonstrably does not do.
    #
    # Normalised over land, not over latitude. The weight averages to one across
    # a uniform globe, the mean of lat^2 being a third, but this world's land is
    # not uniform: the polar taper keeps it in the middle latitudes, where the
    # weight is below one, and a slider set to +10 delivered +6.9 C to the
    # ground. Dividing by the land mean makes the slider mean what it says on
    # land and leaves the pole-to-equator ratio, which is the point of it, alone.
    weight = (1.0 - cfg.temp_polar_amp) + 3.0 * cfg.temp_polar_amp * lat ** 2
    land = h > sea_level
    if land.any():
        weight = weight / max(1e-6, float(weight[land].mean()))
    temp = (base + cfg.temp_offset * weight
            - cfg.temp_lapse * np.maximum(h - sea_level, 0.0))
    # Saturating rather than linear: the moderating reach of an ocean is spent
    # within a few hundred km of it, and past that one more cell inland changes
    # nothing. Linear in the distance instead, the middle of a big continent
    # runs away to a swing no latitude justifies.
    swing = _swing(lat, land, cfg)
    # Sea ice moderates nothing: a frozen ocean has a lid on it and behaves like
    # land, which is why the Siberian coast is continental and Norway's is not.
    # So the distance that sets continentality is measured to *open* water. One
    # pass is enough - the first swing decides where the sea freezes, the second
    # uses it - and the fixed point is not worth chasing: the cells that would
    # flip on a third pass are the ones sitting exactly on the ice edge.
    frozen = ~land & (temp + swing < ICE_C)
    if frozen.any() and not (land | frozen).all():
        swing = _swing(lat, land | frozen, cfg)
    return temp, swing


def _swing(lat, closed, cfg):
    """Seasonal half-range: latitude, damped by how near open water is."""
    cont = 1.0 - np.exp(-grid.edt(closed) / max(1e-6, cfg.temp_cont_reach))
    return (cfg.temp_swing * np.abs(lat)
            * (cfg.temp_maritime + (1.0 - cfg.temp_maritime) * cont))


# Biome ids. Ocean is 0, so an empty grid reads as all sea.
(OCEAN, ICE, TUNDRA, TAIGA, COLD_DESERT, STEPPE, TEMPERATE_FOREST,
 TEMPERATE_RAINFOREST, SHRUBLAND, HOT_DESERT, SAVANNA, TROPICAL_SEASONAL,
 TROPICAL_RAINFOREST) = range(13)

BIOME_NAMES = [
    "ocean", "ice", "tundra", "taiga", "cold desert", "steppe",
    "temperate forest", "temperate rainforest", "shrubland", "hot desert",
    "savanna", "tropical seasonal forest", "tropical rainforest",
]

# Moisture index cuts - rainfall over what the local heat can evaporate.
# Holdridge's humidity provinces, near enough: under a quarter is desert, over
# two is rainforest.
# Shifted up from Holdridge's own cuts, which run 0.25 / 0.5 / 1 / 2, because
# this generator's rain is narrower than Earth's: the wettest land here gets
# 2.4x the mean where a real rainforest gets 3 to 5x, and the driest is nowhere
# near as dry. Left on the textbook numbers the whole map bunches into the two
# middle bands - deserts came out at 5% of land against Earth's 20%, and
# tropical rainforest at 1.5% against 13%. These are the same provinces read
# off a flatter distribution.
MI_BANDS = (0.4, 0.7, 1.1, 1.8)
# Mean annual temperature cuts, in C. The top one is 22 and not the 24 a real
# tropical mean sits above, because the equator here is 27 at sea level and the
# lapse rate takes the rest: at 24 the tropical column is reachable only below
# 0.08 of height and within 20 degrees of the line, which came out at 7% of land
# against Earth's 36%.
T_BANDS = (0.0, 6.0, 12.0, 17.0, 22.0)
ICE_C = 0.0     # warmest month below this and nothing ever thaws
TREE_C = 6.0    # warmest month below this and nothing grows tall

# Short names, so the table below reads as one.
_CD, _HD, _TU, _TA, _ST, _SH, _SV = (COLD_DESERT, HOT_DESERT, TUNDRA, TAIGA,
                                     STEPPE, SHRUBLAND, SAVANNA)
_TF, _TR, _PS, _PR = (TEMPERATE_FOREST, TEMPERATE_RAINFOREST,
                      TROPICAL_SEASONAL, TROPICAL_RAINFOREST)
BIOME_MATRIX = np.array([
    # polar subpolar cool temperate subtropical tropical
    [_CD, _CD, _CD, _CD, _HD, _HD],     # arid       mi < 0.25
    [_TU, _TU, _ST, _ST, _SH, _SV],     # semiarid   mi < 0.5
    [_TU, _TA, _TA, _TF, _PS, _PS],     # subhumid   mi < 1.1
    # Tropical humid is rainforest and not seasonal forest: potential
    # evapotranspiration is so high there that clearing 1.1 takes as much
    # absolute rain as a temperate perhumid cell gets.
    [_TU, _TA, _TF, _TF, _TF, _PR],     # humid      mi < 1.8
    [_TU, _TA, _TR, _TR, _PR, _PR],     # perhumid   mi >= 1.6
], dtype=np.int8)


def precip_mm(cfg):
    """What a runoff of 1.0 means in mm/yr, at this world's temperature.

    A warmer atmosphere holds more water - about 7% more per degree - but global
    rainfall is limited by the energy available to evaporate it, not by what the
    air could carry, so it rises nearer 2 to 3% per degree. `rain_per_degree` is
    that number, and without it the temperature slider only ever raises
    evapotranspiration: rain in millimetres could not move, so the warm end of
    the slider turned the map into a desert rather than the wetter world a real
    hothouse is. Measured over two seeds at 384x288, desert ran 27% of land at
    the middle of the slider and 52% at the top.

    Potential evapotranspiration still climbs faster than this does, so a hot
    world is drier *relative to its own thirst* - which is right, and is why the
    subtropics are where they are - it just no longer runs away.
    """
    return max(1.0, cfg.precip_mean_mm * (1.0 + cfg.rain_per_degree * cfg.temp_offset))


def biomes(h, temp, swing, runoff, cfg, sea_level=0.0):
    """What grows where: temperature against rain *for that temperature*.

    The moisture axis is not millimetres. `runoff` is already precipitation
    normalised to average one over land, so a constant turns it back into a
    depth - but a depth on its own says nothing, because the same rain that
    keeps a cold place in forest leaves a hot one bare. Divide it by potential
    evapotranspiration, which Holdridge reads straight off temperature, and the
    axis becomes how wet somewhere is relative to its own thirst. That single
    division is what puts the Gobi (cold, dry, and arid) and the tundra (cold,
    dry, and yet humid) in different biomes instead of painting both as desert.

    Then the two extremes that no matrix handles, because they are set by the
    warmest month rather than the mean: permanent ice, and the tree line.
    """
    # Biotemperature, not the annual mean: the year averaged with every month
    # below freezing counted as zero, because nothing grows or transpires in
    # those and averaging them in as negatives credits a cold winter for warmth
    # it never had. Twelve samples of the seasonal cycle, summed in place rather
    # than stacked - the stacked version is twelve full grids at once.
    #
    # Taking `clip(temp, 0, 30)` instead, as the annual mean, is not a rounding
    # error: it sends every freezing cell's PET to zero, the moisture index to
    # infinity, and a quarter of the map to taiga.
    biotemp = np.zeros_like(temp)
    for phase in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        biotemp += np.clip(temp + swing * np.cos(phase), 0.0, 30.0)
    biotemp /= 12.0
    # Floored at Holdridge's own polar line. Below it the cell is ice or tundra
    # by the overrides at the bottom of this function anyway, and the only job
    # left for the floor is to keep the division finite.
    pet = np.maximum(58.93 * biotemp, 58.93 * 1.5)      # mm/yr
    mi = precip_mm(cfg) * runoff / pet
    # Banded straight off the raw fields, every hill that crosses a cut puts a
    # lone cell of another biome in the middle of one, and the map comes out
    # speckled rather than regional. A short blur first costs nothing and only
    # takes out features narrower than the blur - a range wide enough to have
    # its own climate is far wider than this, so altitudinal zonation survives.
    out = BIOME_MATRIX[np.searchsorted(MI_BANDS, grid.blur(mi, cfg.biome_blur)),
                       np.searchsorted(T_BANDS, grid.blur(temp, cfg.biome_blur))]
    warmest = temp + swing
    out = np.where(warmest < TREE_C, TUNDRA, out)
    out = np.where(warmest < ICE_C, ICE, out)
    return np.where(h > sea_level, out, OCEAN).astype(np.int8)
