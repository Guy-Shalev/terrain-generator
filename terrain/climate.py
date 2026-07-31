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
    lat = (np.arange(h) / max(1, h - 1) * 2.0 - 1.0)[:, None] + np.zeros((1, w))
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
    lat = np.abs(np.arange(h.shape[0]) / max(1, h.shape[0] - 1) * 2.0 - 1.0)
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
