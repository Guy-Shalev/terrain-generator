"""Invariant checks for the generator. Run: python test_terrain.py"""
import json
import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np
from scipy import ndimage

from terrain import Config, generate
from terrain import climate, export, grid, hydrology, noise, render, rivers, trees, world


def test_grid_wraps():
    a = np.arange(12).reshape(3, 4).astype(float)
    assert np.array_equal(grid.shift(a, 0, 1)[:, -1], a[:, 0]), "x must wrap"
    assert np.array_equal(grid.shift(a, -1, 0)[0], a[0]), "y must clamp"
    m = np.ones((8, 8), bool)
    m[4, 0] = False
    d = grid.edt(m)
    assert d[4, 7] == 1.0, "distance must cross the seam"


def test_noise_seamless():
    rng = np.random.default_rng(1)
    n = noise.fbm(64, 64, rng, periods=4, octaves=4)
    seam = np.abs(n[:, 0] - n[:, -1]).max()
    interior = np.abs(np.diff(n, axis=1)).max()
    assert seam <= interior * 1.5, f"seam discontinuity {seam} vs {interior}"
    assert np.abs(n).max() < 1.6


def test_fill_and_flow():
    rng = np.random.default_rng(3)
    h = -np.hypot(*np.mgrid[-16:16, -16:16]) * 0.05 + rng.normal(0, 0.01, (32, 32))
    h[16, 16] = -0.3  # a pit near the summit that must be filled
    filled = hydrology.fill_depressions(h, sea_level=-9.0)
    # 1e-6 covers the single-precision surface the fill returns against the
    # double-precision one handed to it; the pipeline itself is float32 on both
    # sides of this call, so there the guarantee is exact.
    assert (filled >= h - 1e-6).all(), "filling may only raise the surface"
    assert filled[16, 16] > h[16, 16], "pit was not filled"
    rec, rec_d, _ = hydrology.flow_routing(filled)
    acc = hydrology.accumulate(filled, rec)
    assert acc.min() >= 1.0
    assert acc.max() <= h.size
    # Every drop must end up somewhere: total outflow at sinks == cell count.
    sinks = rec.ravel() == np.arange(h.size)
    assert abs(acc.ravel()[sinks].sum() - h.size) < 1e-6


def _bumpy(shape, seed, scale=4):
    rng = np.random.default_rng(seed)
    return grid.blur(rng.normal(0, 1, shape), scale) * 3.0


def test_fill_warm_start_is_exact():
    """The multigrid warm start may change the cost, never the answer."""
    for shape in ((160, 224), (150, 198)):   # even and odd halvings
        h = _bumpy(shape, 11)
        outlet = h <= 0.0
        outlet[0] = outlet[-1] = True
        exact = hydrology._solve(h, outlet, 1e-5, None, 6000)
        assert np.abs(hydrology.fill_depressions(h) - exact).max() < 1e-12, shape


def test_halo_matches_shift():
    """`Halo` is a speed trick and must return exactly what `shift` returns.

    It replaces the fancy-index gather in the thermal and routing loops, which
    was the single most expensive thing in the generator. The wrap-x and
    clamp-y edges are the whole subtlety: the corners have to inherit the
    wrapped columns, so the buffer's x edges are filled before its y edges.
    """
    for shape in ((5, 7), (2, 3), (9, 4)):
        a = np.arange(shape[0] * shape[1], dtype=float).reshape(shape) * 0.37
        halo = grid.Halo(shape).load(a)
        for dy, dx in grid.NEIGH8:
            assert np.array_equal(halo.at(dy, dx), grid.shift(a, dy, dx)), \
                f"{shape} offset {(dy, dx)}"
            assert np.array_equal(grid.neighbour_index(shape, dy, dx),
                                  grid.shift(np.arange(a.size).reshape(shape),
                                             dy, dx))


def test_carve_outlets_ignores_a_handed_in_routing():
    """Passing the erosion stage's routing in must not change the result.

    The river stage opens by filling and routing the surface erosion just
    filled and routed, so the answer is handed over instead. It is only valid
    before the first notch, which is why carve_outlets drops it after one use.
    """
    cfg = Config(width=128, height=128, seed=4)
    h = _bumpy((128, 128), 7)
    h[40:70, 40:70] -= 0.6                       # a basin to carve out of
    filled = hydrology.fill_depressions(h, 0.0)
    rec, _, _ = hydrology.flow_routing(filled)
    flow = hydrology.accumulate(filled, rec)
    plain = rivers.carve_outlets(h.copy(), cfg)
    handed = rivers.carve_outlets(h.copy(), cfg, routed=(filled, rec, flow))
    assert np.array_equal(plain, handed), "handing the routing in changed the carve"


def test_accumulate_matches_ordered_walk():
    """Level peeling and the elevation-ordered walk are both topological."""
    h = _bumpy((96, 128), 4)
    filled = hydrology.fill_depressions(h, sea_level=-99.0)
    rec, _, _ = hydrology.flow_routing(filled)
    walk = np.ones(filled.size)
    r = rec.ravel()
    for i in np.argsort(filled, axis=None)[::-1]:
        if r[i] != i:
            walk[r[i]] += walk[i]
    assert np.array_equal(hydrology.accumulate(filled, rec), walk.reshape(h.shape))


def test_outlet_carving_drains_basins():
    """Carving must remove impounded volume, never add it.

    The slab drains one way and the basin sits low on it, so the basin has a
    catchment several times its own area and spills. That is the case carving is
    for: a basin whose inflow all evaporates has no outflow to cut a gorge with,
    and `test_lake_water_balance` covers that one.
    """
    cfg = Config(width=96, height=96, seed=4)
    rng = np.random.default_rng(4)
    y, x = np.mgrid[0:96, 0:96]
    h = 0.6 - 0.004 * y + 0.05 * rng.normal(0, 1, (96, 96))
    h = grid.blur(h, 3)
    h[60:76, 36:60] -= 0.12  # a basin with no way out, and well above sea level
    before = (hydrology.fill_depressions(h) - h).sum()
    after_h = rivers.carve_outlets(h.copy(), cfg)
    after = (hydrology.fill_depressions(after_h) - after_h).sum()
    assert after < before, f"carving did not drain anything: {before} -> {after}"
    assert (after_h <= h + 1e-12).all(), "carving may only lower the surface"


def test_rain_shadow_and_normalisation():
    """Rain must fall on the windward side, and average one unit over land.

    The average is what lets `runoff` drop into `accumulate` as weights without
    moving `river_threshold`, which counts cells. The shadow is the whole point
    of marching moisture instead of reading a latitude off the row index.
    """
    h = np.full((120, 240), -0.4)
    h[:, 80:160] = 0.05                      # a continent
    ridge = np.exp(-((np.arange(240) - 110) / 6.0) ** 2)
    h[:, :] += 0.9 * ridge[None, :] * (h > 0)
    cfg = Config(width=240, height=120, ref_width=240, rain_belts=0.0)
    ro, evap = climate.runoff(h, cfg, np.random.default_rng(0))
    land = h > 0
    assert abs(ro[land].mean() - 1.0) < 1e-9, "runoff is not normalised"
    assert (ro[~land] == 0).all(), "the sea contributes runoff"
    # Mid-latitudes blow west to east, so for those rows the far side of the
    # ridge is the dry one; the tropics and poles blow the other way.
    lat = np.abs(np.arange(120) / 119 * 2 - 1)
    west = (lat > 1 / 3) & (lat < 2 / 3)
    up, lee = ro[west, 95:108].mean(), ro[west, 113:126].mean()
    assert up > 2 * lee, f"no rain shadow: windward {up:.2f} vs lee {lee:.2f}"
    assert ro[~west, 113:126].mean() > ro[~west, 95:108].mean(), \
        "the easterly bands shadow the same side as the westerlies"
    # Aridity is what a lake has to survive, so it must run the other way.
    assert evap[land][ro[land] < 0.5].mean() > evap[land][ro[land] > 1.5].mean()


def test_temperature_gradients():
    """Latitude sets it, height takes off it, and the sea holds the year steady.

    The third one is the reason `temperature` returns a pair. Mean annual alone
    cannot tell a maritime coast from an interior at the same latitude, and it
    is the interior's summer - not its average - that decides whether anything
    grows there.
    """
    h = np.full((160, 240), -0.4)
    h[:, 60:180] = 0.05                      # a continent, 120 cells across
    h[60:100, 100:140] = 0.45                # a plateau in the middle of it
    cfg = Config(width=240, height=160, ref_width=240, temp_wobble=0.0)
    t, sw = climate.temperature(h, cfg, np.random.default_rng(0))

    assert t[80].mean() > t[0].mean() and t[80].mean() > t[-1].mean(), \
        "the equator must be warmer than either pole"
    # The plateau against flat land on the same rows, clear of it.
    lifted, flat = t[70:90, 110:130].mean(), t[70:90, 70:90].mean()
    drop = cfg.temp_lapse * 0.40
    assert abs((flat - lifted) - drop) < 1.0, \
        f"lapse rate is off: {flat - lifted:.1f}C over 0.40 of height, want {drop:.1f}"

    # Same rows, so the same latitude and the same share of the pole-ward
    # swing; all that differs is how far the sea is.
    coast, interior = sw[20:60, 62:66].mean(), sw[20:60, 118:122].mean()
    assert interior > coast * 1.5, \
        f"continentality does nothing: coast {coast:.1f}C vs interior {interior:.1f}C"
    assert sw.min() >= 0.0, "a seasonal half-range cannot be negative"
    assert sw[5:15].mean() > sw[75:85].mean(), \
        "the year must swing further at the pole than at the equator"


def test_biome_placement():
    """The classification must key on rain *relative to heat*, not on rain.

    Two cells with the same rainfall, one hot and one cold, are not the same
    biome: the hot one evaporates its way to desert while the cold one is
    merely dry. That division is the whole difference between this and a plain
    temperature-by-precipitation lookup, so it gets its own assertion.
    """
    shape = (5, 4)
    h = np.full(shape, 0.1)
    # No pre-blur: this grid is smaller than the kernel, and what is under test
    # is the classification, not the smoothing.
    cfg = Config(width=4, height=5, ref_width=4, biome_blur=0.0)
    # runoff of 1.0 is the land average, so `precip_mean_mm` mm of rain.
    temp = np.array([[28.0], [28.0], [4.0], [-30.0], [10.0]]) + np.zeros(shape)
    runoff = np.array([[3.0], [0.25], [0.25], [1.0], [0.08]]) + np.zeros(shape)
    swing = np.zeros(shape)
    b = climate.biomes(h, temp, swing, runoff, cfg)[:, 0]

    assert b[0] == climate.TROPICAL_RAINFOREST, "hot and soaked is not rainforest"
    assert b[1] == climate.DESERT, "hot and dry is not desert"
    # Rows 1 and 2 get *the same rainfall*, 24 degrees apart. A lookup on
    # millimetres has to return the same biome for both. Dividing by what the
    # heat can evaporate makes the cold one merely damp - and it is that one
    # assertion which fails if the PET term is ever dropped.
    assert b[2] != climate.DESERT, \
        "the same rain that leaves a desert at 28C must not at 4C"
    assert b[3] == climate.ICE, "nothing above freezing all year must be ice"
    assert b[4] == climate.DESERT, "cold and genuinely arid is still desert"
    assert climate.biomes(np.full(shape, -0.1), temp, swing, runoff, cfg).max() \
        == climate.OCEAN, "below sea level must be ocean whatever the climate"

    # Tundra is cold *and wet* - the case a moisture axis in millimetres gets
    # wrong, because in millimetres it looks like desert.
    cold_wet = climate.biomes(h, np.full(shape, -2.0), np.full(shape, 10.0),
                              np.full(shape, 0.5), cfg)
    assert (cold_wet == climate.TUNDRA).all(), \
        "a cold cell whose summer clears freezing must be tundra, not ice or desert"

    w = generate(Config(width=128, height=96, seed=3), verbose=False)
    assert w.biome.min() >= 0 and w.biome.max() < len(render.BIOME_COLORS), \
        "a biome id has no colour"
    assert ((w.biome != climate.OCEAN) == w.land).all(), \
        "the biome grid and `world.land` disagree about the coastline"


def test_rain_belts():
    """The zonal bands must dry the horse latitudes and not the equator.

    Flat land, so nothing but the belts can vary the rain - but with an ocean
    either side of it, because the bands only show against something that
    resupplies the air. On an all-land world the two even out: a dry belt rains
    less and so keeps more moisture for the next cell, and after the field is
    normalised the row means come back level to two decimal places.
    """
    h = np.full((180, 120), -0.4)
    h[:, 40:80] = 0.05
    kw = dict(width=120, height=180, ref_width=120, rain_wobble=0.0)
    ro, _ = climate.runoff(h, Config(**kw), np.random.default_rng(0))
    band = ro[:, 40:80].mean(axis=1)
    # Row 90 is the equator; the horse latitudes are a third of the way out,
    # which on 180 rows is row 60. Row 30 is 60 degrees - a wet belt, not a dry
    # one - and picking it there gives two readings that match to the decimal.
    eq, horse = band[88:92].mean(), band[58:62].mean()
    assert eq > 1.4 * horse, f"belts are flat: equator {eq:.2f} vs 30 deg {horse:.2f}"
    flat, _ = climate.runoff(h, Config(rain_belts=0.0, **kw),
                             np.random.default_rng(0))
    assert np.ptp(flat[:, 40:80].mean(axis=1)) < np.ptp(band), \
        "rain_belts=0 is not flat"


def _basin_world(shape=(96, 96), seed=4):
    """A cone draining to the edges with one closed basin dug into it."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    h = 0.4 - 0.004 * np.hypot(y - shape[0] / 2, x - shape[1] / 2)
    h = grid.blur(h + 0.05 * rng.normal(0, 1, shape), 3)
    h[36:60, 36:60] -= 0.12
    return h


def test_lake_water_balance():
    """A lake holds what its catchment can keep, and keeps what it holds.

    With no evaporation every basin fills to its rim and spills, which is the
    old behaviour. With evaporation the same basin stands lower, stops spilling
    once its whole inflow goes to the air, and the water that reached it does
    not reappear downstream - the accumulation is gated, not patched afterwards.
    """
    h = _basin_world()
    wet = lambda r: int(((r.lake_id > 0) & (r.level[r.lake_id] > h)).sum())
    full = rivers._route_water(h, Config(width=96, height=96, lake_evap=0.0))
    thin = rivers._route_water(h, Config(width=96, height=96, lake_evap=40.0))
    assert full.n and thin.n, "no basin to balance"
    assert wet(full) > wet(thin) > 0, f"{wet(full)} -> {wet(thin)} wet cells"
    assert full.outflow.max() > 0, "a rim-full lake must spill"
    assert thin.outflow.max() == 0, "an evaporating lake must not spill"
    # Gating is the point: a lake that keeps its inflow removes it from the
    # network, so less than one unit per cell can reach the sinks.
    def to_sinks(r):
        sinks = r.rec.ravel() == np.arange(h.size)
        return r.flow.ravel()[sinks].sum()
    assert abs(to_sinks(full) - h.size) < 1e-6, "nothing should be lost with no evaporation"
    lost = to_sinks(full) - to_sinks(thin)
    assert lost > 100, f"evaporation removed only {lost} cells of runoff"


def test_lake_balance_leaves_no_cycles():
    """Pointing a lake's cells at its exit must not make water circulate.

    Every lake cell is rerouted to the lowest cell of its own filled surface so
    the whole inflow arrives at one gate. That target is strictly below what it
    receives from, which is what keeps the graph acyclic - and a cycle would not
    raise anything, it would quietly drop the water going round it.
    """
    h = _basin_world((80, 112), seed=6)
    r = rivers._route_water(h, Config(width=112, height=80, lake_evap=0.0))
    rec = r.rec.ravel()
    seen = np.zeros(h.size, bool)
    for start in range(0, h.size, 7):       # every seventh cell, walked to a sink
        c, steps = start, 0
        while rec[c] != c:
            c = rec[c]
            steps += 1
            assert steps <= h.size, f"receiver walk from {start} never ends"
        seen[c] = True
    assert seen.any()


def test_lakes_and_rivers():
    w = generate(Config(width=192, height=144, seed=9, erosion_passes=2), verbose=False)
    wat = w.water

    lake = wat.lake_mask
    if lake.any():
        # One flat surface per lake, always at or above its own bed.
        for i in range(1, wat.lake_id.max() + 1):
            sel = wat.lake_id == i
            assert np.ptp(wat.lake_level[sel]) < 1e-9, f"lake {i} surface is not level"
            assert (wat.lake_level[sel] >= w.height[sel] - 1e-9).all()
        assert (wat.lake_depth[lake] > 0).all()
    assert (wat.lake_depth[~lake] == 0).all()

    assert wat.polylines, "no rivers traced"
    for pts in wat.polylines:
        assert len(pts) >= 3
    # Discharge only grows downstream, so width must too. Read off the per-node
    # widths rather than by sampling `flow` under the path: a migrated channel
    # sits a few cells off the D8 line it was traced from, and sampling the grid
    # there reads whatever drains that cell, not what the river carries.
    for wv in wat.widths:
        assert wv[-1] >= wv[0] * 0.9, "path is not running downstream"
    assert wat.river_mask.any()
    assert wat.width[wat.river_mask].min() > 0
    assert wat.width.max() <= w.cfg.river_width_max + 1e-9
    # Rivers are cut into the bed, never raised above it.
    assert (w.height <= w.height_eroded + 1e-9).all()


def test_rain_weighted_erosion_carves_the_wet_side_harder():
    """Two identical flanks, rain on one of them, and only one gets dissected.

    A ridge with sea either side, so both flanks drain the same way down the
    same slope from the same height. The only thing told apart is what falls on
    them. `k` is well above the configured one on purpose: this is asking
    whether the weighting reaches the incision at all, not whether the shipped
    tuning makes it visible - which is a separate question, and the answer is
    that at `erosion_k` the fluvial term is small next to thermal creep.
    """
    y, x = np.mgrid[0:96, 0:160]
    h = 0.6 - 0.012 * np.abs(x - 80) + np.random.default_rng(0).normal(0, 0.004, (96, 160))
    wet = np.where(x < 80, 1.8, 0.2)        # west flank soaked, east flank dry
    wet = np.where(h > 0, wet, 0.0)

    def cut(weights):
        out = hydrology.stream_power(h.copy(), 0.0, passes=4, k=0.06, m=0.5, n=1.0,
                                     thermal_iters=12, talus=0.045, weights=weights)[0]
        d = h - out
        flank = (h > 0.1)
        return d[flank & (x < 80)].mean(), d[flank & (x > 80)].mean()

    fw, fd = cut(None)
    ww, wd = cut(wet)
    assert abs(fw - fd) < 0.25 * max(fw, fd), \
        f"unweighted flanks should erode alike: {fw:.4f} vs {fd:.4f}"
    assert ww / wd > 1.5 * (fw / fd), \
        f"rain did not lopside the ridge: {ww / wd:.2f} against {fw / fd:.2f}"
    assert wd < fd, "the dry flank should be spared, not merely out-cut"


def test_currents_tell_the_two_coasts_apart():
    """A gyre is cold down the east side of its basin and warm up the west one.

    Which is to say: cold off a continent's west coast, warm off its east, and
    only in the subtropics for the cold half. Straight sea, so nothing but the
    shore's direction can be making the difference.
    """
    h = np.full((288, 384), -0.5)
    h[:, 150:250] = 0.3
    cfg = Config(width=384, height=288, ref_width=384)
    a = climate.currents(h, cfg)
    sea = h <= 0
    # 1e-6, not 0: the field is single precision, so what is left after the mean
    # is subtracted is rounding on a quantity of order half a degree.
    assert abs(float(a[sea].mean())) < 1e-6, "currents must move heat, not add it"
    assert not a[~sea].any(), "the anomaly is the sea's, not the land's"

    lat = np.abs(grid.latitude(h.shape))[:, 0]
    sub = int(np.argmin(np.abs(lat - cfg.current_lat)))
    west, east = a[sub, 140], a[sub, 260]        # water either side of the land
    assert west < -2.0, f"no upwelling off the west coast: {west:.2f}"
    assert east > 0.5, f"no warm current off the east coast: {east:.2f}"
    assert abs(a[sub, 60]) < abs(west), "open ocean should be nearer the mean"
    # Upwelling is a subtropical band; the warm limb keeps going poleward.
    pol = int(np.argmin(np.abs(lat - 0.8)))
    assert abs(a[pol, 140]) < abs(west), "upwelling reached the pole"
    assert a[pol, 260] > east, "the warm limb should strengthen poleward"


def test_currents_change_the_map():
    """Measured, because a field that nothing downstream reads is decoration.

    Same seed either way, and the current field draws no random numbers, so the
    surface going *into* erosion is identical and the cells compare one to one.
    It does not survive erosion: currents reach the rain, the rain weights the
    carving, and the finished terrain is theirs too.
    """
    kw = dict(seed=7, width=192, height=144, erosion_passes=2)
    off = generate(Config(current_cold=0.0, current_warm=0.0, **kw), verbose=False)
    on = generate(Config(**kw), verbose=False)
    assert np.array_equal(off.height_pre, on.height_pre), \
        "currents moved the terrain before erosion; they may only move the climate"
    land = on.land
    d = np.abs(on.runoff - off.runoff)[land] / np.maximum(off.runoff[land], 1e-6)
    assert np.percentile(d, 90) > 0.05, f"currents barely touched the rain: {d.max():.3f}"
    moved = ((on.biome != off.biome) & land).sum() / land.sum()
    assert moved > 0.01, f"only {moved:.1%} of land changed biome"


def test_warm_seas_feed_more_rain():
    """The air over a warm sea carries more, so the tropics are wet without a
    band being drawn there. Belts off, so capacity is the only thing left."""
    h = np.full((160, 240), -0.4)
    h[:, 90:150] = 0.05                     # a flat continent, nothing to lift air
    kw = dict(width=240, height=160, ref_width=240, rain_belts=0.0)
    temp, _ = climate.temperature(h, Config(**kw), np.random.default_rng(0))
    land = h > 0
    lat = np.abs(grid.latitude(h.shape))

    def by_latitude(cap):
        ro, _ = climate.runoff(h, Config(rain_capacity=cap, **kw),
                               np.random.default_rng(0), temp=temp)
        warm = land & (lat < 0.25)
        cold = land & (lat > 0.75)
        return float(ro[warm].mean()), float(ro[cold].mean())

    warm, cold = by_latitude(0.07)
    assert warm > 2 * cold, f"tropics {warm:.2f} against poles {cold:.2f}"
    flat_warm, flat_cold = by_latitude(0.0)
    assert abs(flat_warm - flat_cold) < abs(warm - cold), \
        "capacity 0 spread the rain as widely as Clausius-Clapeyron did"


def test_wet_slopes_lose_height_gently():
    """A saturated updraught cools at about half the rate a dry one does, so the
    same plateau is warmer on the wet side of a map than on the dry side."""
    h = np.full((120, 200), 0.05)
    h[40:80, :] = 0.45                      # one plateau, spanning both climates
    runoff = np.zeros_like(h) + 0.2
    runoff[:, 100:] = 2.0                   # dry half, wet half
    top = np.zeros(h.shape, bool)
    top[40:80] = True
    dry, wet = top & (runoff < 1), top & (runoff > 1)

    def plateau(moist):
        cfg = Config(width=200, height=120, ref_width=200,
                     temp_lapse_moist=moist, temp_wobble=0.0)
        t, _ = climate.temperature(h, cfg, np.random.default_rng(0), humid=runoff)
        return float(t[dry].mean()), float(t[wet].mean())

    d, w = plateau(0.25)
    assert w > d + 3.0, f"wet plateau {w:.1f} C, dry {d:.1f} C - no difference"
    d0, w0 = plateau(0.0)
    assert abs(w0 - d0) < 0.05, "the flat rate is not flat"
    # The middle of the distribution keeps `temp_lapse`, so the calibration the
    # number was chosen for survives being made local.
    assert abs((d + w) / 2 - (d0 + w0) / 2) < 0.6


def test_temperature_slider_moves_a_whole_climate():
    """Warming must move rain, the gradient and the ice with it.

    The slider used to raise potential evapotranspiration and nothing else:
    rainfall in millimetres could not move, because `runoff` is normalised to
    average one over land and cannot know the world got hotter. Desert ran 3% of
    land at the cold end and 52% at the warm one, which is neither Earth nor
    interesting. Three couplings fix it - rain follows temperature, the offset
    lands hardest on the poles, and a frozen sea stops moderating its coast.
    """
    kw = dict(width=256, height=192, seed=7, erosion_passes=2)
    cold = generate(Config(temp_offset=-10.0, **kw), verbose=False)
    warm = generate(Config(temp_offset=+10.0, **kw), verbose=False)

    def share(w, *biomes):
        land = w.height > 0
        return float(np.isin(w.biome[land], biomes).mean())

    def gradient(w):
        h = w.temp.shape[0]
        return (w.temp[h // 2 - 8:h // 2 + 8].mean() -
                np.concatenate([w.temp[:16], w.temp[-16:]]).mean())

    for w in (cold, warm):
        d = share(w, climate.DESERT)
        assert 0.12 < d < 0.45, f"desert is {d:.0%} of land at {w.cfg.temp_offset:+.0f}"
    # A warm world is a flatter one, and the slider means what it says on land.
    assert gradient(warm) < gradient(cold) - 15, \
        f"gradient did not narrow: {gradient(cold):.0f} -> {gradient(warm):.0f}"
    shift = warm.temp[warm.height > 0].mean() - cold.temp[cold.height > 0].mean()
    assert abs(shift - 20.0) < 3.0, f"20 degrees of slider moved land by {shift:.1f}"
    assert share(warm, climate.ICE) < share(cold, climate.ICE)
    assert share(warm, climate.TROPICAL_RAINFOREST) > \
        share(cold, climate.TROPICAL_RAINFOREST)


def test_frozen_sea_stops_moderating_its_coast():
    """Sea ice has a lid on it and behaves like land, so the coast behind it
    swings like an interior. Measured on the field, not on the biomes: the
    continentality term is what carries it."""
    kw = dict(width=256, height=192, seed=7, erosion_passes=2)
    cold = generate(Config(temp_offset=-12.0, **kw), verbose=False)
    mild = generate(Config(temp_offset=0.0, **kw), verbose=False)
    lat = np.abs(grid.latitude(cold.height.shape))
    for w, name in ((cold, "cold"), (mild, "mild")):
        w.frozen = (w.height <= 0) & (w.temp + w.swing < 0.0)
    assert cold.frozen.mean() > 3 * mild.frozen.mean(), "the cold world froze no sea"
    polar = (cold.height > 0) & (lat > 0.55)
    assert polar.any()
    assert cold.swing[polar].mean() > mild.swing[polar].mean() + 1.5, \
        "polar coasts did not turn continental"


def test_rivers_reach_water():
    """Every drawn river must be part of a body that reaches a lake or the sea.

    A river ending in open ground is the one defect that reads as broken from
    across the map. They came from channels being cut at the rim of the basin
    holding a lake rather than at the water's edge - a quarter of the river
    cells on this seed, a median of 4 to 7 cells short of the pond they were
    running into.

    Labelled on a 3x tiling, so a body spanning the x seam stays one body.
    """
    for kw in (dict(width=512, height=384, seed=1),      # was 25% stray
               dict(width=384, height=288, seed=42),
               dict(width=256, height=192, seed=3)):
        w = generate(Config(**kw), verbose=False)
        wid = w.height.shape[1]
        sea, lake, riv = w.height <= 0, w.lakes, w.rivers
        lab, _ = ndimage.label(np.concatenate([sea | lake | riv] * 3, axis=1),
                               np.ones((3, 3)))
        reaches = set(np.unique(lab[np.concatenate([sea | lake] * 3, axis=1)])) - {0}
        stray = np.isin(lab[:, wid:2 * wid], list(reaches), invert=True) & riv
        assert not stray.any(), \
            f"{int(stray.sum())} of {int(riv.sum())} river cells reach no water: {kw}"


def test_meander_stays_attached():
    """The swing must move the path and leave both ends exactly where they were,
    or every tributary detaches from its trunk."""
    rng = np.random.default_rng(0)
    cfg = Config()
    pts = np.stack([np.arange(80.0), np.zeros(80)], axis=-1)
    out = rivers.meander(pts.copy(), np.full(80, 2.0), cfg, rng)
    ends = np.hypot(*(out - pts)[[0, -1]].T)
    assert (ends < 1e-9).all(), "endpoints moved; tributaries would detach"
    assert np.abs(out[:, 1]).max() > 0.2, "no meander at all"
    assert np.abs(out[:, 1]).max() < cfg.meander_amp * np.sqrt(2.0) * 1.5


_MARGIN_KW = dict(width=256, height=192, seed=7, erosion_passes=1)


def _margin_masks(w):
    """Margin shoreline and open-coast shoreline of a world."""
    t, h = w.tect, w.height_pre
    typ = np.abs(t.cont_self - t.cont_other)
    coast = np.abs(h) < 0.03
    return (coast & (typ > 0.8) & (t.dist < w.cfg.margin_zone),
            coast & (typ < 0.2))


def _shore_dist(w, sel):
    """Mean distance from the cells of `sel` to `w`'s shoreline."""
    land = w.height_pre > 0
    return np.abs(grid.edt(land) - grid.edt(~land))[sel].mean()


def _travel(before, after, sel):
    """How far `sel`'s shoreline moved between two worlds, in cells.

    Measured against `before` compared with itself, not against zero: a cell
    picked by an elevation window is near the contour, not exactly on it, so
    the raw distance carries a couple of cells of offset either way.
    """
    return _shore_dist(after, sel) - _shore_dist(before, sel)


def test_margin_fray_moves_the_shoreline():
    """The fray must actually shift the margin coast, and not the rest.

    Travel is the only honest measure here. Reasoning about it as
    amplitude/slope overstates it about fivefold, because fBm of these octave
    counts has a standard deviation near 0.18 rather than 1 - which is exactly
    why the stage is written against normalised noise.
    """
    # Inlets off in both runs: they reshape the coast themselves, and this is
    # measuring the fray.
    solo = dict(margin_cut_h=0.0, **_MARGIN_KW)
    off = generate(Config(margin_h=0.0, margin_h_fine=0.0, **solo), verbose=False)
    on = generate(Config(**solo), verbose=False)
    margin, plain = _margin_masks(off)
    assert margin.sum() > 100 and plain.sum() > 100, "not enough coast to compare"
    moved, untouched = _travel(off, on, margin), _travel(off, on, plain)
    assert moved > 0.8, f"margin shoreline barely moved: {moved:.2f} cells"
    assert abs(untouched) < 0.3, f"open coast moved too: {untouched:.2f} cells"


def test_margin_cuts_only_remove_land():
    """The inlets may lower the surface and never raise it.

    Checked with the fray off, so what is left is the cut term alone. It is
    clipped non-negative and subtracted, which is what keeps it from filling
    anything in while it carves.
    """
    kw = dict(margin_h=0.0, margin_h_fine=0.0, **_MARGIN_KW)
    off = generate(Config(margin_cut_h=0.0, **kw), verbose=False)
    on = generate(Config(**kw), verbose=False)
    assert (on.height_pre <= off.height_pre + 1e-9).all(), \
        "the cut term raised the surface somewhere"
    assert on.land.mean() < off.land.mean(), "no land was removed"
    drowned = (off.height_pre > 0) & ~(on.height_pre > 0)
    typ = np.abs(off.tect.cont_self - off.tect.cont_other)
    assert drowned.sum() > 20, f"only {drowned.sum()} cells drowned"
    assert drowned[typ < 0.05].sum() < 0.05 * drowned.sum(), \
        "land drowned away from any plate-type boundary"


def test_margin_stage_builds_no_coastal_mountains():
    """The stage reshapes the shoreline; it must not raise hills along it.

    Two terms failed this way. A fringing-island term, since removed, used a
    gaussian window centred offshore but wide enough that most of its weight
    sat on the shoreline. The fray then did it again on its own: uncapped fBm
    reaches ~3.5 standard deviations and the slope gain multiplies it, so the
    largest additions landed exactly where the coast is steepest. Both raised
    mountains along the shore instead of moving it, which is why the noise is
    clipped now.
    """
    off = generate(Config(margin_h=0.0, margin_h_fine=0.0, margin_cut_h=0.0,
                          **_MARGIN_KW), verbose=False)
    on = generate(Config(**_MARGIN_KW), verbose=False)
    assert on.height.max() < off.height.max() + 0.1, \
        f"the stage raised a new summit: {off.height.max():.2f} -> {on.height.max():.2f}"
    added = (on.height_pre - off.height_pre).max()
    assert added < 0.7, f"stage added {added:.2f} of elevation in one cell"


def test_margin_stage_does_not_perforate_the_coast():
    """The stage must bend the coastline, not punch holes through it.

    Two ways it did. Fine noise near the zero contour perforates the coast
    rather than displacing it, and a cut window centred inland strands the
    basin behind it instead of opening a channel to the sea. Together, at
    `margin_periods` 14 and the inlets 5 cells inland, this map went from 14
    enclosed pockets of water to 45. Coarser noise and a cut centred on the
    shoreline give 23, and move the shoreline further while doing it.
    """
    def pockets(w):
        """Bodies of water with no connection to the open ocean."""
        land = w.height > 0
        wid = land.shape[1]
        # Labelled on a 3x tiling so a body spanning the x seam stays one body.
        lab, n = ndimage.label(np.concatenate([~land] * 3, axis=1))
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        ocean = int(np.argmax(sizes)) + 1
        mid = lab[:, wid:2 * wid]
        return len([v for v in np.unique(mid) if v not in (0, ocean)])

    off = generate(Config(margin_h=0.0, margin_h_fine=0.0, margin_cut_h=0.0,
                          **_MARGIN_KW), verbose=False)
    on = generate(Config(**_MARGIN_KW), verbose=False)
    assert pockets(on) < 2 * pockets(off), \
        f"stage perforated the coast: {pockets(off)} pockets -> {pockets(on)}"


def test_margin_fray_stays_on_the_margins():
    """The stage must touch continent-ocean margins and nothing else.

    `cont_self`/`cont_other` are nearest-boundary pulls and so are defined
    across the whole map; without the distance gate this would fire on inland
    coasts far from any plate boundary.
    """
    off = generate(Config(margin_h=0.0, margin_h_fine=0.0, margin_cut_h=0.0,
                          **_MARGIN_KW), verbose=False)
    on = generate(Config(**_MARGIN_KW), verbose=False)
    # The stage draws its noise either way, so the two runs stay in step
    # downstream and the only difference is what it added.
    d = np.abs(on.height_pre - off.height_pre)
    t = on.tect
    typ = np.abs(t.cont_self - t.cont_other)
    inband = (typ > 0.8) & (t.dist < on.cfg.margin_zone)
    # The mask is deliberately smooth, so it has a shoulder: at typ 0.2 a fifth
    # of the effect is still expected and wanted. Only well clear of any type
    # mismatch should the stage be silent.
    outside = typ < 0.05
    assert d[inband].max() > 0.02, "stage did nothing at the margins"
    assert d[outside].max() < 0.1 * d[inband].max(), "stage leaked off the margins"
    # The fray is zero-mean and the inlets only carve, so a small net loss is
    # expected; sea level was fixed by quantile well upstream and does not
    # move to compensate. Only a runaway would be a bug.
    assert abs(on.land.mean() - off.land.mean()) < 0.04, \
        f"land area moved by {on.land.mean() - off.land.mean():+.3f}"


def test_config_scales_with_map_size():
    """Pixel knobs follow the map, and the caller's config is left alone."""
    cfg = Config(width=1024, height=768)
    scaled = world._scale_to_size(cfg)
    assert cfg.plate_warp == 34.0, "the caller's config was mutated"
    assert scaled.plate_warp == 34.0 * 2, "a pixel width did not scale"
    # A rise per cell goes the other way: the same climb spread over twice the
    # cells is half the step.
    assert abs(scaled.coast_slope - cfg.coast_slope / 2) < 1e-12
    # At the reference width it must be a no-op.
    assert world._scale_to_size(Config(width=512)).plate_warp == 34.0


def test_river_width_stops_scaling_at_the_cap():
    """Past the cap the whole width scale holds still, not just the ceiling.

    Clamping `river_width_max` alone would leave every tributary scaled up
    against a ceiling they all reach, drawing the network as a mat.
    """
    small = world._scale_to_size(Config(width=640))       # 4 * 1.25 = 5, under
    assert small.river_width_max == 5.0
    assert small.river_width == 0.85 * 1.25
    for wdt in (1280, 1920, 3840):
        big = world._scale_to_size(Config(width=wdt))
        assert big.river_width_max == big.river_width_cap
        # Same factor on both, so trunk and tributary keep their contrast.
        assert abs(big.river_width / big.river_width_max
                   - Config().river_width / Config().river_width_max) < 1e-12


def test_relief_does_not_sharpen_with_map_size():
    """A bigger map must be the same world in more detail, not a harder one.

    `Config` mixes units - `periods` are relative to the map, everything else
    is pixels - so left unscaled, boundary landforms keep a fixed pixel width
    and narrow into hard creases as the map grows while the terrain around
    them stays put. Measured on seed 7, boundary slope over median slope ran
    1.22 / 1.50 / 1.99 at 384 / 640 / 1024 wide.
    """
    def crease(wdt):
        w = generate(Config(width=wdt, height=int(wdt * 3 / 4) // 2 * 2,
                            seed=7, erosion_passes=1), verbose=False)
        s = np.hypot(*grid.gradient(w.height))
        return np.median(s[w.tect.boundary]) / np.median(s)

    small, big = crease(256), crease(512)
    assert abs(big - small) < 0.25, \
        f"boundary relief sharpened with size: {small:.2f} -> {big:.2f}"


def _read_png16(path):
    """Enough of a PNG reader to check what `export` wrote, and no more.

    Deliberately not the writer run backwards: it walks the chunks itself and
    checks every CRC, so a malformed length or a bad checksum fails here rather
    than in whatever engine loads the file next.
    """
    raw = Path(path).read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    chunks, i = {}, 8
    while i < len(raw):
        n = struct.unpack(">I", raw[i:i + 4])[0]
        tag, body = raw[i + 4:i + 8], raw[i + 8:i + 8 + n]
        assert zlib.crc32(tag + body) == struct.unpack(">I", raw[i + 8 + n:i + 12 + n])[0], tag
        chunks[tag] = chunks.get(tag, b"") + body
        i += 12 + n
    w, h, depth, ctype = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert ctype == 0 and depth in (8, 16), f"want greyscale 8/16, got {depth}/{ctype}"
    n = depth // 8
    rows = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(h, w * n + 1)
    assert not rows[:, 0].any(), "every row must carry filter type 0"
    px = rows[:, 1:].copy()
    return px if n == 1 else px.view(">u2")


def test_heightmap_export():
    """Codes must map back to the heights they came from, within half a step."""
    rng = np.random.default_rng(0)
    # Odd dimensions on purpose: rows are written one at a time, and a width
    # that is not a nice multiple is where a stride mistake would show.
    h = rng.normal(0.1, 0.4, (17, 23))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "h.png"
        meta = export.heightmap(h, path)
        codes = _read_png16(path)
        assert codes.shape == h.shape
        back = meta["lo"] + codes * meta["units_per_code"]
        assert np.abs(back - h).max() <= meta["units_per_code"] * 0.51, "lost more than rounding"
        assert json.loads(Path(str(path)[:-4] + ".json").read_text()) == meta
        # Sea level has to be findable in the codes, or the map is unusable.
        assert abs(meta["sea_code"] * meta["units_per_code"] + meta["lo"]) < 1e-9
        # A shared scale is the whole point of lo/hi: same height, same code.
        export.heightmap(h * 0.5, path, meta["lo"], meta["hi"])
        half = _read_png16(path)
        assert np.abs(half[h > 0].astype(int) - codes[h > 0]).min() > 0
        assert half.max() < codes.max(), "second export ignored the handed-in scale"


def test_indexmap_export():
    """Class indices must survive exactly - they are labels, not measurements."""
    ids = np.arange(12, dtype=np.int8).reshape(3, 4)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "b.png"
        meta = export.indexmap(ids, path, climate.BIOME_NAMES)
        assert np.array_equal(_read_png16(path), ids), "indices must not be rescaled"
        assert meta["classes"] == climate.BIOME_NAMES
        # A byte cannot hold more, and silently wrapping would relabel the map.
        try:
            export.indexmap(np.array([[300]]), path, [])
            assert False, "out-of-range indices must raise"
        except ValueError:
            pass


def test_exported_config_rebuilds_the_world():
    """The written config must rebuild the same map, bit for bit.

    Deliberately at a width other than `ref_width`, because that is the only
    place the bug lives: `generate` keeps the *scaled* config on the world, and
    handing that straight back scales the pixel knobs a second time. At the
    reference width the scaling is a no-op and a broken round-trip passes.
    """
    w = generate(Config(seed=5, width=256, height=192, erosion_passes=2), verbose=False)
    assert w.cfg.width != Config().ref_width, "test must run off the reference width"
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cfg.json"
        export.config(w.cfg, path)
        again = generate(export.load_config(path), verbose=False)
    assert np.array_equal(again.height, w.height), "rebuild differs from the export"
    assert np.array_equal(again.biome, w.biome)
    # And the naive version really is wrong, or the re-anchoring above is
    # cargo cult. Same seed, same knobs, scaled twice.
    twice = generate(world.replace(w.cfg), verbose=False)
    assert not np.array_equal(twice.height, w.height), \
        "double-scaling no longer changes the world; re-anchoring may be dead code"


def test_config_stamp_flags_a_different_generator():
    """A config that says nothing about the code that wrote it is a false promise."""
    import warnings as _w
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cfg.json"
        meta = export.config(Config(seed=1), path)
        assert "src:" in meta["generator"] and "numpy:" in meta["generator"]
        with _w.catch_warnings():
            _w.simplefilter("error")        # a matching stamp must stay quiet
            export.load_config(path)
        doctored = json.loads(path.read_text())
        doctored["generator"] = "src:000000000000 numpy:0 scipy:0"
        doctored["a_knob_from_the_future"] = 1
        path.write_text(json.dumps(doctored))
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            cfg = export.load_config(path)   # must still load, warning or not
        assert cfg.seed == 1
        assert len(caught) == 2, [str(c.message) for c in caught]


def test_trees_grow_where_they_should():
    """Cover has to respect the four things that can forbid a tree outright,
    and the scatter has to land on the cover rather than beside it."""
    w = generate(Config(width=192, height=144, seed=4, erosion_passes=2),
                 verbose=False)
    t = w.trees
    assert 0.0 <= t.cover.min() and t.cover.max() <= 1.0
    for name, forbidden in (("sea", ~w.land), ("lake", w.lakes),
                            ("ice", w.biome == climate.ICE),
                            ("above the tree line",
                             (w.temp + w.swing) < climate.TREE_C)):
        assert t.cover[forbidden].max(initial=0.0) == 0.0, f"canopy on {name}"

    assert len(t) > 0, "no trees scattered at all"
    ys, xs = t.pos[:, 0].astype(int), t.pos[:, 1].astype(int) % w.height.shape[1]
    assert w.land[ys, xs].all(), "a tree was placed in the water"
    assert (t.cover[ys, xs] > 0).all(), "a tree was placed on bare cover"
    assert set(np.unique(t.kind)) <= set(range(len(trees.KIND_NAMES)))

    # The moisture axis has to come through: closed-canopy biomes must carry
    # more of it than the dry ones, or the field is noise with a mask on it.
    wet = np.isin(w.biome, [climate.TEMPERATE_FOREST, climate.TROPICAL_RAINFOREST,
                            climate.TEMPERATE_RAINFOREST, climate.TAIGA])
    dry = np.isin(w.biome, [climate.DESERT, climate.STEPPE])
    if wet.any() and dry.any():
        assert t.cover[wet].mean() > 2 * t.cover[dry].mean(), \
            f"forest {t.cover[wet].mean():.2f} vs dry {t.cover[dry].mean():.2f}"


def test_tree_kinds_blend_across_a_boundary():
    """A species boundary must be a belt of mixed stand, not a stencil edge."""
    cfg = Config(width=192, height=144, seed=4, erosion_passes=2)
    w = generate(cfg, verbose=False)
    mix = trees.kind_mix(w.biome, w.land, w.cfg)
    # A distribution over land, which is the only place it is sampled; out at
    # sea it is zero rather than arbitrary.
    assert np.allclose(mix.sum(-1)[w.land], 1.0), "kinds must be a distribution"

    # The interior of a region stays pure - mixing everything would just be a
    # different way of losing the boundary - while a real share of the land is
    # in transition rather than committed to one kind.
    top = mix.max(-1)[w.land]
    assert top.max() > 0.98, "no pure stand anywhere; the mix blurred everything"
    assert (top < 0.9).mean() > 0.15, f"only {(top < 0.9).mean():.0%} of land is mixed"

    # And the trees actually take it: the hard lookup would put exactly one kind
    # on every cell of a given biome, so finding two inside one biome is the
    # whole difference.
    t = w.trees
    ys, xs = t.pos[:, 0].astype(int), t.pos[:, 1].astype(int) % w.height.shape[1]
    b = w.biome[ys, xs]
    mixed = [len(set(t.kind[b == v].tolist())) for v in np.unique(b)
             if (b == v).sum() > 40]
    assert max(mixed) > 1, "every biome grew a single species; kinds did not blend"

    # Turning the knob off has to restore the stencil, or the test above is
    # passing on something other than the mixing.
    hard = generate(Config(**{**cfg.__dict__, "tree_mix": 0.0}), verbose=False)
    hm = trees.kind_mix(hard.biome, hard.land, hard.cfg)
    assert hm.max(-1)[hard.land].min() == 1.0, "tree_mix=0 must be a hard lookup"


def test_gallery_forest_follows_the_rivers():
    """The one thing here that a biome map cannot say: trees somewhere its own
    rainfall would not keep them, because a river runs through it."""
    w = generate(Config(width=192, height=144, seed=4, erosion_passes=2),
                 verbose=False)
    d = grid.edt(~(w.rivers | w.lakes))
    # Dry country only. In a rainforest the term is designed to do nothing, so
    # averaging over all land would drown the effect in ground that was already
    # covered.
    arid = w.land & (w.runoff < np.median(w.runoff[w.land]))
    bank, inland = arid & (d <= 2), arid & (d > 8)
    assert bank.any() and inland.any()
    assert w.trees.cover[bank].mean() > 1.5 * w.trees.cover[inland].mean(), \
        f"bank {w.trees.cover[bank].mean():.2f} vs inland " \
        f"{w.trees.cover[inland].mean():.2f}"
    # And it must stop at the tree line rather than riding a river over it.
    cold = w.land & ((w.temp + w.swing) < climate.TREE_C) & (d <= 2)
    assert w.trees.cover[cold].max(initial=0.0) == 0.0, "gallery crossed the tree line"


def test_biome_legend_matches_the_map():
    """A key is only worth drawing if every biome on the map is in it, and if
    no two swatches are close enough to be mistaken for each other."""
    w = generate(Config(width=256, height=192, seed=7, erosion_passes=2),
                 verbose=False)
    ids = render.present_biomes(w)
    assert ids == sorted(set(w.biome[w.land].tolist()) - {climate.OCEAN}), \
        "the key and the land disagree about which biomes are here"
    # Every pair, not just the ones this seed happens to show: a new colour that
    # collides with an old one has to fail when it is added, not two seeds later.
    c = render.LEGEND_COLORS[1:]
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    i, j = np.unravel_index(d.argmin(), d.shape)
    assert d.min() > 40, (f"{climate.BIOME_NAMES[i + 1]} and "
                          f"{climate.BIOME_NAMES[j + 1]} are {d.min():.0f} apart")
    # Borders run between two land biomes and nowhere else - not round the
    # coast, which the sea ramp already draws, and not across the water.
    edges = render.biome_edges(w)
    assert edges.any(), "no biome borders found"
    assert not (edges & (~w.land | w.rivers | w.lakes)).any(), "border off the land"
    # The key itself has to be on the image, and over the emptiest corner.
    img = render.legend_biome_map(w)
    assert (img == np.round(render.LEGEND_PAPER).astype(np.uint8)).all(-1).any(), \
        "no legend panel was drawn"


def test_world():
    w = generate(Config(width=128, height=96, seed=5, erosion_passes=2), verbose=False)
    assert w.height.shape == (96, 128)
    assert np.isfinite(w.height).all()
    assert 0.15 < w.land.mean() < 0.5, f"land fraction off: {w.land.mean()}"
    # y does not wrap, so the poles must close the world with ocean rather than
    # leaving continents sliced off at the top and bottom edges.
    assert not w.land[[0, 1, -2, -1]].any(), "land runs off the top or bottom edge"
    assert w.tect.plate.min() == 0 and w.tect.plate.max() == w.cfg.n_plates - 1
    assert w.tect.boundary.any() and w.tect.conv.max() > 0
    # Erosion must remove more than it adds on land.
    delta = (w.height_eroded - w.height_pre)[w.land]
    assert delta.mean() < 0, "erosion should lower land on average"
    assert w.rivers.any(), "no rivers formed"
    for i in range(len(render.LAYERS)):
        name, img = render.layer(w, i)
        assert img.shape == (96, 128, 3) and img.dtype == np.uint8, name
    # The whole export bundle, on the world that is already built.
    with tempfile.TemporaryDirectory() as d:
        out = export.bundle(w, d)
        assert out.name == f"seed{w.cfg.seed}"
        for f in ("height16.png", "lakes16.png", "biome.png", "canopy.png",
                  "rivers.json", "config.json"):
            assert (out / f).exists(), f
        # A second export must not land on the first: these are kept, and an
        # overwrite is the one mistake that cannot be undone from outside.
        again = export.bundle(w, d)
        assert again != out and again.name == f"seed{w.cfg.seed}-2", again
        assert (out / "config.json").exists(), "first export was clobbered"
        assert np.array_equal(_read_png16(out / "biome.png"), w.biome), \
            "biome indices did not round-trip"
        js = json.loads((out / "rivers.json").read_text())
        assert len(js["rivers"]) == len(w.water.polylines)
        # Points are [x, y], the transpose of how the arrays hold them; getting
        # that backwards on a non-square map is silent until something loads it.
        first = js["rivers"][0]
        assert np.allclose(first["points"][0][::-1], w.water.polylines[0][0], atol=1e-3)
        assert len(first["width"]) == len(first["points"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
