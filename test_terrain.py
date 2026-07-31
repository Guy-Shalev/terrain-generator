"""Invariant checks for the generator. Run: python test_terrain.py"""
import numpy as np
from scipy import ndimage

from terrain import Config, generate
from terrain import climate, grid, hydrology, noise, render, rivers, world


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
    assert (filled >= h - 1e-9).all(), "filling may only raise the surface"
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
