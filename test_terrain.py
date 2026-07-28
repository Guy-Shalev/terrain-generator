"""Invariant checks for the generator. Run: python test_terrain.py"""
import numpy as np
from scipy import ndimage

from terrain import Config, generate
from terrain import grid, hydrology, noise, render, rivers, world


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
    plain = rivers.carve_outlets(h.copy(), cfg)
    handed = rivers.carve_outlets(h.copy(), cfg, routed=(filled, rec))
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
    """Carving must remove impounded volume, never add it."""
    cfg = Config(width=96, height=96, seed=4)
    rng = np.random.default_rng(4)
    y, x = np.mgrid[0:96, 0:96]
    h = 0.4 - 0.004 * np.hypot(y - 48, x - 48) + 0.05 * rng.normal(0, 1, (96, 96))
    h = grid.blur(h, 3)
    h[36:60, 36:60] -= 0.12  # a basin with no way out
    before = (hydrology.fill_depressions(h) - h).sum()
    after_h = rivers.carve_outlets(h.copy(), cfg)
    after = (hydrology.fill_depressions(after_h) - after_h).sum()
    assert after < before, f"carving did not drain anything: {before} -> {after}"
    assert (after_h <= h + 1e-12).all(), "carving may only lower the surface"


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
    # Discharge only grows downstream, so width must too.
    for pts in wat.polylines:
        q = wat.flow[np.clip(pts[:, 0].astype(int), 0, w.height.shape[0] - 1),
                     pts[:, 1].astype(int) % w.height.shape[1]]
        assert q[-1] >= q[0] * 0.5, "path is not running downstream"
    assert wat.river_mask.any()
    assert wat.width[wat.river_mask].min() > 0
    assert wat.width.max() <= w.cfg.river_width_max + 1e-9
    # Rivers are cut into the bed, never raised above it.
    assert (w.height <= w.height_eroded + 1e-9).all()


def test_meander_stays_attached():
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
